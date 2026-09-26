"""ニューラルスコアラの共通部分.

★ノードスコアリングは必ず一括で行う (docs/pitfalls.md R-12).
1 文につき ContextEncoder は 1 回だけ, ノードは全件まとめて 1 回.

  for node in nodes: model(node)   ← 動くが 1 文で数百 ms を超える

★テンソルだけを受ける score_from_tensors() を経路の中心にする.
ONNX export の対象がここになるため, Python の制御フローを入れない
(docs/architecture.md §7-2).
"""

import torch
from torch import nn

from lattice_g2p.encoder import pad_batch
from lattice_g2p.lattice import Node
from lattice_g2p.scorer.transition import DEFAULT_CONTEXT_IDS, TransitionScorer
from lattice_g2p.vocab import CharVocab

SOURCE_IDS = {"": 0, "pron": 1, "kana": 2, "both": 3, "aug": 4}
"""読みの由来を ID にする.

★pron 由来と kana 由来の読みは同じコストになり, それだけでは区別できない.
ベースライン (UnigramScorer) では, これが最大の誤り要因だった
(docs/architecture.md §3-1).
"""

N_SOURCES = len(SOURCE_IDS)

DICT_COST_SCALE = 100.0
"""辞書コストを割る数. UnigramScorer の cost_scale と揃える.

★UniDic のコストは頻度の代理指標であり,「その表記のその読みがどれくらい
一般的か」という強い事前分布になる. UnigramScorer はこれだけで
benchmark の対象語で 93% 台を出す (docs/results.md §5).

ニューラルスコアラはこれをまったく与えられていなかったため,
`虎 -> ドラ`・`庫 -> コ` のように低頻度の読みを根拠なく選んでいた.
"""

DICT_PRIOR_INIT = 0.05
"""辞書事前分布の重みの初期値.

1.0 にすると UnigramScorer の順位付けをそのまま使うことになり,
ニューラル側のスコア (O(1)) が埋もれる. 小さく始めて学習させる.
"""

EMPTY_READING_ID = CharVocab.UNK
"""空の読み (句読点) を表す ID.

読み用の語彙は固定のカタカナなので, 正規化済みの読みに UNK が現れることはない.
その空きスロットを「読みなし」に流用する.

空列のまま渡すと全位置がマスクされた行になり, 下流で扱いが面倒になる
(encoder 側でも nan を防いでいるが, 表現として 0 ベクトルになってしまう).
"""


class NodeFeatures:
    """ノード列をテンソルに変換したもの.

    ラティスに依存しない形にしておくと, ONNX の入力と 1 対 1 で対応する.
    """

    def __init__(
        self,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
        reading_ids: torch.Tensor,
        reading_mask: torch.Tensor,
        node_lengths: torch.Tensor,
        source_ids: torch.Tensor,
        is_unk: torch.Tensor,
        batch_index: torch.Tensor,
        dict_prior: torch.Tensor,
    ) -> None:
        self.span_starts = span_starts
        self.span_ends = span_ends
        self.reading_ids = reading_ids
        self.reading_mask = reading_mask
        self.node_lengths = node_lengths
        self.source_ids = source_ids
        self.is_unk = is_unk
        self.batch_index = batch_index
        self.dict_prior = dict_prior
        """-cost / DICT_COST_SCALE. 大きいほど一般的な読み."""


class NeuralScorer(nn.Module):
    """文脈エンコーダと読みエンコーダを共有する基底."""

    def __init__(
        self,
        context_encoder: nn.Module,
        reading_encoder: nn.Module,
        text_vocab: CharVocab,
        kana_vocab: CharVocab,
        transition_dim: int = 0,
        transition_ids: int = 0,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.reading_encoder = reading_encoder
        self.text_vocab = text_vocab
        self.kana_vocab = kana_vocab
        self.dict_prior_weight = nn.Parameter(torch.full((1,), DICT_PRIOR_INIT))
        """辞書事前分布をどれだけ効かせるか. ★学習可能。データに決めさせる."""

        self.transition: TransitionScorer | None = None
        """CRF の遷移項 (M6). None なら経路スコアはノードスコアの単純和.

        ★MeCab の連接行列 (480 MB) に相当する部分を低次元に埋め込んで学習する
        (docs/architecture.md §4-9).
        """
        if transition_dim > 0:
            self.transition = TransitionScorer(
                n_context_ids=transition_ids or DEFAULT_CONTEXT_IDS, dim=transition_dim
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # --- 特徴量の組み立て ---

    def encode_context(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = pad_batch([self.text_vocab.encode(t) for t in texts], self.device)
        return self.context_encoder(ids, mask), mask

    def build_features(
        self, nodes_per_text: list[list[Node]], width: int
    ) -> NodeFeatures:
        """ノード列を ONNX の入力と同じ形のテンソルにする."""
        flat: list[Node] = []
        batch_index: list[int] = []
        for b, nodes in enumerate(nodes_per_text):
            flat.extend(nodes)
            batch_index.extend([b] * len(nodes))

        device = self.device

        del width  # span は境界だけで表す (span_pool の docstring を参照)

        reading_ids, reading_mask = pad_batch(
            [
                self.kana_vocab.encode(node.reading) or [EMPTY_READING_ID]
                for node in flat
            ],
            device,
        )

        return NodeFeatures(
            span_starts=torch.tensor(
                [node.start for node in flat], dtype=torch.long, device=device
            ),
            span_ends=torch.tensor(
                [node.end for node in flat], dtype=torch.long, device=device
            ),
            reading_ids=reading_ids,
            reading_mask=reading_mask,
            node_lengths=torch.tensor(
                [float(node.end - node.start) for node in flat],
                dtype=torch.float32,
                device=device,
            ),
            source_ids=torch.tensor(
                [SOURCE_IDS.get(node.source, 0) for node in flat],
                dtype=torch.long,
                device=device,
            ),
            is_unk=torch.tensor(
                [1.0 if node.entry_id < 0 else 0.0 for node in flat],
                dtype=torch.float32,
                device=device,
            ),
            batch_index=torch.tensor(batch_index, dtype=torch.long, device=device),
            dict_prior=torch.tensor(
                [-node.cost / DICT_COST_SCALE for node in flat],
                dtype=torch.float32,
                device=device,
            ),
        )

    # --- スコアリング ---

    def transitions(self, nodes: list[Node]) -> torch.Tensor | None:
        """(N, N) の遷移スコア. 遷移項を持たない構成では None.

        ★crf の各関数に transitions=None を渡すと従来の実装がそのまま動く.
        """
        return None if self.transition is None else self.transition(nodes)

    def score(self, text: str, nodes: list[Node]) -> torch.Tensor:
        """1 文分のスコア. 推論経路."""
        return self.score_batch([text], [nodes])[0]

    def score_batch(
        self, texts: list[str], nodes_per_text: list[list[Node]]
    ) -> list[torch.Tensor]:
        """複数文のスコアを一括で計算する. 学習ループはこちらを使う."""
        if not any(nodes_per_text):
            return [torch.zeros(0, device=self.device) for _ in texts]

        context, context_mask = self.encode_context(texts)
        features = self.build_features(nodes_per_text, context.shape[1])
        scores = self.score_from_tensors(context, context_mask, features)

        out: list[torch.Tensor] = []
        offset = 0
        for nodes in nodes_per_text:
            out.append(scores[offset : offset + len(nodes)])
            offset += len(nodes)
        return out

    def encode_readings(
        self, reading_ids: torch.Tensor, reading_mask: torch.Tensor
    ) -> torch.Tensor:
        """読みだけから (N, H) の表現を作る.

        ★入力文に依存しない. 推論時はここをキャッシュできる
        (docs/architecture.md §4-3). 91 文字の文で 25 ms を占めており,
        キャッシュしないと BiEncoder を選んだ意味がない.
        """
        raise NotImplementedError

    def score_from_reading_vecs(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        features: NodeFeatures,
        reading_vec: torch.Tensor,
    ) -> torch.Tensor:
        """読み表現を受け取ってスコアにする. 文脈に依存する部分."""
        raise NotImplementedError

    def score_from_tensors(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        features: NodeFeatures,
    ) -> torch.Tensor:
        return self.score_from_reading_vecs(
            context,
            context_mask,
            features,
            self.encode_readings(features.reading_ids, features.reading_mask),
        )

    def add_dict_prior(
        self, scores: torch.Tensor, features: NodeFeatures
    ) -> torch.Tensor:
        """辞書の頻度事前分布を足す.

        ★各スコアラの score_from_tensors の最後で呼ぶこと.
        ONNX のグラフに含めるため, score_batch 側では足さない.
        """
        return scores + self.dict_prior_weight * features.dict_prior


def span_pool(
    context: torch.Tensor,
    batch_index: torch.Tensor,
    span_starts: torch.Tensor,
    span_ends: torch.Tensor,
) -> torch.Tensor:
    """(B, L, H) から各ノードの span を平均プーリングして (N, H) にする.

    ★累積和で O(N·H) にしている. (N, L) のマスクを掛けて足す素直な実装は
    O(N·L·H) で, 91 文字 / 391 ノードだと 36 MB の中間テンソルを作って
    1 文あたり 11 ms かかっていた (docs/architecture.md §4-3).

    span は必ず連続区間なので, 区間和は累積和の差で取れる.
    """
    zeros = context.new_zeros((context.shape[0], 1, context.shape[2]))
    cum = torch.cat([zeros, context.cumsum(dim=1)], dim=1)  # (B, L+1, H)
    lengths = (span_ends - span_starts).clamp(min=1).unsqueeze(-1).to(context.dtype)
    return (cum[batch_index, span_ends] - cum[batch_index, span_starts]) / lengths
