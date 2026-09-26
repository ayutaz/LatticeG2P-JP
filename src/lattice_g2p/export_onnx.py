"""ONNX への 3 グラフ分割エクスポート.

  グラフ 1: context_encoder   (1, L) -> (1, L, H)      可変次元: 文長 L
  グラフ 2: reading_encoder   (M, R) -> (M, H)         可変次元: 読みの種類 M
  グラフ 3: node_scorer       ノード特徴 -> (N,)        可変次元: ノード数 N

★分ける理由は 2 つある.

1. L・M・N は独立に変わる. 1 つにまとめると動的形状の扱いが複雑になる
2. ★**読み表現は入力文に依存しない**ので, グラフ 2 の出力はキャッシュできる
   (docs/architecture.md §4-3). 91 文字の文ではここが 25 ms を占めており,
   キャッシュしないと BiEncoder を選んだ意味がない

★CRF / Viterbi は ONNX に含めない. DAG 構造が動的なため
(docs/architecture.md §7-2). Rust / C++ へ移植するときもこの分担を保つ.
"""

import shutil
from pathlib import Path

import torch
from torch import nn

from lattice_g2p.scorer.neural_base import NeuralScorer, NodeFeatures

OPSET = 17

GRAPHS = ("context_encoder.onnx", "reading_encoder.onnx", "node_scorer.onnx")
"""書き出す ONNX グラフ. ★増やしたら量子化 (quantize) も追随すること."""

TRANSITION_GRAPH = "transition.onnx"
"""CRF の遷移項 (M6). 持つモデルだけ書き出す.

★ホスト側の Viterbi / forward が参照する. ONNX に CRF は入れない
(不変条件 6). 入力は連接 ID の列で, 出力は (N, N) のスコア.
"""

VOCABS = ("text_vocab.json", "kana_vocab.json")

EXPORT_OPTS = {"opset_version": OPSET, "external_data": False, "dynamo": True}
"""★external_data=False で 1 ファイルに閉じる.

既定では重みが .onnx.data に分かれる. 配布と INT8 量子化の両方で,
ファイルが 1 つのほうが扱いやすい (サイズは同じ).
"""

NODE_INPUTS = (
    "context",
    "context_mask",
    "span_starts",
    "span_ends",
    "reading_vec",
    "node_lengths",
    "source_ids",
    "is_unk",
    "batch_index",
    "dict_prior",
)
"""node_scorer.onnx の入力名. NodeFeatures のフィールドと 1 対 1 に対応する.

reading_ids / reading_mask はここに現れない. 読みは reading_encoder.onnx で
先に (N, H) にしてから渡す (キャッシュのため).
"""


class _ContextWrapper(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(ids, mask)


class _TransitionWrapper(nn.Module):
    """遷移スコアラ. ★出力は (N, N) で, ホスト側の Viterbi が参照する."""

    def __init__(self, transition: nn.Module) -> None:
        super().__init__()
        self.transition = transition

    def forward(self, right_ids: torch.Tensor, left_ids: torch.Tensor) -> torch.Tensor:
        right = self.transition.right_emb(right_ids)
        left = self.transition.left_emb(left_ids)
        return (right @ left.T) * self.transition._scale


class _ReadingWrapper(nn.Module):
    """読みエンコーダ. ★出力は入力文に依存しないのでキャッシュできる."""

    def __init__(self, scorer: NeuralScorer) -> None:
        super().__init__()
        self.scorer = scorer

    def forward(self, reading_ids: torch.Tensor, reading_mask: torch.Tensor) -> torch.Tensor:
        return self.scorer.encode_readings(reading_ids, reading_mask)


class _ScorerWrapper(nn.Module):
    """NodeFeatures をテンソル引数に展開したもの.

    NodeFeatures は値を持つだけの入れ物なので, ここで組み立て直しても
    トレースされるグラフは変わらない.
    """

    def __init__(self, scorer: NeuralScorer) -> None:
        super().__init__()
        self.scorer = scorer

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
        reading_vec: torch.Tensor,
        node_lengths: torch.Tensor,
        source_ids: torch.Tensor,
        is_unk: torch.Tensor,
        batch_index: torch.Tensor,
        dict_prior: torch.Tensor,
    ) -> torch.Tensor:
        features = NodeFeatures(
            span_starts=span_starts,
            span_ends=span_ends,
            reading_ids=torch.zeros(0),  # 未使用 (読みは reading_vec で渡す)
            reading_mask=torch.zeros(0),
            node_lengths=node_lengths,
            source_ids=source_ids,
            is_unk=is_unk,
            batch_index=batch_index,
            dict_prior=dict_prior,
        )
        return self.scorer.score_from_reading_vecs(
            context, context_mask, features, reading_vec
        )


def _dummy_node_inputs(hidden: int, length: int, n: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(1, length, hidden),
        torch.ones(1, length, dtype=torch.bool),
        torch.zeros(n, dtype=torch.long),
        torch.ones(n, dtype=torch.long),
        torch.zeros(n, hidden),
        torch.ones(n),
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n),
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n),
    )


def export_modules(scorer: NeuralScorer, out_dir: Path) -> None:
    """学習済みスコアラを 2 つの ONNX グラフと語彙に書き出す."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer.eval()

    scorer.text_vocab.save(out_dir / "text_vocab.json")
    scorer.kana_vocab.save(out_dir / "kana_vocab.json")

    hidden = scorer.context_encoder.hidden_size

    with torch.no_grad():
        torch.onnx.export(
            _ContextWrapper(scorer.context_encoder),
            (torch.ones(1, 8, dtype=torch.long), torch.ones(1, 8, dtype=torch.bool)),
            str(out_dir / "context_encoder.onnx"),
            input_names=["ids", "mask"],
            output_names=["context"],
            dynamic_axes={"ids": {1: "L"}, "mask": {1: "L"}, "context": {1: "L"}},
            **EXPORT_OPTS,
        )

        torch.onnx.export(
            _ReadingWrapper(scorer),
            (torch.ones(3, 4, dtype=torch.long), torch.ones(3, 4, dtype=torch.bool)),
            str(out_dir / "reading_encoder.onnx"),
            input_names=["reading_ids", "reading_mask"],
            output_names=["reading_vec"],
            dynamic_axes={
                "reading_ids": {0: "M", 1: "R"},
                "reading_mask": {0: "M", 1: "R"},
                "reading_vec": {0: "M"},
            },
            **EXPORT_OPTS,
        )

        if getattr(scorer, "transition", None) is not None:
            torch.onnx.export(
                _TransitionWrapper(scorer.transition),
                (torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)),
                str(out_dir / TRANSITION_GRAPH),
                input_names=["right_ids", "left_ids"],
                output_names=["transitions"],
                dynamic_axes={
                    "right_ids": {0: "N"},
                    "left_ids": {0: "N"},
                    "transitions": {0: "N", 1: "N"},
                },
                **EXPORT_OPTS,
            )

        torch.onnx.export(
            _ScorerWrapper(scorer),
            _dummy_node_inputs(hidden, length=8, n=3),
            str(out_dir / "node_scorer.onnx"),
            input_names=list(NODE_INPUTS),
            output_names=["scores"],
            dynamic_axes={
                "context": {1: "L"},
                "context_mask": {1: "L"},
                "span_starts": {0: "N"},
                "span_ends": {0: "N"},
                "reading_vec": {0: "N"},
                "node_lengths": {0: "N"},
                "source_ids": {0: "N"},
                "is_unk": {0: "N"},
                "batch_index": {0: "N"},
                "dict_prior": {0: "N"},
                "scores": {0: "N"},
            },
            **EXPORT_OPTS,
        )


KEEP_FP32 = ("reading_encoder.onnx",)
"""量子化しないグラフ（R-26）.

★動的量子化は活性の尺度をバッチ全体の min / max で決める. 読みエンコーダの出力は
キャッシュされるので、INT8 にすると同じ読みでも「どの読みと同じバッチで最初に
計算されたか」でベクトルが変わり、**同じ文のスコアが評価の履歴に依存する**
（同じ文で最大 0.12 ずれた）. 読みはキャッシュされるので、FP32 のままでも速度への
影響は初めて見る読みを含む文だけに限られる. サイズは size-m で 2.0 -> 7.0 MB.
"""


def quantize(src_dir: Path, dst_dir: Path) -> None:
    """動的量子化で INT8 にする. ★読みエンコーダは FP32 のまま写す（KEEP_FP32）.

    ★サイズだけ見て採用しない. 量子化後の精度は必ず測り直すこと
    (docs/architecture.md §7-3).
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in GRAPHS + (TRANSITION_GRAPH,):
        if not (src_dir / name).exists():
            continue  # ★遷移項を持たないモデルでは transition.onnx が無い
        if name in KEEP_FP32:
            shutil.copyfile(src_dir / name, dst_dir / name)
            continue
        quantize_dynamic(
            model_input=src_dir / name,
            model_output=dst_dir / name,
            weight_type=QuantType.QInt8,
        )
    for name in VOCABS:
        shutil.copyfile(src_dir / name, dst_dir / name)

