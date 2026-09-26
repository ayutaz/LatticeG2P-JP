"""バイエンコーダ版スコアラ (軽量 / 速度の攻め手).

    score(v) = <u_span, v_reading> / sqrt(d) + 各種バイアス

★読み表現は入力文に依存しないため, 辞書エントリごとに事前計算・キャッシュできる.
推論時はノードごとに内積 1 回で済む (docs/architecture.md §4-3).

精度上の懸念: 読みと文脈の相互作用を内積でしか表現できない.
どこまで落ちるかは CrossEncoder と同条件で比較して測る.
"""

import math

import torch
from torch import nn

from lattice_g2p.scorer.neural_base import (
    N_SOURCES,
    NeuralScorer,
    NodeFeatures,
    span_pool,
)
from lattice_g2p.vocab import CharVocab


class BiEncoderScorer(NeuralScorer):
    def __init__(
        self,
        context_encoder: nn.Module,
        reading_encoder: nn.Module,
        text_vocab: CharVocab,
        kana_vocab: CharVocab,
        transition_dim: int = 0,
        transition_ids: int = 0,
    ) -> None:
        super().__init__(
            context_encoder,
            reading_encoder,
            text_vocab,
            kana_vocab,
            transition_dim=transition_dim,
            transition_ids=transition_ids,
        )
        hidden = context_encoder.hidden_size
        self.span_proj = nn.Linear(hidden, hidden)
        self.reading_proj = nn.Linear(reading_encoder.hidden_size, hidden)
        self.source_bias = nn.Embedding(N_SOURCES, 1)
        self.length_bias = nn.Parameter(torch.zeros(1))
        self.unk_bias = nn.Parameter(torch.zeros(1))
        self._scale = 1.0 / math.sqrt(hidden)

        # 0 初期化にすると source が初手で効かない. 小さな値から始める
        nn.init.normal_(self.source_bias.weight, std=0.01)

    def encode_readings(
        self, reading_ids: torch.Tensor, reading_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.reading_proj(self.reading_encoder(reading_ids, reading_mask))

    def score_from_reading_vecs(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        features: NodeFeatures,
        reading_vec: torch.Tensor,
    ) -> torch.Tensor:
        del context_mask  # バイエンコーダは文脈全体を参照しない

        u = self.span_proj(
            span_pool(
                context, features.batch_index, features.span_starts, features.span_ends
            )
        )

        scores = (
            (u * reading_vec).sum(dim=-1) * self._scale
            + self.source_bias(features.source_ids).squeeze(-1)
            + self.length_bias * features.node_lengths
            + self.unk_bias * features.is_unk
        )
        return self.add_dict_prior(scores, features)
