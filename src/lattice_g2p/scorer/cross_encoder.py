"""クロスエンコーダ版スコアラ (論文準拠 / 精度の基準).

    span 表現 + 読み表現 -> Transformer Encoder ×2
                         -> Transformer Decoder ×2（文全体 Context と Cross Attention）
                         -> Pooling -> score

精度の基準 (reference implementation) として置き, BiEncoder の劣化を測る対象にする.
"""

import torch
from torch import nn

from lattice_g2p.scorer.neural_base import (
    N_SOURCES,
    NeuralScorer,
    NodeFeatures,
    span_pool,
)
from lattice_g2p.vocab import CharVocab


class CrossEncoderScorer(NeuralScorer):
    def __init__(
        self,
        context_encoder: nn.Module,
        reading_encoder: nn.Module,
        text_vocab: CharVocab,
        kana_vocab: CharVocab,
        transition_dim: int = 0,
        transition_ids: int = 0,
        layers: int = 2,
        heads: int = 8,
        ffn: int = 2048,
        dropout: float = 0.1,
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
        self.reading_proj = nn.Linear(reading_encoder.hidden_size, hidden)
        self.source_embed = nn.Embedding(N_SOURCES, hidden)
        self.extra_proj = nn.Linear(2, hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pair_encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

        # 0 初期化にすると source が初手で効かない. 小さな値から始める
        nn.init.normal_(self.source_embed.weight, std=0.01)

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
        span_vec = span_pool(
            context, features.batch_index, features.span_starts, features.span_ends
        )
        extra = self.extra_proj(
            torch.stack([features.node_lengths, features.is_unk], dim=-1)
        )
        source_vec = self.source_embed(features.source_ids)

        # (N, 3, H): [span, reading, その他の特徴] の 3 トークン列として扱う
        pair = torch.stack([span_vec, reading_vec + source_vec, extra], dim=1)
        pair = self.pair_encoder(pair)

        memory = context[features.batch_index]  # (N, L, H)
        memory_padding = ~context_mask[features.batch_index]  # (N, L)
        out = self.decoder(pair, memory, memory_key_padding_mask=memory_padding)

        return self.add_dict_prior(self.head(out.mean(dim=1)).squeeze(-1), features)
