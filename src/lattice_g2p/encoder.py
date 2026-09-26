"""文脈エンコーダと読みエンコーダ.

★ONNX export 可能な構造を保つこと (docs/architecture.md §7-2).
forward に動的な Python 制御フローを入れない.

★パディング位置が実トークンの表現に影響しないこと.
ここが壊れるとバッチの組み方で結果が変わり, 精度劣化として現れるが
原因が分かりにくい.
"""

from typing import Protocol

import torch
from torch import nn

from lattice_g2p.vocab import CharVocab


def pad_batch(
    seqs: list[list[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """可変長の id 列をパディングして (ids, mask) にする.

    空の列 (句読点の読みなど) が混ざっても落ちないこと.
    """
    if not seqs:
        return (
            torch.zeros((0, 1), dtype=torch.long, device=device),
            torch.zeros((0, 1), dtype=torch.bool, device=device),
        )

    width = max(1, max(len(s) for s in seqs))
    ids = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), width), dtype=torch.bool, device=device)
    for i, seq in enumerate(seqs):
        if seq:
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, : len(seq)] = True
    return ids, mask


class ContextEncoder(Protocol):
    """文全体を文字単位の文脈表現に変換する."""

    hidden_size: int

    def __call__(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, L) -> (B, L, H)"""
        ...


class _TransformerStack(nn.Module):
    """埋め込み + 位置埋め込み + TransformerEncoder."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        layers: int,
        heads: int,
        ffn: int,
        max_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        h = self.dropout(self.norm(self.embed(ids) + self.pos(positions)))

        # ★全位置がマスクされた行があると attention の softmax が nan を返す.
        #   句読点の読みは空文字なので, パディングだけの行が普通に発生する.
        #   便宜的に先頭を有効にして計算し, 出力は元のマスクで 0 にする.
        #   「動くが nan が混ざる」という形で現れるため気づきにくいバグ.
        safe_mask = mask | (~mask.any(dim=1, keepdim=True))

        h = self.encoder(h, src_key_padding_mask=~safe_mask)
        # パディング位置の出力を 0 にする. 下流のプーリングを単純にするため
        return h * mask.unsqueeze(-1)


class ScratchCharEncoder(nn.Module):
    """スクラッチ学習する文字単位の文脈エンコーダ.

    事前学習モデルを使わない理由は docs/pitfalls.md R-08 を参照
    (要点: ku-nlp の文字単位モデルは CC BY-SA 4.0 で, 寛容なライセンスでの配布と両立しない).

    tiny 構成の目安: hidden=256, layers=4, heads=4, ffn=1024 -> 約 5M params
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int = 256,
        layers: int = 4,
        heads: int = 4,
        ffn: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stack = _TransformerStack(vocab_size, hidden, layers, heads, ffn, max_len, dropout)
        self.hidden_size = hidden

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.stack(ids, mask)


class ReadingEncoder(nn.Module):
    """カナ列 -> 1 本のベクトル.

    ★(表記, 読み) ペアごとの埋め込みテーブルにしないこと.
    151,000 entry × 256 次元 × 4 byte = 約 155 MB になり, 軽量化目標を
    単独で破綻させる (docs/architecture.md §4-4).

    パラメータはカナ文字種 (約 100) に比例するだけで済み,
    結果は辞書エントリごとにキャッシュできる.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int = 256,
        layers: int = 2,
        heads: int = 4,
        ffn: int = 1024,
        max_len: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stack = _TransformerStack(vocab_size, hidden, layers, heads, ffn, max_len, dropout)
        self.hidden_size = hidden

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.stack(ids, mask)
        # 空の読み (句読点) は全部パディングになる. 0 除算を避ける
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return h.sum(dim=1) / lengths


# --- 事前学習エンコーダ（研究用の参照） ---
#
# ★出荷経路ではない. ku-nlp の文字単位モデルは CC BY-SA 4.0 (copyleft) で、
# 寛容なライセンスで配布する経路には使えない (docs/pitfalls.md R-08)。
# 「事前学習ありでどこまで出るか」という上限を測るためだけに使う。


def build_remap(tokenizer, vocab: CharVocab) -> list[int]:  # noqa: ANN001
    """こちらの語彙 ID -> 事前学習モデルの語彙 ID の対応表を作る.

    ★ここがずれると全文字が別の字として入力され、原因の分からない
    精度低下になる。PAD と UNK の位置は CharVocab の定義に合わせること。

    こちらの語彙は学習データと辞書から作るので、向こうの語彙に無い文字が
    必ず出る。それらは UNK に寄せる。
    """
    remap = [0] * len(vocab)
    remap[CharVocab.PAD] = tokenizer.pad_token_id
    remap[CharVocab.UNK] = tokenizer.unk_token_id
    for i, ch in enumerate(vocab.chars):
        token_id = tokenizer.convert_tokens_to_ids(ch)
        remap[i + 2] = tokenizer.unk_token_id if token_id is None else token_id
    return remap


class PretrainedCharEncoder(nn.Module):
    """文字単位の事前学習モデルを ContextEncoder として使う.

    ★トークナイザが純粋な文字単位であることが前提。サブワード分割だと
    文字位置とラティスの位置が 1 対 1 に対応しない (docs/pitfalls.md R-08)。
    ku-nlp/deberta-v2-tiny-japanese-char-wwm は実測で文字単位であることを確認済み。
    """

    def __init__(
        self,
        model_name: str,
        vocab: CharVocab,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # 任意依存 (--extra hf)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.model.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "remap", torch.tensor(build_remap(tokenizer, vocab), dtype=torch.long)
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.model(
            input_ids=self.remap[ids], attention_mask=mask.long()
        ).last_hidden_state
        return self.dropout(out) * mask.unsqueeze(-1)
