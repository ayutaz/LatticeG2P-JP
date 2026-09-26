import pytest
import torch

from lattice_g2p.encoder import ReadingEncoder, ScratchCharEncoder, pad_batch
from lattice_g2p.vocab import CharVocab


def test_pad_batch_shapes() -> None:
    ids, mask = pad_batch([[1, 2, 3], [4]], torch.device("cpu"))

    assert ids.shape == (2, 3)
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert ids[1, 1:].tolist() == [0, 0], "パディングは PAD=0"


def test_pad_batch_empty_list() -> None:
    ids, mask = pad_batch([], torch.device("cpu"))
    assert ids.shape[0] == 0
    assert mask.shape[0] == 0


def test_pad_batch_with_empty_sequence() -> None:
    """空の読み（句読点のノード）があっても落ちないこと。"""
    ids, mask = pad_batch([[], [1, 2]], torch.device("cpu"))

    assert ids.shape == (2, 2)
    assert mask[0].tolist() == [False, False]


def test_context_encoder_output_shape() -> None:
    enc = ScratchCharEncoder(vocab_size=100, hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    ids = torch.randint(2, 100, (3, 10))
    mask = torch.ones(3, 10, dtype=torch.bool)

    out = enc(ids, mask)

    assert out.shape == (3, 10, 32)
    assert enc.hidden_size == 32


def test_context_encoder_respects_padding() -> None:
    """★パディング位置が実トークンの表現に影響しないこと。

    ここが壊れていると、バッチの組み方で結果が変わる。精度劣化として
    現れるが原因が分かりにくい。
    """
    enc = ScratchCharEncoder(vocab_size=100, hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    enc.eval()

    ids_short = torch.tensor([[5, 6, 7]])
    mask_short = torch.ones(1, 3, dtype=torch.bool)
    ids_pad = torch.tensor([[5, 6, 7, 0, 0]])
    mask_pad = torch.tensor([[True, True, True, False, False]])

    with torch.no_grad():
        a = enc(ids_short, mask_short)
        b = enc(ids_pad, mask_pad)[:, :3]

    assert torch.allclose(a, b, atol=1e-5)


def test_context_encoder_zeroes_padding_output() -> None:
    enc = ScratchCharEncoder(vocab_size=100, hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    enc.eval()
    ids = torch.tensor([[5, 6, 0, 0]])
    mask = torch.tensor([[True, True, False, False]])

    with torch.no_grad():
        out = enc(ids, mask)

    assert torch.all(out[0, 2:] == 0)


def test_context_encoder_is_differentiable() -> None:
    enc = ScratchCharEncoder(vocab_size=100, hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    out = enc(torch.randint(2, 100, (2, 5)), torch.ones(2, 5, dtype=torch.bool))
    out.sum().backward()

    assert any(p.grad is not None for p in enc.parameters())


def test_reading_encoder_pools_to_one_vector() -> None:
    enc = ReadingEncoder(vocab_size=120, hidden=32, layers=1, heads=2, ffn=64, max_len=32)
    ids = torch.randint(2, 120, (5, 6))
    mask = torch.ones(5, 6, dtype=torch.bool)

    assert enc(ids, mask).shape == (5, 32)


def test_reading_encoder_ignores_padding_in_pooling() -> None:
    """★平均プーリングがパディングを含めないこと。"""
    enc = ReadingEncoder(vocab_size=120, hidden=32, layers=1, heads=2, ffn=64, max_len=32)
    enc.eval()

    short = enc(torch.tensor([[5, 6]]), torch.ones(1, 2, dtype=torch.bool))
    padded = enc(
        torch.tensor([[5, 6, 0, 0]]),
        torch.tensor([[True, True, False, False]]),
    )

    with torch.no_grad():
        assert torch.allclose(short, padded, atol=1e-5)


def test_reading_encoder_handles_all_padding_row() -> None:
    """空の読み（句読点）は全部パディングになる。0 除算しないこと。"""
    enc = ReadingEncoder(vocab_size=120, hidden=32, layers=1, heads=2, ffn=64, max_len=32)

    out = enc(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.bool))

    assert out.shape == (1, 32)
    assert torch.isfinite(out).all()


def test_reading_encoder_is_small() -> None:
    """★カナ文字種に比例するサイズであること（埋め込みテーブルにしない証拠）。"""
    enc = ReadingEncoder(vocab_size=120, hidden=256, layers=2, heads=4, ffn=1024, max_len=32)
    n = sum(p.numel() for p in enc.parameters())

    assert n < 3_000_000, f"{n} params: 大きすぎる"


def test_scratch_encoder_param_count_in_target_range() -> None:
    """tiny 構成が 3〜10M に収まること（目標とした規模）。"""
    enc = ScratchCharEncoder(vocab_size=6000, hidden=256, layers=4, heads=4, ffn=1024, max_len=512)
    n = sum(p.numel() for p in enc.parameters())

    assert 1_000_000 < n < 10_000_000, f"{n / 1e6:.1f}M"


def test_encoders_are_onnx_exportable() -> None:
    """★動的な Python 制御フローを forward に入れないこと（M4 で ONNX にする）。

    M4 まで待つと「実は export できない構造だった」と分かったときの手戻りが大きい。
    """
    import io

    pytest.importorskip("onnxscript", reason="uv sync --extra onnx が必要")

    enc = ScratchCharEncoder(vocab_size=100, hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    enc.eval()
    buf = io.BytesIO()

    torch.onnx.export(
        enc,
        (torch.ones(1, 8, dtype=torch.long), torch.ones(1, 8, dtype=torch.bool)),
        buf,
        input_names=["ids", "mask"],
        output_names=["context"],
        dynamic_axes={"ids": {1: "L"}, "mask": {1: "L"}, "context": {1: "L"}},
        opset_version=17,
    )

    assert buf.tell() > 0


def test_all_masked_row_does_not_produce_nan() -> None:
    """★全位置がマスクされた行があっても nan にならないこと。

    句読点の読みは空文字なので、パディングだけの行になる。
    attention の softmax が全マスク行で nan を返すのが原因で、
    しかも「動くが nan が混ざる」という形で現れるため気づきにくい。
    """
    enc = ScratchCharEncoder(vocab_size=100, hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    enc.eval()

    ids = torch.tensor([[5, 6, 0], [0, 0, 0]])
    mask = torch.tensor([[True, True, False], [False, False, False]])

    with torch.no_grad():
        out = enc(ids, mask)

    assert torch.isfinite(out).all(), "全マスク行が nan を生んでいる"
    assert torch.all(out[1] == 0), "全マスク行の出力は 0 になること"


def test_all_masked_row_does_not_break_other_rows() -> None:
    """全マスク行が混ざっても、他の行の結果が変わらないこと。"""
    enc = ScratchCharEncoder(vocab_size=100, hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    enc.eval()

    with torch.no_grad():
        alone = enc(torch.tensor([[5, 6]]), torch.tensor([[True, True]]))
        mixed = enc(
            torch.tensor([[5, 6], [0, 0]]),
            torch.tensor([[True, True], [False, False]]),
        )

    assert torch.allclose(alone[0], mixed[0], atol=1e-5)


def test_reading_encoder_all_masked_row_is_finite() -> None:
    enc = ReadingEncoder(vocab_size=120, hidden=32, layers=2, heads=2, ffn=64, max_len=32)
    enc.eval()

    with torch.no_grad():
        out = enc(
            torch.tensor([[5, 6], [0, 0]]),
            torch.tensor([[True, True], [False, False]]),
        )

    assert torch.isfinite(out).all()


# --- 事前学習エンコーダ（M3c Task 4。研究用の参照であり出荷経路ではない） ---


class _FakeTokenizer:
    """HF トークナイザの最小の模造。ネットワークに触らずに対応表を検証する。"""

    pad_token_id = 0
    unk_token_id = 3

    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, self.unk_token_id)


def test_build_remap_maps_our_ids_to_hf_ids() -> None:
    """★語彙 ID の対応表が正しいこと。ここがずれると全文字が別の字になる。"""
    from lattice_g2p.encoder import build_remap

    vocab = CharVocab(["米", "国"])  # 米 -> 2, 国 -> 3（PAD=0, UNK=1）
    tok = _FakeTokenizer({"米": 100, "国": 200})

    remap = build_remap(tok, vocab)

    assert remap[CharVocab.PAD] == tok.pad_token_id
    assert remap[CharVocab.UNK] == tok.unk_token_id
    assert remap[2] == 100
    assert remap[3] == 200
    assert len(remap) == len(vocab)


def test_build_remap_falls_back_to_unk() -> None:
    """★事前学習モデルの語彙に無い文字は UNK に寄せる。

    こちらの語彙は学習データと辞書から作るので、向こうに無い字が必ず出る。
    """
    from lattice_g2p.encoder import build_remap

    vocab = CharVocab(["米", "𠮟"])
    tok = _FakeTokenizer({"米": 100})

    remap = build_remap(tok, vocab)

    assert remap[2] == 100
    assert remap[3] == tok.unk_token_id
