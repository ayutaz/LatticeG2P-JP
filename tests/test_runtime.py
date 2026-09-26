"""ONNX export と ONNX Runtime 推論のテスト.

★test_onnx_scores_match_pytorch が最重要.
ここがずれていると, 精度が落ちたときに「量子化のせい」か「export バグ」かを
区別できなくなる (docs/architecture.md §7-2).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from lattice_g2p.crf import viterbi, viterbi_numpy
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.encoder import ReadingEncoder, ScratchCharEncoder
from lattice_g2p.lattice import build_lattice
from lattice_g2p.scorer.bi_encoder import BiEncoderScorer
from lattice_g2p.scorer.cross_encoder import CrossEncoderScorer
from lattice_g2p.vocab import CharVocab

ort = pytest.importorskip("onnxruntime", reason="onnxruntime は --extra runtime で入る")

from lattice_g2p.export_onnx import export_modules  # noqa: E402
from lattice_g2p.runtime import OnnxG2P  # noqa: E402

TEXT = "米国の記者。"


def _dictionary() -> Dictionary:
    entries = [
        Entry(surface="米", reading="コメ", pos="名詞", entry_id=0, cost=5000, source="kana"),
        Entry(surface="米", reading="ベイ", pos="名詞", entry_id=1, cost=3000, source="kana"),
        Entry(surface="米国", reading="ベイコク", pos="名詞", entry_id=2, cost=2000, source="both"),
        Entry(surface="国", reading="クニ", pos="名詞", entry_id=3, cost=4000, source="pron"),
        Entry(surface="の", reading="ノ", pos="助詞", entry_id=4, cost=100, source="both"),
        Entry(surface="記者", reading="キシャ", pos="名詞", entry_id=5, cost=2500, source="both"),
        Entry(surface="。", reading="", pos="補助記号", entry_id=6, cost=10, source="pron"),
    ]
    return Dictionary.from_entries(entries)


def _build(scorer_cls, tmp_path: Path):  # noqa: ANN001, ANN202
    torch.manual_seed(0)
    text_vocab = CharVocab.build([TEXT])
    kana_vocab = CharVocab.build(["コメベイクニノキシャ"])
    ctx = ScratchCharEncoder(len(text_vocab), hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    rd = ReadingEncoder(len(kana_vocab), hidden=32, layers=1, heads=2, ffn=64, max_len=16)
    model = scorer_cls(ctx, rd, text_vocab, kana_vocab)
    model.eval()

    out_dir = tmp_path / scorer_cls.__name__
    export_modules(model, out_dir)
    return _dictionary(), model, out_dir


def test_export_creates_three_graphs(tmp_path: Path) -> None:
    _, _, out_dir = _build(BiEncoderScorer, tmp_path)
    for name in (
        "context_encoder.onnx",
        "reading_encoder.onnx",
        "node_scorer.onnx",
        "text_vocab.json",
        "kana_vocab.json",
    ):
        assert (out_dir / name).exists(), name


@pytest.mark.parametrize("scorer_cls", [BiEncoderScorer, CrossEncoderScorer])
def test_onnx_scores_match_pytorch(scorer_cls, tmp_path: Path) -> None:  # noqa: ANN001
    """★ONNX と PyTorch のスコアが一致すること."""
    dic, model, out_dir = _build(scorer_cls, tmp_path)
    runtime = OnnxG2P(out_dir, dic)

    lat = build_lattice(TEXT, dic)
    with torch.no_grad():
        expected = model.score(lat.text, lat.nodes).numpy()
    actual = runtime.score_nodes(lat.text, lat.nodes)

    assert actual.shape == expected.shape
    assert np.abs(expected - actual).max() < 1e-4


def test_onnx_g2p_matches_pytorch_path(tmp_path: Path) -> None:
    """経路まで含めて PyTorch と一致すること."""
    dic, model, out_dir = _build(BiEncoderScorer, tmp_path)
    lat = build_lattice(TEXT, dic)
    with torch.no_grad():
        expected_path = viterbi(lat, model.score(lat.text, lat.nodes))
    expected = "".join(lat.nodes[i].reading for i in expected_path)

    assert OnnxG2P(out_dir, dic).g2p(TEXT) == expected


def test_timing_breakdown_has_all_sections(tmp_path: Path) -> None:
    dic, _, out_dir = _build(BiEncoderScorer, tmp_path)
    _, timing = OnnxG2P(out_dir, dic).g2p_with_timing(TEXT)

    assert set(timing) == {"lattice", "context", "scoring", "viterbi", "total"}
    assert timing["total"] > 0


def test_viterbi_numpy_matches_torch(tmp_path: Path) -> None:
    """★推論経路から torch を外しても結果が変わらないこと."""
    dic, model, _ = _build(BiEncoderScorer, tmp_path)
    lat = build_lattice(TEXT, dic)
    with torch.no_grad():
        scores = model.score(lat.text, lat.nodes)

    assert viterbi_numpy(lat, scores.numpy()) == viterbi(lat, scores)


def test_viterbi_numpy_empty_text() -> None:
    assert viterbi_numpy(build_lattice("", _dictionary()), np.zeros(0)) == []


def test_reading_cache_is_reused(tmp_path: Path) -> None:
    """★同じ読みを 2 度エンコードしないこと.

    キャッシュしないと 91 文字の文で読みエンコーダが 25 ms を占める
    (docs/architecture.md §4-3).
    """
    dic, _, out_dir = _build(BiEncoderScorer, tmp_path)
    runtime = OnnxG2P(out_dir, dic)

    assert runtime.cache_size == 0
    runtime.g2p(TEXT)
    first = runtime.cache_size
    assert first > 0

    calls = 0
    original = runtime.reading_session.run

    def counting(*args, **kw):  # noqa: ANN002, ANN003, ANN202
        nonlocal calls
        calls += 1
        return original(*args, **kw)

    runtime.reading_session.run = counting
    runtime.g2p(TEXT)

    assert calls == 0, "2 度目は読みエンコーダを呼ばないはず"
    assert runtime.cache_size == first


def test_reading_cache_matches_uncached(tmp_path: Path) -> None:
    """キャッシュの有無でスコアが変わらないこと."""
    dic, _, out_dir = _build(BiEncoderScorer, tmp_path)
    lat = build_lattice(TEXT, dic)

    cold = OnnxG2P(out_dir, dic).score_nodes(TEXT, lat.nodes)
    warm = OnnxG2P(out_dir, dic)
    warm.g2p("米の国")
    assert np.abs(cold - warm.score_nodes(TEXT, lat.nodes)).max() < 1e-6


def _build_with_transitions(tmp_path: Path):  # noqa: ANN202
    """遷移項つきのモデルを書き出す（M6）."""
    torch.manual_seed(0)
    text_vocab = CharVocab.build([TEXT])
    kana_vocab = CharVocab.build(["コメベイクニノキシャ"])
    ctx = ScratchCharEncoder(len(text_vocab), hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    rd = ReadingEncoder(len(kana_vocab), hidden=32, layers=1, heads=2, ffn=64, max_len=16)
    model = BiEncoderScorer(ctx, rd, text_vocab, kana_vocab, transition_dim=8, transition_ids=64)
    # ★初期値は 0 付近なので、遷移が効いていることを見るために散らす
    torch.nn.init.normal_(model.transition.left_emb.weight, std=1.0)
    torch.nn.init.normal_(model.transition.right_emb.weight, std=1.0)
    model.eval()

    out_dir = tmp_path / "with_transitions"
    export_modules(model, out_dir)
    return _dictionary(), model, out_dir


def test_transition_graph_is_exported(tmp_path: Path) -> None:
    _, _, out_dir = _build_with_transitions(tmp_path)

    assert (out_dir / "transition.onnx").exists()


def test_transition_graph_is_absent_without_transitions(tmp_path: Path) -> None:
    """★遷移項を持たないモデルでは書き出さないこと（既存モデルとの互換）."""
    _, _, out_dir = _build(BiEncoderScorer, tmp_path)

    assert not (out_dir / "transition.onnx").exists()
    assert OnnxG2P(out_dir, _dictionary()).transitions(
        build_lattice(TEXT, _dictionary()).nodes
    ) is None


def test_onnx_transitions_match_pytorch(tmp_path: Path) -> None:
    dic, model, out_dir = _build_with_transitions(tmp_path)
    lat = build_lattice(TEXT, dic)

    with torch.no_grad():
        expected = model.transitions(lat.nodes).numpy()
    actual = OnnxG2P(out_dir, dic).transitions(lat.nodes)

    assert np.abs(expected - actual).max() < 1e-4


def test_onnx_path_matches_pytorch_with_transitions(tmp_path: Path) -> None:
    """★遷移項を入れても経路まで一致すること。"""
    dic, model, out_dir = _build_with_transitions(tmp_path)
    lat = build_lattice(TEXT, dic)

    with torch.no_grad():
        expected_path = viterbi(
            lat, model.score(lat.text, lat.nodes), model.transitions(lat.nodes)
        )
    expected = "".join(lat.nodes[i].reading for i in expected_path)

    assert OnnxG2P(out_dir, dic).g2p(TEXT) == expected
