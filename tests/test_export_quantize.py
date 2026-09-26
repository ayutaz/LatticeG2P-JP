"""INT8 量子化のテスト（R-26）.

★動的量子化は活性の尺度をバッチ全体の min / max で決める。読みエンコーダを量子化すると、
同じ読みでも「どの読みと同じバッチで最初に計算されたか」でベクトルが変わり、それが
読みキャッシュに残る。**同じ文のスコアが評価の履歴で変わる**（docs/pitfalls.md R-26）。
FP32 での同じ確認（test_runtime.py の test_reading_cache_matches_uncached）はあったが、
INT8 では確かめていなかった。
"""

from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime", reason="onnxruntime は --extra onnx で入る")
onnx = pytest.importorskip("onnx", reason="onnx は --extra onnx で入る")

from lattice_g2p.export_onnx import quantize  # noqa: E402
from lattice_g2p.lattice import build_lattice  # noqa: E402
from lattice_g2p.runtime import OnnxG2P  # noqa: E402
from lattice_g2p.scorer.bi_encoder import BiEncoderScorer  # noqa: E402
from tests.test_runtime import TEXT, _build  # noqa: E402

WARMUP = ["米の国", "記者の米", "国の記者", "米国", "の国の米の記者"]
"""TEXT と読みを共有しつつ、別の組み合わせで読みエンコーダに流す文."""


def _int8(tmp_path: Path):  # noqa: ANN202
    dic, _, fp32 = _build(BiEncoderScorer, tmp_path)
    int8 = tmp_path / "int8"
    quantize(fp32, int8)
    return dic, fp32, int8


def test_int8_scores_do_not_depend_on_history(tmp_path: Path) -> None:
    """★同じ文は、その前に何を処理したかによらず同じスコアになること."""
    dic, _, int8 = _int8(tmp_path)
    lat = build_lattice(TEXT, dic)

    fresh = OnnxG2P(int8, dic).score_nodes(TEXT, lat.nodes)
    warm = OnnxG2P(int8, dic)
    for text in WARMUP:
        warm.g2p(text)

    assert np.array_equal(fresh, warm.score_nodes(TEXT, lat.nodes))


def test_reading_encoder_stays_fp32(tmp_path: Path) -> None:
    """読みエンコーダは量子化せずに写す. 読みはキャッシュされるので速度への影響は小さい."""
    _, fp32, int8 = _int8(tmp_path)

    assert (int8 / "reading_encoder.onnx").read_bytes() == (
        fp32 / "reading_encoder.onnx"
    ).read_bytes()


def test_other_graphs_are_still_quantized(tmp_path: Path) -> None:
    """文脈エンコーダとノードスコアラは INT8 のまま（サイズとレイテンシの本体）."""
    _, _, int8 = _int8(tmp_path)

    for name in ("context_encoder.onnx", "node_scorer.onnx"):
        ops = {n.op_type for n in onnx.load(int8 / name).graph.node}
        assert ops & {"DynamicQuantizeLinear", "MatMulInteger", "DynamicQuantizeMatMul"}, name
