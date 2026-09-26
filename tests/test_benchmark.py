"""レイテンシ計測ユーティリティのテスト.

計測値そのものは環境依存なので, ここでは「形」だけを確認する.
実測は scripts/run_benchmark.py を★ローカルで★走らせて行う (R-13).
"""

from lattice_g2p.benchmark import SECTIONS, BenchmarkResult, environment_info, measure


def _fake(text: str) -> tuple[str, dict[str, float]]:
    return text, {k: 1.0 for k in SECTIONS}


def test_measure_returns_percentiles() -> None:
    result = measure(_fake, ["あいう"], label="fake", warmup=2, trials=20)

    assert result.n_trials == 20
    assert result.n_chars == 3
    assert result.p50 > 0
    assert result.p95 >= result.p50
    assert set(result.breakdown_p50) == set(SECTIONS)


def test_measure_without_breakdown() -> None:
    result = measure(lambda t: (t, None), ["あ"], label="none", warmup=1, trials=5)

    assert result.breakdown_p50 == {}
    assert "内訳" not in result.format()


def test_measure_cycles_through_texts() -> None:
    seen: list[str] = []

    def record(text: str) -> tuple[str, None]:
        seen.append(text)
        return text, None

    measure(record, ["あ", "い"], label="cycle", warmup=0, trials=4)
    assert seen[:4] == ["あ", "い", "あ", "い"]


def test_format_has_all_sections() -> None:
    result = BenchmarkResult(
        label="x",
        n_chars=10,
        n_trials=1,
        n_nodes_mean=3.0,
        p50=2.0,
        p95=4.0,
        breakdown_p50={k: 0.5 for k in SECTIONS},
    )
    text = result.format()
    for key in SECTIONS:
        assert key in text
    assert "p95" in text


def test_environment_info_without_model() -> None:
    info = environment_info()
    assert "platform" in info
    assert "model size" not in info
