"""benchmark の公式評価ツール（jkyb-eval）を呼ぶ部分のテスト（R-28）.

★公式ツールそのものはネットワーク越しに入れるので、ここでは呼び出し（runner）を差し替えて、
予測の書き出し・データセットの切り出し・集計の読み取りだけを確かめる.
実物での確認は docs/evaluation.md §1-5（OpenJTalk が論文と 3 列とも一致した）.
"""

import json
from pathlib import Path

import pytest

from lattice_g2p.official_eval import (
    JKYB_EVAL,
    OfficialMetrics,
    build_command,
    evaluate_many,
    parse_summary,
    write_dataset,
    write_predictions,
)


def _benchmark(tmp_path: Path) -> Path:
    path = tmp_path / "bench.jsonl"
    rows = [{"key": k, "text": t, "yomi": y} for k, t, y in
            [("a", "米を炊く", "コメヲタク"), ("b", "米国", "ベイコク"), ("c", "生", "ナマ")]]
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return path


def _summary(accuracy: float, relaxed: float, target: float, sentence: float) -> dict:
    return {
        "metrics": {
            "accuracy": {"rate": accuracy},
            "relaxed_accuracy": {"rate": relaxed},
            "target_kana_cer": {"raw": 0.9, "cer_at_1": 0.8},
            "relaxed_target_kana_cer": {"raw": 0.9, "cer_at_1": target},
            "sentence_kana_cer": {"raw": 0.9, "cer_at_1": sentence},
        }
    }


def test_write_predictions_uses_the_official_format(tmp_path: Path) -> None:
    """{"key": ..., "yomi": 文全体の読み} を 1 行ずつ、キーの順に書く."""
    write_predictions(tmp_path / "p.jsonl", ["a", "b"], ["コメヲタク", "ベイコク"])

    rows = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert rows == [{"key": "a", "yomi": "コメヲタク"}, {"key": "b", "yomi": "ベイコク"}]


def test_write_dataset_keeps_rows_verbatim_in_key_order(tmp_path: Path) -> None:
    """★部分集合で測るときは、その文だけのデータセットを渡す.

    公式ツールは、データセットの全行に予測が無いと止まる.
    """
    bench = _benchmark(tmp_path)
    write_dataset(bench, ["c", "a"], tmp_path / "d.jsonl")

    rows = [json.loads(line) for line in (tmp_path / "d.jsonl").read_text().splitlines()]
    assert [r["key"] for r in rows] == ["c", "a"]
    assert rows[1] == {"key": "a", "text": "米を炊く", "yomi": "コメヲタク"}


def test_write_dataset_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        write_dataset(_benchmark(tmp_path), ["zzz"], tmp_path / "d.jsonl")


def test_parse_summary_maps_the_paper_columns(tmp_path: Path) -> None:
    """★論文の 3 列 = accuracy / relaxed_target_kana_cer / sentence_kana_cer.

    後の 2 つは cer_at_1（文ごとに上限 1.0）.
    """
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary(0.966312, 0.967272, 0.030003, 0.008015)))

    m = parse_summary(path)

    assert m == OfficialMetrics(
        accuracy=pytest.approx(96.6312),
        relaxed_accuracy=pytest.approx(96.7272),
        target_per=pytest.approx(3.0003),
        sentence_per=pytest.approx(0.8015),
    )


def test_build_command_runs_in_an_isolated_environment(tmp_path: Path) -> None:
    """★--no-project で動かす（huggingface-hub<1 が依存と衝突する）. 版は固定する."""
    cmd = build_command(tmp_path / "p.jsonl", tmp_path / "out", tmp_path / "d.jsonl")

    assert cmd[:4] == ["uv", "run", "--no-project", "--with"]
    assert cmd[4] == JKYB_EVAL and "@" in JKYB_EVAL.split("git+")[1]
    assert cmd[5:7] == ["jkyb-eval", "g2p"]
    assert cmd[cmd.index("--dataset") + 1] == str(tmp_path / "d.jsonl")


def test_evaluate_many_runs_each_system_once(tmp_path: Path) -> None:
    """系統ごとに予測を書き、公式ツールを 1 回ずつ呼び、集計を読む."""
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> None:
        calls.append(cmd)
        out = Path(cmd[cmd.index("--output-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        acc = 0.9 if "0_" in out.name else 0.5
        (out / "summary.json").write_text(json.dumps(_summary(acc, acc, 0.01, 0.02)))

    results = evaluate_many(
        {"系統 A": ["コメヲタク", "ベイコク"], "系統 B（b）": ["コメヲタク", "ベイコク"]},
        keys=["a", "b"],
        benchmark_path=_benchmark(tmp_path),
        work_dir=tmp_path / "work",
        runner=fake_runner,
    )

    assert len(calls) == 2
    assert results["系統 A"].accuracy == pytest.approx(90.0)
    assert results["系統 B（b）"].accuracy == pytest.approx(50.0)
    dataset = Path(calls[0][calls[0].index("--dataset") + 1])
    assert [json.loads(line)["key"] for line in dataset.read_text().splitlines()] == ["a", "b"]


def test_evaluate_many_rejects_mismatched_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        evaluate_many({"A": ["コメ"]}, keys=["a", "b"], benchmark_path=_benchmark(tmp_path),
                      work_dir=tmp_path / "w", runner=lambda cmd: None)
