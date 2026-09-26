"""benchmark の公式評価ツール（jkyb-eval）で測る（R-28）.

★論文の数値は公式ツールの値である。自前の「対象語」の数え方（対象語にかかるノードの読みに
正解が部分文字列として含まれるか）は公式より 0.2〜0.6 pt 甘く、甘さは構成で違う
（docs/evaluation.md §1-5）。**論文や他の報告と比べるときは、ここで測った値を使う。**

論文の 3 列との対応（自前で走らせた OpenJTalk が論文と 3 列とも一致して確かめた）:

    Accuracy       <- accuracy（自然な読みとの完全一致）
    Target PER     <- relaxed_target_kana_cer の cer_at_1（文ごとに上限 1.0）
    Sentence PER   <- sentence_kana_cer の cer_at_1

公式ツールは huggingface-hub<1 を要求し、このプロジェクトの依存と衝突するので、
`uv run --no-project --with` の一時環境で動かす（依存に足さない）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

JKYB_EVAL = (
    "jkyb-eval @ git+https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition"
    "@9606471bc15f052bdda7c18eba6396cd31020528"
)
"""公式評価ツール（jkyb-eval 0.2.0）. ★版を固定する. 指標の定義が変わると数字が変わる."""


@dataclass(frozen=True)
class OfficialMetrics:
    """公式ツールの指標（%）."""

    accuracy: float
    relaxed_accuracy: float
    target_per: float
    sentence_per: float

    def format(self) -> str:
        return (
            f"Accuracy {self.accuracy:6.2f}%  Target PER {self.target_per:5.2f}%  "
            f"Sentence PER {self.sentence_per:5.2f}%  "
            f"（Relaxed Accuracy {self.relaxed_accuracy:6.2f}%）"
        )


def write_predictions(path: Path, keys: Sequence[str], sentences: Sequence[str]) -> None:
    """公式ツールの入力（{"key": ..., "yomi": 文全体の読み} の JSONL）を書く."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key, yomi in zip(keys, sentences, strict=True):
            f.write(json.dumps({"key": key, "yomi": yomi}, ensure_ascii=False) + "\n")


def write_dataset(benchmark_path: Path, keys: Sequence[str], path: Path) -> None:
    """benchmark から keys の行だけを、その順に、そのまま書き出す.

    ★公式ツールは、データセットの全行に予測が無いと止まる. 部分集合（--limit）で測るときも、
    その文だけのデータセットを渡せば同じ規則で測れる.
    """
    rows: dict[str, str] = {}
    for line in Path(benchmark_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows[json.loads(line)["key"]] = line
    missing = [k for k in keys if k not in rows]
    if missing:
        raise KeyError(f"benchmark に無いキーがあります: {missing[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(rows[k] + "\n" for k in keys), encoding="utf-8")


def build_command(pred_path: Path, out_dir: Path, dataset_path: Path) -> list[str]:
    return [
        "uv", "run", "--no-project", "--with", JKYB_EVAL,
        "jkyb-eval", "g2p", str(pred_path),
        "--output-dir", str(out_dir),
        "--dataset", str(dataset_path),
    ]


def parse_summary(path: Path) -> OfficialMetrics:
    m = json.loads(Path(path).read_text(encoding="utf-8"))["metrics"]
    return OfficialMetrics(
        accuracy=m["accuracy"]["rate"] * 100,
        relaxed_accuracy=m["relaxed_accuracy"]["rate"] * 100,
        target_per=m["relaxed_target_kana_cer"]["cer_at_1"] * 100,
        sentence_per=m["sentence_kana_cer"]["cer_at_1"] * 100,
    )


def _run(cmd: list[str]) -> None:
    """公式ツールを動かす. ★親の仮想環境（VIRTUAL_ENV）を引き継がない（依存が衝突する）."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-15:])
        raise RuntimeError(
            f"公式評価ツールが失敗しました（終了コード {result.returncode}）。"
            f"ネットワークか uv を確認してください:\n{tail}"
        )


def evaluate_many(
    systems: Mapping[str, Sequence[str]],
    keys: Sequence[str],
    benchmark_path: Path,
    work_dir: Path,
    runner: Callable[[list[str]], None] = _run,
) -> dict[str, OfficialMetrics]:
    """系統ごとの文全体の読み（keys と同じ順）を公式ツールで測る."""
    for name, sentences in systems.items():
        if len(sentences) != len(keys):
            raise ValueError(f"{name}: 予測 {len(sentences)} 件とキー {len(keys)} 件が合いません")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset = work_dir / "dataset.jsonl"
    write_dataset(benchmark_path, keys, dataset)

    results: dict[str, OfficialMetrics] = {}
    for i, (name, sentences) in enumerate(systems.items()):
        safe = re.sub(r"[^\w\-（）()\[\]]+", "_", name)[:60]
        out = work_dir / f"{i}_{safe}"
        pred = work_dir / f"{i}_{safe}.predictions.jsonl"
        write_predictions(pred, keys, sentences)
        runner(build_command(pred, out, dataset))
        results[name] = parse_summary(out / "summary.json")
    return results
