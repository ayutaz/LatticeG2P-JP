"""Hard Set をカテゴリ別に評価する（M4 Task 4 Step 4）.

    uv run python scripts/run_hard_set.py --onnx models/m4-int8

★期待される結果: 固有名詞と数詞で Accuracy が大きく落ちること。それが正しい。
全体の数値だけを見て「よく出た」と判断しないための装置である（R-11）。

★`seen_in_train` で「学習データに出た表記か」を分けて出す。
数詞は意図的に教えているので, 分けないと般化の有無が見えない。
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from lattice_g2p.baselines import (
    find_unexpected_chars,
    openjtalk_readings,
    span_reading_from_morphemes,
)
from lattice_g2p.benchmark_data import load_joyo_benchmark
from lattice_g2p.crf import viterbi
from lattice_g2p.decode import span_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import normalize_kana, phonetic_key
from lattice_g2p.lattice import build_lattice
from lattice_g2p.runtime import OnnxScorer
from lattice_g2p.train import load_scorer

sys.stdout.reconfigure(line_buffering=True)

CATEGORIES = (
    "proper_noun_person",
    "proper_noun_place",
    "proper_noun_org",
    "numeral",
    "homograph",
)


def hit(span: str, item) -> bool:  # noqa: ANN001
    """compare_systems.py と同じ規約にすること. 違うと比較にならない."""
    if not span:
        return False
    accept = {phonetic_key(normalize_kana(r)) for r in item.acceptable_readings()}
    return any(a in phonetic_key(span) for a in accept)


def rate(hits: list[bool]) -> str:
    if not hits:
        return "   —    "
    return f"{sum(hits) / len(hits) * 100:6.2f}% "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path, nargs="*")
    ap.add_argument("--onnx", type=Path, nargs="*", default=[])
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--hard-set", type=Path, default=Path("data/hard_set"))
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)

    systems: list[tuple[str, object]] = [("OpenJTalk", None)]
    for run_dir in args.checkpoints:
        systems.append(load_scorer(run_dir, "cpu"))
    for model_dir in args.onnx:
        systems.append((f"ONNX: {model_dir.name}", OnnxScorer(model_dir, dic)))

    rows: dict[str, list[str]] = {name: [] for name, _ in systems}
    header: list[str] = []

    for category in CATEGORIES:
        path = args.hard_set / f"{category}.jsonl"
        if not path.exists():
            continue
        items = load_joyo_benchmark(path)
        seen = [
            json.loads(line)["seen_in_train"] for line in path.open(encoding="utf-8")
        ]
        n_seen = sum(seen)
        header.append(f"{category} ({len(items)} 文 / 学習済み {n_seen})")

        for name, scorer in systems:
            hits: list[bool] = []
            unexpected: set[str] = set()
            for it in items:
                if scorer is None:
                    morphemes = openjtalk_readings(it.text)
                    unexpected |= find_unexpected_chars("".join(r for _, r in morphemes))
                    span = span_reading_from_morphemes(
                        morphemes, it.target_start, it.target_end
                    )
                else:
                    lat = build_lattice(it.text, dic)
                    getter = getattr(scorer, "transitions", None)
                    with torch.no_grad():
                        path_ = viterbi(
                            lat,
                            scorer.score(lat.text, lat.nodes),
                            None if getter is None else getter(lat.nodes),
                        )
                    span = span_reading(lat, path_, it.target_start, it.target_end)
                    del path_
                hits.append(hit(span, it))
            if unexpected:
                raise SystemExit(f"★OpenJTalk の出力に未知の文字: {sorted(unexpected)}")

            unseen_hits = [h for h, s in zip(hits, seen, strict=True) if not s]
            rows[name].append(f"{rate(hits)}({rate(unseen_hits).strip()} 未学習)")

    width = max(len(n) for n in rows)
    print("カテゴリ:")
    for i, h in enumerate(header):
        print(f"  {i + 1}. {h}")
    print()
    print(" " * width + "".join(f"{i + 1:>22}" for i in range(len(header))))
    for name, cells in rows.items():
        print(f"{name:<{width}}" + "".join(f"{c:>22}" for c in cells))
    print("\n★括弧内は学習データに出なかった表記だけの値。数詞は意図的に教えている。")


if __name__ == "__main__":
    main()
