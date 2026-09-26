"""benchmark の読み表記の文字種を集計する.

カナ正規化規則を決めるための調査スクリプト. docs/evaluation.md §2 参照.
"""

import argparse
import unicodedata
from collections import Counter
from pathlib import Path

from lattice_g2p.benchmark_data import load_joyo_benchmark

# 判断が必要な文字 (docs/evaluation.md §2)
WATCH = {
    "ー": "長音符",
    "ヲ": "助詞「を」",
    "ヂ": "ヂ",
    "ヅ": "ヅ",
    "ワ": "助詞「は」→ワ の可能性",
    "エ": "助詞「へ」→エ の可能性",
    "ヴ": "ヴ",
}

KATAKANA_START = "ァ"
KATAKANA_END = "ヺ"
PROLONGED = "ー"


def is_katakana(ch: str) -> bool:
    return KATAKANA_START <= ch <= KATAKANA_END or ch == PROLONGED


def report(name: str, readings: list[str]) -> None:
    counter: Counter[str] = Counter()
    for r in readings:
        counter.update(r)

    print(f"\n## {name}")
    print(f"件数          : {len(readings)}")
    print(f"異なり文字数   : {len(counter)}")
    print(f"総文字数      : {sum(counter.values())}")

    print("\n### 判断が必要な文字")
    for ch, desc in WATCH.items():
        print(f"  {ch} ({desc}): {counter.get(ch, 0)}")

    non_kana = {ch: n for ch, n in counter.items() if not is_katakana(ch)}
    print(f"\n### ★カタカナ以外の文字: {len(non_kana)} 種")
    for ch, n in sorted(non_kana.items(), key=lambda kv: -kv[1]):
        try:
            name_of = unicodedata.name(ch)
        except ValueError:
            name_of = "?"
        print(f"  {ch!r}  {n:>8}  U+{ord(ch):04X}  {name_of}")

    small = [c for c in counter if c in "ァィゥェォッャュョヮヵヶ"]
    print(f"\n### 小書き文字: {sorted(small)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", type=Path)
    args = ap.parse_args()

    items = load_joyo_benchmark(args.benchmark)
    report("文全体の読み (yomi)", [it.reading for it in items])
    report("対象語の読み (target_reading)", [it.target_reading for it in items])

    print("\n## 対象語の読み候補数の分布")
    dist = Counter(len(it.natural_readings) for it in items)
    for n, count in sorted(dist.items()):
        print(f"  natural {n} 個: {count} 件")
    n_marginal = sum(1 for it in items if it.marginal_readings)
    print(f"  marginal を持つ行: {n_marginal} 件")

    print("\n## reading_category の分布")
    for cat, count in Counter(it.reading_category for it in items).most_common():
        print(f"  {cat}: {count}")

    print("\n## source の分布")
    for src, count in Counter(it.source for it in items).most_common():
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
