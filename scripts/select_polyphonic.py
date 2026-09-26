"""UniDic から多音語を抽出して JSONL に書き出す."""

import argparse
from dataclasses import asdict
from pathlib import Path

from lattice_g2p.data.schema import write_jsonl
from lattice_g2p.data.select import select_polyphonic
from lattice_g2p.dictionary import Dictionary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--out", type=Path, default=Path("data/polyphonic.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-readings", type=int, default=2)
    ap.add_argument("--max-readings", type=int, default=8)
    ap.add_argument("--max-surface-len", type=int, default=4)
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)
    words = select_polyphonic(
        dic,
        min_readings=args.min_readings,
        max_readings=args.max_readings,
        limit=args.limit,
        max_surface_len=args.max_surface_len,
    )

    write_jsonl(args.out, [asdict(w) for w in words])
    print(f"多音語 {len(words):,} 語を {args.out} に書き出した")

    by_len: dict[int, int] = {}
    for w in words:
        by_len[len(w.surface)] = by_len.get(len(w.surface), 0) + 1
    print("\n表記の長さ別:")
    for n in sorted(by_len):
        print(f"  {n} 文字: {by_len[n]:,}")

    print("\n読み候補数の分布:")
    by_readings: dict[int, int] = {}
    for w in words:
        by_readings[len(w.readings)] = by_readings.get(len(w.readings), 0) + 1
    for n in sorted(by_readings):
        print(f"  {n} 個: {by_readings[n]:,}")

    print("\n単漢字の例:")
    for w in [w for w in words if len(w.surface) == 1][:15]:
        print(f"  {w.surface}: {list(w.readings)}")


if __name__ == "__main__":
    main()
