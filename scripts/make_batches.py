"""サブエージェント生成用のバッチファイルを作る.

使い方:
    uv run python scripts/make_batches.py --limit 200 --batch-size 20
    # -> data/batches/batch_000.jsonl .. batch_009.jsonl
    # サブエージェントが各バッチを読んで batch_NNN.out.jsonl を書く
    uv run python scripts/collect_batches.py
"""

import argparse
from pathlib import Path

from lattice_g2p.data.batches import batch_summary, make_batches
from lattice_g2p.data.schema import read_jsonl
from lattice_g2p.data.select import PolyphonicWord


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polyphonic", type=Path, default=Path("data/polyphonic.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/batches"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()

    words = [
        PolyphonicWord(
            surface=r["surface"],
            readings=tuple(r["readings"]),
            pos_list=tuple(r["pos_list"]),
        )
        for r in read_jsonl(args.polyphonic)
    ][args.offset : args.offset + args.limit]

    paths = make_batches(words, args.out_dir, args.batch_size)
    print(f"多音語 {len(words):,} 語を {len(paths)} バッチに分割 -> {args.out_dir}")
    for p in paths:
        n = sum(1 for _ in read_jsonl(p))
        print(f"  {p.name}: {n} 語")
    print(f"\n{batch_summary(args.out_dir)}")


if __name__ == "__main__":
    main()
