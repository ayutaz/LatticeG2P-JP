"""学習データを生成する.

Phase A（スモーク）:
    uv run python scripts/generate_data.py --limit 200 --n-per-group 10
Phase B（本番）:
    uv run python scripts/generate_data.py --limit 5000 --n-per-group 15

★Phase A が通ってから Phase B を実行すること.
生成を先にまとめて走らせると, フィルタで大量に落ちたときに丸ごと無駄になる.
"""

import argparse
from pathlib import Path

from lattice_g2p.data.client import DEFAULT_MODEL, ClaudeClient
from lattice_g2p.data.generate import generate_reading_groups, generate_sentences
from lattice_g2p.data.schema import ReadingGroup, read_jsonl
from lattice_g2p.data.select import PolyphonicWord


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polyphonic", type=Path, default=Path("data/polyphonic.jsonl"))
    ap.add_argument("--groups-out", type=Path, default=Path("data/reading_groups.jsonl"))
    ap.add_argument("--sentences-out", type=Path, default=Path("data/generated.jsonl"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-per-group", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    words = [
        PolyphonicWord(
            surface=r["surface"],
            readings=tuple(r["readings"]),
            pos_list=tuple(r["pos_list"]),
        )
        for r in read_jsonl(args.polyphonic)
    ][: args.limit]
    print(f"対象 {len(words):,} 語  model={args.model}")

    client = ClaudeClient(model=args.model)

    generate_reading_groups(client, words, args.groups_out)
    groups = [ReadingGroup(**r) for r in read_jsonl(args.groups_out)]
    valid = [g for g in groups if g.valid]
    print(f"reading group {len(groups):,} 件（うち valid {len(valid):,} 件）")

    generate_sentences(client, groups, args.sentences_out, args.n_per_group)
    n = sum(1 for _ in read_jsonl(args.sentences_out))
    print(f"生成文 {n:,} 件 -> {args.sentences_out}")


if __name__ == "__main__":
    main()
