"""サブエージェントのバッチ出力を回収し、フィルタして学習データにする."""

import argparse
from pathlib import Path

from lattice_g2p.data.batches import batch_summary, merge_batch_outputs
from lattice_g2p.data.consistency import drop_inconsistent
from lattice_g2p.data.filter import drop_held_out_chars, filter_sentences
from lattice_g2p.data.prepare import prepare_all
from lattice_g2p.data.schema import write_jsonl
from lattice_g2p.dictionary import Dictionary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batch-dir",
        type=Path,
        nargs="+",
        default=[Path("data/batches")],
        help="複数指定できる（Phase A と Phase B をまとめて回収する場合）",
    )
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--generated-out", type=Path, default=Path("data/generated.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--model", default="claude-code-subagent")
    ap.add_argument(
        "--held-out-chars",
        default="",
        help="★dev に取り分けた文字。これを本文に含む例を落とす（R-20）。例: 四六八",
    )
    args = ap.parse_args()

    rows = []
    for d in args.batch_dir:
        print(f"{d}: {batch_summary(d)}")
        rows.extend(merge_batch_outputs(d, model=args.model))
    if not rows:
        print("バッチ出力がありません")
        return
    write_jsonl(args.generated_out, rows)
    print(f"回収 {len(rows):,} 文 -> {args.generated_out}\n")

    if args.held_out_chars:
        rows, dropped_chars = drop_held_out_chars(rows, set(args.held_out_chars))
        print(f"★held-out 文字 {args.held_out_chars} を含む {dropped_chars:,} 件を除外 "
              f"-> 残り {len(rows):,} 件\n")

    dic = Dictionary.load(args.dict_cache)
    kept, stats = filter_sentences(rows, dic)
    print(stats.format())

    if stats.lattice_failures:
        print(f"\nラティス整合で落ちた例（{len(stats.lattice_failures)} 件中 先頭 15）:")
        for f in stats.lattice_failures[:15]:
            print(f"  {f}")

    # ★フィルタを通過しても、同じ語に矛盾する読みが付いていることがある
    kept, consistency = drop_inconsistent(kept)
    print()
    print(consistency.format())

    write_jsonl(args.out, kept)
    print(f"\n学習データ {len(kept):,} 件 -> {args.out}")

    prepared, dropped = prepare_all(kept, dic)
    print(f"前処理成功: {len(prepared):,} / {len(kept):,}  (落ちた: {dropped})")
    if dropped:
        raise SystemExit("★フィルタを通ったのに前処理で落ちる例がある")

    lattice_rate = stats.lattice_ok / stats.format_ok if stats.format_ok else 0.0
    yield_rate = stats.dedup_ok / stats.generated if stats.generated else 0.0
    print("\nゲート:")
    print(
        f"  ラティス整合 > 95%  : {'✓' if lattice_rate > 0.95 else '✗'}"
        f"  {lattice_rate * 100:.1f}%"
    )
    print(f"  最終歩留まり > 85%  : {'✓' if yield_rate > 0.85 else '✗'}  {yield_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
