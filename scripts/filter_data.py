"""生成データをフィルタして学習データにする."""

import argparse
from pathlib import Path

from lattice_g2p.data.filter import filter_sentences
from lattice_g2p.data.prepare import prepare_all
from lattice_g2p.data.schema import GeneratedSentence, read_jsonl, write_jsonl
from lattice_g2p.dictionary import Dictionary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--generated", type=Path, default=Path("data/generated.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/train.jsonl"))
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)
    rows = [GeneratedSentence(**r) for r in read_jsonl(args.generated)]
    if not rows:
        print(f"{args.generated} が空です")
        return

    kept, stats = filter_sentences(rows, dic)
    print(stats.format())

    if stats.lattice_failures:
        print("\nラティス整合で落ちた例（先頭 20 件）:")
        for f in stats.lattice_failures[:20]:
            print(f"  {f}")
        print("\n★通過率が 95% を下回るなら、LLM の品質ではなく")
        print("  辞書かコードの問題を先に疑うこと（docs/training.md §3）")

    write_jsonl(args.out, kept)
    print(f"\n学習データ {len(kept):,} 件 -> {args.out}")

    prepared, dropped = prepare_all(kept, dic)
    print(f"前処理成功: {len(prepared):,} / {len(kept):,}  (落ちた: {dropped})")
    if dropped:
        raise SystemExit("★フィルタを通ったのに前処理で落ちる例がある。フィルタの実装を確認")


if __name__ == "__main__":
    main()
