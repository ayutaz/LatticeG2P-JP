"""学習データを train / dev に分割する.

★エントリ単位で分割すること. 文単位で分けると同じ語が両側に出て,
実力を過大評価する (docs/pitfalls.md R-09).

★held-out にしてよいのは「推論できるもの」だけである (docs/pitfalls.md R-21).
数詞 + 助数詞の音便 (六本 -> ロッポン) は規則ではなく語彙的知識なので,
数詞を dev に回すと原理的に解けない問題を測ることになる.
--dev-exclude-chars で dev から外す.
"""

import argparse
import random
from pathlib import Path

from lattice_g2p.data.schema import TrainingExample, read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-all", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--train-out", type=Path, default=Path("data/train_split.jsonl"))
    # ★出力先は現行の dev（data/dev_r21.jsonl）にしない。分割をやり直すと現行の dev を上書きする。
    #   dev を固定して学習データだけ作り直すなら --dev-from data/dev_r21.jsonl を使う
    ap.add_argument("--dev-out", type=Path, default=Path("data/dev.jsonl"))
    ap.add_argument("--dev-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dev-exclude-chars",
        default="",
        help="この文字を含む表記は dev に入れず train に回す（R-21）",
    )
    ap.add_argument(
        "--dev-from",
        type=Path,
        default=None,
        help="既存の dev の語を使う。★データ構成を変えて比較するときは必須"
        "（分割をやり直すと dev が変わって比較にならない）",
    )
    args = ap.parse_args()

    rows = [TrainingExample(**r) for r in read_jsonl(args.train_all)]
    surfaces = sorted({r.target_surface for r in rows})

    if args.dev_from is not None:
        # ★dev を固定する。学習データを変えて比較するとき、分割をやり直すと
        #   dev まで変わってしまい「何が効いたのか」が分からなくなる
        dev_surfaces = {r["target_surface"] for r in read_jsonl(args.dev_from)}
        n_dev = len(dev_surfaces)
        print(f"dev を {args.dev_from} から固定: {n_dev:,} 語")
    else:
        # ★推論できない語は dev に入れない（R-21）
        excluded = set(args.dev_exclude_chars)
        eligible = [s for s in surfaces if not (excluded and any(c in s for c in excluded))]
        if excluded:
            print(f"dev から除外する文字 {args.dev_exclude_chars}: "
                  f"対象語 {len(surfaces) - len(eligible):,} 語を train に回す")
        random.Random(args.seed).shuffle(eligible)
        n_dev = max(1, int(len(eligible) * args.dev_ratio))
        dev_surfaces = set(eligible[:n_dev])

    dev = [r for r in rows if r.target_surface in dev_surfaces]
    train = [r for r in rows if r.target_surface not in dev_surfaces]

    write_jsonl(args.train_out, train)
    write_jsonl(args.dev_out, dev)

    print(f"train {len(train):,} 文 / {len(set(surfaces) - dev_surfaces):,} 語 -> {args.train_out}")
    print(f"dev   {len(dev):,} 文 / {n_dev:,} 語 -> {args.dev_out}")

    leaked = dev_surfaces & {r.target_surface for r in train}
    if leaked:
        raise SystemExit(f"★語が両側に漏れている: {sorted(leaked)[:10]}")
    print("OK: エントリ単位で分割されている")


if __name__ == "__main__":
    main()
