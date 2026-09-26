"""matrix.bin の軸の向きを、matrix.def（テキスト）と突き合わせて検証する（M7 Task 1 Step 5）.

★軸を取り違えても値は「それらしく」出る。**1 行でも食い違ったら先に進まない。**

matrix.def（3.7 GB）は展開せず、zip から間引いて取り出してから渡す:

    unzip -p data/unidic-src/unidic-cwj-3.1.1-full.zip unidic-cwj-3.1.1-full/matrix.def \\
      | awk 'NR == 2 || (NR > 1 && NR % 240000 == 0)' > <scratch>/matrix_sample.txt
    uv run python scripts/verify_matrix.py --sample <scratch>/matrix_sample.txt

matrix.def の各行は `prev_right_id next_left_id cost`。

✅ UniDic CWJ 3.1.1 で 1,002 行 / 食い違い 0。★この検証には判別力がある:
軸を取り違えて引くと 1,002 行中 1 行しか一致しない（範囲外 15 行）。
"""

import argparse
import sys
from pathlib import Path

from lattice_g2p.connection import ConnectionMatrix

DEFAULT_MATRIX = Path("data/unidic-src/unidic-cwj-3.1.1-full/matrix.bin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = ap.parse_args()

    cm = ConnectionMatrix.load(args.matrix)
    print(f"matrix.bin: lsize = {cm.lsize:,} / rsize = {cm.rsize:,}")

    checked = mismatched = 0
    seen_prev: set[int] = set()
    seen_next: set[int] = set()
    for line in args.sample.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        prev_right, next_left, expected = (int(x) for x in parts)
        got = cm.cost(prev_right, next_left)
        checked += 1
        seen_prev.add(prev_right)
        seen_next.add(next_left)
        if got != expected:
            mismatched += 1
            if mismatched <= 5:
                print(f"  ✗ ({prev_right}, {next_left}): matrix.def={expected} matrix.bin={got}")

    print(f"突き合わせ: {checked:,} 行 / 食い違い {mismatched}")
    print(f"  覆った範囲: prev_right {min(seen_prev)}〜{max(seen_prev)} / "
          f"next_left {min(seen_next)}〜{max(seen_next)}")
    if checked < 100:
        print("★サンプルが少なすぎる。間引き方を見直すこと")
        return 1
    if mismatched:
        print("★軸の向きが違う。先に進まないこと")
        return 1
    print("✓ 軸の向きは matrix.def と一致する")
    return 0


if __name__ == "__main__":
    sys.exit(main())
