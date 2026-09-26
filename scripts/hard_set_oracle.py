"""Hard Set の★語レベル Oracle: その読みが辞書ラティスで生成できるか.

論文が弱点として認めている領域（固有名詞・数詞・同表記異読）で,
**辞書のカバレッジそのもの**を測る. ここが低ければモデルの問題ではない
(docs/evaluation.md §3-3).

★文を介さない. LLM を介さないので, 正解読みは人手で書いた表そのものである.
「物差し自体が間違っている」可能性を最小にするための設計.

    uv run python scripts/hard_set_oracle.py
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from lattice_g2p.crf import build_constrained_graph
from lattice_g2p.data.schema import read_jsonl
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import normalize_kana
from lattice_g2p.lattice import build_lattice

sys.stdout.reconfigure(line_buffering=True)


def reachable(surface: str, reading: str, dic: Dictionary) -> bool:
    return build_constrained_graph(
        build_lattice(surface, dic), normalize_kana(reading)
    ).is_reachable()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("data/hard_set/table.jsonl"))
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--out", type=Path, default=Path("data/hard_set/oracle.jsonl"))
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)
    by_category: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)

    for row in read_jsonl(args.table):
        for reading in row["readings"]:
            by_category[row["category"]].append(
                (row["surface"], reading, reachable(row["surface"], reading, dic))
            )

    print(f"{'カテゴリ':<22}{'Oracle':>9}  {'成功':>4}/{'件数':>4}")
    print("-" * 48)
    total_ok = total = 0
    for category in sorted(by_category):
        rows = by_category[category]
        ok = sum(1 for _, _, r in rows if r)
        total_ok += ok
        total += len(rows)
        print(f"{category:<22}{ok / len(rows) * 100:8.1f}%  {ok:4d}/{len(rows):4d}")
    print("-" * 48)
    print(f"{'合計':<22}{total_ok / total * 100:8.1f}%  {total_ok:4d}/{total:4d}")

    print("\n★辞書に無い読み（= Oracle 失敗。モデルを直しても出せない）:")
    for category in sorted(by_category):
        misses = [(s, r) for s, r, ok in by_category[category] if not ok]
        if misses:
            joined = "、".join(f"{s}→{r}" for s, r in misses)
            print(f"  [{category}] {joined}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        import json

        for category, rows in sorted(by_category.items()):
            for surface, reading, ok in rows:
                f.write(
                    json.dumps(
                        {
                            "category": category,
                            "surface": surface,
                            "reading": reading,
                            "oracle": ok,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"\n結果を {args.out} に書き出した")


if __name__ == "__main__":
    main()
