"""数詞 + 助数詞の例文生成用バッチを作る（M3d）.

★M3c で「数詞 + 助数詞の誤りが 3 モデルとも同じ 34 件」と分かった。
ルビで語彙を 8 倍にしても、事前学習の知識を入れても変わらない。
学習データに `一本`・`三匹`・`八階` のような例が 1 つも無いのが原因
（→ docs/pitfalls.md R-21）。

★読みは事前に作った表で確定させる。音便（三本→サンボン）は
ラティス整合フィルタで検出できない。`本` には `ホン` も `ボン` も辞書にあるので、
`サンホン` と書かれても通ってしまう。
**知識（小さい表・高品質）と量（文・安価）を分離する。**

★dev が `四`・`六`・`八` を held-out にしているので、それらは表から除く。
`一`・`二`・`三`・`五`・`七`・`九`・`十` で音便を教え、般化するかを見る。
"""

import argparse
import json
from pathlib import Path

from lattice_g2p.crf import build_constrained_graph
from lattice_g2p.data.schema import read_jsonl, write_jsonl
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.lattice import build_lattice

HELD_OUT_NUMERALS = ("四", "六", "八")
"""dev に取り分けてある数詞. 学習データに入れてはいけない."""


def verify(surface: str, reading: str, dic: Dictionary) -> bool:
    """その読みが辞書ラティスで生成できるか.

    ★これは「読みが正しいか」の検証ではない. 生成できることの確認だけ.
    音便の正しさは表の品質に依存する（フィルタでは検出できない）。
    """
    return build_constrained_graph(build_lattice(surface, dic), reading).is_reachable()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("data/num_table_agent.jsonl"))
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/batches_num"))
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument(
        "--held-out",
        default="".join(HELD_OUT_NUMERALS),
        help="dev に取り分けてある数詞。空文字にすると除外しない（R-21 の対応後）",
    )
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)
    items: list[dict] = []
    skipped: list[str] = []
    unreachable: list[str] = []

    for row in read_jsonl(args.table):
        surface = row["surface"]
        readings = row.get("readings")
        if not readings:
            continue
        if args.held_out and any(ch in surface for ch in args.held_out):
            skipped.append(surface)
            continue
        for reading in readings:
            if not verify(surface, reading, dic):
                unreachable.append(f"{surface}->{reading}")
                continue
            items.append({"surface": surface, "reading": reading})

    print(f"表 {sum(1 for _ in read_jsonl(args.table)):,} 行 -> 生成対象 {len(items):,} 件")
    if skipped:
        print(f"★dev の数詞を含むため除外: {len(skipped)} 件 {skipped[:8]}")
    if unreachable:
        print(f"★ラティスで生成できない読み: {len(unreachable)} 件 {unreachable[:8]}")
        print("  辞書に部品が無い。表の誤りか、辞書の不足（どちらかを確認すること）")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), args.batch_size):
        chunk = items[i : i + args.batch_size]
        write_jsonl(args.out_dir / f"batch_{i // args.batch_size:03d}.jsonl", chunk)
    n = (len(items) + args.batch_size - 1) // args.batch_size
    print(f"\n{n} バッチを {args.out_dir} に書き出した")
    print(json.dumps(items[:3], ensure_ascii=False))


if __name__ == "__main__":
    main()
