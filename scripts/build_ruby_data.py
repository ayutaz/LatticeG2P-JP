"""青空文庫のテキストからルビ由来の学習データを作る.

    テキスト -> 本文抽出 -> 文分割 -> ルビ抽出 -> ラティス整合 -> JSONL

★ラティス整合チェックは品質管理ではなく動作要件である.
制約を満たす経路が存在しない例は CRF の分子が計算できず, 学習が壊れる.

ここで落ちるのは主に次の 3 つで, いずれも「落ちて正しい」.

    義訓ルビ           運命《さだめ》     著者独自の読み. 辞書に無い
    ルビの位置ずれ     解析の失敗
    辞書に無い語       固有名詞など (docs/pitfalls.md R-04)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from lattice_g2p.data.prepare import prepare_ruby_all
from lattice_g2p.data.ruby import (
    RubyExample,
    drop_held_out_spans,
    extract_body,
    filter_ruby_examples,
    iter_ruby_examples,
)
from lattice_g2p.data.schema import read_jsonl, write_jsonl
from lattice_g2p.dictionary import Dictionary


def collect(text_dir: Path, min_len: int, max_len: int) -> list[RubyExample]:
    rows: list[RubyExample] = []
    for path in sorted(text_dir.glob("*.txt")):
        body = extract_body(path.read_text(encoding="utf-8"))
        rows.extend(
            iter_ruby_examples(body, source=path.stem, min_len=min_len, max_len=max_len)
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dir", type=Path, default=Path("data/aozora/texts"))
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--out", type=Path, default=Path("data/ruby.jsonl"))
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument(
        "--held-out",
        type=Path,
        default=Path("data/dev_r21.jsonl"),
        help="★この JSONL の target_surface にルビが付いた span を落とす（dev リーク対策）。"
        "★現行の dev を指すこと（旧 dev.jsonl と dev_r21 は語がほぼ重ならない）",
    )
    ap.add_argument("--show-dropped", type=int, default=15)
    args = ap.parse_args()

    rows = collect(args.text_dir, args.min_len, args.max_len)
    works = len(list(args.text_dir.glob("*.txt")))
    print(f"{works:,} 作品から ルビ付きの文 {len(rows):,} 件を抽出\n")
    if not rows:
        raise SystemExit("ルビ付きの文が 1 件もありません")

    dic = Dictionary.load(args.dict_cache)
    kept, stats = filter_ruby_examples(rows, dic)
    print(stats.format())

    if args.show_dropped:
        # ★何が落ちたかを必ず目視する。件数だけでは「落ちて正しい」のか
        #   「解析の失敗」なのか区別できない（docs/training.md §5）
        kept_spans = {r.text: set(r.spans) for r in kept}
        dropped_spans: Counter[tuple[str, str]] = Counter()
        for r in rows:
            for span in r.spans:
                if span not in kept_spans.get(r.text, ()):
                    dropped_spans[(r.text[span[0] : span[1]], span[2])] += 1
        print(f"\n落ちた span（{len(dropped_spans)} 種類中 上位 {args.show_dropped}）:")
        for (surface, reading), n in dropped_spans.most_common(args.show_dropped):
            print(f"  {surface} -> {reading}  ({n} 件)")

    # ★dev の語にルビが付いた例を落とす。残すと dev の読みを教えることになる
    if args.held_out and args.held_out.exists():
        held = {row["target_surface"] for row in read_jsonl(args.held_out)}
        kept, leaked = drop_held_out_spans(kept, held)
        print(
            f"\ndev リーク対策  {len(held):,} 語 -> "
            f"span {leaked:,} 件を除外、残り {len(kept):,} 文"
        )

    write_jsonl(args.out, [{"text": r.text, "spans": r.spans, "source": r.source} for r in kept])
    print(f"\nルビ学習データ {len(kept):,} 件 -> {args.out}")

    prepared, dropped = prepare_ruby_all(kept, dic)
    print(f"前処理成功: {len(prepared):,} / {len(kept):,}  (落ちた: {dropped})")
    if dropped:
        raise SystemExit("★フィルタを通ったのに前処理で落ちる例がある")

    surfaces = Counter(r.text[s:e] for r in kept for s, e, _ in r.spans)
    print(f"\n対象語 {len(surfaces):,} 種類 / span 合計 {sum(surfaces.values()):,}")
    print("上位 15:", json.dumps(dict(surfaces.most_common(15)), ensure_ascii=False))


if __name__ == "__main__":
    main()
