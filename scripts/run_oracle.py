"""Oracle 評価: 辞書ラティス内に正解読みが存在する割合を測る.

ニューラルネットを使わない. 手法の前提が成立しているかの検証
(docs/evaluation.md §1-4).

M0 のゲート: Joyo benchmark に対して Oracle Accuracy >= 99.5%
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from lattice_g2p.benchmark_data import BenchmarkItem, load_joyo_benchmark
from lattice_g2p.crf import build_constrained_graph
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import normalize_kana
from lattice_g2p.lattice import UNK_READING, build_lattice

OUT_OF_SCOPE = re.compile(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]")
"""数字・ラテン文字を含む文は第 1 マイルストーンの範囲外.

前段の Text Normalizer が担当する (docs/pitfalls.md R-05).
論文自身も Arabic numerals / Latin-script loanwords は upstream で処理すべきとしている.
"""


def evaluate_oracle(
    dic: Dictionary, items: list[BenchmarkItem]
) -> tuple[int, list[dict[str, object]], list[int]]:
    """(正解件数, 失敗の詳細, ノード数の一覧) を返す."""
    ok = 0
    failures: list[dict[str, object]] = []
    node_counts: list[int] = []

    for item in items:
        gold = normalize_kana(item.reading)
        lattice = build_lattice(item.text, dic)
        node_counts.append(len(lattice.nodes))

        if build_constrained_graph(lattice, gold).is_reachable():
            ok += 1
            continue

        failures.append(
            {
                "key": item.key,
                "text": item.text,
                "gold_reading": gold,
                "raw_reading": item.reading,
                "target_surface": item.target_surface,
                "target_reading": item.target_reading,
                "n_nodes": len(lattice.nodes),
                "has_unk_node": any(n.reading == UNK_READING for n in lattice.nodes),
                "has_full_path": lattice.has_full_path(),
                "normalization_changed": gold != item.reading,
                "target_in_dict": normalize_kana(item.target_reading)
                in dic.readings_of(item.target_surface),
                "source": item.source,
            }
        )

    return ok, failures, node_counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument(
        "--benchmark", type=Path, default=Path("data/benchmark/common_kanji_source.jsonl")
    )
    ap.add_argument("--failures-out", type=Path, default=Path("data/oracle_failures.jsonl"))
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)
    items = load_joyo_benchmark(args.benchmark)
    print(f"辞書 {len(dic):,} エントリ / benchmark {len(items):,} 件\n")

    t = time.perf_counter()
    ok, failures, node_counts = evaluate_oracle(dic, items)
    elapsed = time.perf_counter() - t

    n = len(items)
    out_of_scope = [f for f in failures if OUT_OF_SCOPE.search(str(f["text"]))]
    n_out_items = sum(1 for it in items if OUT_OF_SCOPE.search(it.text))
    n_in_items = n - n_out_items
    ok_in = n_in_items - (len(failures) - len(out_of_scope))

    print(f"Oracle Accuracy (全体)     : {ok / n * 100:.2f}%  ({ok:,} / {n:,})")
    print(
        f"Oracle Accuracy (スコープ内): {ok_in / n_in_items * 100:.2f}%"
        f"  ({ok_in:,} / {n_in_items:,})"
    )
    print(f"  ※ スコープ内 = 数字・ラテン文字を含まない文（範囲外は {n_out_items:,} 件）")
    print(f"失敗            : {len(failures):,}  (うち範囲外 {len(out_of_scope):,})")
    print(f"平均ノード数     : {sum(node_counts) / len(node_counts):.1f}")
    print(f"最大ノード数     : {max(node_counts)}")
    print(f"処理時間        : {elapsed:.1f}s  ({elapsed / n * 1000:.2f} ms/文)")

    if failures:
        print("\n失敗例の内訳:")
        n_unk = sum(1 for f in failures if f["has_unk_node"])
        n_norm = sum(1 for f in failures if f["normalization_changed"])
        n_nopath = sum(1 for f in failures if not f["has_full_path"])
        n_nodict = sum(1 for f in failures if not f["target_in_dict"])
        print(f"  UNK ノードを含む          : {n_unk}")
        print(f"  正規化で変化した          : {n_norm}")
        print(f"  経路自体がない            : {n_nopath}")
        print(f"  対象語の読みが辞書にない  : {n_nodict}")

        print("\n  source 別:")
        for src, count in Counter(f["source"] for f in failures).most_common():
            print(f"    {src}: {count}")

        print("\n先頭 20 件:")
        for f in failures[:20]:
            print(f"  [{f['key']}] {f['text']}")
            print(f"    gold={f['gold_reading']}")

        args.failures_out.parent.mkdir(parents=True, exist_ok=True)
        with args.failures_out.open("w", encoding="utf-8") as out:
            for f in failures:
                out.write(json.dumps(f, ensure_ascii=False) + "\n")
        print(f"\n失敗例を {args.failures_out} に書き出した")

    print("\nゲート (Oracle >= 99.5%):")
    print(f"  全体       : {'✓' if ok / n >= 0.995 else '✗'}  {ok / n * 100:.2f}%")
    print(
        f"  スコープ内 : {'✓' if ok_in / n_in_items >= 0.995 else '✗'}  "
        f"{ok_in / n_in_items * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
