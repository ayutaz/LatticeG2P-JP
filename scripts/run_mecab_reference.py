"""実 MeCab + UniDic 3.1.1 で benchmark を読む（M7 Task 2 Step 6b・★参考値）.

    uv run --with fugashi python scripts/run_mecab_reference.py

★計画では解釈表の (i) のときだけ行う測定だった。**結果を見た後に参考値として足した**
（解釈表の判定は変えない）。

理由: U（こちらの MeCab 近似）はラティスを (表記, 読み) で重複排除し、最小コストの
エントリの連接 ID だけを残す。**品詞違いのエントリが消える。** 例えば「実施される」の
`さ` は終助詞（最小コスト）の 1 ノードしか残らず、サ変の未然形の `さ` が無いので、
U は `実/施さ/れる` と読む。実 MeCab は `実施/さ/れる`。
**U は「辞書 + 連接行列」の下限でしかない可能性がある。** その差を測る。

★読みの取り出し方で数字が変わる（R-22 と同じ構図）。UniDic は pron（発音形, f[9]）と
kana（仮名形, f[20]）を持ち、benchmark は `を→ヲ`（kana）・`は→ワ`（pron）を使う
（不変条件 1）。対象語は kana で見る（U の source_bonus が kana を優先するのと同じ）。
文全体は「助詞は pron（は→ワ・へ→エ）、ただし を は kana（ヲ）、それ以外は kana」も
併記する。benchmark の表記規則そのもの（docs/evaluation.md §4）で、精度の調整ではない。
★最初は「助詞だけ pron」にして を→オ になり、文一致を 42.82% と過小評価した（正しくは 85% 前後）。
★未知の文字が読みに残ったら止める（find_unexpected_chars）。
"""

import argparse
import json
import sys
from pathlib import Path

import fugashi

from lattice_g2p.baselines import find_unexpected_chars, span_reading_from_morphemes
from lattice_g2p.benchmark_data import load_joyo_benchmark
from lattice_g2p.kana import is_kana_only, normalize_kana

sys.path.insert(0, str(Path(__file__).parent))
from run_m7 import bench_metrics, hit  # noqa: E402  ★run_m7.py と同じ規約で数える

_KANA = 20
_PRON = 9


def readings(tagger: fugashi.GenericTagger, text: str) -> tuple[list, list, int]:
    """(kana の形態素列, 助詞は pron（を は kana）の形態素列, 未知語の数)."""
    kana: list[tuple[str, str]] = []
    mixed: list[tuple[str, str]] = []
    unknown = 0
    for w in tagger(text):
        f = w.feature
        if len(f) <= _KANA:  # 未知語（素性が 6 個しかない）
            unknown += 1
            k = normalize_kana(w.surface)
            k = k if k and is_kana_only(k) else ""
            kana.append((w.surface, k))
            mixed.append((w.surface, k))
            continue
        k = "" if f[_KANA] == "*" else f[_KANA]
        p = "" if f[_PRON] == "*" else f[_PRON]
        kana.append((w.surface, k))
        mixed.append((w.surface, p if f[0] == "助詞" and k != "ヲ" else k))
    return kana, mixed, unknown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unidic", type=Path, default=Path("data/unidic-src/unidic-cwj-3.1.1-full"))
    ap.add_argument(
        "--benchmark", type=Path, default=Path("data/benchmark/common_kanji_source.jsonl")
    )
    ap.add_argument("--hard-set", type=Path, default=Path("data/hard_set"))
    ap.add_argument("--out", type=Path, default=Path("data/m7/mecab_reference.json"))
    args = ap.parse_args()

    tagger = fugashi.GenericTagger(f"-r /dev/null -d {args.unidic}")
    items = load_joyo_benchmark(args.benchmark)

    preds_kana: list[tuple[str, str]] = []
    preds_mixed: list[tuple[str, str]] = []
    unexpected: set[str] = set()
    n_unknown = 0
    for it in items:
        kana, mixed, unknown = readings(tagger, it.text)
        n_unknown += unknown
        for morphemes, out in ((kana, preds_kana), (mixed, preds_mixed)):
            sentence = "".join(r for _, r in morphemes)
            unexpected |= find_unexpected_chars(sentence)
            out.append(
                (sentence, span_reading_from_morphemes(morphemes, it.target_start, it.target_end))
            )
    if unexpected:
        raise SystemExit(f"★MeCab の読みに未知の文字が残っている: {sorted(unexpected)}")

    result = {
        "benchmark": {
            "MeCab（kana）": bench_metrics(preds_kana, items),
            "MeCab（助詞は pron・を は ヲ）": bench_metrics(preds_mixed, items),
        },
        "unknown_tokens": n_unknown,
        "hard_set": {},
    }
    print(f"benchmark {len(items):,} 件 / 未知語 {n_unknown} 形態素")
    for name, m in result["benchmark"].items():
        print(f"  {name:<24} 対象語 {m['対象語']:6.2f}  文一致 {m['文一致']:6.2f}  "
              f"Sentence PER {m['Sentence PER']:5.2f}")

    for cat in ("proper_noun_person", "proper_noun_place", "proper_noun_org",
                "numeral", "homograph"):
        path = args.hard_set / f"{cat}.jsonl"
        if not path.exists():
            continue
        hs = load_joyo_benchmark(path)
        hits = 0
        for it in hs:
            kana, _, _ = readings(tagger, it.text)
            hits += hit(span_reading_from_morphemes(kana, it.target_start, it.target_end), it)
        result["hard_set"][cat] = hits / len(hs) * 100
        print(f"  Hard Set {cat:<20} {result['hard_set'][cat]:6.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    with (args.out.parent / "mecab_predictions.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"config": "MeCab（kana）", "benchmark": preds_kana},
                           ensure_ascii=False) + "\n")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
