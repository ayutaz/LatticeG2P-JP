"""既存手法と自前モデルを benchmark で同じ土俵で比較する（M4 Task 0b）.

    uv run python scripts/compare_systems.py checkpoints/m4

★論文値と比べない。同じ正規化規則・同じ評価コードで自前で走らせた値と比べる。
★既定で benchmark の公式評価ツール（jkyb-eval）の値も出す。**論文や他の報告と比べるときは
  こちらを使う**（自前の「対象語」の数え方は公式より 0.2〜0.6 pt 甘い。R-28）。
  止めるときは --no-official。
★比較対象の出力形式も検証する。OpenJTalk の pron はアクセント記号を含み、
  これを落とし損ねて一度 OpenJTalk を大幅に過小評価した
  （docs/pitfalls.md R-22）。
"""

import argparse
import sys
from pathlib import Path

import torch

from lattice_g2p.baselines import (
    find_unexpected_chars,
    openjtalk_readings,
    span_reading_from_morphemes,
)
from lattice_g2p.benchmark_data import load_joyo_benchmark
from lattice_g2p.crf import viterbi
from lattice_g2p.decode import path_reading, span_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import normalize_kana, phonetic_key
from lattice_g2p.lattice import build_lattice
from lattice_g2p.metrics import corpus_per
from lattice_g2p.official_eval import evaluate_many
from lattice_g2p.runtime import OnnxScorer
from lattice_g2p.scorer import UnigramScorer
from lattice_g2p.train import load_scorer

sys.stdout.reconfigure(line_buffering=True)  # リダイレクト時もその場で出す


def report(name: str, preds: list[tuple[str, str]], items: list) -> None:  # noqa: ANN001
    n = len(items)
    sent = tgt = 0
    ps: list[str] = []
    gs: list[str] = []
    for (sentence, span), item in zip(preds, items, strict=True):
        gold = phonetic_key(normalize_kana(item.reading))
        ps.append(phonetic_key(sentence))
        gs.append(gold)
        sent += phonetic_key(sentence) == gold
        accept = {phonetic_key(normalize_kana(r)) for r in item.acceptable_readings()}
        tgt += any(a in phonetic_key(span) for a in accept) if span else False
    print(
        f"{name:<30} 文一致 {sent / n * 100:6.2f}%  対象語 {tgt / n * 100:6.2f}%  "
        f"Sentence PER {corpus_per(ps, gs) * 100:5.2f}%  ※自前の物差し"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path, nargs="*", help="学習済みモデルのディレクトリ")
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument(
        "--benchmark", type=Path, default=Path("data/benchmark/common_kanji_source.jsonl")
    )
    ap.add_argument(
        "--onnx", type=Path, nargs="*", default=[], help="ONNX モデルのディレクトリ（複数可）"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--no-official", action="store_true", help="公式評価ツール（jkyb-eval）で測らない"
    )
    ap.add_argument(
        "--official-dir", type=Path, default=Path("data/official_eval/compare_systems")
    )
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)
    items = load_joyo_benchmark(args.benchmark)
    if args.limit:
        items = items[: args.limit]
    print(f"benchmark {len(items):,} 件\n")

    # --- OpenJTalk ---
    oj: list[tuple[str, str]] = []
    unexpected: set[str] = set()
    for it in items:
        morphemes = openjtalk_readings(it.text)
        sentence = "".join(r for _, r in morphemes)
        unexpected |= find_unexpected_chars(sentence)
        oj.append(
            (sentence, span_reading_from_morphemes(morphemes, it.target_start, it.target_end))
        )
    if unexpected:
        # ★ここが空でないなら計測が歪んでいる。黙って続けない
        raise SystemExit(f"★OpenJTalk の出力に未知の文字が残っている: {sorted(unexpected)}")
    report("OpenJTalk", oj, items)
    collected: dict[str, list[str]] = {"OpenJTalk": [s for s, _ in oj]}

    # --- ラティス系 ---
    def run(scorer, label: str) -> None:  # noqa: ANN001
        out: list[tuple[str, str]] = []
        with torch.no_grad():
            for it in items:
                lat = build_lattice(it.text, dic)
                getter = getattr(scorer, "transitions", None)
                path = viterbi(
                    lat,
                    scorer.score(lat.text, lat.nodes),
                    None if getter is None else getter(lat.nodes),
                )
                out.append(
                    (
                        path_reading(lat, path),
                        span_reading(lat, path, it.target_start, it.target_end),
                    )
                )
        report(label, out, items)
        collected[label] = [s for s, _ in out]

    run(
        UnigramScorer(
            cost_scale=100.0,
            unk_penalty=5.0,
            node_penalty=30.0,
            source_bonus={"kana": 0.5, "both": 0.5, "pron": 0.0, "aug": 0.0},
        ),
        "UnigramScorer（辞書のみ）",
    )
    for run_dir in args.checkpoints:
        name, model = load_scorer(run_dir, "cpu")
        run(model, name)
    for model_dir in args.onnx:
        run(OnnxScorer(model_dir, dic), f"ONNX: {model_dir.name}")

    if args.no_official:
        return
    print("\n公式の物差し（jkyb-eval。★論文の数値と同じ定義）:")
    official = evaluate_many(
        collected, [it.key for it in items], args.benchmark, args.official_dir
    )
    for name, m in official.items():
        print(f"{name:<30} {m.format()}")


if __name__ == "__main__":
    main()
