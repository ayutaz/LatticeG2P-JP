"""任意の NodeScorer で benchmark を評価する.

誤り分析を 3 分類で出す (docs/evaluation.md §6-3).
  - Oracle 失敗   : 辞書に正解がない -> 辞書の問題. モデルを直しても改善しない
  - 経路選択の誤り : 正解は辞書にある -> モデルの問題. ここが改善対象
  - 正規化の不一致 : バグ
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from lattice_g2p.analysis import ErrorCategory, classify_error, summarize
from lattice_g2p.benchmark_data import BenchmarkItem, load_joyo_benchmark
from lattice_g2p.crf import viterbi
from lattice_g2p.decode import path_reading, span_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import normalize_kana
from lattice_g2p.lattice import build_lattice
from lattice_g2p.metrics import EvalResult, accuracy_any, corpus_per
from lattice_g2p.runtime import OnnxScorer
from lattice_g2p.scorer import NodeScorer, UnigramScorer
from lattice_g2p.train import load_scorer

OUT_OF_SCOPE = re.compile(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]")
"""数字・ラテン文字を含む文は範囲外 (docs/pitfalls.md R-05)."""


def _transitions(scorer, nodes):  # noqa: ANN001, ANN202
    """CRF の遷移項 (M6). 持たないスコアラでは None."""
    getter = getattr(scorer, "transitions", None)
    return None if getter is None else getter(nodes)


def evaluate(
    scorer: NodeScorer, dic: Dictionary, items: list[BenchmarkItem]
) -> tuple[EvalResult, list[dict[str, object]]]:
    pred_targets: list[str] = []
    gold_targets: list[tuple[str, ...]] = []
    pred_sentences: list[str] = []
    gold_sentences: list[str] = []
    errors: list[dict[str, object]] = []

    for item in items:
        gold_sentence = normalize_kana(item.reading)
        lattice = build_lattice(item.text, dic)

        with torch.no_grad():
            scores = scorer.score(lattice.text, lattice.nodes)
            transitions = _transitions(scorer, lattice.nodes)
        path = viterbi(lattice, scores, transitions)

        pred_sentence = path_reading(lattice, path)

        # ★対象語の読みは「経路のノードから取り出す」のではなく
        #   「予測した文全体の読みの, 正解が示すカナ位置」を見る.
        #   経路が target span をまたぐノードを選ぶことがあり,
        #   ノードの読みをそのまま取ると対象外の読みまで含んでしまうため
        #   (「固める -> カタメル」の 1 ノードで target が「固」だけ, など).
        pred_target = pred_sentence[item.target_kana_start : item.target_kana_end]
        node_target = span_reading(lattice, path, item.target_start, item.target_end)

        pred_sentences.append(pred_sentence)
        gold_sentences.append(gold_sentence)
        pred_targets.append(pred_target)
        gold_targets.append(tuple(normalize_kana(r) for r in item.acceptable_readings()))

        if pred_target not in set(gold_targets[-1]):
            category = classify_error(item.text, gold_sentence, dic)
            errors.append(
                {
                    "key": item.key,
                    "text": item.text,
                    "target_surface": item.target_surface,
                    "gold_target": list(gold_targets[-1]),
                    "pred_target": pred_target,
                    "node_target": node_target,
                    "gold_sentence": gold_sentence,
                    "pred_sentence": pred_sentence,
                    "category": category.value,
                    "out_of_scope": bool(OUT_OF_SCOPE.search(item.text)),
                }
            )

    result = EvalResult(
        accuracy=accuracy_any(pred_targets, gold_targets),
        target_per=corpus_per(pred_targets, [g[0] for g in gold_targets]),
        sentence_per=corpus_per(pred_sentences, gold_sentences),
        n=len(items),
    )
    return result, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument(
        "--benchmark", type=Path, default=Path("data/benchmark/common_kanji_source.jsonl")
    )
    ap.add_argument("--errors-out", type=Path, default=Path("data/eval_errors.jsonl"))
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="学習済みモデルのディレクトリ。省略すると UnigramScorer（ベースライン）",
    )
    ap.add_argument(
        "--onnx", type=Path, default=None, help="ONNX モデルのディレクトリ（--checkpoint と排他）"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cost-scale", type=float, default=100.0)
    ap.add_argument("--unk-penalty", type=float, default=5.0)
    ap.add_argument("--node-penalty", type=float, default=30.0)
    ap.add_argument("--kana-bonus", type=float, default=0.5)
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)
    items = load_joyo_benchmark(args.benchmark)
    print(f"辞書 {len(dic):,} エントリ / benchmark {len(items):,} 件")

    if args.onnx is not None:
        scorer = OnnxScorer(args.onnx, dic)
        print(f"ONNX Runtime: {args.onnx}\n")
    elif args.checkpoint is not None:
        name, scorer = load_scorer(args.checkpoint, args.device)
        n_params = sum(p.numel() for p in scorer.parameters())
        print(f"ニューラルスコアラ: {name}  {n_params / 1e6:.2f}M params\n")
    else:
        scorer = UnigramScorer(
            cost_scale=args.cost_scale,
            unk_penalty=args.unk_penalty,
            node_penalty=args.node_penalty,
            source_bonus={
                "kana": args.kana_bonus,
                "both": args.kana_bonus,
                "pron": 0.0,
                "aug": 0.0,
            },
        )
        print(
            f"UnigramScorer(cost_scale={args.cost_scale}, unk_penalty={args.unk_penalty}, "
            f"node_penalty={args.node_penalty}, kana_bonus={args.kana_bonus})\n"
        )

    t = time.perf_counter()
    result, errors = evaluate(scorer, dic, items)
    elapsed = time.perf_counter() - t

    print(result.format())
    print(f"処理時間      : {elapsed:.1f}s  ({elapsed / len(items) * 1000:.2f} ms/文)")

    in_scope = [e for e in errors if not e["out_of_scope"]]
    n_out_items = sum(1 for it in items if OUT_OF_SCOPE.search(it.text))
    n_in_items = len(items) - n_out_items
    print(
        f"\nAccuracy (スコープ内): "
        f"{(n_in_items - len(in_scope)) / n_in_items * 100:.2f}%"
        f"  ({n_in_items - len(in_scope):,} / {n_in_items:,})"
    )

    print("\n誤り分析:")
    counts = summarize([ErrorCategory(e["category"]) for e in errors])
    for category, count in counts.items():
        print(f"  {category.value}: {count:,}")

    print("\n誤り例（先頭 15 件）:")
    for e in errors[:15]:
        print(f"  {e['target_surface']}: pred={e['pred_target']} gold={e['gold_target']}")
        print(f"    {e['text'][:40]}")

    args.errors_out.parent.mkdir(parents=True, exist_ok=True)
    with args.errors_out.open("w", encoding="utf-8") as out:
        for e in errors:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\n誤り例を {args.errors_out} に書き出した")

    print(f"\nゲート (Accuracy > 90%): {'✓' if result.accuracy > 0.90 else '✗'}")


if __name__ == "__main__":
    main()
