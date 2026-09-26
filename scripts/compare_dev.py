"""学習済みチェックポイントを dev で比較し、誤りを 3 分類する.

    uv run python scripts/compare_dev.py checkpoints/no_ruby checkpoints/ruby

★数値だけの比較は無意味（docs/evaluation.md §6-3）。
Accuracy の差がどこから来ているかを、

    Oracle 失敗    辞書に正解がない。モデルを直しても改善しない
    経路選択の誤り  モデルの問題。ここが改善対象
    正規化の不一致  バグ

に分けて出す。加えて「片方だけが正解した例」を列挙する。
平均値が同じでも中身が違うことがあるため。
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from lattice_g2p.analysis import ErrorCategory, classify_error
from lattice_g2p.crf import viterbi
from lattice_g2p.data.audit import load_accepts, matches_accepted
from lattice_g2p.data.prepare import PreparedExample
from lattice_g2p.decode import path_reading, span_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.train import load_prepared, load_scorer


@torch.no_grad()
def predict(model: torch.nn.Module, examples: list[PreparedExample]) -> list[str]:
    out: list[str] = []
    for ex in examples:
        lattice = ex.lattice()
        path = viterbi(lattice, model.score(ex.text, ex.nodes))
        out.append(path_reading(lattice, path))
    return out


@torch.no_grad()
def predict_spans(
    model: torch.nn.Module, examples: list[PreparedExample], rows: list[dict]
) -> list[str]:
    """対象 span の予測読み."""
    out: list[str] = []
    for ex, row in zip(examples, rows, strict=True):
        lattice = ex.lattice()
        path = viterbi(lattice, model.score(ex.text, ex.nodes))
        out.append(span_reading(lattice, path, row["target_start"], row["target_end"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--dev-path", type=Path, default=Path("data/dev_r21.jsonl"))
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--accepts",
        type=Path,
        default=Path("data/dev_audit_r21.jsonl"),
        help="監査結果。複数正解として認める読みを読み込む（無ければ無視）。★dev と組で替えること",
    )
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)
    examples, dropped = load_prepared(args.dev_path, dic)
    print(f"dev {len(examples):,} 文（正解経路なしで除外 {dropped}）\n")

    dev_rows = [json.loads(line) for line in args.dev_path.open(encoding="utf-8")]
    accepts = load_accepts(args.accepts) if args.accepts and args.accepts.exists() else {}
    if accepts:
        print(f"監査結果 {len(accepts)} 件を「複数正解」として適用する ({args.accepts})")
        print("★これは監査したモデルが外した例だけを覆う。補正後の値は下限である\n")

    names: list[str] = []
    preds: list[list[str]] = []
    spans: list[list[str]] = []
    for run_dir in args.run_dirs:
        name, model = load_scorer(run_dir, args.device)
        names.append(name)
        preds.append(predict(model, examples))
        spans.append(predict_spans(model, examples, dev_rows))

    for name, span in zip(names, spans, strict=True):
        raw = sum(
            matches_accepted(sp, row["target_reading"], None)
            for sp, row in zip(span, dev_rows, strict=True)
        )
        fixed = sum(
            matches_accepted(
                sp,
                row["target_reading"],
                accepts.get((row["target_surface"], row["target_reading"])),
            )
            for sp, row in zip(span, dev_rows, strict=True)
        )
        n = len(examples)
        print(f"{name}  対象語一致  監査前 {raw / n * 100:.2f}%"
              f"  ->  監査後 {fixed / n * 100:.2f}%  (+{(fixed - raw) / n * 100:.2f} pt)")
    print()

    for name, pred in zip(names, preds, strict=True):
        correct = [p == e.gold_reading for p, e in zip(pred, examples, strict=True)]
        acc = sum(correct) / len(correct)
        cats = Counter(
            classify_error(e.text, e.gold_reading, dic)
            for e, ok in zip(examples, correct, strict=True)
            if not ok
        )
        print(f"{name}\n  文一致 {acc * 100:.2f}%  ({sum(correct)}/{len(correct)})")
        for cat in ErrorCategory:
            print(f"    {cat.value}: {cats.get(cat, 0)}")
        print()

    if len(preds) != 2:
        return

    # ★平均が同じでも中身が違うことがある。片方だけ正解した例を見る
    a, b = preds
    only_a = [e for e, x, y in zip(examples, a, b, strict=True)
              if x == e.gold_reading and y != e.gold_reading]
    only_b = [e for e, x, y in zip(examples, a, b, strict=True)
              if x != e.gold_reading and y == e.gold_reading]
    both = sum(1 for e, x, y in zip(examples, a, b, strict=True)
               if x == e.gold_reading and y == e.gold_reading)

    print(f"両方正解 {both} / {names[0]} だけ {len(only_a)} / {names[1]} だけ {len(only_b)}")
    for label, rows in ((names[0], only_a), (names[1], only_b)):
        if not rows:
            continue
        print(f"\n{label} だけが正解した例（上位 {args.show}）:")
        for e in rows[: args.show]:
            print(f"  {e.text}  -> {e.gold_reading}")


if __name__ == "__main__":
    main()
