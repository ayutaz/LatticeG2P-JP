"""dev のラベルを監査する（M3c Task 1-2）.

    uv run python scripts/audit_dev.py --dry-run          候補数を閾値ごとに見る
    uv run python scripts/audit_dev.py --make-batches     判定用バッチを書き出す
    uv run python scripts/audit_dev.py --collect          判定結果を集計する

★元の dev は書き換えない。結果は別ファイルに出す（既定: data/dev_r21.jsonl の監査を
  data/audit_r21/ で判定し、data/dev_audit_r21.jsonl に集計する）。
★判定済みのバッチ（audit_*.out.jsonl）があるディレクトリには --make-batches で書き込まない
  （--overwrite を付けたときだけ）。判定をやり直すのは高くつく（opus で行う）。
★ラベルを「モデルの予測に合わせて」直さない。それは評価を壊す行為である。
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from lattice_g2p.config import TrainConfig
from lattice_g2p.crf import viterbi
from lattice_g2p.data.audit import _readings_by_cost, flag_suspicious, reading_rank
from lattice_g2p.data.schema import TrainingExample, read_jsonl, write_jsonl
from lattice_g2p.decode import span_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import phonetic_key
from lattice_g2p.train import build_model, load_prepared
from lattice_g2p.vocab import CharVocab

VERDICTS = ("label_ok", "both_ok", "label_wrong")


def _load_model(run_dir: Path):  # noqa: ANN202
    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k != "extra"})
    model = build_model(
        config,
        CharVocab.load(run_dir / "text_vocab.json"),
        CharVocab.load(run_dir / "kana_vocab.json"),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def _predictions(run_dir: Path, dev_path: Path, dic: Dictionary) -> dict[str, str]:
    """本文 -> 対象 span の予測読み."""
    examples, _ = load_prepared(dev_path, dic)
    rows = [json.loads(line) for line in dev_path.open(encoding="utf-8")]
    model = _load_model(run_dir)

    out: dict[str, str] = {}
    for ex, row in zip(examples, rows, strict=True):
        lattice = ex.lattice()
        path = viterbi(lattice, model.score(ex.text, ex.nodes))
        out[ex.text] = span_reading(lattice, path, row["target_start"], row["target_end"])
    return out


def _error_groups(
    rows: list[TrainingExample], preds: dict[str, str], dic: Dictionary
) -> list[dict]:
    """モデルが外した例を (表記, dev ラベル, 予測) でまとめる.

    ★判定にかけるのは誤りだけでよい. 天井を説明するのが目的なので、
    モデルが当てた例を監査しても説明は増えない.

    ★同じ語の例文は 4 文ずつあり、判定はほぼ同じになる。語ごとにまとめて
    判定回数を 130 -> 約 50 に減らす.
    """
    groups: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        pred = preds.get(row.text, "")
        if pred and phonetic_key(row.target_reading) in phonetic_key(pred):
            continue
        groups.setdefault((row.target_surface, row.target_reading, pred), []).append(row.text)

    out: list[dict] = []
    for (surface, label, pred), texts in groups.items():
        rank, total = reading_rank(dic, surface, label)
        pred_rank, _ = reading_rank(dic, surface, pred)
        out.append(
            {
                "target_surface": surface,
                "dev_label": label,
                "model_pred": pred,
                "examples": texts[:3],
                "n_sentences": len(texts),
                "dict_readings": _readings_by_cost(dic, surface),
                "label_rank": [rank, total],
                "pred_rank": pred_rank,
            }
        )
    out.sort(key=lambda d: -d["n_sentences"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-path", type=Path, default=Path("data/dev_r21.jsonl"))
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--model-dir", type=Path, default=Path("checkpoints/finetune"))
    # ★dev・判定バッチ・集計結果は組で替える（旧 dev は data/audit -> data/dev_audit.jsonl）
    ap.add_argument("--batch-dir", type=Path, default=Path("data/audit_r21"))
    ap.add_argument("--out", type=Path, default=Path("data/dev_audit_r21.jsonl"))
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="判定済みのバッチがあるディレクトリにも --make-batches で書き込む",
    )
    ap.add_argument("--min-rank-ratio", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--make-batches", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)
    rows = [TrainingExample(**r) for r in read_jsonl(args.dev_path)]

    if args.dry_run:
        print(f"dev {len(rows):,} 文\n")
        print("閾値ごとの候補数（★419 文の半分を超えるなら閾値がおかしい）")
        for ratio in (0.3, 0.5, 0.7, 0.9, 0.99):
            n = len(flag_suspicious(rows, dic, min_rank_ratio=ratio))
            print(f"  min_rank_ratio={ratio:<5} -> {n:>4} 件 ({n / len(rows) * 100:.1f}%)")

        flagged = flag_suspicious(rows, dic, min_rank_ratio=args.min_rank_ratio)
        missing = [c for c in flagged if c.rank[0] == 0]
        print(f"\nうち辞書に無い読み: {len(missing)} 件（★最も疑わしい）")
        for c in missing[:10]:
            print(f"  {c.target_surface} -> {c.target_reading}  辞書: {c.alternatives}")
        print(f"\n候補の例（min_rank_ratio={args.min_rank_ratio}）:")
        for c in flagged[:10]:
            print(f"  {c.text}")
            print(f"    {c.target_surface} -> {c.target_reading}  "
                  f"順位 {c.rank[0]}/{c.rank[1]}  辞書: {c.alternatives}")
        return

    if args.make_batches:
        judged = sorted(args.batch_dir.glob("audit_*.out.jsonl"))
        if judged and not args.overwrite:
            raise SystemExit(
                f"★{args.batch_dir} には判定済みのバッチが {len(judged)} 個あります。"
                "上書きすると判定と候補が食い違います。別の --batch-dir を指定するか、"
                "--overwrite を付けてください。"
            )
        preds = _predictions(args.model_dir, args.dev_path, dic)
        items = _error_groups(rows, preds, dic)
        n_sent = sum(i["n_sentences"] for i in items)
        print(f"誤り {n_sent} 件 -> {len(items)} グループ（表記 x ラベル x 予測）")
        args.batch_dir.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(items), args.batch_size):
            chunk = items[i : i + args.batch_size]
            write_jsonl(args.batch_dir / f"audit_{i // args.batch_size:03d}.jsonl", chunk)
        n_batches = (len(items) + args.batch_size - 1) // args.batch_size
        print(f"候補 {len(items):,} 件 -> {n_batches} バッチ を {args.batch_dir} に書き出した")
        return

    if args.collect:
        results: list[dict] = []
        for path in sorted(args.batch_dir.glob("audit_*.out.jsonl")):
            results.extend(read_jsonl(path))
        if not results:
            raise SystemExit(f"{args.batch_dir} に audit_*.out.jsonl がありません")

        counts = Counter(r.get("verdict") for r in results)
        print(f"判定 {len(results):,} 件 / dev {len(rows):,} 文\n")
        for v in VERDICTS:
            n = counts.get(v, 0)
            print(f"  {v:<12} {n:>4} 件  ({n / len(rows) * 100:.1f}% of dev)")
        unknown = set(counts) - set(VERDICTS)
        if unknown:
            print(f"  ★未知の判定: {sorted(unknown)}")

        write_jsonl(args.out, results)
        print(f"\n-> {args.out}")
        return

    ap.error("--dry-run / --make-batches / --collect のいずれかを指定してください")


if __name__ == "__main__":
    main()
