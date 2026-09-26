"""BiEncoder と CrossEncoder を同条件で比較する.

★同じデータ・同じ epoch 数・同じ学習率で走らせること。
条件が違うと比較にならない（docs/plans/M3 Task 8）。

    uv run python scripts/compare_scorers.py --epochs 10 --device cpu
"""

import argparse
import json
import time
from pathlib import Path

from lattice_g2p.config import TrainConfig
from lattice_g2p.train import train

CONFIGS = {
    "bi_encoder": Path("configs/tiny.yaml"),
    "cross_encoder": Path("configs/tiny-cross.yaml"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    # ★現行の分割。train と dev は組で替える（旧 train_split には dev_r21 の語が 349 行ある）
    ap.add_argument("--train-path", type=Path, default=Path("data/train_m5_split.jsonl"))
    ap.add_argument("--dev-path", type=Path, default=Path("data/dev_r21.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints/compare"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results: dict[str, dict] = {}

    for name, config_path in CONFIGS.items():
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        config = TrainConfig.load(config_path)
        config.train_path = args.train_path
        config.dev_path = args.dev_path
        config.device = args.device
        config.seed = args.seed
        config.out_dir = args.out_dir / name
        if args.epochs is not None:
            config.epochs = args.epochs

        started = time.time()
        metrics = train(config)
        metrics["wall_clock_s"] = time.time() - started
        results[name] = metrics

    print(f"\n{'=' * 72}")
    print(f"{'scorer':<16}{'params':>10}{'train':>10}{'dev':>10}{'best dev':>10}{'time':>10}")
    print("-" * 72)
    for name, m in results.items():
        print(
            f"{name:<16}{m['params_m']:>9.2f}M{m['train_accuracy'] * 100:>9.2f}%"
            f"{m['dev_accuracy'] * 100:>9.2f}%{m['best_dev_accuracy'] * 100:>9.2f}%"
            f"{m['wall_clock_s']:>9.0f}s"
        )

    out = args.out_dir / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果を {out} に書き出した")


if __name__ == "__main__":
    main()
