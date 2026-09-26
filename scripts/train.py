"""学習を実行する.

    uv run python scripts/train.py --config configs/tiny.yaml

★GPU が必要な処理は vast.ai で実行する（docs/training.md §9）。
"""

import argparse
import json
import sys
from pathlib import Path

from lattice_g2p.config import TrainConfig
from lattice_g2p.train import train


def main() -> None:
    # ★ログをファイルにリダイレクトするとブロックバッファになり、
    #   学習が終わるまで 1 行も見えない。課金されている GPU の進捗が
    #   分からないのは困るので行バッファに固定する。
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--train-path", type=Path, default=None)
    ap.add_argument("--dev-path", type=Path, default=None)
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="別の学習結果（out_dir）から重みと語彙を引き継ぐ。optimizer は引き継がない",
    )
    ap.add_argument(
        "--ruby-path",
        type=Path,
        default=None,
        help="青空文庫のルビ由来データ。★学習データにだけ足される（dev には使えない）",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--init-from-prefer",
        choices=("best", "latest"),
        default=None,
        help="init_from で best.pt と latest.pt のどちらを使うか（★docstring 参照）",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="★同一条件を seed だけ変えて反復すると、ノイズ床が測れる。"
        "この計測系では対象語で約 0.3 pt（docs/pitfalls.md R-24）",
    )
    ap.add_argument("--metrics-out", type=Path, default=None)
    args = ap.parse_args()

    config = TrainConfig.load(args.config)
    for name in (
        "train_path",
        "dev_path",
        "ruby_path",
        "init_from",
        "init_from_prefer",
        "seed",
        "out_dir",
        "device",
        "epochs",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)

    print(f"config: {args.config}")
    metrics = train(config)

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(
                {"config": config.to_dict(), "metrics": metrics},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n指標を {args.metrics_out} に書き出した")


if __name__ == "__main__":
    main()
