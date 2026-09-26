"""学習済みチェックポイントを ONNX の 2 グラフに書き出す.

    uv run python scripts/export_model.py --checkpoint checkpoints/m4 --out models/m4-fp32

★書き出したあと, 同じ文で PyTorch と ONNX のスコアを突き合わせる.
ここを省くと, 後段で精度が落ちたときに量子化のせいか export バグかを
区別できなくなる (docs/architecture.md §7-2).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from lattice_g2p.dictionary import Dictionary
from lattice_g2p.export_onnx import export_modules, quantize
from lattice_g2p.lattice import build_lattice
from lattice_g2p.runtime import OnnxG2P
from lattice_g2p.train import load_scorer

CHECK_TEXTS = [
    "米国の記者が来た。",
    "生の魚を生で食べる。",
    "今日は一日中雨だった。",
]

TOLERANCE = 1e-3

def _report_size(out_dir: Path) -> float:
    total = 0.0
    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size / 1e6
        total += size
        print(f"  {f.name:24s} {size:8.2f} MB")
    print(f"  {'合計':24s} {total:8.2f} MB")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument(
        "--quantize-out", type=Path, default=None, help="指定すると INT8 版も書き出す"
    )
    args = ap.parse_args()

    name, scorer = load_scorer(args.checkpoint, "cpu")
    n_params = sum(p.numel() for p in scorer.parameters())
    print(f"{name}  {n_params / 1e6:.2f}M params  <- {args.checkpoint}")

    export_modules(scorer, args.out)
    fp32_size = _report_size(args.out)

    if args.quantize_out is not None:
        print(f"\nINT8 量子化 -> {args.quantize_out}")
        quantize(args.out, args.quantize_out)
        int8_size = _report_size(args.quantize_out)
        print(
            f"  圧縮率 {int8_size / fp32_size * 100:.1f}%"
            f"（{fp32_size:.1f} -> {int8_size:.1f} MB）"
        )

    dic = Dictionary.load(args.dict_cache)
    runtime = OnnxG2P(args.out, dic)

    print("\nPyTorch と ONNX の突き合わせ:")
    worst = 0.0
    for text in CHECK_TEXTS:
        lat = build_lattice(text, dic)
        with torch.no_grad():
            expected = scorer.score(lat.text, lat.nodes).numpy()
        diff = float(np.abs(expected - runtime.score_nodes(lat.text, lat.nodes)).max())
        worst = max(worst, diff)
        print(f"  {text}  ノード {len(lat.nodes):4d}  最大差 {diff:.2e}  -> {runtime.g2p(text)}")

    if worst > TOLERANCE:
        raise SystemExit(f"★スコアが一致しない（最大差 {worst:.2e} > {TOLERANCE:.0e}）")
    print(f"\n✓ 一致（最大差 {worst:.2e}）")


if __name__ == "__main__":
    sys.exit(main())
