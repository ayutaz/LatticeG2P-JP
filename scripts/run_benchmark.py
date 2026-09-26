"""CPU レイテンシベンチマーク. ★必ずローカルで実行すること.

    uv run python scripts/run_benchmark.py --model models/m4-int8

vast.ai 上で測ってはいけない (docs/pitfalls.md R-13). Go/No-Go 判定の
半分がレイテンシなので, 測定場所を間違えると判定そのものが無意味になる.
"""

import argparse
from pathlib import Path

from lattice_g2p.baselines import openjtalk_readings
from lattice_g2p.benchmark import environment_info, measure, run_benchmark
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.runtime import OnnxG2P

TEXTS: dict[int, list[str]] = {
    20: ["今日は天気がいいので散歩に行きます。"],
    50: [
        "今日は天気がいいので、近くの公園まで散歩に行こうと思いますが、"
        "夕方から雨が降るかもしれません。"
    ],
    100: [
        "今日は天気がいいので、近くの公園まで散歩に行こうと思いますが、"
        "夕方から雨が降るかもしれないので折り畳み傘を持っていくことにしました。"
        "帰りには米と魚を買って、夕食の支度をする予定です。"
    ],
}
"""20 / 50 / 100 文字の評価用文 (docs/evaluation.md §5-1)."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, nargs="+", required=True)
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--openjtalk", action="store_true", help="OpenJTalk も同じ文で測る")
    args = ap.parse_args()

    print(environment_info())
    print(f"threads      : {args.threads}")
    print(f"trials       : {args.trials}\n")

    dic = Dictionary.load(args.dict_cache)

    for n_chars, texts in TEXTS.items():
        print(f"=== 約 {n_chars} 文字（実際 {len(texts[0])} 文字）===")
        for model_dir in args.model:
            g2p = OnnxG2P(model_dir, dic, num_threads=args.threads)
            result = run_benchmark(g2p, texts, trials=args.trials)
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1e6
            print(f"\n--- {model_dir.name}（{size:.1f} MB）---")
            print(result.format())
        if args.openjtalk:
            result = measure(
                lambda t: (openjtalk_readings(t), None),
                texts,
                label="OpenJTalk",
                trials=args.trials,
            )
            print("\n--- OpenJTalk（pyopenjtalk）---")
            print(result.format())
        print()


if __name__ == "__main__":
    main()
