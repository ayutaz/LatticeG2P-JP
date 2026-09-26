"""UniDic の lex CSV を読み込んでキャッシュを作る.

233 MB の CSV を毎回パースすると数十秒かかるため, 一度だけ変換して保存する.
"""

import argparse
import time
from pathlib import Path

from lattice_g2p.dict_augment import augment_dictionary
from lattice_g2p.dictionary import Dictionary

DEFAULT_CSV = Path("data/unidic-src/unidic-cwj-3.1.1-full/lex_3_1.csv")
DEFAULT_CACHE = Path("data/dic_cache.pkl")

AUGMENT_ROUNDS = 1
AUGMENT_MAX_SURFACE_LEN = 4
"""複合語からの単漢字読み抽出の設定.

rounds を増やすと Oracle はほぼ改善しないのにラティスのノード数だけ増える:

    構成             entries   Oracle全体  ｽｺｰﾌﾟ内  平均node
    補完なし         874,018     99.14%    99.58%     52.4
    rounds=1         885,203     99.46%    99.90%     83.3   <- 採用
    rounds=2         887,587     99.47%    99.91%     90.9
    rounds=3         887,894     99.47%    99.91%     93.3

ノード数は M4 の推論レイテンシに直結するため, rounds=1 を採用する.
"""


def load_dictionary(cache: Path = DEFAULT_CACHE, csv_path: Path = DEFAULT_CSV) -> Dictionary:
    """キャッシュがあればそこから, なければ CSV から読み込む."""
    if cache.exists():
        return Dictionary.load(cache)
    dic = Dictionary.from_unidic_csv(csv_path)
    dic.save(cache)
    return dic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--reading-fields", nargs="+", default=["pron", "kana"])
    ap.add_argument("--augment-rounds", type=int, default=AUGMENT_ROUNDS)
    ap.add_argument(
        "--keep-pos",
        action="store_true",
        help="品詞違い（連接 ID 違い）のエントリを別に残す（R-27）。出力は --out で分けること",
    )
    args = ap.parse_args()
    if args.keep_pos and args.out == DEFAULT_CACHE:
        raise SystemExit(
            f"★--keep-pos のときは --out を {DEFAULT_CACHE} 以外にしてください"
            "（既定の辞書を上書きしない）"
        )

    t = time.perf_counter()
    dic = Dictionary.from_unidic_csv(
        args.csv, reading_fields=tuple(args.reading_fields), keep_pos=args.keep_pos
    )
    print(f"parse: {time.perf_counter() - t:.1f}s")
    print(f"entries (生)  : {len(dic):,}")

    if args.augment_rounds:
        t = time.perf_counter()
        dic, added = augment_dictionary(
            dic, max_surface_len=AUGMENT_MAX_SURFACE_LEN, rounds=args.augment_rounds
        )
        print(f"単漢字読みの補完: +{added:,}  ({time.perf_counter() - t:.1f}s)")

    print(f"entries      : {len(dic):,}")
    print(f"surfaces     : {len(dic.surfaces()):,}")
    print(f"max surface  : {dic.max_surface_length}")

    dic.save(args.out)
    print(f"saved -> {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")

    for s in ["米", "は", "へ", "を", "人", "生", "。"]:
        print(f"  {s}: {dic.readings_of(s)[:8]}")


if __name__ == "__main__":
    main()
