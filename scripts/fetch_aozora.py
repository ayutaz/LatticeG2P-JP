"""青空文庫からルビ付きテキストを取得する.

★ルビは実際の出版物で人間が付けた読みなので hallucination がない.
LLM 生成の弱点 (フィルタで検出できない文脈的な誤読) がそもそも存在しない.

取得条件は 2 つとも必須である.

    文字遣い種別 = 新字新仮名   歴史的仮名遣い (今日《けふ》) を避ける
    作品著作権フラグ = なし     パブリックドメインのものだけ

★旧仮名の作品を混ぜてはいけない. ルビが ケフ / ヰ / ヱ で書かれていると
UniDic の読みと一致せず, ラティス整合フィルタで全部落ちる. 歩留まりが
下がるだけでなく「落ちた理由」の切り分けが難しくなる.

サーバに負荷をかけないよう 1 件ずつ間隔を空けて取得し, 既にあるファイルは
飛ばす (途中再開できる).
"""

import argparse
import csv
import io
import random
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

INDEX_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
INDEX_NAME = "list_person_all_extended_utf8.csv"

USER_AGENT = "LatticeG2P-JP/0.1 (research; contact via repository)"
SLEEP_SECONDS = 0.6
"""1 件ごとの待ち時間. 青空文庫は個人運営のサイトなので詰めない."""


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as res:  # noqa: S310
        return res.read()


def load_index(path: Path) -> list[dict[str, str]]:
    """作品一覧 CSV を読む. 無ければ取得する."""
    if not path.exists():
        print(f"作品一覧を取得: {INDEX_URL}")
        blob = _get(INDEX_URL, timeout=180)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            data = zf.read(INDEX_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_works(
    rows: list[dict[str, str]], limit: int, per_author: int, seed: int
) -> list[dict[str, str]]:
    """新字新仮名・著作権なし・ルビ付きの作品を選ぶ.

    ★1 人の作家に偏らせない. 文体と語彙の多様性が学習データの価値を決める
    (docs/pitfalls.md R-09).
    """
    usable = [
        r
        for r in rows
        if r["文字遣い種別"] == "新字新仮名"
        and r["作品著作権フラグ"] == "なし"
        and r["テキストファイル符号化方式"] == "ShiftJIS"
        and "_ruby_" in r["テキストファイルURL"]
    ]
    usable.sort(key=lambda r: (r["人物ID"], r["作品ID"]))
    random.Random(seed).shuffle(usable)

    seen: dict[str, int] = defaultdict(int)
    picked: list[dict[str, str]] = []
    for row in usable:
        if seen[row["人物ID"]] >= per_author:
            continue
        seen[row["人物ID"]] += 1
        picked.append(row)
        if len(picked) >= limit:
            break
    return picked


def fetch_work(row: dict[str, str], out_dir: Path) -> Path | None:
    """1 作品を取得して UTF-8 のテキストとして保存する.

    Returns:
        保存先. 既にあれば取得せずそのパスを返す. 失敗したら None.
    """
    out = out_dir / f"{row['作品ID']}.txt"
    if out.exists():
        return out

    try:
        blob = _get(row["テキストファイルURL"])
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not names:
                return None
            raw = zf.read(names[0])
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {row['作品ID']} {row['作品名']}: {e}")
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(raw.decode("cp932", errors="replace"), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/aozora/index.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/aozora/texts"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--per-author", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_index(args.index)
    works = select_works(rows, args.limit, args.per_author, args.seed)
    print(f"対象 {len(works):,} 作品（新字新仮名・著作権なし・ルビ付き）\n")

    fetched = 0
    for i, row in enumerate(works, 1):
        before = (args.out_dir / f"{row['作品ID']}.txt").exists()
        path = fetch_work(row, args.out_dir)
        if path is None:
            continue
        fetched += 1
        if not before:
            time.sleep(SLEEP_SECONDS)
        if i % 20 == 0 or i == len(works):
            print(f"  {i:>4}/{len(works)}  {row['姓']}{row['名']}「{row['作品名']}」")

    print(f"\n取得 {fetched:,} 作品 -> {args.out_dir}")


if __name__ == "__main__":
    main()
