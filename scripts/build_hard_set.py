"""Hard Set（独自評価セット）を benchmark 形式に組み立てる（M4 Task 4）.

論文が弱点として認めている領域を実際に測るためのもの:
固有名詞（人名・地名・企業）・数詞 + 助数詞・同表記異読。

★これは**物差し**である。物差し自体が間違っていると測定の意味が無くなるので、
少しでも怪しい行は落とす。歩留まりより正しさを優先する。

  1. 対象語の読みは人手で書いた表（data/hard_set/table.jsonl）で確定
  2. 文はサブエージェントが生成
  3. ここで機械的に検証する:
     - 対象表記が文中にちょうど 1 回だけ出る
     - 対象語の読みが文全体の読みにちょうど 1 回だけ出る（カナ位置が一意）
     - 文全体の読みが辞書ラティスで生成できる
     - その位置でその読みになる経路が存在する

    uv run python scripts/build_hard_set.py
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lattice_g2p.baselines import (
    clean_reading,
    openjtalk_readings,
    span_reading_from_morphemes,
)
from lattice_g2p.crf import build_constrained_graph, build_partially_constrained_graph
from lattice_g2p.data.schema import read_jsonl
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import normalize_kana, phonetic_key
from lattice_g2p.lattice import build_lattice

sys.stdout.reconfigure(line_buffering=True)

REASONS = (
    "表記が文中に無い/複数ある",
    "指定と違う読みが書かれている",
    "読みのカナ位置が一意でない",
    "文全体の読みがラティスで生成できない",
    "その位置でその読みになる経路が無い",
    "★対象語の前後の読みが OpenJTalk と食い違う",
    "★語そのものを話題にした文（読みが定まらない）",
    "★文が壊れている（ひらがなが 1 文字も無い）",
)
"""落とす理由. ★どれも「怪しいから落とす」であって「間違いを証明した」ではない."""


META = ("読み", "音読", "訓読", "呼び方", "呼ばれ", "表記", "と書く", "と書いて", "別の音",
        "という名", "という苗字", "という地名", "という字", "難読", "当て字", "由来する",
        "数字の名前")
"""★語そのものを話題にした文を弾く語.

「東海林という地名もある」「東海林は音読みでも読める」のような文は,
語の**使用例**ではなく**言及**であり, 文脈が読みを決めない.
評価データとしては役に立たない (M3c と同じ教訓: 怪しいラベルは評価側を直す).
"""


def is_meta(text: str, surface: str) -> bool:
    return any(w in text for w in META)


def is_broken(text: str) -> bool:
    """★ひらがなが 1 文字も無い文を弾く.

    「一日ニ開店スル。」のようにカタカナ書きされた文が混ざる.
    ラティス整合は通ってしまう (UniDic はカタカナのエントリも持つ) が,
    評価データとしては壊れている.
    """
    return not any("\u3041" <= ch <= "\u309f" for ch in text)


def context_agrees(text: str, reading: str, start: int, end: int, kana_start: int,
                   kana_end: int) -> bool:
    """対象語の**前後**の読みが OpenJTalk と一致するか.

    ★ここを見る理由: 評価は「予測した文全体の読みの, 正解が示すカナ位置」で行う.
    対象語より前の読みが 1 文字でもずれていると, カナ位置がずれて
    **正しい予測が不正解になる**. 由緒 -> ユショ のような周辺語の誤りは
    ラティス整合では検出できない (辞書が 由=ユ, 緒=ショ を持つため).

    ★対象語そのものは比較しない. そこは人手の表で確定しており,
    OpenJTalk が間違うことこそ測りたいものだから.

    ★比較は phonetic_key で行う. OpenJTalk は長音を チメー と書き,
    こちらは チメイ と書く. これは表記の違いであって読みの違いではない
    (docs/architecture.md §9 不変条件 9).

    ★★形態素が span の境界をまたぐ側は比較しない.
    span_reading_from_morphemes は span と重なる形態素を丸ごと含めるので,
    OpenJTalk が対象語を隣と融合させると前後に対象語の読みが混ざり,
    **必ず不一致になる**. `一二三` を「123」と読む, `心地よい` を 1 語にする, など.

    これを弾くと「OpenJTalk が大きく外す語ほど Hard Set から消える」ことになり,
    **比較相手に有利なバイアスがかかる**. 境界をまたぐ側はチェックを諦め,
    またがない側だけを見る (両側またぐなら素通しする).
    """
    morphemes = openjtalk_readings(text)
    bounds = set()
    pos = 0
    for surface, _ in morphemes:
        bounds.add(pos)
        pos += len(surface)
    bounds.add(pos)

    same = lambda a, b: phonetic_key(normalize_kana(a)) == phonetic_key(b)  # noqa: E731
    if start in bounds:
        oj_prefix = clean_reading(span_reading_from_morphemes(morphemes, 0, start))
        if not same(oj_prefix, reading[:kana_start]):
            return False
    if end in bounds:
        oj_suffix = clean_reading(span_reading_from_morphemes(morphemes, end, len(text)))
        if not same(oj_suffix, reading[kana_end:]):
            return False
    return True


def check(row: dict, allowed: dict[str, set[str]], dic: Dictionary,
          verify_context: bool) -> tuple[str | None, dict]:
    """1 行を検証する. (落とす理由 or None, 追加情報) を返す."""
    text, surface = row["text"], row["target_surface"]
    reading = normalize_kana(row["reading"])
    target = normalize_kana(row["target_reading"])
    start = int(row["target_start"])
    end = start + len(surface)

    if text.count(surface) != 1 or text[start:end] != surface:
        return REASONS[0], {}
    if target not in allowed.get(surface, set()):
        return REASONS[1], {}
    if reading.count(target) != 1:
        return REASONS[2], {}
    if is_meta(text, surface):
        return REASONS[6], {}
    if is_broken(text):
        return REASONS[7], {}

    lattice = build_lattice(text, dic)
    if not build_constrained_graph(lattice, reading).is_reachable():
        return REASONS[3], {}
    if not build_partially_constrained_graph(lattice, [(start, end, target)]).is_reachable():
        return REASONS[4], {}

    kana_start = reading.index(target)
    kana_end = kana_start + len(target)
    if verify_context and not context_agrees(text, reading, start, end, kana_start, kana_end):
        return REASONS[5], {}

    return None, {
        "text": text,
        "tagged_text": text[:start] + f"<{surface}>" + text[end:],
        "yomi": reading,
        "tagged_yomi": (
            reading[:kana_start] + f"<{target}>" + reading[kana_end:]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("data/hard_set/table.jsonl"))
    ap.add_argument(
        "--batch-dir", type=Path, nargs="+", default=[Path("data/batches_hard")]
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/hard_set"))
    ap.add_argument("--train", type=Path, default=Path("data/train_m5_split.jsonl"))
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument(
        "--no-verify-context",
        action="store_true",
        help="対象語の前後を OpenJTalk と突き合わせない（--extra bench が無いとき）",
    )
    args = ap.parse_args()

    dic = Dictionary.load(args.dict_cache)

    category: dict[str, str] = {}
    allowed: dict[str, set[str]] = defaultdict(set)
    free: dict[str, list[str]] = {}
    """★自由異読の表記 -> 認める読みの一覧.

    三階 (サンガイ / サンカイ) のように指す内容が同じで発音の差しかない組は,
    文脈に手がかりが残らない. 片方に固定すると**原理的に解けない問題**になるので,
    両方を正解として認める. 音便の知識そのものは測れる.
    """
    for row in read_jsonl(args.table):
        category[row["surface"]] = row["category"]
        readings = [normalize_kana(r) for r in row["readings"]]
        allowed[row["surface"]].update(readings)
        if row.get("free_variants"):
            free[row["surface"]] = readings

    trained = {json.loads(line)["target_surface"] for line in args.train.open(encoding="utf-8")}

    outs = sorted(p for d in args.batch_dir for p in d.glob("*.out.jsonl"))
    print(f"バッチ {len(outs)} 本")

    kept: dict[str, list[dict]] = defaultdict(list)
    dropped: Counter[str] = Counter()
    seen_keys: Counter[str] = Counter()
    n_total = 0

    for path in outs:
        for row in read_jsonl(path):
            n_total += 1
            reason, extra = check(row, allowed, dic, not args.no_verify_context)
            if reason:
                dropped[reason] += 1
                continue
            surface = row["target_surface"]
            cat = category[surface]
            key = f"{surface}_{normalize_kana(row['target_reading'])}"
            seen_keys[key] += 1
            kept[cat].append(
                {
                    "key": f"{key}_{seen_keys[key] - 1}",
                    **extra,
                    "reading_category": cat,
                    "readings": {
                        "natural": free.get(
                            surface, [normalize_kana(row["target_reading"])]
                        ),
                        "marginal": [],
                    },
                    "seen_in_train": surface in trained,
                    "source": "hard_set_generated",
                }
            )

    n_kept = sum(len(v) for v in kept.values())
    rate = n_kept / max(n_total, 1) * 100
    print(f"生成 {n_total:,} 文 -> 採用 {n_kept:,} 文  （歩留まり {rate:.1f}%）")
    for reason in REASONS:
        if dropped[reason]:
            print(f"  落とした: {reason}  {dropped[reason]:,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'カテゴリ':<22}{'文':>6}{'うち学習済み':>12}")
    for cat in sorted(kept):
        rows = kept[cat]
        seen = sum(1 for r in rows if r["seen_in_train"])
        print(f"{cat:<22}{len(rows):6d}{seen:12d}")
        with (args.out_dir / f"{cat}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{args.out_dir}/<カテゴリ>.jsonl に書き出した")


if __name__ == "__main__":
    main()
