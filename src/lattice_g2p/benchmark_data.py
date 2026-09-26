"""Joyo-Kanji-Yomi Benchmark (Parakeet Edition) のロード.

データ形式は data/benchmark/SCHEMA.md を参照.
評価対象の漢字は ``tagged_text`` 中で ``<...>`` に囲まれており, 対応する読みが
``tagged_yomi`` 中で同様に囲まれている.

``readings.natural`` は複数ありうる. 「憧憬」が「ショウケイ」とも「ドウケイ」とも
読めるように, 文脈から一意に決まらない場合が存在するため, Accuracy の判定では
このリストのいずれかに一致すれば正解として扱う.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from lattice_g2p.kana import normalize_kana

_OPEN = "<"
_CLOSE = ">"


def split_tagged(tagged: str) -> tuple[str, int, int]:
    """``<...>`` で囲まれた文字列を, タグなし文字列と span に分解する.

    Returns:
        (タグを除去した文字列, 開始位置, 終了位置). 終了位置は exclusive.

    Raises:
        ValueError: タグがない, 複数ある, または閉じていない場合
    """
    if tagged.count(_OPEN) != 1 or tagged.count(_CLOSE) != 1:
        raise ValueError(f"タグはちょうど 1 組でなければなりません: {tagged!r}")

    start = tagged.index(_OPEN)
    close = tagged.index(_CLOSE)
    if close < start:
        raise ValueError(f"タグの開閉が逆です: {tagged!r}")
    if close == start + 1:
        raise ValueError(f"タグの中身が空です: {tagged!r}")

    inner = tagged[start + 1 : close]
    plain = tagged[:start] + inner + tagged[close + 1 :]
    return plain, start, start + len(inner)


@dataclass(frozen=True)
class BenchmarkItem:
    """評価データ 1 件."""

    key: str
    text: str
    reading: str
    """文全体の読み. 正規化前の生の値."""

    target_start: int
    target_end: int
    """text 上の評価対象の span. end は exclusive."""

    target_surface: str
    target_reading: str
    """tagged_yomi から取り出した, この文における対象語の読み."""

    target_kana_start: int
    target_kana_end: int
    """正規化した ``reading`` の中で対象語の読みが占める位置.

    ★経路が target span をまたぐノードを選ぶことがあるため, ノードの読みを
    そのまま取り出すと対象外の読みまで含んでしまう
    (「固める -> カタメル」の 1 ノードで target が「固」だけ, など).
    benchmark 自身が tagged_yomi でカナ位置を示しているので, それを使う.
    """

    natural_readings: tuple[str, ...]
    """この文脈で自然に許容できる読み. 複数ありうる."""

    marginal_readings: tuple[str, ...]
    """辞書上は許容できるが, この文脈では一般的でない読み."""

    reading_category: str
    source: str

    def acceptable_readings(self, include_marginal: bool = False) -> tuple[str, ...]:
        """正解として扱う読みの一覧."""
        if include_marginal:
            return self.natural_readings + self.marginal_readings
        return self.natural_readings


def load_joyo_benchmark(path: Path) -> list[BenchmarkItem]:
    """benchmark の JSONL を読み込む.

    Raises:
        ValueError: tagged_text と text が矛盾する行があった場合
    """
    items: list[BenchmarkItem] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            plain_text, start, end = split_tagged(obj["tagged_text"])
            if plain_text != obj["text"]:
                raise ValueError(
                    f"{path}:{lineno}: tagged_text からタグを除いたものが text と一致しません\n"
                    f"  text        : {obj['text']!r}\n"
                    f"  tagged 除去後: {plain_text!r}"
                )

            plain_yomi, yomi_start, yomi_end = split_tagged(obj["tagged_yomi"])
            target_reading = plain_yomi[yomi_start:yomi_end]

            # 正規化で句読点が消えるぶん, カナ位置がずれる
            kana_start = len(normalize_kana(plain_yomi[:yomi_start]))
            kana_end = kana_start + len(normalize_kana(target_reading))

            readings = obj.get("readings", {})
            items.append(
                BenchmarkItem(
                    key=obj["key"],
                    text=obj["text"],
                    reading=obj["yomi"],
                    target_start=start,
                    target_end=end,
                    target_surface=obj["text"][start:end],
                    target_reading=target_reading,
                    target_kana_start=kana_start,
                    target_kana_end=kana_end,
                    natural_readings=tuple(readings.get("natural", [])),
                    marginal_readings=tuple(readings.get("marginal", [])),
                    reading_category=obj.get("reading_category", ""),
                    source=obj.get("source", ""),
                )
            )
    return items
