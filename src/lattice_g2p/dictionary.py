"""辞書のロードと共通接頭辞検索.

★UniDic は pron (発音形) と kana (仮名形) の両方を持ち, 両者は一致しない.

    表記   pron   kana   benchmark の表記
    は     ワ     ハ     ワ      ← pron が正しい
    へ     エ     ヘ     エ      ← pron が正しい
    を     オ     ヲ     ヲ      ← kana が正しい
    米     ベー   ベイ   ベイ    ← kana が正しい (長音の書き分け)

どちらか一方では Oracle が落ちるため, **既定では両方を読み候補にする**.
経路の選択はニューラルスコアラの仕事であり, 辞書は候補を漏らさないことを優先する.
詳細は docs/architecture.md §3-1 を参照.
"""

import csv
import pickle
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import marisa_trie

from lattice_g2p.kana import normalize_kana

# UniDic lex CSV の列番号.
# 形式は surface, left_id, right_id, cost, f[0]..f[28] で, f[n] の定義は dicrc にある.
COL_SURFACE = 0
COL_LEFT_ID = 1
COL_RIGHT_ID = 2
"""★連接 ID. MeCab / UniDic はこの 2 つで連接行列 (480 MB) を引く.

M6 でこれを低次元に埋め込んで学習する. 行列そのものは持たない
(docs/architecture.md §4-9)。
"""

COL_COST = 3
COL_POS1 = 4
COL_POS4 = 8
"""pos1..pos4 は 4..7 列 (COL_POS4 は exclusive な終端)."""
COL_PRON = 13
"""f[9] = pron (発音形出現形)."""
COL_KANA = 24
"""f[20] = kana (仮名形出現形)."""

_MIN_COLUMNS = 25

_CACHE_VERSION = 4
"""★4 で left_id / right_id を追加した（M6）。古いキャッシュは作り直しが要る。"""

_CACHE_FIELDS = (
    "surface",
    "reading",
    "pos",
    "entry_id",
    "cost",
    "source",
    "left_id",
    "right_id",
)
"""キャッシュに書き出す Entry のフィールド.

★Entry にフィールドを足したらここも足し、_CACHE_VERSION を上げること。
足し忘れると読み込み時に既定値（0 や空文字）になり、**黙って壊れる**。
M6 で left_id / right_id を足したとき、実際にこれで丸一手戻った。
"""


@dataclass(frozen=True)
class Entry:
    """辞書エントリ 1 件. 表記と読みのペア.

    cost は UniDic の生起コスト (小さいほど出現しやすい).
    """

    surface: str
    reading: str
    pos: str
    entry_id: int
    cost: int = 0

    source: str = ""
    """読みの由来. "pron" / "kana" / "both" / "aug" のいずれか.

    ★pron と kana は一致しない (を -> オ/ヲ, 米 -> ベー/ベイ).
    ラティスには両方を入れないと Oracle が成立しないが, 両者は同コストになるため
    スコアラが区別できない. 由来を特徴として渡せるようにする.
    """

    left_id: int = 0
    right_id: int = 0
    """★連接 ID (M6). 遷移スコアを

        transitions[p, c] = <right_emb[p.right_id], left_emb[c.left_id]>

    で作る. MeCab / UniDic はこの 2 つで 480 MB の連接行列を引くが,
    こちらは低次元に埋め込んで学習する. 補完エントリ (source="aug") は 0.
    """


class Dictionary:
    """表記から読み候補を引く辞書."""

    def __init__(self, entries: list[Entry], keep_pos: bool = False) -> None:
        self._entries = entries
        self.keep_pos = keep_pos
        """品詞違い（連接 ID 違い）のエントリを別に残しているか（R-27）.

        ★キャッシュに保存し、補完（dict_augment）でも引き継ぐ. 黙って既定に戻ると、
        品詞を保ったつもりの測定が品詞を潰した辞書で行われる（R-25 と同じ構図）.
        """
        self._by_surface: dict[str, list[Entry]] = {}
        for e in entries:
            self._by_surface.setdefault(e.surface, []).append(e)
        self._trie = marisa_trie.Trie(self._by_surface.keys())
        self.max_surface_length = max((len(s) for s in self._by_surface), default=0)

    # --- 構築 ---

    @classmethod
    def from_entries(cls, entries: Iterable[Entry], keep_pos: bool = False) -> "Dictionary":
        """エントリ列から辞書を作る.

        読みを正規化し, (表記, 読み) で重複排除する. 重複時は最小コストを残す.

        Args:
            keep_pos: True なら (表記, 読み, left_id, right_id) で重複排除し、
                品詞違いのエントリを別に残す（R-27）. ★既定は False（従来どおり）.
                連接行列を使う構成のためのもの. 品詞を潰すと「実施さ(れる)」の
                サ変の さ が終助詞の さ に潰れ、連接が正しく引けない.
        """
        best: dict[tuple, Entry] = {}
        order: list[tuple] = []
        for e in entries:
            if not e.surface:
                continue
            reading = normalize_kana(e.reading)
            if not reading and not allows_empty_reading(e.surface):
                continue
            key = (e.surface, reading, e.left_id, e.right_id) if keep_pos else (e.surface, reading)
            current = best.get(key)
            if current is None:
                best[key] = replace(e, reading=reading, entry_id=0)
                order.append(key)
            else:
                # ★低コスト側の属性 (pos / 連接 ID) を残す。
                #   pos だけ選んで left_id / right_id を取りこぼすと、
                #   遷移項が「どの語の ID か」を取り違える (M6)。
                keep = current if current.cost <= e.cost else e
                best[key] = replace(
                    keep,
                    reading=reading,
                    entry_id=0,
                    cost=min(current.cost, e.cost),
                    source=_merge_sources(current.source, e.source),
                )

        normalized = [replace(best[k], entry_id=i) for i, k in enumerate(order)]
        return cls(normalized, keep_pos=keep_pos)

    @classmethod
    def from_unidic_csv(
        cls,
        csv_path: Path,
        reading_fields: Sequence[str] = ("pron", "kana"),
        keep_pos: bool = False,
    ) -> "Dictionary":
        """UniDic の lex CSV から辞書を作る.

        Args:
            csv_path: ``lex_3_1.csv`` へのパス
            reading_fields: 読みとして採用する列. ``"pron"`` / ``"kana"`` を指定する.
                既定は両方 (モジュール docstring 参照).
        """
        columns: list[tuple[int, str]] = []
        for field in reading_fields:
            if field == "pron":
                columns.append((COL_PRON, "pron"))
            elif field == "kana":
                columns.append((COL_KANA, "kana"))
            else:
                raise ValueError(f"未知の reading_field: {field!r}")

        csv.field_size_limit(sys.maxsize)
        raw: list[Entry] = []
        with csv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) < _MIN_COLUMNS or not row[COL_SURFACE]:
                    continue
                pos = "-".join(p for p in row[COL_POS1:COL_POS4] if p and p != "*")
                cost = int(row[COL_COST]) if _is_int(row[COL_COST]) else 0
                left_id = int(row[COL_LEFT_ID]) if _is_int(row[COL_LEFT_ID]) else 0
                right_id = int(row[COL_RIGHT_ID]) if _is_int(row[COL_RIGHT_ID]) else 0
                for col, source in columns:
                    raw.append(
                        Entry(
                            surface=row[COL_SURFACE],
                            reading=row[col],
                            pos=pos,
                            entry_id=0,
                            cost=cost,
                            source=source,
                            left_id=left_id,
                            right_id=right_id,
                        )
                    )
        return cls.from_entries(raw, keep_pos=keep_pos)

    # --- 検索 ---

    def common_prefix_search(self, text: str, start: int) -> list[Entry]:
        """text[start:] の接頭辞になっている全エントリを返す."""
        if start >= len(text):
            return []
        out: list[Entry] = []
        for surface in self._trie.prefixes(text[start:]):
            out.extend(self._by_surface[surface])
        return out

    def readings_of(self, surface: str) -> list[str]:
        """表記に対する読み候補 (登録順, 重複なし).

        ★keep_pos では同じ読みのエントリが品詞違いで複数あるので、読みで重複を除く.
        """
        return list(dict.fromkeys(e.reading for e in self._by_surface.get(surface, [])))

    def entries_of(self, surface: str) -> list[Entry]:
        return list(self._by_surface.get(surface, []))

    def surfaces(self) -> list[str]:
        return list(self._by_surface.keys())

    def all_entries(self) -> list[Entry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # --- キャッシュ ---

    def save(self, path: Path) -> None:
        """辞書をキャッシュに保存する.

        ``lex_3_1.csv`` は 233 MB あり, 毎回パースすると数十秒かかる.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _CACHE_VERSION,
            # ★Entry にフィールドを足したらここと _CACHE_VERSION も直すこと。
            #   タプル化しているので、足し忘れると黙って既定値になる（M6 で踏んだ）。
            "fields": _CACHE_FIELDS,
            "keep_pos": self.keep_pos,
            "entries": [
                tuple(getattr(e, name) for name in _CACHE_FIELDS) for e in self._entries
            ],
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "Dictionary":
        with path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("version") != _CACHE_VERSION:
            raise ValueError(
                f"キャッシュのバージョンが違います: {payload.get('version')} != {_CACHE_VERSION}"
            )
        fields = payload.get("fields", _CACHE_FIELDS)
        if tuple(fields) != _CACHE_FIELDS:
            raise ValueError(
                f"キャッシュの列が違います: {tuple(fields)} != {_CACHE_FIELDS}。"
                "scripts/build_dict_cache.py で作り直してください。"
            )
        return cls(
            [Entry(**dict(zip(fields, row, strict=True))) for row in payload["entries"]],
            # ★方針を持たない以前のキャッシュは既定（品詞を潰す）
            keep_pos=bool(payload.get("keep_pos", False)),
        )


def _merge_sources(a: str, b: str) -> str:
    """同じ (表記, 読み) が複数の列から来た場合に由来をまとめる."""
    parts = {p for p in (a, b) if p}
    if parts == {"pron", "kana"} or "both" in parts:
        return "both"
    if len(parts) == 1:
        return next(iter(parts))
    return ""


def _is_int(s: str) -> bool:
    return s.lstrip("-").isdigit()


def _is_speakable(ch: str) -> bool:
    """仮名または漢字か. 読みを持つべき文字かどうかの判定に使う."""
    return (
        "ぁ" <= ch <= "ゖ"  # ひらがな
        or "ァ" <= ch <= "ヺ"  # カタカナ
        or ch == "ー"  # ー
        or "一" <= ch <= "鿿"  # CJK 統合漢字
        or "㐀" <= ch <= "䶿"  # CJK 拡張 A
        or "豈" <= ch <= "﫿"  # CJK 互換漢字
    )


def allows_empty_reading(surface: str) -> bool:
    """その表記に空の読みを許してよいか.

    空読みが正当なのは句読点や記号だけである.

    UniDic は顔文字 (補助記号-ＡＡ) の部品として「人」のような漢字も
    読みなしで登録している. これを残すとラティス上に「人」を読み飛ばす経路ができ,
    モデルが文字を脱落させた読みを出力できてしまう.

    pos の分類に頼らず表記の文字種で判定する. 分類の揺れに影響されないため.
    """
    return not any(_is_speakable(ch) for ch in surface)
