"""UniDic から学習対象の多音語を選ぶ.

方針 (docs/training.md §2-1):
  1. 読み候補が 2〜5 個のもの (多すぎるものは辞書ノイズの可能性)
  2. 固有名詞を除外 (論文と同じ. 第 1 マイルストーンの範囲外)
  3. 1 文字の助詞・記号を除外

★学習の価値は多音語の文脈判別にある. 単一読みの語は辞書を引けば正解するので,
生成コストを割く意味が薄い.
"""

import re
from dataclasses import dataclass

from lattice_g2p.dict_augment import is_kanji
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import phonetic_key

UNUSABLE_POS_PREFIXES = (
    "名詞-固有名詞",
    "補助記号",
    "記号",
    "空白",
)
"""学習対象の読みとして数えない品詞.

★「その品詞のエントリがあれば表記ごと除外」ではない.
「米」は普通名詞 (コメ) でありながら固有名詞 (米国 -> ベイ) でもあり,
「人」は人名由来の読みを大量に持つ. any() で除外すると,
最も学習価値の高い常用漢字が軒並み落ちる.

補完由来 ("補完-単漢字") は学習対象として有効なので含めない.
"""

EXCLUDED_SINGLE_CHAR_POS_PREFIXES = (
    "助詞",
    "助動詞",
)
"""1 文字のときだけ対象外にする品詞."""

OUT_OF_SCOPE_SURFACE = re.compile(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]")
"""数字・ラテン文字を含む表記は範囲外 (docs/pitfalls.md R-05)."""


def fold_variant_readings(readings: list[str]) -> list[str]:
    """長音の書き分けだけが違う読みを畳む (表記の昇順で返す).

    UniDic の pron 形と kana 形は同じ読みを別表記で持つことがある
    (亢 -> コウ / コー). これを別の読みとして数えると, 多音語の抽出結果が
    異表記ペアだらけになる.

    各グループの代表として, 長音符を含まない表記を優先する
    (benchmark の表記が仮名形のため).
    """
    return [r for r, _ in _fold_with_cost([(r, 0) for r in readings])]


def _fold_with_cost(pairs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """(読み, コスト) を畳み, コストの小さい順に返す.

    同じ音の異表記のうち, 長音符を含まない表記を代表にする.
    グループのコストは最小値を採る.
    """
    groups: dict[str, list[tuple[str, int]]] = {}
    for reading, cost in pairs:
        groups.setdefault(phonetic_key(reading), []).append((reading, cost))

    folded: list[tuple[str, int]] = []
    for group in groups.values():
        representative = min(group, key=lambda rc: (rc[0].count("ー"), rc[0]))[0]
        folded.append((representative, min(c for _, c in group)))

    return sorted(folded, key=lambda rc: (rc[1], rc[0]))


@dataclass(frozen=True)
class PolyphonicWord:
    surface: str
    readings: tuple[str, ...]
    pos_list: tuple[str, ...]


def select_polyphonic(
    dic: Dictionary,
    min_readings: int = 2,
    max_readings: int = 8,
    limit: int | None = None,
    max_surface_len: int | None = None,
) -> list[PolyphonicWord]:
    """多音語を抽出する. 表記の昇順で返す (再現性のため).

    ★長音の書き分けだけが違う読みは畳んでから数える.
    そうしないと「亢 -> コウ / コー」のような異表記ペアが大量に混ざる.

    ★読みが多い語を除外しない. コストの小さい順に max_readings 個だけ採る.
    「生」は 46 通りの読みを持つが, 最も学習価値の高い多音語でもある.
    読みの妥当性は LLM の valid フラグで判定する設計なので, ここで
    絞り込む必要はない.
    """
    scored: list[tuple[int, str, PolyphonicWord]] = []

    for surface in sorted(dic.surfaces()):
        if max_surface_len is not None and len(surface) > max_surface_len:
            continue
        if OUT_OF_SCOPE_SURFACE.search(surface):
            continue
        # ★漢字を含まない表記は対象外.
        #   「ボツクス / ボックス」はカタカナの綴り揺れであって読みの曖昧性ではない.
        if not any(is_kanji(ch) for ch in surface):
            continue

        entries = dic.entries_of(surface)
        usable = [
            e
            for e in entries
            if e.reading and not e.pos.startswith(UNUSABLE_POS_PREFIXES)
        ]
        if not usable:
            continue

        folded = _fold_with_cost([(e.reading, e.cost) for e in usable])
        if len(folded) < min_readings:
            continue
        readings = [r for r, _ in folded[:max_readings]]

        pos_list = sorted({e.pos for e in usable})
        if len(surface) == 1 and all(
            p.startswith(EXCLUDED_SINGLE_CHAR_POS_PREFIXES) for p in pos_list
        ):
            continue

        scored.append(
            (
                folded[0][1],
                surface,
                PolyphonicWord(
                    surface=surface, readings=tuple(readings), pos_list=tuple(pos_list)
                ),
            )
        )

    # ★頻度の高い語 (コストの小さい語) から返す.
    #   表記の昇順だと --limit N が珍しい漢字ばかりを拾ってしまう.
    scored.sort(key=lambda t: (t[0], t[1]))
    out = [w for _, _, w in scored]
    return out[:limit] if limit is not None else out
