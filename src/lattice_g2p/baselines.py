"""既存手法（OpenJTalk など）を同じ土俵で走らせる.

★論文値と比べない。同じ正規化規則・同じ評価コードで自前で走らせた値と比べる
(docs/evaluation.md §4).

★比較対象の出力形式も検証すること.
OpenJTalk の `pron` は**アクセント記号を含む**（`マイツ’キ` / `イツツ’`）。
これを落とさずに比較したせいで、一度 OpenJTalk の精度を
対象語 96.85% -> 93.50% と大幅に過小評価した
(docs/pitfalls.md R-22)。

「相手が弱い」という数字が出たときこそ, 自分の計測を疑うこと.
"""

from lattice_g2p.kana import is_kana_only, normalize_kana

ACCENT_MARKS = "’’'́゛゜"
"""OpenJTalk の pron に現れるアクセント記号. 読みではないので落とす."""

_ACCENT_TABLE = str.maketrans("", "", ACCENT_MARKS)


def clean_reading(reading: str) -> str:
    """外部ツールの読み出力を, このプロジェクトの規約に揃える.

    ★正規化しても非カナ文字が残るなら, それは未知の記号である.
    黙って通さずに呼び出し側で検査できるよう, そのまま返す.
    """
    return normalize_kana(reading).translate(_ACCENT_TABLE)


def find_unexpected_chars(reading: str) -> set[str]:
    """clean_reading を通しても残る非カナ文字を返す.

    ★空でなければ計測が歪んでいる可能性がある. 必ず確認すること.
    """
    return {ch for ch in clean_reading(reading) if not is_kana_only(ch)}


def openjtalk_readings(text: str) -> list[tuple[str, str]]:
    """(表記, 読み) の形態素列を返す.

    pyopenjtalk は任意依存 (--extra bench). 読みは `pron` を使う
    （`read` は仮名形で, 助詞の「は」が ハ のままになる）.
    """
    import pyopenjtalk  # 任意依存

    return [
        (m["string"], clean_reading(m["pron"])) for m in pyopenjtalk.run_frontend(text)
    ]


def span_reading_from_morphemes(
    morphemes: list[tuple[str, str]], start: int, end: int
) -> str:
    """span と重なる形態素の読みを連結する.

    ★lattice_g2p.decode.span_reading と同じ規約にすること.
    規約が違うと比較にならない.
    """
    parts: list[str] = []
    pos = 0
    for surface, reading in morphemes:
        nxt = pos + len(surface)
        if pos < end and nxt > start:
            parts.append(reading)
        pos = nxt
    return "".join(parts)
