"""外部ツールの出力を揃える処理のテスト.

★OpenJTalk のアクセント記号を落とし損ねて、一度 OpenJTalk の精度を
対象語 96.85% -> 93.50% と過小評価した（docs/pitfalls.md R-22）。
同じ失敗を繰り返さないための回帰テスト。
"""

from lattice_g2p.baselines import (
    clean_reading,
    find_unexpected_chars,
    span_reading_from_morphemes,
)


def test_strips_accent_marks() -> None:
    """★OpenJTalk の pron はアクセント記号を含む。読みではないので落とす。"""
    assert clean_reading("マイツ’キ") == "マイツキ"
    assert clean_reading("イツツ’") == "イツツ"


def test_keeps_prolonged_mark() -> None:
    """長音符は読みの一部。落としてはいけない。"""
    assert clean_reading("リューコー") == "リューコー"


def test_removes_punctuation_like_normalize_kana() -> None:
    assert clean_reading("カレワハシッタ。") == "カレワハシッタ"


def test_find_unexpected_chars_detects_leftovers() -> None:
    """★正規化しても残る非カナ文字を検出できること。

    これを確認していれば、アクセント記号の混入に気づけた。
    """
    assert find_unexpected_chars("マイツ’キ") == set()
    assert find_unexpected_chars("マイツ#キ") == {"#"}


def test_span_reading_includes_overlapping_morphemes() -> None:
    """★span と重なる形態素を丸ごと含める。decode.span_reading と同じ規約。"""
    morphemes = [("彼", "カレ"), ("は", "ワ"), ("股関節", "コカンセツ"), ("を", "ヲ")]

    assert span_reading_from_morphemes(morphemes, 2, 3) == "コカンセツ"
    assert span_reading_from_morphemes(morphemes, 0, 1) == "カレ"
    assert span_reading_from_morphemes(morphemes, 0, 2) == "カレワ"
