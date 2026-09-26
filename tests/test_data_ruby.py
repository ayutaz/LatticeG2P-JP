"""青空文庫のルビ抽出.

★ルビは「実際の出版物に人間が付けた読み」なので hallucination がない。
一方で文の一部にしか付かないので、文全体の読みを前提とする M2 のデータとは
別の扱いが要る（部分制約付き積グラフ）。
"""

import pytest

from lattice_g2p.data.ruby import (
    RubyExample,
    drop_held_out_spans,
    extract_body,
    extract_ruby,
    filter_ruby_examples,
    iter_ruby_examples,
    iter_sentences,
    usable_spans,
)
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice

# --- ルビの抽出 ---


def test_extracts_ruby_with_explicit_delimiter() -> None:
    """｜ で範囲が明示されている場合。"""
    text, spans = extract_ruby("｜生《なま》ビールを飲む")

    assert text == "生ビールを飲む"
    assert spans == [(0, 1, "ナマ")]


def test_attaches_ruby_to_preceding_kanji_run() -> None:
    """｜ がなければ直前の漢字の連続に付く。"""
    text, spans = extract_ruby("彼は生真面目《きまじめ》な男だ")

    assert text == "彼は生真面目な男だ"
    assert spans == [(2, 6, "キマジメ")]


def test_kanji_run_stops_at_kana() -> None:
    """漢字の連続は仮名で止まること。"""
    text, spans = extract_ruby("お茶漬け飯《めし》")

    assert text == "お茶漬け飯"
    assert spans == [(4, 5, "メシ")]


def test_multiple_ruby_in_one_line() -> None:
    text, spans = extract_ruby("生真面目《きまじめ》な男が米《こめ》を炊く")

    assert text == "生真面目な男が米を炊く"
    assert spans == [(0, 4, "キマジメ"), (7, 8, "コメ")]


def test_hiragana_ruby_is_normalized_to_katakana() -> None:
    _, spans = extract_ruby("｜東京《とうきょう》")
    assert spans == [(0, 2, "トウキョウ")]


def test_katakana_ruby_is_kept() -> None:
    _, spans = extract_ruby("｜本気《マジ》")
    assert spans == [(0, 2, "マジ")]


def test_ruby_without_base_is_dropped() -> None:
    """★直前に漢字も ｜ もなければ、どこに付くルビか決められない。推測しない。"""
    text, spans = extract_ruby("あの《なぞ》")

    assert text == "あの"
    assert spans == []


def test_annotations_are_stripped() -> None:
    """［＃...］ は入力者注。本文ではないので落とす。"""
    text, spans = extract_ruby("米［＃「米」に傍点］を炊く")

    assert text == "米を炊く"
    assert spans == []


def test_annotation_between_base_and_ruby() -> None:
    text, spans = extract_ruby("｜米［＃ここから１字下げ］《こめ》")

    assert text == "米"
    assert spans == [(0, 1, "コメ")]


def test_line_without_ruby_is_unchanged() -> None:
    text, spans = extract_ruby("今日はいい天気だ")

    assert text == "今日はいい天気だ"
    assert spans == []


def test_unclosed_ruby_is_left_alone() -> None:
    """閉じ括弧がない壊れた行で例外を出さないこと。"""
    text, spans = extract_ruby("｜米《こめ")

    assert spans == []
    assert "米" in text


def test_iteration_mark_counts_as_kanji() -> None:
    """々 は漢字の連続に含めること。"""
    _, spans = extract_ruby("時々《ときどき》")
    assert spans == [(0, 2, "トキドキ")]


# --- 本文の切り出し ---


RAW = """吾輩は猫である
夏目漱石

-------------------------------------------------------
【テキスト中に現れる記号について】

《》：ルビ
-------------------------------------------------------

吾輩《わがはい》は猫である。名前はまだ無い。

底本：「吾輩は猫である」岩波文庫
入力：誰か
"""


def test_extract_body_drops_header_and_footer() -> None:
    body = extract_body(RAW)

    assert "吾輩《わがはい》は猫である。" in body
    assert "夏目漱石" not in body
    assert "ルビ" not in body, "凡例を含めないこと"
    assert "底本" not in body


def test_extract_body_without_fences() -> None:
    body = extract_body("米を炊く。\n底本：どこか\n")
    assert body.strip() == "米を炊く。"


# --- 文への分割 ---


def test_splits_on_sentence_enders() -> None:
    got = list(iter_sentences("米を炊く。旨い。\n塩を振る！"))
    assert got == ["米を炊く。", "旨い。", "塩を振る！"]


def test_strips_quote_brackets() -> None:
    got = list(iter_sentences("「米を炊く」と言った。"))
    assert got == ["米を炊くと言った。"]


def test_drops_sentence_without_ender() -> None:
    """文末記号のない断片は落とす（見出しや行の途中）。"""
    assert list(iter_sentences("第一章")) == []


# --- 文単位の抽出 ---


def test_iter_ruby_examples_keeps_only_sentences_with_ruby() -> None:
    body = "米を炊く。｜生《なま》ビールを飲む。"
    got = list(iter_ruby_examples(body, source="test"))

    assert got == [RubyExample(text="生ビールを飲む。", spans=[(0, 1, "ナマ")], source="test")]


def test_spans_are_relative_to_the_sentence() -> None:
    """★span の位置は文の先頭からの位置であること（行の先頭ではない）。"""
    body = "旨い米を炊く。彼は｜生真面目《きまじめ》な男だ。"
    got = list(iter_ruby_examples(body, source="test"))

    assert len(got) == 1
    assert got[0].text == "彼は生真面目な男だ。"
    assert got[0].spans == [(2, 6, "キマジメ")]


@pytest.mark.parametrize(
    "sentence",
    [
        "｜生《なま》ビールを（三本）飲む。",
        "｜生《なま》ビールとＡＢＣを見る。",
        "｜生《なま》ビールを3本飲む。",
        "｜生《なま》ビールを一本……。",
    ],
)
def test_rejects_out_of_scope_characters(sentence: str) -> None:
    """★数字・英字・括弧・記号を含む文は範囲外（R-05）。ルビがあっても落とす。"""
    assert list(iter_ruby_examples(sentence, source="t")) == []


def test_keeps_clean_sentence_with_long_vowel_mark() -> None:
    """長音符「ー」は本文として正当。落とさないこと。"""
    got = list(iter_ruby_examples("｜生《なま》ビールを飲む、旨い。", source="t"))
    assert len(got) == 1


def test_rejects_too_short_and_too_long() -> None:
    assert list(iter_ruby_examples("｜生《なま》。", source="t")) == []
    long_text = "｜生《なま》ビールを飲む" + "、旨い" * 30 + "。"
    assert list(iter_ruby_examples(long_text, source="t")) == []


# --- ラティス整合フィルタ ---



def _dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("生", "ナマ", "名詞", 0),
            Entry("生", "セイ", "名詞", 1),
            Entry("ビール", "ビール", "名詞", 2),
            Entry("を", "ヲ", "助詞", 3),
            Entry("飲む", "ノム", "動詞", 4),
            Entry("運命", "ウンメイ", "名詞", 5),
            Entry("。", "*", "補助記号-句点", 6),
        ]
    )


def test_usable_spans_keeps_verifiable_spans() -> None:
    lat = build_lattice("生ビールを飲む", _dic())
    assert usable_spans(lat, [(0, 1, "ナマ")]) == [(0, 1, "ナマ")]


def test_usable_spans_drops_reading_not_in_dictionary() -> None:
    """★義訓ルビ（運命《さだめ》）は辞書に無いので落ちる。

    辞書制約という設計がそのまま品質フィルタになっている。
    """
    lat = build_lattice("運命を飲む", _dic())
    assert usable_spans(lat, [(0, 2, "サダメ")]) == []


def test_usable_spans_keeps_the_good_half() -> None:
    """★1 つの span が駄目でも、残りの span は使えること。

    文全体の読みを要求する M2 のフィルタなら文ごと捨てるしかない。
    部分制約なら span 単位で落とせる。
    """
    lat = build_lattice("運命の生ビール", _dic())
    got = usable_spans(lat, [(0, 2, "サダメ"), (3, 4, "ナマ")])

    assert got == [(3, 4, "ナマ")]


def test_filter_drops_examples_with_no_usable_span() -> None:
    rows = [
        RubyExample(text="生ビールを飲む。", spans=[(0, 1, "ナマ")], source="a"),
        RubyExample(text="運命を飲む。", spans=[(0, 2, "サダメ")], source="b"),
    ]
    kept, stats = filter_ruby_examples(rows, _dic())

    assert len(kept) == 1
    assert kept[0].source == "a"
    assert stats.total == 2
    assert stats.kept == 1
    assert stats.spans_total == 2
    assert stats.spans_kept == 1


def test_filter_removes_duplicates() -> None:
    row = RubyExample(text="生ビールを飲む。", spans=[(0, 1, "ナマ")], source="a")
    kept, stats = filter_ruby_examples([row, row], _dic())

    assert len(kept) == 1
    assert stats.duplicates == 1


# --- dev リーク対策 ---


def test_drop_spans_of_held_out_surfaces() -> None:
    """★dev の語にルビが付いた例は、その読みを教えてしまう。span ごと落とす。

    dev はエントリ単位で分離している（R-09）。ルビ側でその原則が破れると
    dev の精度が過大評価され、「ルビを足したら伸びた」の解釈ができなくなる。
    """
    rows = [
        RubyExample(text="生ビールを飲む。", spans=[(0, 1, "ナマ")], source="a"),
        RubyExample(
            text="運命の生ビール。",
            spans=[(0, 2, "ウンメイ"), (3, 4, "ナマ")],
            source="b",
        ),
    ]
    kept, dropped = drop_held_out_spans(rows, {"生"})

    assert dropped == 2
    assert len(kept) == 1
    assert kept[0].source == "b"
    assert kept[0].spans == [(0, 2, "ウンメイ")], "dev の語の span だけ落とす"


def test_held_out_surfaces_in_text_are_kept() -> None:
    """★本文に出てくるだけなら残す。ルビが無ければ読みは教えていない。

    CRF は span の外を周辺化するので、ラベルは付かない。
    ここまで落とすとルビの 25% が消える。
    """
    rows = [RubyExample(text="生ビールと運命。", spans=[(5, 7, "ウンメイ")], source="a")]
    kept, dropped = drop_held_out_spans(rows, {"生"})

    assert dropped == 0
    assert kept == rows
