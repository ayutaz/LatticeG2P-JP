from lattice_g2p.dict_augment import extract_single_kanji_readings, split_reading
from lattice_g2p.dictionary import Dictionary, Entry


def test_split_reading_enumerates_all_partitions() -> None:
    splits = set(split_reading("アイウ", 2, max_part_len=6))
    assert splits == {("ア", "イウ"), ("アイ", "ウ")}


def test_split_reading_respects_max_part_len() -> None:
    splits = set(split_reading("アイウエ", 2, max_part_len=2))
    assert splits == {("アイ", "ウエ")}


def test_split_reading_single_part() -> None:
    assert set(split_reading("アイ", 1, max_part_len=6)) == {("アイ",)}


def test_split_reading_too_short() -> None:
    assert list(split_reading("ア", 2, max_part_len=6)) == []


def test_extracts_reading_from_two_char_compound() -> None:
    """★「撮了 -> サツリョウ」で「了 -> リョウ」が既知なら「撮 -> サツ」を学べる。

    論文が mpaligner で約 16,600 件抽出している処理に相当する。
    """
    dic = Dictionary.from_entries(
        [
            Entry("了", "リョウ", "名詞", 0, cost=5000),
            Entry("撮了", "サツリョウ", "名詞", 1, cost=6000),
        ]
    )

    new = extract_single_kanji_readings(dic)

    assert ("撮", "サツ") in {(e.surface, e.reading) for e in new}


def test_extracts_rendaku_variant() -> None:
    """連濁した形もそのまま抽出すること（縦桟 -> タテザン なら 桟 -> ザン）。"""
    dic = Dictionary.from_entries(
        [
            Entry("縦", "タテ", "名詞", 0),
            Entry("縦桟", "タテザン", "名詞", 1),
        ]
    )

    new = extract_single_kanji_readings(dic)

    assert ("桟", "ザン") in {(e.surface, e.reading) for e in new}


def test_skips_when_compound_is_fully_explained() -> None:
    """既知の読みだけで説明できる複合語からは何も学ばないこと。"""
    dic = Dictionary.from_entries(
        [
            Entry("米", "ベイ", "名詞", 0),
            Entry("国", "コク", "名詞", 1),
            Entry("米国", "ベイコク", "名詞", 2),
        ]
    )

    assert extract_single_kanji_readings(dic) == []


def test_skips_when_ambiguous() -> None:
    """★未知の割り当てが一意に決まらない場合は採用しないこと。

    「甲乙 -> アイウ」で甲も乙も未知なら、ア/イウ とも アイ/ウ とも割れる。
    推測で辞書を汚さない。
    """
    dic = Dictionary.from_entries([Entry("甲乙", "アイウ", "名詞", 0)])

    assert extract_single_kanji_readings(dic) == []


def test_skips_non_kanji_surface() -> None:
    """ひらがな・カタカナを含む表記は対象外。活用語尾を誤って取り込むため。"""
    dic = Dictionary.from_entries(
        [
            Entry("見", "ミ", "動詞", 0),
            Entry("見る", "ミル", "動詞", 1),
        ]
    )

    new = extract_single_kanji_readings(dic)

    assert all(e.surface != "る" for e in new)


def test_skips_existing_readings() -> None:
    """既に登録済みの読みは新規として返さないこと。"""
    dic = Dictionary.from_entries(
        [
            Entry("了", "リョウ", "名詞", 0),
            Entry("撮", "サツ", "名詞", 1),
            Entry("撮了", "サツリョウ", "名詞", 2),
        ]
    )

    assert extract_single_kanji_readings(dic) == []


def test_respects_max_surface_length() -> None:
    dic = Dictionary.from_entries(
        [
            Entry("一", "イチ", "名詞", 0),
            Entry("二", "ニ", "名詞", 1),
            Entry("三", "サン", "名詞", 2),
            Entry("一二三四", "イチニサンヨン", "名詞", 3),
        ]
    )

    assert extract_single_kanji_readings(dic, max_surface_len=3) == []
    assert ("四", "ヨン") in {
        (e.surface, e.reading) for e in extract_single_kanji_readings(dic, max_surface_len=4)
    }


def test_extracted_entries_have_high_cost() -> None:
    """★抽出した読みは実エントリより不利なコストにすること。推測由来のため。"""
    dic = Dictionary.from_entries(
        [
            Entry("了", "リョウ", "名詞", 0, cost=5000),
            Entry("撮了", "サツリョウ", "名詞", 1, cost=6000),
        ]
    )

    new = extract_single_kanji_readings(dic, extracted_cost=9999)

    assert all(e.cost == 9999 for e in new)


def test_extracted_entries_are_deduped() -> None:
    dic = Dictionary.from_entries(
        [
            Entry("了", "リョウ", "名詞", 0),
            Entry("撮了", "サツリョウ", "名詞", 1),
            Entry("撮了会", "サツリョウカイ", "名詞", 2),
            Entry("会", "カイ", "名詞", 3),
        ]
    )

    new = extract_single_kanji_readings(dic)
    keys = [(e.surface, e.reading) for e in new]

    assert len(keys) == len(set(keys))


def test_rejects_empty_part() -> None:
    """空の読みを割り当てないこと。文字の脱落経路を作ってしまう。"""
    dic = Dictionary.from_entries(
        [
            Entry("了", "サツリョウ", "名詞", 0),
            Entry("撮了", "サツリョウ", "名詞", 1),
        ]
    )

    new = extract_single_kanji_readings(dic)

    assert all(e.reading for e in new)
