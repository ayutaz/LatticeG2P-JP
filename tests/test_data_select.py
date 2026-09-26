from lattice_g2p.data.select import PolyphonicWord, select_polyphonic
from lattice_g2p.dictionary import Dictionary, Entry


def _dic() -> Dictionary:
    entries = [
        Entry("米", "コメ", "名詞-普通名詞-一般", 0),
        Entry("米", "ベイ", "名詞-普通名詞-一般", 0),
        Entry("犬", "イヌ", "名詞-普通名詞-一般", 0),
        Entry("田中", "タナカ", "名詞-固有名詞-人名-姓", 0),
        Entry("田中", "タナカン", "名詞-固有名詞-人名-姓", 0),
        Entry("の", "ノ", "助詞-格助詞", 0),
        Entry("の", "ン", "助詞-格助詞", 0),
        Entry("。", "*", "補助記号-句点", 0),
        *[Entry("生", r, "名詞-普通名詞-一般", 0) for r in "アイウエオカキク"],
    ]
    return Dictionary.from_entries(entries)


def test_selects_only_polyphonic_words() -> None:
    surfaces = {w.surface for w in select_polyphonic(_dic())}

    assert "米" in surfaces
    assert "犬" not in surfaces, "単一読みの語は含めない"


def test_excludes_proper_nouns() -> None:
    """★固有名詞は除外する（論文と同じ。第 1 マイルストーンの範囲外）。"""
    assert "田中" not in {w.surface for w in select_polyphonic(_dic())}


def test_excludes_symbols() -> None:
    assert "。" not in {w.surface for w in select_polyphonic(_dic())}


def test_excludes_single_char_particles() -> None:
    """1 文字の助詞は文脈判別の学習価値が低い。"""
    assert "の" not in {w.surface for w in select_polyphonic(_dic())}


def test_many_readings_are_truncated_not_excluded() -> None:
    """読み候補が多くても除外せず、上位だけを採ること。"""
    words = {w.surface: w for w in select_polyphonic(_dic(), max_readings=5)}

    assert "生" in words
    assert len(words["生"].readings) == 5


def test_excludes_empty_readings() -> None:
    dic = Dictionary.from_entries(
        [Entry("々", "*", "記号-一般", 0), Entry("々", "ノマ", "記号-一般", 0)]
    )
    assert select_polyphonic(dic) == []


def test_collects_all_readings() -> None:
    kome = next(w for w in select_polyphonic(_dic()) if w.surface == "米")
    assert sorted(kome.readings) == ["コメ", "ベイ"]


def test_limit_truncates() -> None:
    assert len(select_polyphonic(_dic(), limit=1)) == 1


def test_result_is_deterministic() -> None:
    """★同じ辞書からは同じ順序で返ること（再現性のため）。"""
    assert select_polyphonic(_dic()) == select_polyphonic(_dic())


def test_returns_polyphonic_word_dataclass() -> None:
    words = select_polyphonic(_dic())
    assert all(isinstance(w, PolyphonicWord) for w in words)
    assert all(isinstance(w.readings, tuple) for w in words)


# --- 異表記の除外 ---


def test_excludes_orthographic_variants_only() -> None:
    """★pron 形と kana 形の違いだけの語は多音語ではない。

    「亢 → コウ / コー」は同じ読みの異表記であり、文脈で読み分けるものではない。
    学習データを生成しても無意味なので除外する。
    """
    dic = Dictionary.from_entries(
        [
            Entry("亢", "コウ", "名詞-普通名詞-一般", 0),
            Entry("亢", "コー", "名詞-普通名詞-一般", 0),
        ]
    )
    assert select_polyphonic(dic) == []


def test_keeps_word_with_real_and_variant_readings() -> None:
    """異表記を畳んでも 2 つ以上残るなら多音語として扱う。"""
    dic = Dictionary.from_entries(
        [
            Entry("米", "ベイ", "名詞-普通名詞-一般", 0),
            Entry("米", "ベー", "名詞-普通名詞-一般", 0),
            Entry("米", "コメ", "名詞-普通名詞-一般", 0),
        ]
    )
    words = select_polyphonic(dic)

    assert len(words) == 1
    assert len(words[0].readings) == 2, "ベイ/ベー は畳んで 1 つに数える"


def test_excludes_digit_and_latin_surfaces() -> None:
    """★数字・ラテン文字は範囲外（docs/pitfalls.md R-05）。"""
    dic = Dictionary.from_entries(
        [
            Entry("8", "ハチ", "名詞-数詞", 0),
            Entry("8", "ハッ", "名詞-数詞", 0),
            Entry("Ａ", "エー", "名詞-普通名詞-一般", 0),
            Entry("Ａ", "エイ", "名詞-普通名詞-一般", 0),
        ]
    )
    assert select_polyphonic(dic) == []


def test_max_surface_length() -> None:
    """長すぎる表記は例文生成の対象として扱いにくいので除外できること。"""
    dic = Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0),
            Entry("米", "ベイ", "名詞", 0),
            Entry("非常食料備蓄", "ヒジョウショクリョウビチク", "名詞", 0),
            Entry("非常食料備蓄", "ヒジョウショクリョウビチュク", "名詞", 0),
        ]
    )
    surfaces = {w.surface for w in select_polyphonic(dic, max_surface_len=4)}

    assert surfaces == {"米"}


# --- 固有名詞の扱い ---


def test_keeps_word_that_is_also_a_proper_noun() -> None:
    """★固有名詞の読みも持つというだけで表記ごと除外しないこと。

    「米」は普通名詞（コメ）でありながら固有名詞（米国→ベイ）でもある。
    any() で除外すると、最も学習価値の高い常用漢字が軒並み落ちる。
    """
    dic = Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞-普通名詞-一般", 0),
            Entry("米", "ベイ", "名詞-普通名詞-一般", 0),
            Entry("米", "ヨネ", "名詞-固有名詞-人名-姓", 0),
        ]
    )
    words = select_polyphonic(dic)

    assert [w.surface for w in words] == ["米"]
    assert set(words[0].readings) == {"コメ", "ベイ"}, "固有名詞由来の読みは数えない"


def test_excludes_word_that_is_only_a_proper_noun() -> None:
    dic = Dictionary.from_entries(
        [
            Entry("田中", "タナカ", "名詞-固有名詞-人名-姓", 0),
            Entry("田中", "タナカン", "名詞-固有名詞-人名-姓", 0),
        ]
    )
    assert select_polyphonic(dic) == []


def test_counts_readings_from_usable_entries_only() -> None:
    """★読み候補の数は、固有名詞由来を除いて数えること。

    「人」は人名由来の読み（サト、キト…）を大量に持つ。これを数えると
    max_readings を超えて除外されてしまう。
    """
    dic = Dictionary.from_entries(
        [
            Entry("人", "ヒト", "名詞-普通名詞-一般", 0),
            Entry("人", "ジン", "名詞-普通名詞-一般", 0),
            *[
                Entry("人", r, "名詞-固有名詞-人名-姓", 0)
                for r in ["サト", "キト", "ウト", "ウド", "オト", "ト", "ド"]
            ],
        ]
    )
    words = select_polyphonic(dic, max_readings=5)

    assert [w.surface for w in words] == ["人"]
    assert set(words[0].readings) == {"ヒト", "ジン"}


def test_still_excludes_when_usable_readings_are_single() -> None:
    dic = Dictionary.from_entries(
        [
            Entry("犬", "イヌ", "名詞-普通名詞-一般", 0),
            Entry("犬", "ケン", "名詞-固有名詞-人名-姓", 0),
        ]
    )
    assert select_polyphonic(dic) == []


def test_augmented_readings_are_usable() -> None:
    """★補完で得た単漢字の読みは学習対象として有効。

    これを除外すると、辞書補完の効果が学習データに反映されない。
    """
    dic = Dictionary.from_entries(
        [
            Entry("撮", "サツ", "補完-単漢字", 0, source="aug"),
            Entry("撮", "トリ", "名詞-普通名詞-一般", 0),
        ]
    )
    assert [w.surface for w in select_polyphonic(dic)] == ["撮"]


# --- 読み候補の打ち切り ---


def test_truncates_readings_instead_of_excluding_word() -> None:
    """★読みが多い語を除外せず、コストの小さい順に上位だけを採る。

    「生」は 46 通りの読みを持つが、最も学習価値の高い多音語でもある。
    読み数で除外すると、常用漢字が軒並み落ちてしまう。
    読みの妥当性の判定は LLM の valid フラグに任せる設計なので、
    ここで絞り込む必要はない。
    """
    dic = Dictionary.from_entries(
        [
            Entry("生", "セイ", "名詞-普通名詞-一般", 0, cost=1000),
            Entry("生", "ショウ", "名詞-普通名詞-一般", 0, cost=2000),
            Entry("生", "ナマ", "名詞-普通名詞-一般", 0, cost=3000),
            Entry("生", "アヤ", "名詞-普通名詞-一般", 0, cost=9000),
            Entry("生", "アレ", "名詞-普通名詞-一般", 0, cost=9500),
        ]
    )
    words = select_polyphonic(dic, max_readings=3)

    assert len(words) == 1, "読みが多いことを理由に除外しないこと"
    assert list(words[0].readings) == ["セイ", "ショウ", "ナマ"], "コストの小さい順"


def test_truncation_keeps_order_by_cost() -> None:
    dic = Dictionary.from_entries(
        [
            Entry("甲", "ウ", "名詞-普通名詞-一般", 0, cost=5000),
            Entry("甲", "ア", "名詞-普通名詞-一般", 0, cost=1000),
            Entry("甲", "イ", "名詞-普通名詞-一般", 0, cost=3000),
        ]
    )
    words = select_polyphonic(dic, max_readings=2)

    assert list(words[0].readings) == ["ア", "イ"]


def test_still_requires_min_readings() -> None:
    dic = Dictionary.from_entries([Entry("犬", "イヌ", "名詞-普通名詞-一般", 0)])
    assert select_polyphonic(dic) == []


def test_words_are_ordered_by_frequency() -> None:
    """★頻度の高い語から返すこと。

    --limit N で上位 N 語を取るとき、表記の昇順だと珍しい漢字ばかりになる。
    Phase A の 200 語が学習価値の低い語で埋まってしまう。
    """
    dic = Dictionary.from_entries(
        [
            Entry("稀", "マレ", "名詞-普通名詞-一般", 0, cost=9000),
            Entry("稀", "キ", "名詞-普通名詞-一般", 0, cost=9500),
            Entry("生", "ナマ", "名詞-普通名詞-一般", 0, cost=1000),
            Entry("生", "セイ", "名詞-普通名詞-一般", 0, cost=1200),
        ]
    )
    assert [w.surface for w in select_polyphonic(dic)] == ["生", "稀"]


def test_limit_takes_most_frequent() -> None:
    dic = Dictionary.from_entries(
        [
            Entry("稀", "マレ", "名詞-普通名詞-一般", 0, cost=9000),
            Entry("稀", "キ", "名詞-普通名詞-一般", 0, cost=9500),
            Entry("生", "ナマ", "名詞-普通名詞-一般", 0, cost=1000),
            Entry("生", "セイ", "名詞-普通名詞-一般", 0, cost=1200),
        ]
    )
    assert [w.surface for w in select_polyphonic(dic, limit=1)] == ["生"]


def test_requires_at_least_one_kanji() -> None:
    """★表記に漢字を含まない語は対象外。

    「ボツクス / ボックス」「レヴュー / レビュー」はカタカナの綴り揺れであって
    読みの曖昧性ではない。このタスクは漢字の読み推定である。
    """
    dic = Dictionary.from_entries(
        [
            Entry("ボツクス", "ボックス", "名詞-普通名詞-一般", 0, cost=100),
            Entry("ボツクス", "ボツクス", "名詞-普通名詞-一般", 0, cost=200),
            Entry("①", "イチ", "名詞-数詞", 0, cost=100),
            Entry("①", "イッ", "名詞-数詞", 0, cost=200),
            Entry("米", "コメ", "名詞-普通名詞-一般", 0, cost=1000),
            Entry("米", "ベイ", "名詞-普通名詞-一般", 0, cost=1100),
        ]
    )
    assert [w.surface for w in select_polyphonic(dic)] == ["米"]


def test_mixed_kanji_kana_surface_is_kept() -> None:
    """漢字を含めば送り仮名付きでも対象にすること。"""
    dic = Dictionary.from_entries(
        [
            Entry("行った", "イッタ", "動詞-一般", 0, cost=1000),
            Entry("行った", "オコナッタ", "動詞-一般", 0, cost=1100),
        ]
    )
    assert [w.surface for w in select_polyphonic(dic)] == ["行った"]
