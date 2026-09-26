from lattice_g2p.data.filter import FilterStats, filter_sentences
from lattice_g2p.data.schema import GeneratedSentence
from lattice_g2p.dictionary import Dictionary, Entry


def _dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0),
            Entry("米", "ベイ", "名詞", 0),
            Entry("を", "ヲ", "助詞", 0),
            Entry("炊く", "タク", "動詞", 0),
            Entry("国", "コク", "名詞", 0),
            Entry("。", "*", "補助記号-句点", 0),
        ]
    )


def _sentence(**kwargs) -> GeneratedSentence:  # noqa: ANN003
    base = {
        "text": "米を炊く",
        "reading": "コメヲタク",
        "target_start": 0,
        "target_end": 1,
        "target_surface": "米",
        "target_reading": "コメ",
        "prompt_version": "v1",
        "model": "test",
        "generated_at": "2026-01-01",
    }
    base.update(kwargs)
    return GeneratedSentence(**base)


def test_valid_sentence_passes() -> None:
    kept, stats = filter_sentences([_sentence()], _dic(), min_len=3)

    assert len(kept) == 1
    assert stats.dedup_ok == 1
    assert kept[0].text == "米を炊く"


def test_rejects_non_kana_reading() -> None:
    kept, stats = filter_sentences([_sentence(reading="コメwoタク")], _dic(), min_len=3)

    assert kept == []
    assert stats.format_ok == 0


def test_rejects_wrong_target_position() -> None:
    kept, _ = filter_sentences([_sentence(target_start=1, target_end=2)], _dic(), min_len=3)
    assert kept == []


def test_rejects_digits_and_latin() -> None:
    rows = [_sentence(text="米を3合炊く", reading="コメヲサンゴウタク")]
    assert filter_sentences(rows, _dic(), min_len=3)[0] == []


def test_rejects_too_short_and_too_long() -> None:
    assert filter_sentences([_sentence()], _dic(), min_len=10)[0] == []
    assert filter_sentences([_sentence()], _dic(), min_len=3, max_len=3)[0] == []


def test_rejects_reading_not_in_lattice() -> None:
    """★ラティス整合チェック: 辞書で生成できない読みを落とすこと。"""
    kept, stats = filter_sentences([_sentence(reading="イリガビタク")], _dic(), min_len=3)

    assert kept == []
    assert stats.format_ok == 1, "形式は通る"
    assert stats.lattice_ok == 0, "ラティス整合で落ちる"
    assert stats.lattice_failures


def test_rejects_when_target_reading_differs_on_gold_path() -> None:
    """★正解経路上で対象語が意図した読みになっていない例を落とすこと。"""
    row = _sentence(reading="ベイヲタク", target_reading="コメ")
    kept, stats = filter_sentences([row], _dic(), min_len=3)

    assert kept == []
    assert stats.lattice_ok == 1, "ラティス整合は通る（ベイヲタク は生成可能）"
    assert stats.target_ok == 0, "対象語の読み一致で落ちる"


def test_accepts_orthographic_variant_of_target_reading() -> None:
    """長音の書き分けだけの違いは同じ読みとして通すこと。"""
    dic = Dictionary.from_entries(
        [
            Entry("米", "ベー", "名詞", 0),
            Entry("国", "コク", "名詞", 0),
        ]
    )
    row = _sentence(
        text="米国", reading="ベーコク", target_surface="米", target_reading="ベイ"
    )
    kept, _ = filter_sentences([row], dic, min_len=2)

    assert len(kept) == 1


def test_removes_duplicates() -> None:
    kept, stats = filter_sentences([_sentence(), _sentence()], _dic(), min_len=3)

    assert len(kept) == 1
    assert stats.lattice_ok == 2
    assert stats.dedup_ok == 1


def test_stats_are_monotonic() -> None:
    rows = [
        _sentence(),
        _sentence(reading="イリガビタク"),
        _sentence(text="米3"),
        _sentence(),
    ]
    _, s = filter_sentences(rows, _dic(), min_len=3)

    assert s.generated >= s.format_ok >= s.lattice_ok >= s.target_ok >= s.dedup_ok


def test_stats_format_reports_yield() -> None:
    stats = FilterStats(
        generated=100, format_ok=90, lattice_ok=80, target_ok=75, dedup_ok=70
    )
    out = stats.format()

    assert "100" in out
    assert "70" in out


def test_kept_examples_have_normalized_readings() -> None:
    kept, _ = filter_sentences([_sentence(reading="こめをたく")], _dic(), min_len=3)

    assert len(kept) == 1
    assert kept[0].reading == "コメヲタク"


# --- 対象語の読みが分離できない場合 ---


def test_accepts_when_target_is_inside_a_longer_node() -> None:
    """★対象語が長いノードの内側にあり、どの正解経路でも分離できない場合。

    「チームの結束を固める」で「固」だけを対象にすると、
    「固める → カタメル」が唯一の経路であり span の境界が存在しない。

    このとき CRF は文全体の読みで学習するので、例としては有効である。
    分離できないことを理由に捨てると歩留まりが 24% 落ちる。
    LLM が指定した読みを使ったかどうかだけを確認する。
    """
    dic = Dictionary.from_entries(
        [
            Entry("固める", "カタメル", "動詞", 0),
            Entry("を", "ヲ", "助詞", 0),
            Entry("心", "ココロ", "名詞", 0),
        ]
    )
    row = _sentence(
        text="心を固める",
        reading="ココロヲカタメル",
        target_start=2,
        target_end=3,
        target_surface="固",
        target_reading="カタ",
    )
    kept, stats = filter_sentences([row], dic, min_len=2)

    assert len(kept) == 1
    assert stats.target_ok == 1


def test_rejects_when_target_reading_absent_from_sentence() -> None:
    """★指定した読みが文の読みのどこにも現れないなら、LLM が指示を無視している。"""
    dic = Dictionary.from_entries(
        [
            Entry("固める", "カタメル", "動詞", 0),
            Entry("を", "ヲ", "助詞", 0),
            Entry("心", "ココロ", "名詞", 0),
        ]
    )
    row = _sentence(
        text="心を固める",
        reading="ココロヲカタメル",
        target_start=2,
        target_end=3,
        target_surface="固",
        target_reading="コ",
    )
    assert filter_sentences([row], dic, min_len=2)[0] == []


def test_prefers_separable_match_over_substring() -> None:
    """分離できる場合は、分離した読みが一致しなければ落とすこと。"""
    dic = Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0),
            Entry("米", "ベイ", "名詞", 0),
            Entry("国", "コク", "名詞", 0),
        ]
    )
    # 文の読みは ベイコク。「米」は分離でき、その読みは ベイ。
    # コメ を期待した例は落ちること（コメ は文の読みに現れない）
    row = _sentence(
        text="米国",
        reading="ベイコク",
        target_start=0,
        target_end=1,
        target_surface="米",
        target_reading="コメ",
    )
    assert filter_sentences([row], dic, min_len=2)[0] == []


# --- 助詞の読みチェック ---


def _particle_dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("私", "ワタシ", "代名詞", 0, cost=1000),
            Entry("は", "ワ", "助詞-係助詞", 0, cost=-904),
            Entry("は", "ハ", "助詞-係助詞", 0, cost=5329),
            Entry("葉", "ハ", "名詞-普通名詞-一般", 0, cost=3000),
            Entry("へ", "エ", "助詞-格助詞", 0, cost=2244),
            Entry("へ", "ヘ", "助詞-格助詞", 0, cost=6489),
            Entry("家", "イエ", "名詞-普通名詞-一般", 0, cost=2000),
            Entry("学生", "ガクセイ", "名詞-普通名詞-一般", 0, cost=1000),
            Entry("行く", "イク", "動詞-一般", 0, cost=1500),
        ]
    )


def test_rejects_topic_particle_read_as_ha() -> None:
    """★助詞の「は」を ハ と読む文を落とすこと。

    UniDic には助詞「は」の pron=ハ エントリも（低頻度で）存在するため、
    ラティス整合チェックでは落ちない。しかし助詞「は」は日本語で最頻出の語で、
    これが学習データに混ざると助詞の読みを誤って学習してしまう。
    """
    row = _sentence(
        text="私は学生",
        reading="ワタシハガクセイ",
        target_start=2,
        target_end=4,
        target_surface="学生",
        target_reading="ガクセイ",
    )
    kept, stats = filter_sentences([row], _particle_dic(), min_len=3)

    assert kept == []
    assert stats.lattice_ok == 1, "ラティス整合は通る（だから見逃される）"
    assert stats.target_ok == 0


def test_accepts_topic_particle_read_as_wa() -> None:
    row = _sentence(
        text="私は学生",
        reading="ワタシワガクセイ",
        target_start=2,
        target_end=4,
        target_surface="学生",
        target_reading="ガクセイ",
    )
    assert len(filter_sentences([row], _particle_dic(), min_len=3)[0]) == 1


def test_rejects_direction_particle_read_as_he() -> None:
    row = _sentence(
        text="家へ行く",
        reading="イエヘイク",
        target_start=0,
        target_end=1,
        target_surface="家",
        target_reading="イエ",
    )
    assert filter_sentences([row], _particle_dic(), min_len=3)[0] == []


def test_accepts_direction_particle_read_as_e() -> None:
    row = _sentence(
        text="家へ行く",
        reading="イエエイク",
        target_start=0,
        target_end=1,
        target_surface="家",
        target_reading="イエ",
    )
    assert len(filter_sentences([row], _particle_dic(), min_len=3)[0]) == 1


def test_does_not_reject_noun_ha() -> None:
    """★「葉」のように ハ と読む名詞は落とさないこと。

    表記が「は」の助詞だけを見る。
    """
    row = _sentence(
        text="葉は緑",
        reading="ハワミドリ",
        target_start=0,
        target_end=1,
        target_surface="葉",
        target_reading="ハ",
    )
    dic = Dictionary.from_entries(
        [
            *_particle_dic().all_entries(),
            Entry("緑", "ミドリ", "名詞-普通名詞-一般", 0, cost=1500),
        ]
    )
    assert len(filter_sentences([row], dic, min_len=3)[0]) == 1


# --- held-out 文字の除外（M3d） ---


def test_drops_rows_containing_held_out_chars() -> None:
    """★dev に取り分けた文字を含む文を落とすこと。

    生成データは文全体の読みが教師なので、その文字が本文に出るだけで
    読みが教えられてしまう（docs/pitfalls.md R-20）。
    ルビ（span 局所）とは条件が違う。
    """
    from lattice_g2p.data.filter import drop_held_out_chars

    rows = [
        _sentence(text="米を炊く", reading="コメヲタク"),
        _sentence(text="六合の米を炊く", reading="ロクゴウノコメヲタク"),
    ]
    kept, dropped = drop_held_out_chars(rows, {"四", "六", "八"})

    assert len(kept) == 1
    assert kept[0].text == "米を炊く"
    assert dropped == 1


def test_no_held_out_chars_keeps_everything() -> None:
    from lattice_g2p.data.filter import drop_held_out_chars

    rows = [_sentence()]
    assert drop_held_out_chars(rows, set()) == (rows, 0)


def test_does_not_reject_word_starting_with_ha() -> None:
    """★「はがき」のように語頭が「は」の語を落とさないこと。

    ラティスには は(助詞, pron=ハ) + がき という経路も立つので、
    「正解経路のどれかに助詞ハが含まれる」で落とすと有効なデータが消える。
    正しくは「すべての正解経路で避けられない」ときだけ落とす。

    実測で M3d の生成データがこれで落ちていた。
    """
    dic = Dictionary.from_entries(
        [
            Entry("はがき", "ハガキ", "名詞-普通名詞-一般", 0, cost=2000),
            Entry("は", "ワ", "助詞-係助詞", 1, cost=-904),
            Entry("は", "ハ", "助詞-係助詞", 2, cost=5329),
            Entry("がき", "ガキ", "名詞-普通名詞-一般", 3, cost=8000),
            Entry("が", "ガ", "助詞-格助詞", 4, cost=-900),
            Entry("届く", "トドク", "動詞-一般", 5, cost=2000),
        ]
    )
    row = _sentence(
        text="はがきが届く",
        reading="ハガキガトドク",
        target_start=0,
        target_end=3,
        target_surface="はがき",
        target_reading="ハガキ",
    )
    kept, stats = filter_sentences([row], dic, min_len=3)

    assert stats.lattice_ok == 1
    assert len(kept) == 1, "はがき を名詞として読む正解経路があるので通すこと"


def test_still_rejects_when_particle_ha_is_unavoidable() -> None:
    """★避けられない場合は落とすこと。「私は学生」を ワタシハ と読む文。"""
    kept, stats = filter_sentences(
        [
            _sentence(
                text="私は学生",
                reading="ワタシハガクセイ",
                target_start=2,
                target_end=4,
                target_surface="学生",
                target_reading="ガクセイ",
            )
        ],
        _particle_dic(),
        min_len=3,
    )

    assert kept == []
    assert stats.lattice_ok == 1
    assert stats.target_ok == 0
