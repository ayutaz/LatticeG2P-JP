"""dev のラベル監査.

★評価セットも実装物である。モデルと同じように疑う（docs/pitfalls.md R-19）。

M3b で学習データを 4 倍にしても対象語精度が 68.02% → 68.97% しか動かなかった。
「モデルが悪い」以外に「dev のラベルが悪い」可能性を切り分ける。
"""

from lattice_g2p.data.audit import flag_suspicious, reading_rank
from lattice_g2p.data.schema import TrainingExample
from lattice_g2p.dictionary import Dictionary, Entry


def _example(**kwargs) -> TrainingExample:  # noqa: ANN003
    base = {
        "text": "ダミー",
        "reading": "ダミー",
        "target_start": 0,
        "target_end": 1,
        "target_surface": "米",
        "target_reading": "コメ",
    }
    base.update(kwargs)
    return TrainingExample(**base)


def _houshi_dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("法師", "ホウシ", "名詞", 0, cost=1000),
            Entry("法師", "ボウシ", "名詞", 1, cost=9000),
        ]
    )


# --- reading_rank ---


def test_reading_rank_orders_by_cost() -> None:
    """★コストが小さいほど高頻度。1 が最頻。"""
    dic = _houshi_dic()

    assert reading_rank(dic, "法師", "ホウシ") == (1, 2)
    assert reading_rank(dic, "法師", "ボウシ") == (2, 2)


def test_reading_rank_returns_zero_for_unknown_reading() -> None:
    assert reading_rank(_houshi_dic(), "法師", "ガッコウ") == (0, 2)


def test_reading_rank_folds_long_vowel_variants() -> None:
    """★コウ と コー を別の読みとして数えないこと。"""
    dic = Dictionary.from_entries(
        [
            Entry("亢", "コウ", "名詞", 0, cost=1000),
            Entry("亢", "コー", "名詞", 1, cost=1100),
        ]
    )
    assert reading_rank(dic, "亢", "コー") == (1, 1)


# --- flag_suspicious ---


def test_flags_low_frequency_reading() -> None:
    """★辞書で低頻度の読みがラベルになっている例を候補に挙げること。"""
    rows = [_example(target_surface="法師", target_reading="ボウシ")]

    flagged = flag_suspicious(rows, _houshi_dic())

    assert len(flagged) == 1
    assert flagged[0].rank == (2, 2)
    assert flagged[0].alternatives == ["ホウシ", "ボウシ"]


def test_does_not_flag_most_frequent_reading() -> None:
    rows = [_example(target_surface="法師", target_reading="ホウシ")]

    assert flag_suspicious(rows, _houshi_dic()) == []


def test_single_reading_is_never_flagged() -> None:
    """読みが 1 つしかない語は疑いようがない。"""
    dic = Dictionary.from_entries([Entry("一方", "イッポウ", "名詞", 0, cost=1000)])
    rows = [_example(target_surface="一方", target_reading="イッポウ")]

    assert flag_suspicious(rows, dic) == []


def test_long_vowel_variant_is_not_flagged() -> None:
    """★異表記は 1 つの読みとして数えるので候補にならない。"""
    dic = Dictionary.from_entries(
        [
            Entry("亢", "コウ", "名詞", 0, cost=1000),
            Entry("亢", "コー", "名詞", 1, cost=1100),
        ]
    )
    rows = [_example(target_surface="亢", target_reading="コー")]

    assert flag_suspicious(rows, dic) == []


def test_reading_absent_from_dictionary_is_flagged() -> None:
    """★辞書に無い読みは最も疑わしい。rank=0 で必ず挙げること。"""
    rows = [_example(target_surface="法師", target_reading="ホシ")]

    flagged = flag_suspicious(rows, _houshi_dic())

    assert len(flagged) == 1
    assert flagged[0].rank == (0, 2)


def test_min_rank_ratio_controls_strictness() -> None:
    """3 読みのうち 2 位は、閾値によって候補に入ったり入らなかったりする。"""
    dic = Dictionary.from_entries(
        [
            Entry("生", "セイ", "名詞", 0, cost=1000),
            Entry("生", "ナマ", "名詞", 1, cost=2000),
            Entry("生", "キ", "名詞", 2, cost=3000),
        ]
    )
    rows = [_example(target_surface="生", target_reading="ナマ")]

    # rank 2 / total 3 = 0.667。閾値より大きいものを候補にする
    assert len(flag_suspicious(rows, dic, min_rank_ratio=0.5)) == 1, "0.5 なら候補"
    assert flag_suspicious(rows, dic, min_rank_ratio=0.9) == [], "0.9 は厳しすぎて対象外"


def test_keeps_the_original_row_for_review() -> None:
    """判定用に本文をそのまま持つこと。文脈が無いと妥当性を判断できない。"""
    rows = [_example(text="法師の修行は厳しい", target_surface="法師", target_reading="ボウシ")]

    flagged = flag_suspicious(rows, _houshi_dic())

    assert flagged[0].text == "法師の修行は厳しい"


# --- 四つ仮名（ヅ/ズ・ヂ/ジ）は同じ読みとして数える ---


def _kanzume_dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("カン詰め", "カンズメ", "名詞", 0, cost=1000),
            Entry("カン詰め", "カンヅメ", "名詞", 1, cost=1100),
        ]
    )


def test_yotsugana_variants_are_one_reading() -> None:
    """★ヅ と ズ は現代標準語では同じ音。別の読みとして数えないこと。

    これを畳まないと、連濁する語（カン詰め・ビン詰め・モノ作り）が
    すべて「疑わしいラベル」として候補に挙がってしまう（実測で候補の大半）。
    """
    assert reading_rank(_kanzume_dic(), "カン詰め", "カンヅメ") == (1, 1)


def test_yotsugana_variant_is_not_flagged() -> None:
    rows = [_example(target_surface="カン詰め", target_reading="カンヅメ")]

    assert flag_suspicious(rows, _kanzume_dic()) == []


def test_still_flags_genuinely_different_reading() -> None:
    """畳んでも別物の読みは候補に残ること。"""
    rows = [_example(target_surface="法師", target_reading="ボウシ")]

    assert len(flag_suspicious(rows, _houshi_dic())) == 1


# --- 監査結果を評価に反映する ---


def test_load_accepts_maps_surface_and_label_to_readings(tmp_path) -> None:  # noqa: ANN001
    """★(表記, dev ラベル) -> 正解として認める読み の対応を作ること。"""
    from lattice_g2p.data.audit import load_accepts

    path = tmp_path / "audit.jsonl"
    path.write_text(
        '{"target_surface": "大麻", "dev_label": "オオヌサ", "verdict": "both_ok",'
        ' "accept": ["オオヌサ", "タイマ"]}\n'
        '{"target_surface": "法師", "dev_label": "ボウシ", "verdict": "label_wrong",'
        ' "accept": ["ホウシ"]}\n',
        encoding="utf-8",
    )
    accepts = load_accepts(path)

    assert accepts[("大麻", "オオヌサ")] == ["オオヌサ", "タイマ"]
    assert accepts[("法師", "ボウシ")] == ["ホウシ"], "label_wrong は置き換える"


def test_load_accepts_unions_duplicate_keys(tmp_path) -> None:  # noqa: ANN001
    """★同じ (表記, ラベル) が複数行に出ることがある。

    監査は (表記, ラベル, 予測) で分けたので、予測が違えば別の行になる。
    評価側は予測を知らないので和を取る。★わずかに甘くなる近似である。
    """
    from lattice_g2p.data.audit import load_accepts

    path = tmp_path / "audit.jsonl"
    path.write_text(
        '{"target_surface": "軽々", "dev_label": "ケイケイ", "verdict": "label_wrong",'
        ' "accept": ["カルガル"]}\n'
        '{"target_surface": "軽々", "dev_label": "ケイケイ", "verdict": "label_wrong",'
        ' "accept": ["カルガル", "カルガルシイ"]}\n',
        encoding="utf-8",
    )
    assert load_accepts(path)[("軽々", "ケイケイ")] == ["カルガル", "カルガルシイ"]


def test_load_accepts_ignores_label_ok(tmp_path) -> None:  # noqa: ANN001
    """label_ok は dev のラベルのままなので、対応表に入れる必要がない。"""
    from lattice_g2p.data.audit import load_accepts

    path = tmp_path / "audit.jsonl"
    path.write_text(
        '{"target_surface": "文字", "dev_label": "モジ", "verdict": "label_ok",'
        ' "accept": ["モジ"]}\n',
        encoding="utf-8",
    )
    assert load_accepts(path) == {}
