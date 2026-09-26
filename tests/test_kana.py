import pytest

from lattice_g2p.kana import is_kana_only, normalize_kana, phonetic_key


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("こめ", "コメ"),  # ひらがな → カタカナ
        ("コメ", "コメ"),  # カタカナはそのまま
        ("ｺﾒ", "コメ"),  # 半角カナ → 全角
        ("ｶﾞｯｺｳ", "ガッコウ"),  # 半角濁点の合成
        ("コメ ヲ タク", "コメヲタク"),  # 半角空白を除去
        ("コメ　ヲ", "コメヲ"),  # 全角空白も除去
        ("コンニチワ。", "コンニチワ"),  # 句点を除去
        ("ソレハ、コレダ", "ソレハコレダ"),  # 読点を除去するだけ。助詞の変換はしない
        ("「ソウ」ト イッタ！", "ソウトイッタ"),  # 括弧・感嘆符を除去
        ("ホントウ？", "ホントウ"),  # 全角疑問符を除去
        ("ガッコウ", "ガッコウ"),  # 促音を保持
        ("キョウ", "キョウ"),  # 拗音を保持
        ("ヴァイオリン", "ヴァイオリン"),  # ヴ を保持
        ("コーヒー", "コーヒー"),  # 既にある長音符は保持
        ("トウキョウ", "トウキョウ"),  # 長音は表記どおり（ー に変換しない）
        ("ヲ", "ヲ"),  # 助詞「を」は ヲ のまま
        ("ツヅク", "ツヅク"),  # ヅ を変換しない
        ("*", ""),  # UniDic の「読みなし」マーカ
        ("", ""),  # 空文字
    ],
)
def test_normalize_kana(raw: str, expected: str) -> None:
    assert normalize_kana(raw) == expected


def test_prolonged_mark_is_never_removed() -> None:
    """★長音符を除去すると「コーヒー」が「コヒ」になる。"""
    assert "ー" in normalize_kana("ラーメン")


def test_particles_are_not_converted() -> None:
    """★助詞の は→ワ / へ→エ は kana.py では変換しない。

    UniDic の pron が既に解決しているため、正規化側で文脈判断をしてはいけない。
    """
    assert normalize_kana("ハ") == "ハ"
    assert normalize_kana("ヘ") == "ヘ"


@pytest.mark.parametrize("raw", ["こめ", "ｶﾞｯｺｳ", "コメ ヲ タク", "コーヒー", "*", ""])
def test_normalize_is_idempotent(raw: str) -> None:
    """2 回かけても結果が変わらないこと。

    辞書側と評価側で適用回数が異なっても同じ結果になる必要がある。
    """
    once = normalize_kana(raw)
    assert normalize_kana(once) == once


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("コメ", True),
        ("コーヒー", True),
        ("ヴァイオリン", True),
        ("ガッコウ", True),
        ("", True),
        ("こめ", False),  # ひらがなは正規化前なので False
        ("コメ123", False),
        ("コメABC", False),
        ("コメ ヲ", False),  # 空白を含む
        ("コメ。", False),  # 句点を含む
        ("米", False),  # 漢字
    ],
)
def test_is_kana_only(s: str, expected: bool) -> None:
    assert is_kana_only(s) is expected


def test_normalized_benchmark_readings_are_kana_only() -> None:
    """正規化した結果は必ず is_kana_only を満たすこと。"""
    samples = [
        "アタラシイプロジェクトヲススメルタメ、チームノケッソクヲカタメルヒツヨウガアル。",
        "「ソウ」ト イッタ！",
        "ホントウ？",
    ]
    for s in samples:
        assert is_kana_only(normalize_kana(s)), s


# --- 異表記の同一視 ---


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("コウ", "コー"),  # オ段 + ウ  ≡  オ段 + ー
        ("トウキョウ", "トーキョー"),
        ("ヘイ", "ヘー"),  # エ段 + イ  ≡  エ段 + ー
        ("セイ", "セー"),
        ("ニイ", "ニー"),  # イ段 + イ  ≡  イ段 + ー
        ("ゴオ", "ゴー"),  # オ段 + オ  ≡  オ段 + ー
        ("サア", "サー"),  # ア段 + ア  ≡  ア段 + ー
        ("スウ", "スー"),  # ウ段 + ウ  ≡  ウ段 + ー
        ("キョウ", "キョー"),  # 拗音のあと
    ],
)
def test_phonetic_key_unifies_long_vowel_spellings(a: str, b: str) -> None:
    """★UniDic の pron 形と kana 形は長音の書き方が違うだけのことがある。

    「亢 → コウ / コー」は多音語ではない。音として同じ読みを同一視できないと、
    学習データ生成の対象語が異表記ペアだらけになる。
    """
    from lattice_g2p.kana import phonetic_key

    assert phonetic_key(a) == phonetic_key(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("コメ", "ベイ"),  # 本当に違う読み
        ("ショウ", "ジョウ"),  # 清濁の違いは保つ
        ("キョウ", "キョウカ"),  # 長さが違う
        ("コウ", "コイ"),  # オ段 + イ は長音ではない
        ("サイ", "サー"),  # ア段 + イ は長音ではない
    ],
)
def test_phonetic_key_keeps_real_differences(a: str, b: str) -> None:
    from lattice_g2p.kana import phonetic_key

    assert phonetic_key(a) != phonetic_key(b)


def test_phonetic_key_is_idempotent() -> None:
    from lattice_g2p.kana import phonetic_key

    for s in ["コウ", "コー", "トーキョー", "ラーメン", "ッ", ""]:
        assert phonetic_key(phonetic_key(s)) == phonetic_key(s)


def test_phonetic_key_handles_leading_prolonged_mark() -> None:
    """先頭の長音符は直前の母音がないのでそのまま残すこと。"""
    from lattice_g2p.kana import phonetic_key

    assert phonetic_key("ーン") == "ーン"


def test_phonetic_key_keeps_sokuon_and_hatsuon() -> None:
    from lattice_g2p.kana import phonetic_key

    assert phonetic_key("ガッコウ") != phonetic_key("ガコウ")
    assert phonetic_key("カンジ") != phonetic_key("カジ")


def test_phonetic_key_folds_yotsugana() -> None:
    """★ヂ = ジ, ヅ = ズ は現代標準語で同じ音。表記の選択でしかない。

    縮む は現代仮名遣いでは チヂム、表音式では チジム。
    benchmark は後者を使っており、畳まないと 33 件を不正解にしていた
    （OpenJTalk との差の 5.1%）。
    """
    assert phonetic_key("チヂム") == phonetic_key("チジム")
    assert phonetic_key("ツヅミ") == phonetic_key("ツズミ")
    assert phonetic_key("カンヅメ") == phonetic_key("カンズメ")


def test_normalize_kana_keeps_yotsugana() -> None:
    """★出力する読みでは畳まない。phonetic_key は比較専用（不変条件 9）。"""
    assert normalize_kana("ツヅク") == "ツヅク"
    assert normalize_kana("チヂム") == "チヂム"
