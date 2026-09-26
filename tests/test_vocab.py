from pathlib import Path

from lattice_g2p.vocab import CharVocab


def test_pad_and_unk_are_reserved() -> None:
    v = CharVocab.build(["米国"])

    assert v.PAD == 0
    assert v.UNK == 1
    assert min(v.encode("米国")) >= 2


def test_unknown_char_maps_to_unk() -> None:
    v = CharVocab.build(["米国"])
    assert v.encode("Ω") == [v.UNK]


def test_encode_is_per_character() -> None:
    v = CharVocab.build(["米国の"])
    assert len(v.encode("米国の")) == 3


def test_encode_empty() -> None:
    assert CharVocab.build(["米"]).encode("") == []


def test_build_is_deterministic() -> None:
    """★語彙の順序が入力順に依存してはいけない。

    語彙 ID がずれると、保存したモデルと語彙の対応が壊れる。
    """
    a = CharVocab.build(["米国", "日本"])
    b = CharVocab.build(["日本", "米国"])

    assert a.chars == b.chars


def test_len_includes_special_tokens() -> None:
    v = CharVocab.build(["米国"])
    assert len(v) == 2 + 2  # 「米」「国」の 2 文字 + PAD + UNK


def test_build_from_multiple_sources() -> None:
    v = CharVocab.build(["米", "国"])
    assert set(v.chars) == {"米", "国"}


def test_roundtrip(tmp_path: Path) -> None:
    v = CharVocab.build(["米国の収穫"])
    path = tmp_path / "vocab.json"
    v.save(path)

    loaded = CharVocab.load(path)
    assert loaded.chars == v.chars
    assert loaded.encode("米国") == v.encode("米国")


def test_min_count_drops_rare_chars() -> None:
    """出現回数が少ない文字を落とせること（語彙の肥大を抑える）。"""
    v = CharVocab.build(["米米米国"], min_count=2)

    assert "米" in v.chars
    assert "国" not in v.chars


def test_kana_vocab_covers_all_katakana() -> None:
    """読み用の語彙は固定でよい（カタカナは文字種が限られる）。"""
    v = CharVocab.katakana()

    for ch in "アイウエオカキクケコガギグゲゴャュョッンーヴ":
        assert v.encode(ch) != [v.UNK], ch
    assert len(v) < 150, "カタカナの語彙は小さいこと"


def test_katakana_vocab_is_deterministic() -> None:
    assert CharVocab.katakana().chars == CharVocab.katakana().chars
