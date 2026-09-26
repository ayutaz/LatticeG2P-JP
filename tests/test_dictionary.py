from pathlib import Path

import pytest

from lattice_g2p.dictionary import Dictionary, Entry


@pytest.fixture
def dic() -> Dictionary:
    entries = [
        Entry(surface="米", reading="コメ", pos="名詞-普通名詞-一般", entry_id=0, cost=1853),
        Entry(surface="米", reading="ベイ", pos="名詞-普通名詞-一般", entry_id=1, cost=1548),
        Entry(surface="米国", reading="ベイコク", pos="名詞-固有名詞-地名", entry_id=2, cost=1000),
        Entry(surface="国", reading="クニ", pos="名詞-普通名詞-一般", entry_id=3, cost=2000),
        Entry(surface="の", reading="ノ", pos="助詞-格助詞", entry_id=4, cost=-500),
        Entry(surface="は", reading="ワ", pos="助詞-係助詞", entry_id=5, cost=-904),
    ]
    return Dictionary.from_entries(entries)


def test_common_prefix_search_finds_all_lengths(dic: Dictionary) -> None:
    found = dic.common_prefix_search("米国の", 0)
    assert sorted({e.surface for e in found}) == ["米", "米国"]


def test_common_prefix_search_returns_all_readings(dic: Dictionary) -> None:
    found = [e for e in dic.common_prefix_search("米国の", 0) if e.surface == "米"]
    assert sorted(e.reading for e in found) == ["コメ", "ベイ"]


def test_common_prefix_search_from_offset(dic: Dictionary) -> None:
    found = dic.common_prefix_search("米国の", 1)
    assert sorted({e.surface for e in found}) == ["国"]


def test_common_prefix_search_no_match(dic: Dictionary) -> None:
    assert dic.common_prefix_search("XYZ", 0) == []


def test_common_prefix_search_past_end(dic: Dictionary) -> None:
    assert dic.common_prefix_search("米", 1) == []


def test_readings_of(dic: Dictionary) -> None:
    assert sorted(dic.readings_of("米")) == ["コメ", "ベイ"]
    assert dic.readings_of("存在しない語") == []


def test_particle_ha_reads_as_wa(dic: Dictionary) -> None:
    """★係助詞「は」の読みが「ワ」であること（pron を使っている証拠）。"""
    assert dic.readings_of("は") == ["ワ"]


def test_from_entries_normalizes_readings() -> None:
    dic = Dictionary.from_entries([Entry("米", "こめ", "名詞", 0)])
    assert dic.readings_of("米") == ["コメ"]


def test_duplicate_surface_reading_pairs_are_deduped() -> None:
    dic = Dictionary.from_entries(
        [Entry("米", "コメ", "名詞", 0, cost=5000), Entry("米", "コメ", "名詞", 1, cost=1000)]
    )
    assert dic.readings_of("米") == ["コメ"]


def test_dedup_keeps_minimum_cost() -> None:
    """★重複排除では最小コストを残す。UnigramScorer のベースラインに効く。"""
    dic = Dictionary.from_entries(
        [Entry("米", "コメ", "名詞", 0, cost=5000), Entry("米", "コメ", "名詞", 1, cost=1000)]
    )
    assert dic.entries_of("米")[0].cost == 1000


def test_empty_reading_is_kept() -> None:
    """★句読点は pron が '*' で読みなし。空読みのエントリとして保持すること。

    除外すると句読点の位置でラティスの経路が途切れる。
    """
    dic = Dictionary.from_entries([Entry("。", "*", "補助記号-句点", 0)])
    assert dic.readings_of("。") == [""]


def test_entry_id_is_reassigned_sequentially() -> None:
    dic = Dictionary.from_entries(
        [Entry("米", "コメ", "名詞", 999), Entry("国", "クニ", "名詞", 999)]
    )
    assert sorted(e.entry_id for e in dic.all_entries()) == [0, 1]


def test_surfaces_and_entries_of(dic: Dictionary) -> None:
    assert set(dic.surfaces()) == {"米", "米国", "国", "の", "は"}
    assert len(dic.entries_of("米")) == 2
    assert dic.entries_of("存在しない") == []


def test_len(dic: Dictionary) -> None:
    assert len(dic) == 6


def test_max_surface_length(dic: Dictionary) -> None:
    assert dic.max_surface_length == 2


# --- UniDic CSV のパース ---

_CSV_HEADER_COLS = 33
"""surface, left, right, cost, f[0]..f[28] の 33 列。"""


def _unidic_row(surface: str, cost: int, pos: list[str], pron: str, kana: str) -> str:
    """UniDic lex CSV の 1 行を組み立てる（テスト用）。"""
    cols = ["*"] * _CSV_HEADER_COLS
    cols[0] = surface
    cols[1] = "1"
    cols[2] = "1"
    cols[3] = str(cost)
    cols[4:8] = pos + ["*"] * (4 - len(pos))
    cols[13] = pron
    cols[24] = kana
    return ",".join(cols)


def test_from_unidic_csv_reads_pron_and_kana(tmp_path: Path) -> None:
    """★pron と kana の両方を読み候補にすること。

    UniDic の pron は「を」→オ、「米」→ベー だが、benchmark は ヲ / ベイ を使う。
    どちらの表記も候補に含めないと Oracle が落ちる
    （docs/architecture.md §3-1 参照）。
    """
    csv_path = tmp_path / "lex.csv"
    csv_path.write_text(
        "\n".join(
            [
                _unidic_row("を", -1331, ["助詞", "格助詞"], "オ", "ヲ"),
                _unidic_row("米", 1548, ["名詞", "普通名詞"], "ベー", "ベイ"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dic = Dictionary.from_unidic_csv(csv_path)

    assert set(dic.readings_of("を")) == {"オ", "ヲ"}
    assert set(dic.readings_of("米")) == {"ベー", "ベイ"}


def test_from_unidic_csv_can_restrict_to_pron(tmp_path: Path) -> None:
    csv_path = tmp_path / "lex.csv"
    csv_path.write_text(_unidic_row("を", -1331, ["助詞", "格助詞"], "オ", "ヲ") + "\n", "utf-8")

    dic = Dictionary.from_unidic_csv(csv_path, reading_fields=("pron",))

    assert dic.readings_of("を") == ["オ"]


def test_from_unidic_csv_joins_pos_levels(tmp_path: Path) -> None:
    csv_path = tmp_path / "lex.csv"
    csv_path.write_text(_unidic_row("は", -904, ["助詞", "係助詞"], "ワ", "ハ") + "\n", "utf-8")

    dic = Dictionary.from_unidic_csv(csv_path)

    assert dic.entries_of("は")[0].pos == "助詞-係助詞"


def test_from_unidic_csv_reads_cost(tmp_path: Path) -> None:
    csv_path = tmp_path / "lex.csv"
    csv_path.write_text(_unidic_row("米", 1548, ["名詞"], "ベー", "ベイ") + "\n", "utf-8")

    dic = Dictionary.from_unidic_csv(csv_path)

    assert dic.entries_of("米")[0].cost == 1548


def test_from_unidic_csv_skips_empty_surface(tmp_path: Path) -> None:
    csv_path = tmp_path / "lex.csv"
    csv_path.write_text(
        "\n".join(
            [
                _unidic_row("", 0, ["名詞"], "コメ", "コメ"),
                _unidic_row("米", 1548, ["名詞"], "ベー", "ベイ"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dic = Dictionary.from_unidic_csv(csv_path)

    assert dic.surfaces() == ["米"]


def test_roundtrip_save_load(tmp_path: Path, dic: Dictionary) -> None:
    """★233MB の CSV を毎回パースしないためのキャッシュ。"""
    path = tmp_path / "dic.pkl"
    dic.save(path)
    loaded = Dictionary.load(path)

    assert len(loaded) == len(dic)
    assert sorted(loaded.readings_of("米")) == ["コメ", "ベイ"]
    assert sorted({e.surface for e in loaded.common_prefix_search("米国の", 0)}) == ["米", "米国"]


# --- 空読みの制限 ---


def test_empty_reading_dropped_for_kanji_surface() -> None:
    """★漢字の表記に空読みを許さないこと。

    UniDic は「人」を顔文字の部品（補助記号-ＡＡ）としても登録しており、
    その読みは '*'（空）になる。これを残すと、モデルが「人」を読み飛ばして
    文字を脱落させる経路を選べてしまう。
    """
    dic = Dictionary.from_entries(
        [
            Entry("人", "ヒト", "名詞-普通名詞", 0, cost=1000),
            Entry("人", "*", "補助記号-ＡＡ-一般", 1, cost=8334),
        ]
    )
    assert dic.readings_of("人") == ["ヒト"]


def test_empty_reading_dropped_for_kana_surface() -> None:
    """「を」も補助記号として空読みで登録されている。"""
    dic = Dictionary.from_entries(
        [
            Entry("を", "ヲ", "助詞-格助詞", 0, cost=-1331),
            Entry("を", "*", "補助記号-一般", 1, cost=6037),
        ]
    )
    assert dic.readings_of("を") == ["ヲ"]


def test_empty_reading_kept_for_punctuation() -> None:
    """★句読点・記号は空読みを保持する。

    除外すると句読点の位置でラティスの経路が途切れる。
    """
    for surface in ["。", "、", "「", "」", "…", "!?"]:
        dic = Dictionary.from_entries([Entry(surface, "*", "補助記号", 0)])
        assert dic.readings_of(surface) == [""], surface


def test_surface_with_only_empty_reading_is_kept() -> None:
    """空読みしか持たない漢字表記は、エントリごと消えること。"""
    dic = Dictionary.from_entries(
        [
            Entry("人", "*", "補助記号-ＡＡ-一般", 0),
            Entry("米", "コメ", "名詞", 1),
        ]
    )
    assert dic.surfaces() == ["米"]


# --- 読みの由来 (source) ---


def test_source_records_which_field_the_reading_came_from(tmp_path: Path) -> None:
    """★pron 由来か kana 由来かを記録すること。

    UniDic の pron と kana は一致しない（を→オ/ヲ、米→ベー/ベイ）。
    ラティスには両方を入れる必要があるが（Oracle のため）、同コストになるため
    スコアラが区別できない。由来を特徴として渡せるようにする。
    """
    csv_path = tmp_path / "lex.csv"
    csv_path.write_text(_unidic_row("を", -1331, ["助詞", "格助詞"], "オ", "ヲ") + "\n", "utf-8")

    dic = Dictionary.from_unidic_csv(csv_path)
    by_reading = {e.reading: e.source for e in dic.entries_of("を")}

    assert by_reading["オ"] == "pron"
    assert by_reading["ヲ"] == "kana"


def test_source_is_both_when_fields_agree(tmp_path: Path) -> None:
    csv_path = tmp_path / "lex.csv"
    csv_path.write_text(_unidic_row("犬", 1000, ["名詞"], "イヌ", "イヌ") + "\n", "utf-8")

    dic = Dictionary.from_unidic_csv(csv_path)

    assert dic.entries_of("犬")[0].source == "both"


def test_source_merges_on_dedup() -> None:
    dic = Dictionary.from_entries(
        [
            Entry("犬", "イヌ", "名詞", 0, cost=1000, source="pron"),
            Entry("犬", "イヌ", "名詞", 1, cost=2000, source="kana"),
        ]
    )

    assert dic.entries_of("犬")[0].source == "both"


def test_source_defaults_to_empty() -> None:
    dic = Dictionary.from_entries([Entry("犬", "イヌ", "名詞", 0)])
    assert dic.entries_of("犬")[0].source == ""


def test_from_entries_keeps_context_ids() -> None:
    """★from_entries は Entry を作り直すので、フィールドを取りこぼしやすい。

    M6 で left_id / right_id を足したとき、3 箇所の再構築のどれも
    新しいフィールドを渡しておらず、**すべて 0 になっていた**。
    黙って 0 になるので、辞書を作り直すまで気づけない。
    """
    dic = Dictionary.from_entries(
        [Entry("米", "コメ", "名詞", 0, cost=100, left_id=4321, right_id=8765)]
    )

    entry = dic.entries_of("米")[0]

    assert (entry.left_id, entry.right_id) == (4321, 8765)


def test_from_entries_keeps_ids_of_the_cheaper_entry() -> None:
    """★重複時は低コスト側の属性を丸ごと残すこと。

    pos だけ選んで連接 ID を取りこぼすと、遷移項が
    「どの語の ID か」を取り違える。
    """
    dic = Dictionary.from_entries(
        [
            Entry("米", "コメ", "接尾辞", 0, cost=9000, left_id=11, right_id=11),
            Entry("米", "コメ", "名詞", 0, cost=100, left_id=22, right_id=33),
        ]
    )

    entry = dic.entries_of("米")[0]

    assert entry.pos == "名詞"
    assert (entry.left_id, entry.right_id) == (22, 33)


def test_lattice_carries_context_ids() -> None:
    """★辞書からラティスまで連接 ID が届くこと。"""
    from lattice_g2p.lattice import build_lattice

    dic = Dictionary.from_entries(
        [Entry("米", "コメ", "名詞", 0, left_id=4321, right_id=8765)]
    )

    node = next(n for n in build_lattice("米", dic).nodes if n.reading == "コメ")

    assert (node.left_id, node.right_id) == (4321, 8765)


def test_cache_round_trip_keeps_every_field(tmp_path) -> None:  # noqa: ANN001
    """★キャッシュはタプル化しているので、フィールドを足すと取りこぼす。

    M6 で left_id / right_id を足したとき、save() が 6 フィールドしか
    書いておらず、読み込むと**黙って 0 になっていた**。
    辞書を作り直しても直らないので、原因にたどり着くのに一手かかった。
    """
    import dataclasses

    original = Entry("米", "コメ", "名詞", 0, cost=123, source="both", left_id=45, right_id=67)
    path = tmp_path / "dic.pkl"
    Dictionary.from_entries([original]).save(path)

    restored = Dictionary.load(path).entries_of("米")[0]

    for field in dataclasses.fields(Entry):
        assert getattr(restored, field.name) == getattr(original, field.name), field.name


def test_cache_rejects_a_field_mismatch(tmp_path) -> None:  # noqa: ANN001
    """★列が変わったキャッシュは黙って読まず、作り直しを促すこと。"""
    import pickle

    path = tmp_path / "old.pkl"
    with path.open("wb") as f:
        pickle.dump(
            {"version": 4, "fields": ("surface", "reading"), "entries": [("米", "コメ")]}, f
        )

    with pytest.raises(ValueError, match="列が違います"):
        Dictionary.load(path)
