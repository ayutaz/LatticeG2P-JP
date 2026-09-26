import pytest

from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import UNK_READING, Node, build_lattice


@pytest.fixture
def dic() -> Dictionary:
    entries = [
        Entry("米", "コメ", "名詞", 0, cost=1853),
        Entry("米", "ベイ", "名詞", 1, cost=1548),
        Entry("米国", "ベイコク", "名詞", 2, cost=1000),
        Entry("国", "クニ", "名詞", 3, cost=2000),
        Entry("国", "コク", "名詞", 4, cost=1900),
        Entry("の", "ノ", "助詞", 5, cost=-500),
        Entry("。", "*", "補助記号-句点", 6, cost=-3217),
    ]
    return Dictionary.from_entries(entries)


def test_nodes_cover_dictionary_matches(dic: Dictionary) -> None:
    lat = build_lattice("米国", dic)
    spans = {(n.start, n.end, n.reading) for n in lat.nodes}

    assert (0, 1, "コメ") in spans
    assert (0, 1, "ベイ") in spans
    assert (0, 2, "ベイコク") in spans
    assert (1, 2, "クニ") in spans
    assert (1, 2, "コク") in spans


def test_same_surface_different_reading_are_distinct_nodes(dic: Dictionary) -> None:
    lat = build_lattice("米", dic)
    readings = sorted(n.reading for n in lat.nodes if (n.start, n.end) == (0, 1))

    assert readings == ["コメ", "ベイ"]


def test_nodes_carry_cost(dic: Dictionary) -> None:
    lat = build_lattice("米", dic)
    by_reading = {n.reading: n.cost for n in lat.nodes}

    assert by_reading["コメ"] == 1853
    assert by_reading["ベイ"] == 1548


def test_index_consistency(dic: Dictionary) -> None:
    lat = build_lattice("米国の", dic)

    for i, idxs in enumerate(lat.nodes_starting_at):
        for ni in idxs:
            assert lat.nodes[ni].start == i
    for j, idxs in enumerate(lat.nodes_ending_at):
        for ni in idxs:
            assert lat.nodes[ni].end == j


def test_index_arrays_have_right_length(dic: Dictionary) -> None:
    lat = build_lattice("米国の", dic)

    assert len(lat.nodes_starting_at) == len(lat.text) + 1
    assert len(lat.nodes_ending_at) == len(lat.text) + 1


def test_full_path_exists_for_known_text(dic: Dictionary) -> None:
    assert build_lattice("米国の", dic).has_full_path()


def test_unknown_char_still_has_full_path(dic: Dictionary) -> None:
    """★辞書にない文字があっても 0 → L の経路が途切れないこと。"""
    assert build_lattice("米Ω国", dic).has_full_path()


def test_punctuation_has_empty_reading(dic: Dictionary) -> None:
    """★句読点は空読み。経路は途切れないが読み列には寄与しない。"""
    lat = build_lattice("。", dic)

    assert lat.has_full_path()
    assert [n.reading for n in lat.nodes] == [""]


def test_hiragana_fallback_is_katakana(dic: Dictionary) -> None:
    """辞書にないひらがなはカタカナ化してフォールバックすること。"""
    lat = build_lattice("ぬ", dic)

    assert [n.reading for n in lat.nodes] == ["ヌ"]


def test_katakana_fallback_passthrough(dic: Dictionary) -> None:
    lat = build_lattice("ヌ", dic)

    assert [n.reading for n in lat.nodes] == ["ヌ"]


def test_unknown_kanji_falls_back_to_unk_marker(dic: Dictionary) -> None:
    """辞書に読みがない漢字は UNK マーカになること（空読みにはしない）。"""
    lat = build_lattice("鬻", dic)

    assert [n.reading for n in lat.nodes] == [UNK_READING]
    assert UNK_READING != ""


def test_unknown_symbol_falls_back_to_empty_reading(dic: Dictionary) -> None:
    """辞書にない記号は空読み。仮名でも漢字でもないため。"""
    lat = build_lattice("♥", dic)

    assert [n.reading for n in lat.nodes] == [""]


def test_no_duplicate_nodes(dic: Dictionary) -> None:
    """フォールバックが辞書エントリと重複して同じノードを 2 個作らないこと。"""
    lat = build_lattice("の", dic)
    keys = [(n.start, n.end, n.surface, n.reading) for n in lat.nodes]

    assert len(keys) == len(set(keys))


def test_every_position_has_an_outgoing_node(dic: Dictionary) -> None:
    lat = build_lattice("米Ω国の。", dic)

    for i in range(len(lat.text)):
        assert lat.nodes_starting_at[i], f"位置 {i} から出る辺がない"


def test_empty_text(dic: Dictionary) -> None:
    lat = build_lattice("", dic)

    assert lat.nodes == []
    assert lat.has_full_path()


def test_node_is_hashable(dic: Dictionary) -> None:
    lat = build_lattice("米", dic)
    assert len(set(lat.nodes)) == len(lat.nodes)


def test_node_span_invariants(dic: Dictionary) -> None:
    lat = build_lattice("米国の。", dic)
    for node in lat.nodes:
        assert 0 <= node.start < node.end <= len(lat.text)
        assert isinstance(node, Node)
        assert lat.text[node.start : node.end] == node.surface


def test_max_node_length_limits_lattice(dic: Dictionary) -> None:
    """長いエントリを打ち切れること（ノード数の爆発を抑えるため）。"""
    lat = build_lattice("米国", dic, max_node_length=1)
    assert all(n.end - n.start == 1 for n in lat.nodes)
