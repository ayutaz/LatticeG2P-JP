"""品詞を保つ辞書とラティスのテスト（M7 後の続き・候補 D, R-27）.

★辞書は (表記, 読み) で重複排除し、最小コストのエントリの連接 ID だけを残していた。
品詞違いのエントリ（例: 「実施さ(れる)」のサ変の さ と、終助詞の さ）が 1 つに潰れ、
連接行列を使う構成では 1.18 pt 損していた（docs/pitfalls.md R-27）。

keep_pos=True なら (表記, 読み, left_id, right_id) で重複排除し、品詞違いを別ノードに残す。
★既定（keep_pos=False）では辞書もラティスも以前と 1 ビットも変わらないこと。
"""

import pickle
from pathlib import Path

from lattice_g2p.dict_augment import augment_dictionary
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice


def _entries() -> list[Entry]:
    return [
        # さ: 終助詞（安い）とサ変の未然形（高い）。同じ読み・違う連接 ID
        Entry("さ", "サ", "助詞-終助詞", 0, cost=1587, source="both",
              left_id=728, right_id=3231),
        Entry("さ", "サ", "動詞-非自立可能", 0, cost=3000, source="both",
              left_id=900, right_id=901),
        # 同じ読み・同じ連接 ID の重複は、品詞を保っても最小コストに潰す
        Entry("さ", "サ", "動詞-非自立可能", 0, cost=3500, source="pron",
              left_id=900, right_id=901),
        Entry("実施", "ジッシ", "名詞-普通名詞-サ変可能", 0, cost=3389, source="both",
              left_id=6555, right_id=1845),
        Entry("れる", "レル", "助動詞", 0, cost=5278, source="both", left_id=8343, right_id=261),
    ]


# --- 辞書 ---


def test_default_keeps_one_entry_per_reading() -> None:
    """★既定は従来どおり (表記, 読み) で 1 つ. 最小コストのエントリの連接 ID を残す."""
    dic = Dictionary.from_entries(_entries())

    sa = dic.entries_of("さ")
    assert len(sa) == 1
    assert (sa[0].cost, sa[0].left_id, sa[0].right_id) == (1587, 728, 3231)
    assert dic.keep_pos is False


def test_keep_pos_keeps_pos_variants() -> None:
    """品詞違い（連接 ID 違い）は別エントリに残し、同じ ID の重複だけ最小コストに潰す."""
    dic = Dictionary.from_entries(_entries(), keep_pos=True)

    sa = sorted(dic.entries_of("さ"), key=lambda e: e.cost)
    assert [(e.cost, e.left_id, e.right_id) for e in sa] == [(1587, 728, 3231), (3000, 900, 901)]
    assert sa[1].source == "both"  # pron と both をまとめる（従来と同じ規則）
    assert dic.keep_pos is True
    assert dic.readings_of("さ") == ["サ"], "読み候補は重複させない"


def test_keep_pos_survives_the_cache(tmp_path: Path) -> None:
    """★方針はキャッシュに保存され、読み込み後も保たれる（黙って既定に戻らない。R-25）."""
    dic = Dictionary.from_entries(_entries(), keep_pos=True)
    dic.save(tmp_path / "pos.pkl")
    loaded = Dictionary.load(tmp_path / "pos.pkl")

    assert loaded.keep_pos is True
    assert len(loaded.entries_of("さ")) == 2


def test_old_caches_load_as_default(tmp_path: Path) -> None:
    """方針を持たない以前のキャッシュは既定（品詞を潰す）として読む."""
    dic = Dictionary.from_entries(_entries())
    dic.save(tmp_path / "old.pkl")
    payload = pickle.loads((tmp_path / "old.pkl").read_bytes())
    payload.pop("keep_pos", None)
    (tmp_path / "old.pkl").write_bytes(pickle.dumps(payload))

    assert Dictionary.load(tmp_path / "old.pkl").keep_pos is False


def test_augment_keeps_the_policy() -> None:
    """単漢字の補完でも方針を引き継ぐ（補完は from_entries を呼び直す）."""
    entries = _entries() + [
        Entry("実", "ジツ", "名詞", 0, cost=3634, source="both", left_id=10749, right_id=9230),
    ]
    dic, _ = augment_dictionary(Dictionary.from_entries(entries, keep_pos=True))

    assert dic.keep_pos is True
    assert len(dic.entries_of("さ")) == 2


# --- ラティス ---


def test_lattice_has_a_node_per_pos_variant() -> None:
    """品詞違いは別ノードとして立つ（同じ span・同じ読み・違う連接 ID）."""
    dic = Dictionary.from_entries(_entries(), keep_pos=True)
    lat = build_lattice("実施される", dic)

    sa = [n for n in lat.nodes if n.surface == "さ"]
    assert sorted((n.left_id, n.cost) for n in sa) == [(728, 1587), (900, 3000)]


def test_fallback_is_not_duplicated_by_pos_variants() -> None:
    """フォールバックは、同じ読みの辞書ノードがあれば立たない（品詞を保っても従来どおり）."""
    dic = Dictionary.from_entries(_entries(), keep_pos=True)
    lat = build_lattice("実施される", dic)

    assert not [n for n in lat.nodes if n.surface == "さ" and n.entry_id < 0]


def test_default_lattice_is_unchanged() -> None:
    """★既定の辞書では、ラティスは品詞を潰したまま（さ は 1 ノード）."""
    lat = build_lattice("実施される", Dictionary.from_entries(_entries()))

    sa = [n for n in lat.nodes if n.surface == "さ" and n.entry_id >= 0]
    assert [(n.left_id, n.cost) for n in sa] == [(728, 1587)]
