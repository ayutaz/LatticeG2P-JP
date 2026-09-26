"""連接行列を固定の遷移項として足すスコアラのテスト（M7 Task 2）.

★U（UnigramScorer + 連接行列）を「MeCab 相当」と呼べるのは、Viterbi が
MeCab の目的関数（Σ 単語コスト + Σ 連接コスト。BOS -> 先頭・末尾 -> EOS を含む）を
最小にする経路を返すときだけである。**総当たりとの一致で確かめる。**
ここが通らなければ U の数字は意味を持たない。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from lattice_g2p.connection import ConnectionMatrix
from lattice_g2p.crf import viterbi, viterbi_numpy
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import Lattice, build_lattice
from lattice_g2p.scorer import UnigramScorer
from lattice_g2p.scorer.matrix import MatrixTransitionScorer
from tests.helpers import enumerate_paths


def _write_table(path: Path, table: np.ndarray) -> None:
    """table[next_left, prev_right] を MeCab の matrix.bin 形式で書く.

    行優先で並べると添字は prev_right + lsize * next_left になり、MeCab の
    Connector::cost と一致する。
    """
    rsize, lsize = table.shape
    with path.open("wb") as f:
        np.array([lsize, rsize], dtype=np.uint16).tofile(f)
        table.astype(np.int16).ravel().tofile(f)


def _dictionary(*extra: Entry) -> Dictionary:
    return Dictionary.from_entries([
        Entry("米", "コメ", "名詞", 0, cost=10, left_id=1, right_id=1),
        Entry("米", "ベイ", "名詞", 1, cost=12, left_id=2, right_id=2),
        Entry("米国", "ベイコク", "名詞", 2, cost=30, left_id=3, right_id=3),
        Entry("国", "コク", "名詞", 3, cost=8, left_id=1, right_id=1),
        Entry("国", "クニ", "名詞", 4, cost=9, left_id=2, right_id=2),
        *extra,
    ])


def _plain_unigram() -> UnigramScorer:
    """cost_scale = 1・ペナルティ 0 で、score = −単語コスト になる."""
    return UnigramScorer(cost_scale=1.0, node_penalty=0.0, unk_penalty=0.0, source_bonus={})


def _index(lat: Lattice, surface: str, reading: str) -> int:
    (i,) = [
        i for i, n in enumerate(lat.nodes) if (n.surface, n.reading) == (surface, reading)
    ]
    return i


def _mecab_cost(lat: Lattice, cm: ConnectionMatrix, path: list[int]) -> int:
    nodes = [lat.nodes[i] for i in path]
    total = sum(n.cost for n in nodes)
    total += cm.cost(0, nodes[0].left_id) + cm.cost(nodes[-1].right_id, 0)
    total += sum(cm.cost(a.right_id, b.left_id) for a, b in zip(nodes, nodes[1:], strict=False))
    return total


# --- ★MeCab の目的関数 ---


@pytest.mark.parametrize("seed", range(20))
def test_viterbi_minimizes_the_mecab_objective(tmp_path: Path, seed: int) -> None:
    """★UnigramScorer + 連接行列の Viterbi が、MeCab の目的関数の最小経路と一致すること."""
    lat = build_lattice("米国", _dictionary())
    table = np.random.default_rng(seed).integers(-20, 20, size=(4, 4))  # ID 0 は BOS / EOS
    _write_table(tmp_path / "m.bin", table)
    cm = ConnectionMatrix.load(tmp_path / "m.bin")
    scorer = MatrixTransitionScorer(_plain_unigram(), cm, weight=1.0, cost_scale=1.0)

    in_dict = [p for p in enumerate_paths(lat) if all(lat.nodes[i].entry_id >= 0 for i in p)]
    best = min(_mecab_cost(lat, cm, p) for p in in_dict)
    scores = scorer.score(lat.text, lat.nodes)
    trans = scorer.transitions(lat.nodes)

    got = viterbi(lat, scores, trans)
    assert all(lat.nodes[i].entry_id >= 0 for i in got), "フォールバックは競合しない前提"
    assert _mecab_cost(lat, cm, got) == best
    # ★評価は numpy 版で回す。両者が同じ最適経路を返すこと
    got_np = viterbi_numpy(lat, scores.numpy(), trans.numpy())
    assert _mecab_cost(lat, cm, got_np) == best


def test_the_matrix_changes_the_answer_for_some_tables(tmp_path: Path) -> None:
    """★上のテストに判別力があること. 行列を無視しても通るなら何も確かめていない."""
    lat = build_lattice("米国", _dictionary())
    unigram_only = viterbi(lat, _plain_unigram().score(lat.text, lat.nodes))

    changed = 0
    for seed in range(20):
        table = np.random.default_rng(seed).integers(-20, 20, size=(4, 4))
        _write_table(tmp_path / "m.bin", table)
        cm = ConnectionMatrix.load(tmp_path / "m.bin")
        in_dict = [p for p in enumerate_paths(lat) if all(lat.nodes[i].entry_id >= 0 for i in p)]
        best = min(_mecab_cost(lat, cm, p) for p in in_dict)
        changed += _mecab_cost(lat, cm, unigram_only) != best

    assert changed >= 5


# --- 値の組み立て ---


def _distinct_table(tmp_path: Path, size: int = 4) -> ConnectionMatrix:
    """table[next_left, prev_right] = 10 * next_left + prev_right. 軸の取り違えを検出する."""
    table = np.array([[10 * nl + pr for pr in range(size)] for nl in range(size)])
    _write_table(tmp_path / "m.bin", table)
    return ConnectionMatrix.load(tmp_path / "m.bin")


def test_bos_and_eos_are_folded_into_unary(tmp_path: Path) -> None:
    """文頭で始まるノードに BOS -> v、文末で終わるノードに v -> EOS を足す."""
    lat = build_lattice("米国", _dictionary())
    cm = _distinct_table(tmp_path)
    scores = MatrixTransitionScorer(_plain_unigram(), cm, weight=1.0, cost_scale=1.0).score(
        lat.text, lat.nodes
    )

    # BOS -> left_id L は 10 * L、right_id R -> EOS は R
    assert scores[_index(lat, "米", "コメ")] == -10 - 10       # 文頭のみ
    assert scores[_index(lat, "米", "ベイ")] == -12 - 20
    assert scores[_index(lat, "米国", "ベイコク")] == -30 - 30 - 3  # 文頭かつ文末
    assert scores[_index(lat, "国", "コク")] == -8 - 1          # 文末のみ
    assert scores[_index(lat, "国", "クニ")] == -9 - 2


def test_transitions_use_prev_right_and_next_left(tmp_path: Path) -> None:
    """transitions[u, v] = weight × (−cost(u.right_id -> v.left_id) / cost_scale)."""
    lat = build_lattice("米国", _dictionary())
    cm = _distinct_table(tmp_path)
    trans = MatrixTransitionScorer(_plain_unigram(), cm, weight=0.5, cost_scale=10.0).transitions(
        lat.nodes
    )

    kome, bei = _index(lat, "米", "コメ"), _index(lat, "米", "ベイ")
    koku, kuni = _index(lat, "国", "コク"), _index(lat, "国", "クニ")
    assert trans.shape == (len(lat.nodes), len(lat.nodes))
    assert trans[kome, kuni].item() == pytest.approx(0.5 * -(10 * 2 + 1) / 10.0)
    assert trans[bei, koku].item() == pytest.approx(0.5 * -(10 * 1 + 2) / 10.0)


def test_weight_zero_is_the_base(tmp_path: Path) -> None:
    """重み 0 なら base そのもの（N0 と N の差は行列だけから来る）."""
    lat = build_lattice("米国", _dictionary())
    cm = _distinct_table(tmp_path)
    base = UnigramScorer()
    scorer = MatrixTransitionScorer(base, cm, weight=0.0)

    assert torch.equal(scorer.score(lat.text, lat.nodes), base.score(lat.text, lat.nodes))
    assert not scorer.transitions(lat.nodes).any()


def test_base_transitions_are_added(tmp_path: Path) -> None:
    """base が遷移項を持つ（M6 の trans32 など）なら、行列の遷移に足す."""

    class WithTransitions:
        def score(self, text, nodes):  # noqa: ANN001, ANN201
            return torch.zeros(len(nodes))

        def transitions(self, nodes):  # noqa: ANN001, ANN201
            return torch.ones(len(nodes), len(nodes))

    lat = build_lattice("米国", _dictionary())
    cm = _distinct_table(tmp_path)
    alone = MatrixTransitionScorer(_plain_unigram(), cm, weight=1.0).transitions(lat.nodes)
    stacked = MatrixTransitionScorer(WithTransitions(), cm, weight=1.0).transitions(lat.nodes)

    assert torch.allclose(stacked, alone + 1.0)


# --- ★補完ノード・未知語ノード（ID 0 は BOS / EOS と衝突する） ---


UNK = {"KANJI": (5, 6), "HIRAGANA": (7, 8), "DEFAULT": (9, 9)}


def _unresolved_lattice() -> Lattice:
    """補完ノード（国 -> グニ・aug・ID 0）と未知語フォールバック（犬・の）を含む."""
    dic = _dictionary(Entry("国", "グニ", "名詞", 5, cost=50, source="aug"))
    return build_lattice("米国犬の", dic)


def test_unresolved_nodes_take_unk_ids_by_char_class(tmp_path: Path) -> None:
    """主案: 補完・未知語ノードは unk.def の文字種の ID で連接を引く."""
    lat = _unresolved_lattice()
    cm = _distinct_table(tmp_path, size=10)
    scorer = MatrixTransitionScorer(_plain_unigram(), cm, weight=1.0, cost_scale=1.0, unk=UNK)
    trans = scorer.transitions(lat.nodes)

    guni = _index(lat, "国", "グニ")  # aug -> KANJI (5, 6)
    inu = [i for i, n in enumerate(lat.nodes) if n.surface == "犬"][0]  # 未知語 -> KANJI
    no = [i for i, n in enumerate(lat.nodes) if n.surface == "の"][0]  # 未知語 -> HIRAGANA (7, 8)
    kome = _index(lat, "米", "コメ")

    assert lat.nodes[guni].source == "aug" and lat.nodes[inu].entry_id < 0
    assert trans[kome, guni].item() == -(10 * 5 + 1)  # 米(right 1) -> 国/グニ(left 5)
    assert trans[guni, inu].item() == -(10 * 5 + 6)   # 国/グニ(right 6) -> 犬(left 5)
    assert trans[inu, no].item() == -(10 * 7 + 6)     # 犬(right 6) -> の(left 7)
    # の は文末: v -> EOS は right_id 8
    scores = scorer.score(lat.text, lat.nodes)
    assert scores[no].item() == -lat.nodes[no].cost - 8


def test_zero_unresolved_gives_them_no_connection_cost(tmp_path: Path) -> None:
    """感度分析: 補完・未知語ノードが絡む連接（BOS / EOS を含む）はすべて 0."""
    lat = _unresolved_lattice()
    cm = _distinct_table(tmp_path, size=10)
    scorer = MatrixTransitionScorer(
        _plain_unigram(), cm, weight=1.0, cost_scale=1.0, unresolved_cost=0
    )
    trans = scorer.transitions(lat.nodes)
    scores = scorer.score(lat.text, lat.nodes)

    guni = _index(lat, "国", "グニ")
    no = [i for i, n in enumerate(lat.nodes) if n.surface == "の"][0]
    kome = _index(lat, "米", "コメ")

    assert trans[kome, guni].item() == 0.0
    assert not trans[guni].any() and not trans[:, guni].any()
    assert scores[no].item() == -lat.nodes[no].cost  # 文末だが EOS の連接は足さない
    assert trans[kome, _index(lat, "国", "コク")].item() == -(10 * 1 + 1)  # 辞書ノード同士は残る


def test_unresolved_cost_is_a_constant_for_every_connection(tmp_path: Path) -> None:
    """感度分析（定数）: 0 は中立ではないので、平均的な連接コストでも測れるようにする."""
    lat = _unresolved_lattice()
    cm = _distinct_table(tmp_path, size=10)
    scorer = MatrixTransitionScorer(
        _plain_unigram(), cm, weight=1.0, cost_scale=1.0, unresolved_cost=7
    )
    trans = scorer.transitions(lat.nodes)
    scores = scorer.score(lat.text, lat.nodes)

    guni = _index(lat, "国", "グニ")
    no = [i for i, n in enumerate(lat.nodes) if n.surface == "の"][0]
    kome = _index(lat, "米", "コメ")

    assert trans[kome, guni].item() == -7.0
    assert trans[guni, no].item() == -7.0
    assert scores[no].item() == -lat.nodes[no].cost - 7  # 文末: v -> EOS も定数
    assert scores[kome].item() == -10 - 10  # 辞書ノードは行列のまま


def test_unresolved_nodes_need_an_explicit_policy(tmp_path: Path) -> None:
    """★補完・未知語ノードの ID 0 を黙って使わない（BOS / EOS の ID と衝突する）."""
    lat = _unresolved_lattice()
    cm = _distinct_table(tmp_path, size=10)
    scorer = MatrixTransitionScorer(_plain_unigram(), cm, weight=1.0)

    with pytest.raises(ValueError, match="BOS"):
        scorer.transitions(lat.nodes)
    with pytest.raises(ValueError, match="BOS"):
        scorer.score(lat.text, lat.nodes)
