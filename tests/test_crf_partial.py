"""部分制約付き積グラフ.

★青空文庫のルビは文の一部にしか付かない。文全体の読みを持つ M2 の生成データと
違い、「この span だけ読みが分かっている」という制約しか課せない。

    彼は｜生真面目《きまじめ》な男だ
      → text  : 彼は生真面目な男だ
      → spans : [(2, 5, "キマジメ")]   それ以外は未知

build_constrained_graph はこの一般化になっている (span が文全体 1 個の場合)。
"""

import pytest
import torch

from lattice_g2p.crf import (
    build_constrained_graph,
    build_partially_constrained_graph,
    constrained_logsumexp,
    crf_loss,
    lattice_logsumexp,
)
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice
from tests.helpers import paths_satisfying_spans


@pytest.fixture
def dic() -> Dictionary:
    entries = [
        Entry("米", "コメ", "名詞", 0),
        Entry("米", "ベイ", "名詞", 1),
        Entry("米国", "ベイコク", "名詞", 2),
        Entry("国", "コク", "名詞", 3),
        Entry("国", "クニ", "名詞", 4),
        Entry("の", "ノ", "助詞", 5),
    ]
    return Dictionary.from_entries(entries)


def _scores(lattice, value: float = 0.0) -> torch.Tensor:  # noqa: ANN001
    return torch.full((len(lattice.nodes),), value, dtype=torch.float64)


# --- 一般化であることの確認 ---


def test_reduces_to_build_constrained_graph(dic: Dictionary) -> None:
    """★span が文全体 1 個なら build_constrained_graph と同じものになること。

    これが「一般化した」ことの定義。分子の値とノード集合が一致すればよい
    (状態のラベル付けは異なってよい)。
    """
    lat = build_lattice("米国の米", dic)
    gold = "ベイコクノコメ"

    full = build_constrained_graph(lat, gold)
    partial = build_partially_constrained_graph(lat, [(0, len(lat.text), gold)])

    scores = torch.randn(len(lat.nodes), dtype=torch.float64)
    assert torch.allclose(
        constrained_logsumexp(full, scores), constrained_logsumexp(partial, scores)
    )
    assert full.gold_node_indices() == partial.gold_node_indices()


def test_no_spans_means_no_constraint(dic: Dictionary) -> None:
    """制約がなければ分子 = 分母。損失は 0 になる。"""
    lat = build_lattice("米国", dic)
    graph = build_partially_constrained_graph(lat, [])
    scores = torch.randn(len(lat.nodes), dtype=torch.float64)

    assert graph.is_reachable()
    assert torch.allclose(
        constrained_logsumexp(graph, scores), lattice_logsumexp(lat, scores)
    )
    assert torch.allclose(
        crf_loss(lat, graph, scores), torch.zeros((), dtype=torch.float64)
    )


# --- 部分制約の意味 ---


def test_constrains_only_the_given_span(dic: Dictionary) -> None:
    """★span の外は自由であること。

    「米の米」の先頭だけ ベイ と指定する。末尾の「米」は コメ でも ベイ でもよい。
    """
    lat = build_lattice("米の米", dic)
    graph = build_partially_constrained_graph(lat, [(0, 1, "ベイ")])

    used = {(lat.nodes[i].start, lat.nodes[i].reading) for i in graph.gold_node_indices()}
    assert (0, "ベイ") in used
    assert (0, "コメ") not in used, "span 内は制約される"
    assert (2, "ベイ") in used, "span 外は自由"
    assert (2, "コメ") in used


def test_rejects_when_span_reading_is_not_producible(dic: Dictionary) -> None:
    lat = build_lattice("米の米", dic)
    graph = build_partially_constrained_graph(lat, [(0, 1, "マイ")])

    assert not graph.is_reachable()
    assert graph.goal_state is None


def test_multiple_spans_are_all_enforced(dic: Dictionary) -> None:
    lat = build_lattice("米の米", dic)
    graph = build_partially_constrained_graph(lat, [(0, 1, "ベイ"), (2, 3, "コメ")])

    used = {(lat.nodes[i].start, lat.nodes[i].reading) for i in graph.gold_node_indices()}
    assert used == {(0, "ベイ"), (1, "ノ"), (2, "コメ")}


def test_node_straddling_the_span_boundary_is_excluded(dic: Dictionary) -> None:
    """★span の境界をまたぐノードは使えないこと。

    「米国」を 1 ノード (ベイコク) で読む経路は、「米」だけに付いたルビを
    検証できない。カナのどこまでが「米」の読みか決められないため。

    厳密性を優先して落とす。その結果 span を分離できる経路が 1 本もなければ
    到達不能となり、その例は学習に使えない (Oracle 失敗と同じ扱い)。
    """
    lat = build_lattice("米国", dic)
    graph = build_partially_constrained_graph(lat, [(0, 1, "ベイ")])

    used = {(lat.nodes[i].surface, lat.nodes[i].reading) for i in graph.gold_node_indices()}
    assert ("米国", "ベイコク") not in used
    assert ("米", "ベイ") in used
    assert ("国", "コク") in used, "span の外なので自由"
    assert ("国", "クニ") in used


def test_unreachable_when_only_a_straddling_node_exists() -> None:
    """span を分離できるノードが存在しなければ到達不能になること。"""
    d = Dictionary.from_entries([Entry("米国", "ベイコク", "名詞", 0)])
    lat = build_lattice("米国", d)
    graph = build_partially_constrained_graph(lat, [(0, 1, "ベイ")])

    assert not graph.is_reachable()


# --- 素朴な全列挙との一致 ---


@pytest.mark.parametrize(
    "text,spans",
    [
        ("米国の米", []),
        ("米国の米", [(0, 2, "ベイコク")]),
        ("米国の米", [(3, 4, "コメ")]),
        ("米国の米", [(0, 2, "ベイコク"), (3, 4, "ベイ")]),
        ("米の国", [(2, 3, "クニ")]),
        ("米国の米", [(0, 4, "ベイコクノコメ")]),
    ],
)
def test_matches_naive_enumeration(
    dic: Dictionary, text: str, spans: list[tuple[int, int, str]]
) -> None:
    """★仕様は素朴な全列挙で定義する。積グラフはその効率的な計算にすぎない。"""
    lat = build_lattice(text, dic)
    graph = build_partially_constrained_graph(lat, spans)
    expected = paths_satisfying_spans(lat, spans)

    assert graph.is_reachable() == bool(expected)
    if not expected:
        return

    # 分子の値が一致すること
    scores = torch.randn(len(lat.nodes), dtype=torch.float64)
    naive = torch.logsumexp(
        torch.stack([scores[p].sum() for p in expected]), dim=0
    )
    assert torch.allclose(constrained_logsumexp(graph, scores), naive)

    # 使われるノード集合が一致すること
    assert graph.gold_node_indices() == {i for p in expected for i in p}


def test_states_are_topologically_sorted(dic: Dictionary) -> None:
    """★状態 index の昇順がトポロジカル順であること。forward がこれに依存する。"""
    lat = build_lattice("米国の米", dic)
    graph = build_partially_constrained_graph(lat, [(0, 2, "ベイコク")])

    for _, src, dst in graph.edges:
        assert src < dst


# --- 入力の検証 ---


def test_overlapping_spans_are_rejected(dic: Dictionary) -> None:
    lat = build_lattice("米国の米", dic)
    with pytest.raises(ValueError, match="重なって"):
        build_partially_constrained_graph(lat, [(0, 2, "ベイコク"), (1, 3, "コクノ")])


def test_span_out_of_range_is_rejected(dic: Dictionary) -> None:
    lat = build_lattice("米国", dic)
    with pytest.raises(ValueError, match="範囲"):
        build_partially_constrained_graph(lat, [(0, 5, "ベイコク")])


def test_empty_span_is_rejected(dic: Dictionary) -> None:
    lat = build_lattice("米国", dic)
    with pytest.raises(ValueError, match="範囲"):
        build_partially_constrained_graph(lat, [(1, 1, "")])


def test_spans_need_not_be_sorted(dic: Dictionary) -> None:
    lat = build_lattice("米の米", dic)
    a = build_partially_constrained_graph(lat, [(0, 1, "ベイ"), (2, 3, "コメ")])
    b = build_partially_constrained_graph(lat, [(2, 3, "コメ"), (0, 1, "ベイ")])

    assert a.gold_node_indices() == b.gold_node_indices()
