"""CRF の遷移項（pairwise）のテスト.

★経路スコアはこれまでノードスコアの単純和だった。

    s(path) = Σ unary(node)

標準的な linear-chain CRF は pairwise も持つ。

    s(path) = Σ unary(node) + Σ transitions[前のノード, 次のノード]

MeCab / OpenJTalk の連接行列に相当する項で、
「1 語の読みを選ぶ」能力の欠落の原因候補だった
（docs/results.md §3）。

★CRF は目視でデバッグできない。総当たり列挙との一致が唯一の証明手段である。
"""

import pytest
import torch

from lattice_g2p.crf import (
    build_constrained_graph,
    constrained_logsumexp,
    crf_loss,
    lattice_logsumexp,
    viterbi,
    viterbi_numpy,
)
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice
from tests.helpers import enumerate_paths, gold_paths


@pytest.fixture
def dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0),
            Entry("米", "ベイ", "名詞", 1),
            Entry("米国", "ベイコク", "名詞", 2),
            Entry("国", "コク", "名詞", 3),
            Entry("国", "クニ", "名詞", 4),
            Entry("の", "ノ", "助詞", 5),
        ]
    )


def _case(dic: Dictionary, text: str = "米国の", seed: int = 0):  # noqa: ANN202
    lat = build_lattice(text, dic)
    g = torch.Generator().manual_seed(seed)
    n = len(lat.nodes)
    scores = torch.randn(n, generator=g, dtype=torch.float64)
    transitions = torch.randn((n, n), generator=g, dtype=torch.float64)
    return lat, scores, transitions


def path_score(path: list[int], scores, transitions) -> torch.Tensor:  # noqa: ANN001
    """★1 本の経路のスコア。テストの基準はこの定義だけ。"""
    total = scores[path].sum()
    if transitions is not None:
        for a, b in zip(path, path[1:], strict=False):
            total = total + transitions[a, b]
    return total


def _brute(paths, scores, transitions):  # noqa: ANN001, ANN202
    return torch.logsumexp(
        torch.stack([path_score(p, scores, transitions) for p in paths]), dim=0
    )


# --- 分母 ---


def test_denominator_matches_enumeration(dic: Dictionary) -> None:
    """★遷移項を入れても全経路の総当たりと一致すること。"""
    lat, scores, transitions = _case(dic)
    paths = enumerate_paths(lat)
    assert len(paths) > 2, "経路が分岐していること"

    actual = lattice_logsumexp(lat, scores, transitions)

    assert torch.allclose(actual, _brute(paths, scores, transitions), atol=1e-9)


def test_denominator_without_transitions_is_unchanged(dic: Dictionary) -> None:
    """★遷移なしなら現行とまったく同じ結果であること。

    これが回帰を防ぐ唯一の保証。既存の 457 件はこの前提の上に立っている。
    """
    lat, scores, _ = _case(dic)

    assert torch.allclose(
        lattice_logsumexp(lat, scores, None), lattice_logsumexp(lat, scores), atol=1e-12
    )
    assert torch.allclose(
        lattice_logsumexp(lat, scores, torch.zeros((len(lat.nodes),) * 2, dtype=torch.float64)),
        lattice_logsumexp(lat, scores),
        atol=1e-9,
    )


# --- 分子 ---


def test_numerator_matches_enumeration(dic: Dictionary) -> None:
    """★正解経路集合でも総当たりと一致すること。"""
    lat, scores, transitions = _case(dic)
    gold = "ベイコクノ"
    graph = build_constrained_graph(lat, gold)
    paths = gold_paths(lat, gold)
    assert len(paths) > 1, "正解経路が複数あること（分割違い）"

    actual = constrained_logsumexp(graph, scores, transitions)

    assert torch.allclose(actual, _brute(paths, scores, transitions), atol=1e-9)


def test_loss_is_non_negative(dic: Dictionary) -> None:
    """★G ⊆ P なので損失は常に 0 以上。遷移項を入れても成り立つこと。"""
    lat, scores, transitions = _case(dic)
    graph = build_constrained_graph(lat, "ベイコクノ")

    assert crf_loss(lat, graph, scores, transitions).item() >= -1e-9


# --- Viterbi ---


def test_viterbi_matches_enumeration(dic: Dictionary) -> None:
    """★最尤経路が総当たりの最大と一致すること。back pointer の正しさ。"""
    lat, scores, transitions = _case(dic)
    paths = enumerate_paths(lat)
    best = max(paths, key=lambda p: float(path_score(p, scores, transitions)))

    assert viterbi(lat, scores, transitions) == best


def test_viterbi_numpy_matches_torch(dic: Dictionary) -> None:
    """★推論経路（torch 非依存）も一致すること。"""
    lat, scores, transitions = _case(dic)

    assert viterbi_numpy(lat, scores.numpy(), transitions.numpy()) == viterbi(
        lat, scores, transitions
    )


def test_viterbi_without_transitions_is_unchanged(dic: Dictionary) -> None:
    lat, scores, _ = _case(dic)

    assert viterbi(lat, scores, None) == viterbi(lat, scores)


def test_transitions_change_the_chosen_path(dic: Dictionary) -> None:
    """★遷移項が実際に経路選択を変えること（入れた意味があるかの確認）。"""
    lat, scores, _ = _case(dic)
    n = len(lat.nodes)
    base = viterbi(lat, scores)

    # base の最初の遷移に大きな罰則を与えると別の経路になるはず
    transitions = torch.zeros((n, n), dtype=torch.float64)
    transitions[base[0], base[1]] = -100.0

    assert viterbi(lat, scores, transitions) != base


# --- 勾配 ---


def test_gradcheck_through_transitions(dic: Dictionary) -> None:
    """★遷移項について勾配が正しいこと。"""
    lat, scores, transitions = _case(dic, seed=3)
    graph = build_constrained_graph(lat, "ベイコクノ")

    def fn(t: torch.Tensor) -> torch.Tensor:
        return crf_loss(lat, graph, scores, t)

    assert torch.autograd.gradcheck(fn, (transitions.requires_grad_(True),), eps=1e-6)


def test_gradcheck_through_scores_with_transitions(dic: Dictionary) -> None:
    lat, scores, transitions = _case(dic, seed=4)
    graph = build_constrained_graph(lat, "ベイコクノ")

    def fn(s: torch.Tensor) -> torch.Tensor:
        return crf_loss(lat, graph, s, transitions)

    assert torch.autograd.gradcheck(fn, (scores.requires_grad_(True),), eps=1e-6)
