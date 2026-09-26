import pytest
import torch

from lattice_g2p.crf import (
    build_constrained_graph,
    constrained_logsumexp,
    crf_loss,
    lattice_logsumexp,
    viterbi,
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
            Entry("。", "*", "補助記号-句点", 6),
        ]
    )


def _brute_force(scores: torch.Tensor, paths: list[list[int]]) -> torch.Tensor:
    return torch.logsumexp(torch.stack([scores[p].sum() for p in paths]), dim=0)


def test_denominator_matches_brute_force(dic: Dictionary) -> None:
    """★分母が全経路の総当たり logsumexp と一致すること。"""
    lat = build_lattice("米国の", dic)
    torch.manual_seed(0)
    scores = torch.randn(len(lat.nodes), dtype=torch.float64)

    paths = enumerate_paths(lat)
    assert len(paths) > 1, "テストとして意味がある程度に経路が分岐していること"

    assert torch.allclose(lattice_logsumexp(lat, scores), _brute_force(scores, paths), atol=1e-9)


def test_denominator_matches_brute_force_with_punctuation(dic: Dictionary) -> None:
    """空読みノードがあっても分母が一致すること。"""
    lat = build_lattice("米国。", dic)
    torch.manual_seed(7)
    scores = torch.randn(len(lat.nodes), dtype=torch.float64)

    assert torch.allclose(
        lattice_logsumexp(lat, scores), _brute_force(scores, enumerate_paths(lat)), atol=1e-9
    )


def test_numerator_matches_brute_force(dic: Dictionary) -> None:
    """★分子が正解経路だけの総当たり logsumexp と一致すること。"""
    lat = build_lattice("米国", dic)
    gold = "ベイコク"
    torch.manual_seed(1)
    scores = torch.randn(len(lat.nodes), dtype=torch.float64)

    paths = gold_paths(lat, gold)
    assert len(paths) >= 2, "「米国」と「米+国」の 2 経路が正解になること"

    graph = build_constrained_graph(lat, gold)
    assert torch.allclose(
        constrained_logsumexp(graph, scores), _brute_force(scores, paths), atol=1e-9
    )


def test_numerator_matches_brute_force_longer(dic: Dictionary) -> None:
    lat = build_lattice("米国の。", dic)
    gold = "ベイコクノ"
    torch.manual_seed(3)
    scores = torch.randn(len(lat.nodes), dtype=torch.float64)

    paths = gold_paths(lat, gold)
    assert paths

    graph = build_constrained_graph(lat, gold)
    assert torch.allclose(
        constrained_logsumexp(graph, scores), _brute_force(scores, paths), atol=1e-9
    )


def test_numerator_never_exceeds_denominator(dic: Dictionary) -> None:
    """★G ⊆ P なので分子 ≤ 分母 が常に成り立つこと。"""
    lat = build_lattice("米国の", dic)
    graph = build_constrained_graph(lat, "ベイコクノ")

    for seed in range(20):
        torch.manual_seed(seed)
        scores = torch.randn(len(lat.nodes), dtype=torch.float64) * 10
        num = constrained_logsumexp(graph, scores)
        den = lattice_logsumexp(lat, scores)
        assert num <= den + 1e-9, f"seed={seed}: 分子 {num} > 分母 {den}"


def test_loss_is_non_negative(dic: Dictionary) -> None:
    lat = build_lattice("米国の", dic)
    graph = build_constrained_graph(lat, "ベイコクノ")

    for seed in range(20):
        torch.manual_seed(seed)
        scores = torch.randn(len(lat.nodes), dtype=torch.float64) * 10
        assert float(crf_loss(lat, graph, scores)) >= -1e-9


def test_loss_decreases_when_gold_score_increases(dic: Dictionary) -> None:
    """正解経路上のノードのスコアを上げると損失が下がること。"""
    lat = build_lattice("米国", dic)
    graph = build_constrained_graph(lat, "ベイコク")
    scores = torch.zeros(len(lat.nodes), dtype=torch.float64)
    before = float(crf_loss(lat, graph, scores))

    boosted = scores.clone()
    for node_idx in graph.gold_node_indices():
        boosted[node_idx] += 5.0

    assert float(crf_loss(lat, graph, boosted)) < before


def test_unreachable_graph_raises(dic: Dictionary) -> None:
    """★正解経路がないデータは学習に使えない。明示的に失敗すること。"""
    lat = build_lattice("米国", dic)
    graph = build_constrained_graph(lat, "イリガビ")
    scores = torch.zeros(len(lat.nodes), dtype=torch.float64)

    with pytest.raises(ValueError, match="正解経路"):
        constrained_logsumexp(graph, scores)


def test_gradient_matches_numerical(dic: Dictionary) -> None:
    """★勾配が数値微分と一致すること。"""
    lat = build_lattice("米国", dic)
    graph = build_constrained_graph(lat, "ベイコク")

    def fn(s: torch.Tensor) -> torch.Tensor:
        return crf_loss(lat, graph, s)

    torch.manual_seed(0)
    scores = torch.randn(len(lat.nodes), dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(fn, (scores,), eps=1e-6, atol=1e-6)


def test_gradient_is_finite(dic: Dictionary) -> None:
    """損失が nan / inf にならないこと。"""
    lat = build_lattice("米国の。", dic)
    graph = build_constrained_graph(lat, "ベイコクノ")
    torch.manual_seed(11)
    scores = (torch.randn(len(lat.nodes), dtype=torch.float64) * 50).requires_grad_()

    loss = crf_loss(lat, graph, scores)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(scores.grad).all()


def test_viterbi_score_equals_best_path(dic: Dictionary) -> None:
    """★Viterbi の経路スコアが全経路の最大値と一致すること。"""
    lat = build_lattice("米国の", dic)

    for seed in range(10):
        torch.manual_seed(seed)
        scores = torch.randn(len(lat.nodes), dtype=torch.float64)

        best = max(float(scores[p].sum()) for p in enumerate_paths(lat))
        got = float(scores[viterbi(lat, scores)].sum())

        assert got == pytest.approx(best, abs=1e-9)


def test_viterbi_on_empty_text(dic: Dictionary) -> None:
    lat = build_lattice("", dic)
    assert viterbi(lat, torch.zeros(0)) == []
