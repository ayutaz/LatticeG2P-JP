import pytest
import torch

from lattice_g2p.lattice import Node
from lattice_g2p.losses import margin_loss


def _node(start: int, end: int, reading: str) -> Node:
    return Node(start=start, end=end, surface="x", reading=reading, entry_id=0, pos="名詞")


def test_zero_when_margin_is_satisfied() -> None:
    nodes = [_node(0, 1, "コメ"), _node(0, 1, "ベイ")]
    scores = torch.tensor([5.0, 0.0])

    assert float(margin_loss(scores, nodes, [0], margin=1.0)) == 0.0


def test_positive_when_gold_score_is_too_low() -> None:
    nodes = [_node(0, 1, "コメ"), _node(0, 1, "ベイ")]
    scores = torch.tensor([0.0, 0.0])

    assert float(margin_loss(scores, nodes, [0], margin=1.0)) == 1.0


def test_only_compares_within_same_span() -> None:
    """★別の span のノードとは比較しないこと。"""
    nodes = [_node(0, 1, "コメ"), _node(1, 2, "クニ")]
    scores = torch.tensor([0.0, 100.0])

    assert float(margin_loss(scores, nodes, [0], margin=1.0)) == 0.0


def test_zero_when_span_has_no_competitor() -> None:
    assert float(margin_loss(torch.tensor([0.0]), [_node(0, 1, "コメ")], [0], margin=1.0)) == 0.0


def test_zero_when_no_gold_nodes() -> None:
    nodes = [_node(0, 1, "コメ"), _node(0, 1, "ベイ")]
    assert float(margin_loss(torch.tensor([0.0, 0.0]), nodes, [], margin=1.0)) == 0.0


def test_averages_over_pairs() -> None:
    nodes = [_node(0, 1, "コメ"), _node(0, 1, "ベイ"), _node(0, 1, "マイ")]
    scores = torch.tensor([0.0, 0.0, -1.0])
    # コメ vs ベイ: max(0, 1 - 0) = 1.0
    # コメ vs マイ: max(0, 1 - 1) = 0.0

    assert float(margin_loss(scores, nodes, [0], margin=1.0)) == 0.5


def test_is_differentiable() -> None:
    nodes = [_node(0, 1, "コメ"), _node(0, 1, "ベイ")]
    scores = torch.tensor([0.0, 0.0], requires_grad=True)

    margin_loss(scores, nodes, [0], margin=1.0).backward()

    assert scores.grad is not None
    assert float(scores.grad[0]) < 0, "正解ノードのスコアを上げる方向に勾配が向くこと"
    assert float(scores.grad[1]) > 0, "不正解ノードのスコアを下げる方向に勾配が向くこと"


def test_is_never_negative() -> None:
    nodes = [_node(0, 1, "コメ"), _node(0, 1, "ベイ")]
    for seed in range(20):
        torch.manual_seed(seed)
        scores = torch.randn(2) * 10
        assert float(margin_loss(scores, nodes, [0], margin=1.0)) >= 0.0


def test_multiple_gold_nodes_in_same_span() -> None:
    """同じ span に正解ノードが複数ある場合（読みが同じで由来が違う等）。"""
    nodes = [_node(0, 1, "コメ"), _node(0, 1, "コメ"), _node(0, 1, "ベイ")]
    scores = torch.tensor([0.0, 0.0, 5.0])

    loss = float(margin_loss(scores, nodes, [0, 1], margin=1.0))

    assert loss == pytest.approx(6.0), "2 つの正解 × 1 つの不正解の平均"


def test_empty_nodes() -> None:
    assert float(margin_loss(torch.zeros(0), [], [], margin=1.0)) == 0.0


def test_returns_zero_dim_tensor() -> None:
    nodes = [_node(0, 1, "コメ"), _node(0, 1, "ベイ")]
    loss = margin_loss(torch.tensor([0.0, 0.0]), nodes, [0], margin=1.0)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
