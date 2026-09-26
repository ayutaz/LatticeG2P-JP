"""margin loss.

同一 span 上の正解読みノードと不正解読みノードを直接比較する.

    L = max(0, m - (s(v_gold) - s(v_competitor)))

CRF loss は経路全体の確率を最大化するが, 個々の読み候補の比較は間接的にしか
行われない. margin loss はそこを直接押さえる.

論文のアブレーション: Accuracy 99.53% -> 99.62% / Target PER 0.43% -> 0.32%
"""

from collections import defaultdict

import torch

from lattice_g2p.lattice import Node


def margin_loss(
    scores: torch.Tensor,
    nodes: list[Node],
    gold_node_indices: list[int],
    margin: float = 1.0,
) -> torch.Tensor:
    """同一 span 内の正解 / 不正解ペアに対する margin loss の平均.

    ★span が同じノード同士だけを比較する. 別の span のノードと比べても
    意味がない (そもそも同時に選ばれうる).

    Args:
        gold_node_indices: 正解経路上に現れるノードの index
            (ConstrainedGraph.gold_node_indices() から得る)
    """
    if not nodes:
        return torch.zeros((), dtype=torch.float32, device=scores.device)

    gold = set(gold_node_indices)
    by_span: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, node in enumerate(nodes):
        by_span[(node.start, node.end)].append(i)

    terms: list[torch.Tensor] = []
    for indices in by_span.values():
        gold_idx = [i for i in indices if i in gold]
        other_idx = [i for i in indices if i not in gold]
        if not gold_idx or not other_idx:
            continue
        g = scores[gold_idx].unsqueeze(1)  # (G, 1)
        c = scores[other_idx].unsqueeze(0)  # (1, C)
        terms.append(torch.clamp(margin - (g - c), min=0.0).flatten())

    if not terms:
        return torch.zeros((), dtype=scores.dtype, device=scores.device)
    return torch.cat(terms).mean()
