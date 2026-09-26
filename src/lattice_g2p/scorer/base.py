"""ノードスコアラのインターフェース.

★このインターフェースを差し替え可能に保つこと (docs/architecture.md §9 不変条件 3).
  accuracy-latency カーブを取ることがこのプロジェクトの目的の一つであり,
  スコアラを固定すると比較ができなくなる.

★単体ノードを受けるメソッドを追加してはいけない (docs/pitfalls.md R-12).
  for node in nodes: model(node) は動くが 1 文で数百 ms を超える.
"""

from typing import Protocol

import torch

from lattice_g2p.lattice import Node


class NodeScorer(Protocol):
    """ラティスの全ノードにスコアを付ける."""

    def score(self, text: str, nodes: list[Node]) -> torch.Tensor:
        """全ノードのスコアを一括で返す.

        Returns:
            shape (len(nodes),) の 1 次元テンソル. 値が大きいほど尤もらしい.
        """
        ...
