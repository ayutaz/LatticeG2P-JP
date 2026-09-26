"""連接行列を固定の遷移項として足すスコアラ（M7 Task 2）.

★M6 で試したのは連接行列の**低ランク近似**（16 / 32 次元の内積）であって、
行列そのものではなかった。ここでは UniDic の行列（15,626 × 15,388・固定）を
そのまま差し込み、「遷移項で差が埋まるか」の上限を測る（docs/results.md §3）。

    score(v)          = base(v) + weight × (−(BOS -> v と v -> EOS の連接コスト) / cost_scale)
    transitions[u, v] = base の遷移（あれば） + weight × (−cost(u -> v) / cost_scale)

weight = 1 で UnigramScorer（cost_scale = 100・ペナルティ 0）に足すと、経路スコアは
−(Σ 単語コスト + Σ 連接コスト) / 100 になり、Viterbi は **MeCab の目的関数を最小にする
経路**を返す（tests/test_matrix_scorer.py が総当たりと突き合わせている）.

★任意の NodeScorer をラップするだけにする。score() と transitions() を持てば、
compare_systems.py / run_hard_set.py / run_eval.py が変更なしで使える
（M6 で getattr(scorer, "transitions", None) を通してある）。不変条件 3 も保たれる。

★BOS / EOS は unary に畳み込む。CRF の DP は変えない（不変条件 14）。
"""

from __future__ import annotations

import numpy as np
import torch

from lattice_g2p.connection import ConnectionMatrix, is_unresolved, resolve_ids
from lattice_g2p.lattice import Node
from lattice_g2p.scorer.base import NodeScorer


class MatrixTransitionScorer:
    """base のスコアに、UniDic の連接コストを固定の遷移項として足す.

    Args:
        base: 任意の NodeScorer（UnigramScorer / NeuralScorer / OnnxScorer）.
        matrix: 連接行列.
        weight: 連接コストの重み. U（UnigramScorer）では 1.0（MeCab は単語コストと
            連接コストを同じ尺度で学習している）. N（ニューラル）では各 checkpoint が
            学習した dict_prior_weight（単語コストと同じ効かせ方）.
        cost_scale: コストの割る数. UnigramScorer / 辞書事前分布の 100 とそろえる.
        unk: 文字種 -> (left_id, right_id)（connection.unk_ids）. 補完ノード・
            未知語ノードの ID に使う（主案）.
        unresolved_cost: 補完ノード・未知語ノードが絡む連接（BOS / EOS を含む）を
            すべてこの定数にする（感度分析）. 与えると unk より優先する.
            ★0 は中立ではない. UniDic の連接コストは平均 5,319（p5〜p95: 2,701〜7,198）
            なので、0 は補完ノードを隣接 1 つあたり約 5,000 優遇する.

    ★補完ノード（source="aug"）と未知語フォールバック（entry_id < 0）は ID が 0 で、
    **BOS / EOS の ID と衝突する**。unk も unresolved_cost も与えずにそれらが来たら止める.
    """

    def __init__(
        self,
        base: NodeScorer,
        matrix: ConnectionMatrix,
        weight: float,
        cost_scale: float = 100.0,
        unk: dict[str, tuple[int, int]] | None = None,
        unresolved_cost: int | None = None,
    ) -> None:
        self.base = base
        self.matrix = matrix
        self.weight = weight
        self.cost_scale = cost_scale
        self.unk = unk
        self.unresolved_cost = unresolved_cost

    # --- 生の連接コスト（重みと尺度を掛ける前） ---

    def _ids(self, nodes: list[Node]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        unresolved = np.array([is_unresolved(n) for n in nodes], dtype=bool)
        if unresolved.any() and self.unk is None and self.unresolved_cost is None:
            raise ValueError(
                f"連接 ID を持たないノード（補完・未知語）が {int(unresolved.sum())} 個あります。"
                "その ID 0 は BOS / EOS の ID と衝突します。"
                "unk（unk.def の文字種 ID）か unresolved_cost を指定してください。"
            )
        left, right = resolve_ids(nodes, None if self.unresolved_cost is not None else self.unk)
        return left, right, unresolved

    def boundary_costs(self, text: str, nodes: list[Node]) -> np.ndarray:
        """(N,). 文頭で始まるノードは BOS -> v、文末で終わるノードは v -> EOS の連接コスト."""
        left, right, unresolved = self._ids(nodes)
        starts = np.array([n.start == 0 for n in nodes], dtype=bool)
        ends = np.array([n.end == len(text) for n in nodes], dtype=bool)
        out = np.where(starts, self.matrix.bos_costs(left), 0) + np.where(
            ends, self.matrix.eos_costs(right), 0
        )
        if self.unresolved_cost is not None:
            out[unresolved] = (starts[unresolved].astype(np.int64) + ends[unresolved]) * (
                self.unresolved_cost
            )
        return out

    def connection_costs(self, nodes: list[Node]) -> np.ndarray:
        """(N, N). [u, v] = cost(u.right_id -> v.left_id)."""
        left, right, unresolved = self._ids(nodes)
        conn = self.matrix.costs(right, left)
        if self.unresolved_cost is not None:
            conn[unresolved, :] = self.unresolved_cost
            conn[:, unresolved] = self.unresolved_cost
        return conn

    def _as_score(self, costs: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            (self.weight * (-costs.astype(np.float64) / self.cost_scale)).astype(np.float32)
        )

    # --- NodeScorer ---

    def score(self, text: str, nodes: list[Node]) -> torch.Tensor:
        base = self.base.score(text, nodes)
        return base + self._as_score(self.boundary_costs(text, nodes)).to(base.dtype)

    def transitions(self, nodes: list[Node]) -> torch.Tensor:
        out = self._as_score(self.connection_costs(nodes))
        getter = getattr(self.base, "transitions", None)
        base_trans = None if getter is None else getter(nodes)
        if base_trans is not None:
            out = out + torch.as_tensor(base_trans, dtype=out.dtype)
        return out
