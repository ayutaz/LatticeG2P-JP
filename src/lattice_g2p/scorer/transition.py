"""遷移スコア (CRF の pairwise 項).

★MeCab / UniDic は left_id × right_id の**連接行列**で
「隣接する語の組み合わせやすさ」を持っている. UniDic のそれは 480 MB ある.

    経緯（docs/results.md §3）:
      M5 までこの実装には遷移項が無く, 経路スコアがノードスコアの単純和だった.
      文脈エンコーダは「文の文脈」は見ているが「選ばれた分割」を見ていない.

ここでは同じ ID を低次元に埋め込み, 内積で近似する.

    transitions[p, c] = <right_emb[p.right_id], left_emb[c.left_id]> / sqrt(d)

15,000 ID × 16 次元 × 2 = 480K params ≈ fp32 1.9 MB / INT8 約 0.5 MB.
**行列そのものの 1/1000 である.** これが M6 の主張.
"""

import math

import torch
from torch import nn

from lattice_g2p.lattice import Node

DEFAULT_CONTEXT_IDS = 16000
"""UniDic CWJ 3.1.1 の連接 ID の上限より少し大きい値.

★これより大きい ID は clamp する. 語彙を削って配布サイズを下げる運用のため.
"""

DEFAULT_DIM = 16
"""埋め込み次元. 上げると表現力が増すがサイズが線形に増える."""

INIT_STD = 0.01
"""★初期値を小さくする.

学習前から経路選択を動かすと「遷移項を足したことの差分」が測れなくなる.
遷移なしの状態から連続的に始めたい.
"""


class TransitionScorer(nn.Module):
    """ノード列から (N, N) の遷移スコアを作る.

    ★出力は密な (N, N) だが, 実際に参照されるのは隣接する辺の組だけである
    (crf._edge_pass). 密に作るのは autograd とベクトル化のため.
    """

    def __init__(
        self,
        n_context_ids: int = DEFAULT_CONTEXT_IDS,
        dim: int = DEFAULT_DIM,
    ) -> None:
        super().__init__()
        self.n_context_ids = n_context_ids
        self.dim = dim
        self.left_emb = nn.Embedding(n_context_ids, dim)
        self.right_emb = nn.Embedding(n_context_ids, dim)
        self._scale = 1.0 / math.sqrt(dim)

        nn.init.normal_(self.left_emb.weight, std=INIT_STD)
        nn.init.normal_(self.right_emb.weight, std=INIT_STD)

    @property
    def device(self) -> torch.device:
        return self.left_emb.weight.device

    def _ids(self, values: list[int]) -> torch.Tensor:
        """★語彙より大きい ID は clamp する. 落とさない."""
        return torch.tensor(values, dtype=torch.long, device=self.device).clamp_(
            0, self.n_context_ids - 1
        )

    def forward(self, nodes: list[Node]) -> torch.Tensor:
        if not nodes:
            return torch.zeros((0, 0), device=self.device)
        right = self.right_emb(self._ids([n.right_id for n in nodes]))
        left = self.left_emb(self._ids([n.left_id for n in nodes]))
        return (right @ left.T) * self._scale
