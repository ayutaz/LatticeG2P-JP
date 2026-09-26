"""辞書コストだけを使うベースラインスコアラ.

ニューラルネットなしで CRF・Viterbi・評価指標が正しく繋がっているかを確認するために使う.
MeCab に近い挙動になることを期待する (M1 のゲート: Joyo Accuracy > 90%).
"""

import torch

from lattice_g2p.lattice import Node


class UnigramScorer:
    """ノード単体の辞書コストからスコアを作る.

    ★pron 由来と kana 由来の読みは同コストになるため, このスコアラだけでは
    「チュー」と「チュウ」,「オ」と「ヲ」を区別できない. source_bonus で
    表記の流儀を選ぶ. 文脈による本来の読み分けはニューラルスコアラ (M3) の仕事.

    Args:
        length_bonus: 1 文字あたりの加点.
            ★経路上のノード長の総和は必ず文字数 L に等しいため, これは経路に
            よらない定数であり argmax を変えられない (実質 no-op).
            長い語を優先させたいなら node_penalty を使うこと.
        node_penalty: ノード 1 個あたりの減点. 少ないノード数の経路を優先させる.
            MeCab が連接行列から得ている効果の粗い近似.
        cost_scale: UniDic コストの割る数. 小さいほどコスト差が効く
        unk_penalty: 辞書にないフォールバックノードへの減点
        source_bonus: 読みの由来ごとの加点. 既定は kana 形をわずかに優先する
    """

    DEFAULT_SOURCE_BONUS = {"kana": 0.5, "pron": 0.0, "both": 0.5, "aug": 0.0}
    """既定値は Joyo benchmark 上のスイープで決めた (docs/evaluation.md §3-2).

    kana 形をわずかに優先するのは, benchmark の表記が「トウキョウ」「ヲ」のような
    仮名形だから. 助詞の「は->ワ」「へ->エ」だけは pron 形が正しいが,
    ユニグラムでは区別できない. 文脈による読み分けはニューラルスコアラ (M3) の仕事.
    """

    def __init__(
        self,
        length_bonus: float = 0.0,
        cost_scale: float = 100.0,
        unk_penalty: float = 5.0,
        source_bonus: dict[str, float] | None = None,
        node_penalty: float = 30.0,
    ) -> None:
        self.length_bonus = length_bonus
        self.node_penalty = node_penalty
        self.cost_scale = cost_scale
        self.unk_penalty = unk_penalty
        self.source_bonus = (
            dict(self.DEFAULT_SOURCE_BONUS) if source_bonus is None else source_bonus
        )

    def score(self, text: str, nodes: list[Node]) -> torch.Tensor:
        del text  # このスコアラは文脈を使わない
        values = [
            -node.cost / self.cost_scale
            + self.length_bonus * (node.end - node.start)
            - self.node_penalty
            + self.source_bonus.get(node.source, 0.0)
            - (self.unk_penalty if node.entry_id < 0 else 0.0)
            for node in nodes
        ]
        return torch.tensor(values, dtype=torch.float32)
