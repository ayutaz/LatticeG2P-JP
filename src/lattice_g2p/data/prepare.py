"""学習データを CRF がそのまま食える形に前処理する.

★遷移構造は入力文と正解読みだけで決まるので前計算できる.
スコアはモデルパラメータに依存するため forward は毎ステップ必要だが,
「どのノードがどの遷移として有効か」は固定である.
学習ループで毎回ラティスを組み直すと, 学習が辞書引きに律速される
(docs/architecture.md §5-4).
"""

from dataclasses import dataclass

from lattice_g2p.crf import build_constrained_graph, build_partially_constrained_graph
from lattice_g2p.data.ruby import RubyExample
from lattice_g2p.data.schema import TrainingExample
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.lattice import Lattice, Node, build_lattice


@dataclass
class PreparedExample:
    """CRF の遷移構造を前計算した学習データ 1 件."""

    text: str
    gold_reading: str | None
    """文全体の読み. ★青空文庫のルビ由来の例では None になる.

    ルビは文の一部にしか付かないので, 文全体の読みは分からない.
    そのため精度評価 (train.evaluate) には使えず, 学習専用の例になる.
    """

    nodes: list[Node]

    denom_edges: list[tuple[int, int, int]]
    """(node_idx, from_pos, to_pos). 分母 (全経路) 用."""

    numer_edges: list[tuple[int, int, int]]
    """(node_idx, from_state, to_state). 分子 (正解経路) 用の積グラフ."""

    num_positions: int
    num_states: int
    start_state: int
    goal_state: int
    gold_node_indices: list[int]
    target_start: int
    target_end: int

    def lattice(self) -> Lattice:
        """Viterbi や評価のために Lattice を復元する."""
        starting: list[list[int]] = [[] for _ in range(self.num_positions)]
        ending: list[list[int]] = [[] for _ in range(self.num_positions)]
        for idx, node in enumerate(self.nodes):
            starting[node.start].append(idx)
            ending[node.end].append(idx)
        return Lattice(
            text=self.text,
            nodes=self.nodes,
            nodes_starting_at=starting,
            nodes_ending_at=ending,
        )

    def gold_span_reading(self) -> str:
        """正解経路 1 本から target span の読みを取り出す.

        評価時の span_reading と同じ規約 (span と重なるノードを丸ごと含める) を使う.
        """
        nxt: dict[int, tuple[int, int]] = {}
        for node_idx, src, dst in self.numer_edges:
            nxt.setdefault(src, (node_idx, dst))

        parts: list[str] = []
        state = self.start_state
        while state != self.goal_state:
            node_idx, dst = nxt[state]
            node = self.nodes[node_idx]
            if node.start < self.target_end and node.end > self.target_start:
                parts.append(node.reading)
            state = dst
        return "".join(parts)


def prepare(example: TrainingExample, dic: Dictionary) -> PreparedExample | None:
    """学習データ 1 件を前処理する.

    Returns:
        正解経路が存在しない場合は None (呼び出し側で捨てる)
    """
    lattice = build_lattice(example.text, dic)
    graph = build_constrained_graph(lattice, example.reading)
    if graph.goal_state is None:
        return None

    return PreparedExample(
        text=example.text,
        gold_reading=example.reading,
        nodes=lattice.nodes,
        denom_edges=[(i, n.start, n.end) for i, n in enumerate(lattice.nodes)],
        numer_edges=list(graph.edges),
        num_positions=len(example.text) + 1,
        num_states=len(graph.states),
        start_state=graph.start_state,
        goal_state=graph.goal_state,
        gold_node_indices=sorted(graph.gold_node_indices()),
        target_start=example.target_start,
        target_end=example.target_end,
    )


def prepare_ruby(example: RubyExample, dic: Dictionary) -> PreparedExample | None:
    """ルビ由来の例 1 件を前処理する.

    ★分子は部分制約付き積グラフで作る. 文全体の読みは要らない.
    分母 (全経路) は prepare と完全に同じなので, 学習ループは両者を区別せずに
    crf_loss_from_prepared を呼べる.

    target span は先頭のルビにする. 参考値 (evaluate_target) の計算に使うだけで,
    損失にはすべての span が効く.

    Returns:
        使える span が 1 つも無い場合は None (呼び出し側で捨てる)
    """
    lattice = build_lattice(example.text, dic)
    graph = build_partially_constrained_graph(lattice, example.spans)
    if graph.goal_state is None:
        return None

    first = min(example.spans) if example.spans else (0, 0, "")
    return PreparedExample(
        text=example.text,
        gold_reading=None,
        nodes=lattice.nodes,
        denom_edges=[(i, n.start, n.end) for i, n in enumerate(lattice.nodes)],
        numer_edges=list(graph.edges),
        num_positions=len(example.text) + 1,
        num_states=len(graph.states),
        start_state=graph.start_state,
        goal_state=graph.goal_state,
        gold_node_indices=sorted(graph.gold_node_indices()),
        target_start=first[0],
        target_end=first[1],
    )


def prepare_ruby_all(
    examples: list[RubyExample], dic: Dictionary
) -> tuple[list[PreparedExample], int]:
    """ルビ由来の例をまとめて前処理する.

    Returns:
        (前処理できた例, 制約を満たす経路がなくて捨てた件数)
    """
    out: list[PreparedExample] = []
    dropped = 0
    for example in examples:
        prepared = prepare_ruby(example, dic)
        if prepared is None:
            dropped += 1
            continue
        out.append(prepared)
    return out, dropped


def prepare_all(
    examples: list[TrainingExample], dic: Dictionary
) -> tuple[list[PreparedExample], int]:
    """複数件をまとめて前処理する.

    Returns:
        (前処理できた例, 正解経路がなくて捨てた件数)
    """
    out: list[PreparedExample] = []
    dropped = 0
    for example in examples:
        prepared = prepare(example, dic)
        if prepared is None:
            dropped += 1
            continue
        out.append(prepared)
    return out, dropped
