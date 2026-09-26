"""DAG 上の CRF.

M0 では積グラフの構築と到達可能性判定のみ.
M1 で forward (分母・分子) と Viterbi を追加する.
M6 で遷移項 (pairwise) を追加した.

★経路スコアは unary と pairwise の和である.

    s(path) = Σ unary(node) + Σ transitions[前のノード, 次のノード]

M5 まで pairwise が無く, 単なるノードスコアの和だった. MeCab / OpenJTalk が
持つ連接行列に相当するものが無いということであり, 「1 語の読みを選ぶ」能力が
弱い原因の候補だった (docs/results.md §3).

★遷移ありの DP は状態ではなく**辺**で回す (_edge_pass).
状態に「その位置で終わったノード」を含める素直な定式化だと積グラフ
(§5 / 不変条件 5) を作り直すことになるが, 辺で回せばそのまま使える.

★transitions=None のときは従来の実装 (_topological_pass) をそのまま使う.
既存の数値が 1 ビットも変わらないことの保証になる.

★正解経路集合 G を全列挙してはいけない (経路数が指数的に増える).

  状態 (i, k) = 「入力の先頭 i 文字を消費し, 正解カナの先頭 k 文字を生成した」

  ノード v = (start=i, end=j, reading=r) は, 正解カナ列 Y が位置 k から r で
  始まるときだけ (i, k) -> (j, k + len(r)) の遷移になる.
  この積グラフ上で forward を回せば, 正解経路集合の分配関数を厳密に計算できる.

  詳細は docs/architecture.md §5.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from lattice_g2p.lattice import Lattice


@dataclass
class ConstrainedGraph:
    """(文字位置, カナ位置) の積グラフ.

    正解読みを生成する経路だけを残した DAG. 前向き到達と後ろ向き枝刈りの両方を
    適用済みなので, すべての辺が start_state から goal_state への経路上にある.

    states は昇順に並んでおり, その順序がトポロジカル順になっている
    (辺は必ず文字位置を進めるため). forward の実装はこの前提に依存する.
    """

    states: list[tuple[int, int]]
    state_index: dict[tuple[int, int], int]
    edges: list[tuple[int, int, int]]
    """(node_idx, from_state, to_state)."""

    start_state: int
    goal_state: int | None
    """正解読みを生成できない場合は None."""

    def is_reachable(self) -> bool:
        """正解読みを生成する経路が存在するか. Oracle 判定そのもの."""
        return self.goal_state is not None

    def gold_node_indices(self) -> set[int]:
        """正解経路上に現れるラティスノードの index.

        margin loss で「正解読みのノード」を特定するのに使う.
        """
        return {node_idx for node_idx, _, _ in self.edges}


def build_constrained_graph(lattice: Lattice, gold_reading: str) -> ConstrainedGraph:
    """ラティスと正解カナ列から積グラフを構築する.

    正解読みを生成できない場合は ``goal_state=None`` のグラフを返す
    (例外にはしない. Oracle 評価とデータのフィルタリングで日常的に起きるため).
    """
    n_chars = len(lattice.text)
    n_kana = len(gold_reading)
    start = (0, 0)
    goal = (n_chars, n_kana)

    # --- 前向き到達 ---
    out: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    reached = {start}
    stack = [start]
    while stack:
        i, k = stack.pop()
        for node_idx in lattice.nodes_starting_at[i]:
            node = lattice.nodes[node_idx]
            reading = node.reading
            if not gold_reading.startswith(reading, k):
                continue
            nxt = (node.end, k + len(reading))
            out[(i, k)].append((node_idx, nxt))
            if nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)

    return _prune_and_index(out, reached, start, goal)


_Arcs = dict[tuple[int, int], list[tuple[int, tuple[int, int]]]]


def _prune_and_index(
    out: _Arcs,
    reached: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> ConstrainedGraph:
    """前向き到達の結果を後ろ向きに枝刈りし, 状態に連番を振る.

    ★状態は昇順に並べる. 辺は必ず文字位置 (タプルの第 1 要素) を進めるので,
    昇順がそのままトポロジカル順になる. forward の実装がこれに依存する.
    """
    if goal not in reached:
        return ConstrainedGraph(
            states=[start],
            state_index={start: 0},
            edges=[],
            start_state=0,
            goal_state=None,
        )

    # --- 後ろ向き枝刈り: goal に到達できる状態だけ残す ---
    incoming: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for src, arcs in out.items():
        for node_idx, dst in arcs:
            incoming[dst].append((node_idx, src))

    alive = {goal}
    stack = [goal]
    while stack:
        state = stack.pop()
        for _, src in incoming[state]:
            if src not in alive:
                alive.add(src)
                stack.append(src)

    # --- 状態に連番を振り, 生き残った辺を集める ---
    states = sorted(alive)
    state_index = {s: i for i, s in enumerate(states)}
    edges: list[tuple[int, int, int]] = []
    for src, arcs in out.items():
        if src not in alive:
            continue
        for node_idx, dst in arcs:
            if dst not in alive:
                continue
            edges.append((node_idx, state_index[src], state_index[dst]))

    return ConstrainedGraph(
        states=states,
        state_index=state_index,
        edges=edges,
        start_state=state_index[start],
        goal_state=state_index[goal],
    )


def _validate_spans(
    spans: Sequence[tuple[int, int, str]], n_chars: int
) -> list[tuple[int, int, str]]:
    """span を検証し, 開始位置の昇順に並べて返す."""
    for start, end, _ in spans:
        if not (0 <= start < end <= n_chars):
            raise ValueError(
                f"span ({start}, {end}) が文字列の範囲 [0, {n_chars}] に収まっていません。"
            )

    ordered = sorted(spans)
    for (s1, e1, _), (s2, _, _) in zip(ordered, ordered[1:], strict=False):
        if e1 > s2:
            raise ValueError(
                f"span ({s1}, {e1}) と ({s2}, ...) が重なっています。"
                "ルビの範囲は互いに素でなければなりません。"
            )
    return ordered


def build_partially_constrained_graph(
    lattice: Lattice, spans: Sequence[tuple[int, int, str]]
) -> ConstrainedGraph:
    """文の一部だけ読みが分かっている場合の積グラフを構築する.

    ★青空文庫のルビは文の一部にしか付かない.

        彼は|生真面目《きまじめ》な男だ
          -> text  : 彼は生真面目な男だ
          -> spans : [(2, 5, "キマジメ")]     それ以外の読みは未知

    ``build_constrained_graph(lattice, gold)`` は
    ``build_partially_constrained_graph(lattice, [(0, L, gold)])`` と等価である.

    状態は (文字位置 i, その位置を含む span 内で消費したカナ数 k).
    span の外では k = 0 に固定されるので, 制約のない区間ではラティスの
    位置そのものと一対一に対応する. 分母 (全経路) との違いが span の中だけに
    閉じ込められる.

    ★span の境界をまたぐノードは採用しない.
    「米国 -> ベイコク」の 1 ノードで「米」だけにルビが付いている場合,
    カナのどこまでが「米」の読みなのか決められない. 推測せずに落とす.
    その結果 span を分離できる経路が 1 本もなければ到達不能になり,
    その例は学習に使えない (Oracle 失敗と同じ扱い).

    Args:
        spans: (開始文字位置, 終了文字位置, 読み) の列. 互いに素であること.
            空なら制約なし (分子 = 分母) となり, 損失は 0 になる.

    Returns:
        ConstrainedGraph. 制約を満たす経路がなければ ``goal_state=None``.
    """
    n_chars = len(lattice.text)
    ordered = _validate_spans(spans, n_chars)

    # 位置 i を含む span (s <= i < e) の index. 含まれなければ None
    span_at: list[int | None] = [None] * (n_chars + 1)
    for idx, (start, end, _) in enumerate(ordered):
        for i in range(start, end):
            span_at[i] = idx

    # 位置 i 以降で最初に span が始まる位置. なければ n_chars (制約なしと同値)
    next_span_start = [n_chars] * (n_chars + 1)
    nxt = n_chars
    for i in range(n_chars, -1, -1):
        if span_at[i] is not None and (i == 0 or span_at[i - 1] != span_at[i]):
            nxt = i
        next_span_start[i] = nxt

    start_state = (0, 0)
    goal_state = (n_chars, 0)

    out: _Arcs = defaultdict(list)
    reached = {start_state}
    stack = [start_state]
    while stack:
        i, k = stack.pop()
        here = span_at[i] if i < n_chars else None
        for node_idx in lattice.nodes_starting_at[i]:
            node = lattice.nodes[node_idx]
            nxt_state = _partial_transition(node, k, here, ordered, next_span_start[i])
            if nxt_state is None:
                continue
            out[(i, k)].append((node_idx, nxt_state))
            if nxt_state not in reached:
                reached.add(nxt_state)
                stack.append(nxt_state)

    return _prune_and_index(out, reached, start_state, goal_state)


def _partial_transition(
    node,  # noqa: ANN001
    k: int,
    span_idx: int | None,
    spans: list[tuple[int, int, str]],
    next_span_start: int,
) -> tuple[int, int] | None:
    """ノード 1 本ぶんの遷移先を返す. 制約を満たさなければ None."""
    if span_idx is None:
        # 制約区間の外. 次の span の手前で止まるノードだけ使える
        if node.end > next_span_start:
            return None
        return (node.end, 0)

    _, end, reading = spans[span_idx]
    if node.end > end:
        return None  # span の終端をまたぐ
    if not reading.startswith(node.reading, k):
        return None
    k2 = k + len(node.reading)
    if node.end == end:
        return (end, 0) if k2 == len(reading) else None
    return (node.end, k2)


# --- forward / Viterbi ---

NEG_INF = -1e9
"""到達不能な状態のスコア.

-inf を使うと -inf - (-inf) = nan になるため, 有限の大きな負値を使う.
"""


def _topological_pass(
    n_states: int,
    edges: list[tuple[int, int, int]],
    scores: torch.Tensor,
    start: int,
    goal: int,
    use_max: bool = False,
) -> tuple[torch.Tensor, list[tuple[int, int] | None]]:
    """トポロジカル順に alpha を計算する (遷移項なし).

    状態 index の昇順がトポロジカル順であることを前提とする.
    ラティスは文字位置, 積グラフは (文字位置, カナ位置) のソート順であり,
    どちらも辺が必ず文字位置を進めるためこの前提が成り立つ.

    Args:
        use_max: True なら logsumexp の代わりに max を使う (Viterbi)

    Returns:
        (goal の alpha, 各状態のバックポインタ).
        バックポインタは use_max のときのみ意味を持つ (node_idx, from_state).
    """
    incoming: list[list[tuple[int, int]]] = [[] for _ in range(n_states)]
    for node_idx, src, dst in edges:
        incoming[dst].append((node_idx, src))

    alpha: list[torch.Tensor | None] = [None] * n_states
    back: list[tuple[int, int] | None] = [None] * n_states
    alpha[start] = torch.zeros((), dtype=scores.dtype, device=scores.device)

    for state in range(n_states):
        if state == start:
            continue
        terms: list[torch.Tensor] = []
        sources: list[tuple[int, int]] = []
        for node_idx, src in incoming[state]:
            prev = alpha[src]
            if prev is None:
                continue
            terms.append(prev + scores[node_idx])
            sources.append((node_idx, src))
        if not terms:
            continue
        stacked = torch.stack(terms)
        if use_max:
            best = int(torch.argmax(stacked))
            alpha[state] = stacked[best]
            back[state] = sources[best]
        else:
            alpha[state] = torch.logsumexp(stacked, dim=0)

    result = alpha[goal]
    if result is None:
        result = torch.full((), NEG_INF, dtype=scores.dtype, device=scores.device)
    return result, back


def _edge_pass(
    edges: list[tuple[int, int, int]],
    scores: torch.Tensor,
    transitions: torch.Tensor,
    start: int,
    goal: int,
    use_max: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """★遷移項つきの forward / Viterbi. 状態ではなく**辺**で DP を回す.

        beta[e] = unary(e.node)
                + logsumexp over e' (e'.dst == e.src) of
                      (beta[e'] + transitions[e'.node, e.node])

        (e.src == start なら第 2 項は 0)
        答え = logsumexp over e (e.dst == goal) of beta[e]

    ★状態を増やさずに済むのがこの形の利点である.
    「その位置で終わったノード」を状態に含める素直な定式化だと,
    積グラフ (docs/architecture.md §5 / 不変条件 5) を作り直すことになる.
    辺で回せば ConstrainedGraph をそのまま使える.

    ★トポロジカル順は辺の src の昇順でよい.
    辺は必ず文字位置を進めるので, e'.dst == e.src なら e'.src < e.src.

    計算量は O(Σ 入次数 × 出次数). 遷移なしの O(E) より増えるが,
    91 文字 391 ノードで約 4 倍に収まる.

    Returns:
        (答え, 各辺のバックポインタ = 直前の辺の index. 無ければ -1)
    """
    order = sorted(range(len(edges)), key=lambda e: edges[e][1])
    incoming_edges: dict[int, list[int]] = defaultdict(list)
    for e, (_, _, dst) in enumerate(edges):
        incoming_edges[dst].append(e)

    beta: list[torch.Tensor | None] = [None] * len(edges)
    back: list[int] = [-1] * len(edges)

    for e in order:
        node_idx, src, _ = edges[e]
        if src == start:
            beta[e] = scores[node_idx]
            continue

        terms: list[torch.Tensor] = []
        sources: list[int] = []
        for prev_e in incoming_edges[src]:
            prev = beta[prev_e]
            if prev is None:
                continue
            terms.append(prev + transitions[edges[prev_e][0], node_idx])
            sources.append(prev_e)
        if not terms:
            continue

        stacked = torch.stack(terms)
        if use_max:
            best = int(torch.argmax(stacked))
            beta[e] = stacked[best] + scores[node_idx]
            back[e] = sources[best]
        else:
            beta[e] = torch.logsumexp(stacked, dim=0) + scores[node_idx]

    finals = [beta[e] for e in incoming_edges[goal] if beta[e] is not None]
    if not finals:
        # start == goal なら空経路 (スコア 0) が 1 本だけある
        value = torch.zeros((), dtype=scores.dtype, device=scores.device)
        if start != goal:
            value = torch.full((), NEG_INF, dtype=scores.dtype, device=scores.device)
        return value, back

    stacked = torch.stack(finals)
    if use_max:
        best = int(torch.argmax(stacked))
        last = [e for e in incoming_edges[goal] if beta[e] is not None][best]
        back = back + [last]  # 末尾に「最後の辺」を積む (_trace_back_edges が使う)
        return stacked[best], back
    return torch.logsumexp(stacked, dim=0), back


def _run(
    n_states: int,
    edges: list[tuple[int, int, int]],
    scores: torch.Tensor,
    start: int,
    goal: int,
    transitions: torch.Tensor | None,
    use_max: bool = False,
):  # noqa: ANN202
    """遷移項の有無で実装を振り分ける.

    ★transitions=None のときは従来の実装をそのまま使う.
    速いだけでなく, **既存の数値が 1 ビットも変わらないことの保証**になる.
    """
    if transitions is None:
        return _topological_pass(n_states, edges, scores, start, goal, use_max)
    return _edge_pass(edges, scores, transitions, start, goal, use_max)


def _lattice_edges(lattice: Lattice) -> list[tuple[int, int, int]]:
    return [(i, node.start, node.end) for i, node in enumerate(lattice.nodes)]


def lattice_logsumexp(
    lattice: Lattice, scores: torch.Tensor, transitions: torch.Tensor | None = None
) -> torch.Tensor:
    """CRF の分母. 全経路のスコアの logsumexp.

    Args:
        transitions: (N, N) の遷移スコア. transitions[p, c] は
            ノード p の次にノード c が来るときの加点. None なら遷移項なし.
    """
    value, _ = _run(
        n_states=len(lattice.text) + 1,
        edges=_lattice_edges(lattice),
        scores=scores,
        start=0,
        goal=len(lattice.text),
        transitions=transitions,
    )
    return value


def constrained_logsumexp(
    graph: ConstrainedGraph, scores: torch.Tensor, transitions: torch.Tensor | None = None
) -> torch.Tensor:
    """CRF の分子. 正解読みを生成する経路のスコアの logsumexp."""
    if graph.goal_state is None:
        raise ValueError(
            "正解経路が存在しないラティスです。"
            "この例は学習に使えません（Oracle 失敗）。フィルタで除外してください。"
        )
    value, _ = _run(
        n_states=len(graph.states),
        edges=graph.edges,
        scores=scores,
        start=graph.start_state,
        goal=graph.goal_state,
        transitions=transitions,
    )
    return value


def crf_loss(
    lattice: Lattice,
    graph: ConstrainedGraph,
    scores: torch.Tensor,
    transitions: torch.Tensor | None = None,
) -> torch.Tensor:
    """CRF 損失.

        L = log sum_{pi in P} exp(s(pi)) - log sum_{pi in G} exp(s(pi))

    G ⊆ P なので常に 0 以上.
    """
    return lattice_logsumexp(lattice, scores, transitions) - constrained_logsumexp(
        graph, scores, transitions
    )


def viterbi(
    lattice: Lattice, scores: torch.Tensor, transitions: torch.Tensor | None = None
) -> list[int]:
    """最尤経路のノード index 列を返す.

    forward の logsumexp を max に置き換えたもの.
    """
    if not lattice.text:
        return []

    goal = len(lattice.text)
    edges = _lattice_edges(lattice)
    if transitions is None:
        _, back = _topological_pass(
            n_states=goal + 1, edges=edges, scores=scores, start=0, goal=goal, use_max=True
        )
        return _trace_back(back, goal)

    _, back = _edge_pass(edges, scores, transitions, start=0, goal=goal, use_max=True)
    return _trace_back_edges(back, edges)


def viterbi_numpy(lattice: Lattice, scores, transitions=None) -> list[int]:  # noqa: ANN001
    """numpy 配列のスコアで Viterbi を行う. 推論専用.

    ★推論経路から torch を外すためにある. _topological_pass は辺ごとに
    0 次元テンソルを作るので, 推論には重すぎる.

    tests/test_runtime.py と tests/test_crf_transitions.py が
    viterbi() との一致を確認している.
    """
    if not lattice.text:
        return []

    goal = len(lattice.text)
    edges = _lattice_edges(lattice)
    if transitions is not None:
        return _viterbi_numpy_edges(edges, scores, transitions, goal)

    incoming: list[list[tuple[int, int]]] = [[] for _ in range(goal + 1)]
    for node_idx, src, dst in edges:
        incoming[dst].append((node_idx, src))

    alpha: list[float | None] = [None] * (goal + 1)
    back: list[tuple[int, int] | None] = [None] * (goal + 1)
    alpha[0] = 0.0

    for state in range(1, goal + 1):
        best: float | None = None
        for node_idx, src in incoming[state]:
            prev = alpha[src]
            if prev is None:
                continue
            value = prev + float(scores[node_idx])
            if best is None or value > best:
                best = value
                back[state] = (node_idx, src)
        alpha[state] = best

    return _trace_back(back, goal)


def _viterbi_numpy_edges(edges, scores, transitions, goal) -> list[int]:  # noqa: ANN001
    """遷移項つきの Viterbi (numpy). _edge_pass の torch 非依存版."""
    incoming_edges: dict[int, list[int]] = defaultdict(list)
    for e, (_, _, dst) in enumerate(edges):
        incoming_edges[dst].append(e)

    order = sorted(range(len(edges)), key=lambda e: edges[e][1])
    beta: list[float | None] = [None] * len(edges)
    back: list[int] = [-1] * len(edges)

    for e in order:
        node_idx, src, _ = edges[e]
        if src == 0:
            beta[e] = float(scores[node_idx])
            continue
        best: float | None = None
        for prev_e in incoming_edges[src]:
            prev = beta[prev_e]
            if prev is None:
                continue
            value = prev + float(transitions[edges[prev_e][0], node_idx])
            if best is None or value > best:
                best = value
                back[e] = prev_e
        if best is not None:
            beta[e] = best + float(scores[node_idx])

    finals = [e for e in incoming_edges[goal] if beta[e] is not None]
    if not finals:
        raise ValueError("goal に到達する経路がありません。ラティスが不正です。")

    last = max(finals, key=lambda e: beta[e])
    path: list[int] = []
    while last != -1:
        path.append(edges[last][0])
        last = back[last]
    path.reverse()
    return path


def _trace_back_edges(back: list[int], edges: list[tuple[int, int, int]]) -> list[int]:
    """辺の DP のバックポインタから経路のノード index 列を復元する.

    ★back の末尾に _edge_pass が「最後の辺」を積んでいる.
    """
    if len(back) <= len(edges):
        raise ValueError("goal に到達する経路がありません。ラティスが不正です。")

    path: list[int] = []
    e = back[-1]
    while e != -1:
        path.append(edges[e][0])
        e = back[e]
    path.reverse()
    return path


def _trace_back(back: list[tuple[int, int] | None], goal: int) -> list[int]:
    path: list[int] = []
    state = goal
    while state != 0:
        step = back[state]
        if step is None:
            raise ValueError(f"位置 {state} に到達する経路がありません。ラティスが不正です。")
        node_idx, src = step
        path.append(node_idx)
        state = src
    path.reverse()
    return path


def crf_loss_from_prepared(
    prepared,  # noqa: ANN001
    scores: torch.Tensor,
    transitions: torch.Tensor | None = None,
) -> torch.Tensor:
    """前処理済みの遷移構造から CRF 損失を計算する.

    学習ループはこちらを使う. ラティスの再構築が発生しない.
    型注釈を避けているのは data.prepare との循環 import を防ぐため.
    """
    denom, _ = _run(
        n_states=prepared.num_positions,
        edges=prepared.denom_edges,
        scores=scores,
        start=0,
        goal=prepared.num_positions - 1,
        transitions=transitions,
    )
    numer, _ = _run(
        n_states=prepared.num_states,
        edges=prepared.numer_edges,
        scores=scores,
        start=prepared.start_state,
        goal=prepared.goal_state,
        transitions=transitions,
    )
    return denom - numer


def target_readings_on_gold_paths(
    graph: ConstrainedGraph, gold_reading: str, target_start: int, target_end: int
) -> set[str]:
    """正解経路上で対象語の span が取りうる読みをすべて返す.

    ★「任意の 1 本の正解経路」から読みを取り出してはいけない.
    経路によって分割が違うため, たまたま span をまたぐノードを選んだ経路を
    拾うと対象外の読みまで含んでしまう ("固める -> カタメル" で target が「固」だけ).

    積グラフの状態は (文字位置, カナ位置) なので, 文字位置が
    target_start と target_end にちょうど一致する状態の組から
    カナ位置の区間が求まる. その区間の正解カナ列が対象語の読みである.

    Returns:
        取りうる読みの集合. span の境界を持つ経路が 1 本もなければ空集合.
    """
    if graph.goal_state is None:
        return set()

    successors: dict[int, list[int]] = {}
    for _, src, dst in graph.edges:
        successors.setdefault(src, []).append(dst)

    starts = [i for i, (pos, _) in enumerate(graph.states) if pos == target_start]
    end_states = {i for i, (pos, _) in enumerate(graph.states) if pos == target_end}

    readings: set[str] = set()
    for start_state in starts:
        # target_end を超えない範囲で前向きに探索する
        seen = {start_state}
        stack = [start_state]
        while stack:
            state = stack.pop()
            if state in end_states:
                k1 = graph.states[start_state][1]
                k2 = graph.states[state][1]
                readings.add(gold_reading[k1:k2])
                continue
            if graph.states[state][0] > target_end:
                continue
            for nxt in successors.get(state, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

    return readings


def covering_kana_windows(
    graph: ConstrainedGraph,
    lattice: Lattice,
    target_start: int,
    target_end: int,
) -> set[tuple[int, int]]:
    """対象語の span を覆うノードが占めるカナ範囲を返す.

    span の境界を持つ経路が存在しない場合 (「固める -> カタメル」の 1 ノードで
    target が「固」だけ, など), 対象語の読みは分離できない.
    そのときでも「この範囲のどこかに対象語の読みがあるはず」という
    緩い制約は課せる.

    Returns:
        (カナ開始位置, カナ終了位置) の集合
    """
    windows: set[tuple[int, int]] = set()
    for node_idx, src, dst in graph.edges:
        node = lattice.nodes[node_idx]
        if node.start < target_end and node.end > target_start:
            windows.add((graph.states[src][1], graph.states[dst][1]))
    return windows
