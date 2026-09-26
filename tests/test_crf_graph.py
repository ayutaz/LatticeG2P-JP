import pytest

from lattice_g2p.crf import build_constrained_graph
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice


@pytest.fixture
def dic() -> Dictionary:
    entries = [
        Entry("米", "コメ", "名詞", 0),
        Entry("米", "ベイ", "名詞", 1),
        Entry("米国", "ベイコク", "名詞", 2),
        Entry("国", "コク", "名詞", 3),
        Entry("国", "クニ", "名詞", 4),
        Entry("の", "ノ", "助詞", 5),
        Entry("。", "*", "補助記号-句点", 6),
    ]
    return Dictionary.from_entries(entries)


def test_reachable_when_reading_is_producible(dic: Dictionary) -> None:
    lat = build_lattice("米国", dic)
    assert build_constrained_graph(lat, "ベイコク").is_reachable()


def test_multiple_paths_produce_same_reading(dic: Dictionary) -> None:
    """★「米国」→ベイコク は「米国」1 語でも「米+国」2 語でも作れる。両方残ること。

    これが「形態素分割を正解データにしない」という手法の核心。
    """
    lat = build_lattice("米国", dic)
    g = build_constrained_graph(lat, "ベイコク")

    used = {(lat.nodes[ni].surface, lat.nodes[ni].reading) for ni in g.gold_node_indices()}
    assert ("米国", "ベイコク") in used
    assert ("米", "ベイ") in used
    assert ("国", "コク") in used


def test_unreachable_when_reading_not_producible(dic: Dictionary) -> None:
    lat = build_lattice("米国", dic)
    g = build_constrained_graph(lat, "イリガビ")

    assert not g.is_reachable()
    assert g.goal_state is None


def test_wrong_reading_nodes_are_pruned(dic: Dictionary) -> None:
    """正解経路に乗らないノードは gold_node_indices に含まれないこと。"""
    lat = build_lattice("米国", dic)
    g = build_constrained_graph(lat, "コメクニ")

    used = {(lat.nodes[ni].surface, lat.nodes[ni].reading) for ni in g.gold_node_indices()}
    assert used == {("米", "コメ"), ("国", "クニ")}


def test_partial_match_does_not_reach_goal(dic: Dictionary) -> None:
    """途中まで一致しても最後まで到達しなければ不可達であること。"""
    lat = build_lattice("米国", dic)
    assert not build_constrained_graph(lat, "ベイ").is_reachable()


def test_longer_reading_than_producible(dic: Dictionary) -> None:
    lat = build_lattice("米国", dic)
    assert not build_constrained_graph(lat, "ベイコクノ").is_reachable()


def test_empty_reading_on_empty_text(dic: Dictionary) -> None:
    lat = build_lattice("", dic)
    g = build_constrained_graph(lat, "")

    assert g.is_reachable()
    assert g.edges == []


def test_punctuation_consumes_no_kana(dic: Dictionary) -> None:
    """★空読みのノードは文字位置だけ進めてカナ位置を進めないこと。"""
    lat = build_lattice("米。", dic)
    g = build_constrained_graph(lat, "コメ")

    assert g.is_reachable()
    used = {lat.nodes[ni].surface for ni in g.gold_node_indices()}
    assert used == {"米", "。"}


def test_every_edge_lies_on_a_gold_path(dic: Dictionary) -> None:
    """★枝刈り後、すべての辺が start から goal への経路上にあること。"""
    lat = build_lattice("米国の", dic)
    g = build_constrained_graph(lat, "ベイコクノ")
    assert g.is_reachable()

    forward = {g.start_state}
    changed = True
    while changed:
        changed = False
        for _, src, dst in g.edges:
            if src in forward and dst not in forward:
                forward.add(dst)
                changed = True

    backward = {g.goal_state}
    changed = True
    while changed:
        changed = False
        for _, src, dst in g.edges:
            if dst in backward and src not in backward:
                backward.add(src)
                changed = True

    for _, src, dst in g.edges:
        assert src in forward, "start から到達できない辺が残っている"
        assert dst in backward, "goal に到達できない辺が残っている"


def test_states_are_topologically_ordered(dic: Dictionary) -> None:
    """★状態 index の昇順がトポロジカル順であること。

    forward の実装がこの前提に依存している。
    """
    lat = build_lattice("米国の。", dic)
    g = build_constrained_graph(lat, "ベイコクノ")
    assert g.is_reachable()

    for _, src, dst in g.edges:
        assert src < dst, f"辺 {src} -> {dst} が昇順でない"


def test_state_index_is_consistent(dic: Dictionary) -> None:
    lat = build_lattice("米国", dic)
    g = build_constrained_graph(lat, "ベイコク")

    for state, idx in g.state_index.items():
        assert g.states[idx] == state
    assert g.states[g.start_state] == (0, 0)
    assert g.states[g.goal_state] == (len(lat.text), len("ベイコク"))


def test_gold_node_indices_empty_when_unreachable(dic: Dictionary) -> None:
    lat = build_lattice("米国", dic)
    assert build_constrained_graph(lat, "イリガビ").gold_node_indices() == set()


def test_reading_prefix_collision(dic: Dictionary) -> None:
    """読みの前方一致が曖昧でも正しく処理すること。

    「米」を コメ と読む経路と ベイ と読む経路のうち、
    正解カナ列と一致する方だけが残る。
    """
    lat = build_lattice("米国", dic)
    g = build_constrained_graph(lat, "コメコク")

    used = {(lat.nodes[ni].surface, lat.nodes[ni].reading) for ni in g.gold_node_indices()}
    assert used == {("米", "コメ"), ("国", "コク")}


# --- 正解経路上で対象語が取りうる読み ---


def test_target_readings_single_path(dic: Dictionary) -> None:
    from lattice_g2p.crf import target_readings_on_gold_paths

    lat = build_lattice("米国", dic)
    g = build_constrained_graph(lat, "コメクニ")

    assert target_readings_on_gold_paths(g, "コメクニ", 0, 1) == {"コメ"}


def test_target_readings_multiple_segmentations(dic: Dictionary) -> None:
    """★「米国→ベイコク」は 1 語でも 2 語でも作れる。

    2 語の経路なら「米」の読みは「ベイ」として取り出せる。
    1 語の経路では境界がないので取り出せない。両方あるうち、
    取り出せる方の読みを返すこと。
    """
    from lattice_g2p.crf import target_readings_on_gold_paths

    lat = build_lattice("米国", dic)
    g = build_constrained_graph(lat, "ベイコク")

    assert target_readings_on_gold_paths(g, "ベイコク", 0, 1) == {"ベイ"}


def test_target_readings_empty_when_no_boundary(dic: Dictionary) -> None:
    """境界を持つ経路が 1 本もなければ空集合を返すこと。"""
    from lattice_g2p.crf import target_readings_on_gold_paths

    d = Dictionary.from_entries([Entry("米国", "ベイコク", "名詞", 0)])
    lat = build_lattice("米国", d)
    g = build_constrained_graph(lat, "ベイコク")

    assert target_readings_on_gold_paths(g, "ベイコク", 0, 1) == set()


def test_target_readings_with_punctuation(dic: Dictionary) -> None:
    """空読みノードがあっても境界を正しく扱うこと。"""
    from lattice_g2p.crf import target_readings_on_gold_paths

    lat = build_lattice("米。国", dic)
    g = build_constrained_graph(lat, "コメクニ")

    assert target_readings_on_gold_paths(g, "コメクニ", 0, 1) == {"コメ"}


def test_target_readings_unreachable(dic: Dictionary) -> None:
    from lattice_g2p.crf import target_readings_on_gold_paths

    lat = build_lattice("米国", dic)
    g = build_constrained_graph(lat, "イリガビ")

    assert target_readings_on_gold_paths(g, "イリガビ", 0, 1) == set()
