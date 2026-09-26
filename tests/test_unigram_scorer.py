import torch

from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice
from lattice_g2p.scorer import UnigramScorer


def _dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0, cost=5000),
            Entry("米", "ベイ", "名詞", 1, cost=9000),
            Entry("米国", "ベイコク", "名詞", 2, cost=4000),
            Entry("国", "クニ", "名詞", 3, cost=6000),
        ]
    )


def test_score_returns_one_value_per_node() -> None:
    lat = build_lattice("米国", _dic())
    scores = UnigramScorer().score(lat.text, lat.nodes)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (len(lat.nodes),)
    assert scores.dtype == torch.float32


def test_lower_cost_gets_higher_score() -> None:
    """UniDic のコストは小さいほど出やすい。スコアは逆符号になること。"""
    lat = build_lattice("米", _dic())
    scores = UnigramScorer(cost_scale=1000.0).score(lat.text, lat.nodes)

    by_reading = {n.reading: float(s) for n, s in zip(lat.nodes, scores, strict=True)}
    assert by_reading["コメ"] > by_reading["ベイ"]


def test_unknown_node_is_penalized() -> None:
    """辞書にない文字のフォールバックノードにペナルティが乗ること。"""
    lat = build_lattice("Ω", _dic())
    scores = UnigramScorer(unk_penalty=5.0, cost_scale=1e9).score(lat.text, lat.nodes)

    assert all(float(s) <= -5.0 for s in scores)


def test_length_bonus_favors_longer_nodes() -> None:
    lat = build_lattice("米国", _dic())
    scores = UnigramScorer(cost_scale=1e9, length_bonus=1.0).score(lat.text, lat.nodes)

    longest = max(range(len(lat.nodes)), key=lambda i: lat.nodes[i].end - lat.nodes[i].start)
    assert float(scores[longest]) == max(float(s) for s in scores)


def test_empty_nodes() -> None:
    assert UnigramScorer().score("", []).shape == (0,)


def test_scorer_ignores_text() -> None:
    """UnigramScorer は文脈を使わない（ニューラルなしのベースライン）。"""
    lat = build_lattice("米", _dic())
    a = UnigramScorer().score("まったく別の文脈", lat.nodes)
    b = UnigramScorer().score(lat.text, lat.nodes)

    assert torch.equal(a, b)


def test_conforms_to_node_scorer_protocol() -> None:
    from lattice_g2p.scorer import NodeScorer

    scorer: NodeScorer = UnigramScorer()
    lat = build_lattice("米", _dic())
    assert scorer.score(lat.text, lat.nodes).shape == (len(lat.nodes),)
