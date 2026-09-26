"""遷移スコアラのテスト（M6 Task 2）.

★MeCab / UniDic は left_id × right_id の連接行列（480 MB）で
「隣接する語の組み合わせやすさ」を持っている。
こちらは同じ ID を低次元に埋め込んで、内積で近似する。

    transitions[p, c] = <right_emb[p.right_id], left_emb[c.left_id]> / sqrt(d)

15,000 ID × 16 次元 × 2 = 480K params ≈ INT8 0.5 MB。行列の 1/1000。
"""

import torch

from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice
from lattice_g2p.scorer.transition import TransitionScorer


def _dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0, left_id=5, right_id=5),
            Entry("米", "ベイ", "名詞", 1, left_id=5, right_id=5),
            Entry("米国", "ベイコク", "名詞", 2, left_id=7, right_id=7),
            Entry("国", "コク", "名詞", 3, left_id=5, right_id=5),
            Entry("の", "ノ", "助詞", 4, left_id=11, right_id=11),
        ]
    )


def test_shape_is_n_by_n() -> None:
    lat = build_lattice("米国の", _dic())
    scorer = TransitionScorer(n_context_ids=32, dim=8)

    t = scorer(lat.nodes)

    assert t.shape == (len(lat.nodes), len(lat.nodes))


def test_starts_near_zero() -> None:
    """★初期値は 0 付近であること。

    学習前から経路選択を動かすと、既存の挙動からの差分が測れなくなる。
    """
    lat = build_lattice("米国の", _dic())
    scorer = TransitionScorer(n_context_ids=32, dim=8)

    assert scorer(lat.nodes).abs().max().item() < 0.1


def test_same_context_ids_give_same_score() -> None:
    """★スコアは ID の組だけで決まること（表記や読みに依存しない）。"""
    lat = build_lattice("米国の", _dic())
    scorer = TransitionScorer(n_context_ids=32, dim=8)
    torch.nn.init.normal_(scorer.left_emb.weight, std=1.0)
    torch.nn.init.normal_(scorer.right_emb.weight, std=1.0)

    t = scorer(lat.nodes)
    # 「米/コメ」と「米/ベイ」は同じ ID なので、同じ相手への遷移は同じ値になる
    a = next(i for i, n in enumerate(lat.nodes) if n.surface == "米" and n.reading == "コメ")
    b = next(i for i, n in enumerate(lat.nodes) if n.surface == "米" and n.reading == "ベイ")
    c = next(i for i, n in enumerate(lat.nodes) if n.surface == "の")

    assert torch.allclose(t[a, c], t[b, c])


def test_out_of_range_ids_are_clamped() -> None:
    """★辞書の ID が語彙より大きくても落ちないこと。

    n_context_ids を小さく設定して配布サイズを削る運用を想定する。
    """
    lat = build_lattice("米国の", _dic())
    scorer = TransitionScorer(n_context_ids=4, dim=8)

    assert torch.isfinite(scorer(lat.nodes)).all()


def test_gradient_flows() -> None:
    lat = build_lattice("米国の", _dic())
    scorer = TransitionScorer(n_context_ids=32, dim=8)

    scorer(lat.nodes).sum().backward()

    assert scorer.left_emb.weight.grad is not None
    assert scorer.right_emb.weight.grad.abs().sum().item() > 0


def test_param_count_is_small() -> None:
    """★連接行列 480 MB の代わりであることが主張なので、サイズを固定する。"""
    scorer = TransitionScorer(n_context_ids=15000, dim=16)

    n = sum(p.numel() for p in scorer.parameters())

    assert n == 2 * 15000 * 16
    assert n * 4 / 1e6 < 2.0, "fp32 で 2 MB 未満"


def test_empty_nodes() -> None:
    scorer = TransitionScorer(n_context_ids=32, dim=8)

    assert scorer([]).shape == (0, 0)
