import pytest
import torch

from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.encoder import ReadingEncoder, ScratchCharEncoder
from lattice_g2p.lattice import Node, build_lattice
from lattice_g2p.scorer import BiEncoderScorer, CrossEncoderScorer, NodeScorer
from lattice_g2p.vocab import CharVocab


@pytest.fixture
def dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0, cost=1000, source="both"),
            Entry("米", "ベイ", "名詞", 0, cost=1100, source="kana"),
            Entry("国", "コク", "名詞", 0, cost=1500, source="both"),
            Entry("米国", "ベイコク", "名詞", 0, cost=900, source="both"),
            Entry("。", "*", "補助記号-句点", 0, cost=-3000, source="both"),
        ]
    )


def _make(kind: str):  # noqa: ANN202
    text_vocab = CharVocab.build(["米国の収穫時期。"])
    kana_vocab = CharVocab.katakana()
    ctx = ScratchCharEncoder(len(text_vocab), hidden=32, layers=2, heads=2, ffn=64, max_len=64)
    rd = ReadingEncoder(len(kana_vocab), hidden=32, layers=1, heads=2, ffn=64, max_len=16)
    if kind == "bi":
        return BiEncoderScorer(ctx, rd, text_vocab, kana_vocab)
    return CrossEncoderScorer(ctx, rd, text_vocab, kana_vocab, layers=1, heads=2, ffn=64)


KINDS = ["bi", "cross"]


@pytest.mark.parametrize("kind", KINDS)
def test_score_returns_one_value_per_node(kind: str, dic: Dictionary) -> None:
    scorer = _make(kind)
    lat = build_lattice("米国", dic)

    scores = scorer.score(lat.text, lat.nodes)

    assert scores.shape == (len(lat.nodes),)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("kind", KINDS)
def test_conforms_to_node_scorer_protocol(kind: str, dic: Dictionary) -> None:
    scorer: NodeScorer = _make(kind)
    lat = build_lattice("米", dic)
    assert scorer.score(lat.text, lat.nodes).shape == (len(lat.nodes),)


@pytest.mark.parametrize("kind", KINDS)
def test_scores_are_differentiable(kind: str, dic: Dictionary) -> None:
    scorer = _make(kind)
    lat = build_lattice("米国", dic)

    scorer.score(lat.text, lat.nodes).sum().backward()

    assert any(p.grad is not None for p in scorer.parameters())


@pytest.mark.parametrize("kind", KINDS)
def test_same_surface_different_reading_get_different_scores(kind: str, dic: Dictionary) -> None:
    """★同じ span でも読みが違えばスコアが変わること。"""
    scorer = _make(kind)
    scorer.eval()
    lat = build_lattice("米", dic)

    with torch.no_grad():
        scores = scorer.score(lat.text, lat.nodes)

    by_reading = {n.reading: float(s) for n, s in zip(lat.nodes, scores, strict=True)}
    assert by_reading["コメ"] != by_reading["ベイ"]


@pytest.mark.parametrize("kind", KINDS)
def test_score_batch_matches_single(kind: str, dic: Dictionary) -> None:
    """★バッチ処理と 1 件ずつの処理が一致すること。

    ここが壊れていても動いてしまい、精度劣化としてしか現れない。
    """
    scorer = _make(kind)
    scorer.eval()
    lat_a = build_lattice("米国。", dic)
    lat_b = build_lattice("米", dic)

    with torch.no_grad():
        batched = scorer.score_batch([lat_a.text, lat_b.text], [lat_a.nodes, lat_b.nodes])
        single_a = scorer.score(lat_a.text, lat_a.nodes)
        single_b = scorer.score(lat_b.text, lat_b.nodes)

    assert torch.allclose(batched[0], single_a, atol=1e-5)
    assert torch.allclose(batched[1], single_b, atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_empty_nodes(kind: str) -> None:
    assert _make(kind).score("", []).shape == (0,)


@pytest.mark.parametrize("kind", KINDS)
def test_handles_empty_reading_node(kind: str, dic: Dictionary) -> None:
    """句読点は空読み。落ちないこと。"""
    scorer = _make(kind)
    lat = build_lattice("。", dic)

    scores = scorer.score(lat.text, lat.nodes)

    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("kind", KINDS)
def test_source_feature_changes_score(kind: str) -> None:
    """★読みの由来（source）がスコアに影響すること。

    pron 由来と kana 由来の読みは同コストで区別できない。
    ベースラインではこれが最大の誤り要因だった。
    """
    from lattice_g2p.lattice import Node

    scorer = _make(kind)
    scorer.eval()
    nodes = [
        Node(0, 1, "米", "コメ", 0, "名詞", cost=1000, source="pron"),
        Node(0, 1, "米", "コメ", 0, "名詞", cost=1000, source="kana"),
    ]

    with torch.no_grad():
        scores = scorer.score("米", nodes)

    assert float(scores[0]) != float(scores[1])


def test_param_counts_are_in_target_range() -> None:
    """実構成でのパラメータ数が 3〜10M に収まること。"""
    text_vocab = CharVocab(sorted({chr(c) for c in range(0x4E00, 0x4E00 + 3000)}))
    kana_vocab = CharVocab.katakana()
    ctx = ScratchCharEncoder(len(text_vocab), hidden=256, layers=4, heads=4, ffn=1024)
    rd = ReadingEncoder(len(kana_vocab), hidden=256, layers=2, heads=4, ffn=1024)

    bi = BiEncoderScorer(ctx, rd, text_vocab, kana_vocab)
    n_bi = sum(p.numel() for p in bi.parameters())
    assert 3_000_000 < n_bi < 10_000_000, f"BiEncoder: {n_bi / 1e6:.2f}M"


# --- 辞書コストの事前分布（M4 Task 0b の根本原因 1） ---


def test_features_carry_dictionary_prior() -> None:
    """★辞書コストが特徴量に入ること。

    UniDic のコストは頻度の代理指標で、「その表記のその読みがどれくらい
    一般的か」という強い事前分布になる。UnigramScorer はこれだけで
    benchmark の対象語 93.36% を出すが、ニューラルスコアラは
    これをまったく与えられていなかった（M4 Task 0b）。
    """
    from lattice_g2p.scorer.neural_base import DICT_COST_SCALE

    model = _make("bi")
    nodes = [
        Node(0, 1, "虎", "トラ", 1, "名詞", cost=2000),
        Node(0, 1, "虎", "ドラ", 2, "名詞", cost=14000),
    ]
    features = model.build_features([nodes], width=1)

    assert features.dict_prior.shape == (2,)
    assert features.dict_prior[0].item() == -2000 / DICT_COST_SCALE
    assert features.dict_prior[1].item() == -14000 / DICT_COST_SCALE
    assert features.dict_prior[0] > features.dict_prior[1], "低コストの方が高い"


def test_dictionary_prior_moves_scores() -> None:
    """★重みを上げると低コストのノードが有利になること。"""
    import torch

    model = _make("bi")
    nodes = [
        Node(0, 1, "虎", "トラ", 1, "名詞", cost=2000),
        Node(0, 1, "虎", "ドラ", 2, "名詞", cost=14000),
    ]
    with torch.no_grad():
        model.dict_prior_weight.fill_(0.0)
        flat = model.score("虎", nodes)
        model.dict_prior_weight.fill_(1.0)
        with_prior = model.score("虎", nodes)

    assert (with_prior[0] - with_prior[1]) > (flat[0] - flat[1]), (
        "事前分布を効かせると トラ と ドラ の差が広がること"
    )


def test_dictionary_prior_is_learnable() -> None:
    """★重みは学習可能にする。どれだけ効かせるかはデータに決めさせる。"""
    model = _make("bi")
    assert model.dict_prior_weight.requires_grad
