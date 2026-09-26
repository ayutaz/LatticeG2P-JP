import torch

from lattice_g2p.crf import build_constrained_graph, crf_loss, crf_loss_from_prepared
from lattice_g2p.data.prepare import prepare, prepare_all
from lattice_g2p.data.schema import TrainingExample
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice


def _dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0),
            Entry("米", "ベイ", "名詞", 0),
            Entry("を", "ヲ", "助詞", 0),
            Entry("炊く", "タク", "動詞", 0),
            Entry("。", "*", "補助記号-句点", 0),
        ]
    )


def _example(**kwargs) -> TrainingExample:  # noqa: ANN003
    base = {
        "text": "米を炊く",
        "reading": "コメヲタク",
        "target_start": 0,
        "target_end": 1,
        "target_surface": "米",
        "target_reading": "コメ",
    }
    base.update(kwargs)
    return TrainingExample(**base)


def test_prepare_returns_example() -> None:
    p = prepare(_example(), _dic())

    assert p is not None
    assert p.text == "米を炊く"
    assert p.num_positions == 5
    assert len(p.denom_edges) == len(p.nodes)
    assert p.target_start == 0
    assert p.target_end == 1


def test_prepare_returns_none_when_unreachable() -> None:
    """★正解経路がない例は None を返して呼び出し側で捨てられること。"""
    assert prepare(_example(reading="イリガビ"), _dic()) is None


def test_gold_node_indices_point_to_correct_reading() -> None:
    p = prepare(_example(), _dic())

    assert p is not None
    readings = {p.nodes[i].reading for i in p.gold_node_indices}
    assert "コメ" in readings
    assert "ベイ" not in readings


def test_loss_from_prepared_matches_direct_computation() -> None:
    """★前計算した構造での損失が、ラティスから直接計算した損失と一致すること。"""
    ex = _example()
    dic = _dic()
    p = prepare(ex, dic)
    assert p is not None

    lat = build_lattice(ex.text, dic)
    graph = build_constrained_graph(lat, ex.reading)

    torch.manual_seed(0)
    scores = torch.randn(len(lat.nodes), dtype=torch.float64)

    assert torch.allclose(
        crf_loss(lat, graph, scores), crf_loss_from_prepared(p, scores), atol=1e-9
    )


def test_loss_from_prepared_is_differentiable() -> None:
    p = prepare(_example(), _dic())
    assert p is not None

    scores = torch.zeros(len(p.nodes), dtype=torch.float64, requires_grad=True)
    crf_loss_from_prepared(p, scores).backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_prepare_with_punctuation() -> None:
    """空読みノードがあっても前処理できること。"""
    p = prepare(_example(text="米を炊く。", reading="コメヲタク"), _dic())

    assert p is not None
    assert p.num_positions == 6


def test_prepare_all_drops_unreachable() -> None:
    rows = [_example(), _example(reading="イリガビ"), _example(text="米", reading="コメ")]
    prepared, dropped = prepare_all(rows, _dic())

    assert len(prepared) == 2
    assert dropped == 1


def test_lattice_view_reconstructs_indices() -> None:
    """PreparedExample から Viterbi 用の Lattice を復元できること。"""
    from lattice_g2p.crf import viterbi

    p = prepare(_example(), _dic())
    assert p is not None

    lat = p.lattice()
    path = viterbi(lat, torch.zeros(len(p.nodes)))

    assert lat.nodes[path[0]].start == 0
    assert lat.nodes[path[-1]].end == len(p.text)


def test_gold_span_reading() -> None:
    """正解経路から target span の読みを取り出せること。"""
    p = prepare(_example(), _dic())
    assert p is not None
    assert p.gold_span_reading() == "コメ"


# --- ルビ（部分制約）の前処理 ---


def _ruby_dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("生", "ナマ", "名詞", 0),
            Entry("生", "セイ", "名詞", 1),
            Entry("ビール", "ビール", "名詞", 2),
            Entry("を", "ヲ", "助詞", 3),
            Entry("飲む", "ノム", "動詞", 4),
        ]
    )


def test_prepare_ruby_builds_partially_constrained_numerator() -> None:
    """★ルビ例は文全体の読みを持たない。span だけで分子を作れること。"""
    from lattice_g2p.data.prepare import prepare_ruby
    from lattice_g2p.data.ruby import RubyExample

    example = RubyExample(text="生ビールを飲む", spans=[(0, 1, "ナマ")], source="t")
    prepared = prepare_ruby(example, _ruby_dic())

    assert prepared is not None
    assert prepared.gold_reading is None, "文全体の読みは分からない"
    assert prepared.target_start == 0
    assert prepared.target_end == 1

    used = {prepared.nodes[i].reading for i in prepared.gold_node_indices}
    assert "ナマ" in used
    assert "セイ" not in used


def test_prepare_ruby_loss_is_finite_and_positive() -> None:
    """★CRF 損失が計算できること。これが「学習に使える」の定義。"""
    import torch

    from lattice_g2p.crf import crf_loss_from_prepared
    from lattice_g2p.data.prepare import prepare_ruby
    from lattice_g2p.data.ruby import RubyExample

    example = RubyExample(text="生ビールを飲む", spans=[(0, 1, "ナマ")], source="t")
    prepared = prepare_ruby(example, _ruby_dic())
    assert prepared is not None

    scores = torch.zeros(len(prepared.nodes), dtype=torch.float64)
    loss = crf_loss_from_prepared(prepared, scores)

    assert torch.isfinite(loss)
    assert loss.item() > 0, "制約があるので分母 > 分子"


def test_prepare_ruby_returns_none_when_span_unverifiable() -> None:
    from lattice_g2p.data.prepare import prepare_ruby
    from lattice_g2p.data.ruby import RubyExample

    example = RubyExample(text="生ビールを飲む", spans=[(0, 1, "サダメ")], source="t")
    assert prepare_ruby(example, _ruby_dic()) is None


class _ZeroScorer:
    """全ノードのスコアを 0 にするだけのダミー。Viterbi は最短経路を選ぶ。"""

    def score(self, text: str, nodes: list) -> torch.Tensor:  # noqa: ANN001, ARG002
        return torch.zeros(len(nodes), dtype=torch.float64)


def test_evaluate_ignores_examples_without_full_reading() -> None:
    """★ルビ例は学習専用。文全体の読みが無いので精度評価の分母に入れないこと。

    分母に入れると必ず不正解扱いになり、精度が理由もなく下がる。
    """
    from lattice_g2p.data.prepare import prepare_ruby
    from lattice_g2p.data.ruby import RubyExample
    from lattice_g2p.train import evaluate

    ruby = prepare_ruby(
        RubyExample(text="生ビールを飲む", spans=[(0, 1, "ナマ")]), _ruby_dic()
    )
    assert ruby is not None

    full = prepare(
        TrainingExample(
            text="米を炊く",
            reading="コメヲタク",
            target_start=0,
            target_end=1,
            target_surface="米",
            target_reading="コメ",
        ),
        _dic(),
    )
    assert full is not None

    model = _ZeroScorer()
    only_full = evaluate(model, [full])
    with_ruby = evaluate(model, [full, ruby])

    assert with_ruby == only_full, "ルビ例は評価の分母に入らない"
    assert evaluate(model, [ruby]) == 0.0, "評価できる例が無ければ 0.0"


def test_evaluate_target_ignores_examples_without_full_reading() -> None:
    """★evaluate と同じ母集団で測ること。

    ルビ例を混ぜると母集団が実行ごとに変わり、実行間で比較できなくなる。
    """
    from lattice_g2p.data.prepare import prepare_ruby
    from lattice_g2p.data.ruby import RubyExample
    from lattice_g2p.train import evaluate_target

    ruby = prepare_ruby(
        RubyExample(text="生ビールを飲む", spans=[(0, 1, "ナマ")]), _ruby_dic()
    )
    full = prepare(
        TrainingExample(
            text="米を炊く",
            reading="コメヲタク",
            target_start=0,
            target_end=1,
            target_surface="米",
            target_reading="コメ",
        ),
        _dic(),
    )
    assert ruby is not None and full is not None

    model = _ZeroScorer()
    assert evaluate_target(model, [full, ruby]) == evaluate_target(model, [full])
    assert evaluate_target(model, [ruby]) == 0.0
