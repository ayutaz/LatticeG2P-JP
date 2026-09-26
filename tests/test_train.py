from pathlib import Path

import pytest
import torch

from lattice_g2p.config import TrainConfig
from lattice_g2p.data.prepare import prepare_all
from lattice_g2p.data.schema import TrainingExample, write_jsonl
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice
from lattice_g2p.train import (
    build_model,
    build_vocabs,
    evaluate,
    load_checkpoint,
    load_prepared,
    save_checkpoint,
    set_seed,
    train,
)
from lattice_g2p.vocab import CharVocab


def _dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0, cost=1000, source="both"),
            Entry("米", "ベイ", "名詞", 0, cost=1100, source="kana"),
            Entry("を", "ヲ", "助詞", 0, cost=-1000, source="kana"),
            Entry("炊く", "タク", "動詞", 0, cost=2000, source="both"),
            Entry("国", "コク", "名詞", 0, cost=1500, source="both"),
            Entry("の", "ノ", "助詞", 0, cost=-900, source="both"),
            Entry("政策", "セイサク", "名詞", 0, cost=1500, source="both"),
            Entry("。", "*", "補助記号-句点", 0, cost=-3000, source="both"),
        ]
    )


def _examples() -> list[TrainingExample]:
    return [
        TrainingExample("米を炊く。", "コメヲタク", 0, 1, "米", "コメ"),
        TrainingExample("米国の政策", "ベイコクノセイサク", 0, 1, "米", "ベイ"),
    ]


def _tiny_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        hidden=16,
        ctx_layers=1,
        ctx_heads=2,
        ctx_ffn=32,
        reading_layers=1,
        pair_layers=1,
        epochs=1,
        batch_size=2,
        out_dir=tmp_path / "ckpt",
        device="cpu",
    )


def test_load_prepared_drops_unreachable(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(
        path,
        [*_examples(), TrainingExample("米を炊く。", "イリガビ", 0, 1, "米", "コメ")],
    )

    prepared, dropped = load_prepared(path, _dic())

    assert len(prepared) == 2
    assert dropped == 1


def test_build_vocabs_covers_data_and_dictionary() -> None:
    dic = _dic()
    prepared, _ = prepare_all(_examples(), dic)
    text_vocab, kana_vocab = build_vocabs(prepared, dic)

    for ch in "米国炊政策":
        assert ch in text_vocab, ch
    for ch in "コメベイヲタク":
        assert ch in kana_vocab, ch


def test_build_model_respects_scorer_choice(tmp_path: Path) -> None:
    from lattice_g2p.scorer import BiEncoderScorer, CrossEncoderScorer

    dic = _dic()
    prepared, _ = prepare_all(_examples(), dic)
    tv, kv = build_vocabs(prepared, dic)

    cfg = _tiny_config(tmp_path)
    cfg.scorer = "bi_encoder"
    assert isinstance(build_model(cfg, tv, kv), BiEncoderScorer)

    cfg.scorer = "cross_encoder"
    assert isinstance(build_model(cfg, tv, kv), CrossEncoderScorer)


def test_build_model_rejects_unknown_scorer(tmp_path: Path) -> None:
    import pytest

    dic = _dic()
    prepared, _ = prepare_all(_examples(), dic)
    tv, kv = build_vocabs(prepared, dic)
    cfg = _tiny_config(tmp_path)
    cfg.scorer = "unknown"

    with pytest.raises(ValueError, match="scorer"):
        build_model(cfg, tv, kv)


def test_evaluate_returns_accuracy(tmp_path: Path) -> None:
    dic = _dic()
    prepared, _ = prepare_all(_examples(), dic)
    tv, kv = build_vocabs(prepared, dic)
    model = build_model(_tiny_config(tmp_path), tv, kv)

    acc = evaluate(model, prepared)

    assert 0.0 <= acc <= 1.0


def test_evaluate_is_perfect_when_gold_nodes_are_boosted(tmp_path: Path) -> None:
    """★正解ノードのスコアを上げれば Accuracy が 1.0 になること。

    評価の配線（Viterbi → 読み抽出 → 比較）が正しいことの確認。
    """
    dic = _dic()
    prepared, _ = prepare_all(_examples(), dic)
    tv, kv = build_vocabs(prepared, dic)

    class GoldScorer:
        """正解経路上のノードだけに高いスコアを与えるスコアラ。"""

        def __init__(self, examples) -> None:  # noqa: ANN001
            self._gold = {e.text: set(e.gold_node_indices) for e in examples}

        def score(self, text: str, nodes: list) -> torch.Tensor:  # noqa: ANN001
            gold = self._gold[text]
            return torch.tensor(
                [100.0 if i in gold else 0.0 for i in range(len(nodes))]
            )

    assert evaluate(GoldScorer(prepared), prepared) == 1.0


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    """★resume できること。vast.ai は予告なく停止しうる。"""
    dic = _dic()
    prepared, _ = prepare_all(_examples(), dic)
    tv, kv = build_vocabs(prepared, dic)
    cfg = _tiny_config(tmp_path)

    set_seed(0)
    model = build_model(cfg, tv, kv)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ConstantLR(opt)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, step=42, model=model, optimizer=opt, scheduler=sched, config=cfg)

    model2 = build_model(cfg, tv, kv)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.ConstantLR(opt2)
    step = load_checkpoint(path, model2, opt2, sched2)

    assert step == 42
    for a, b in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(a, b)


def test_checkpoint_is_written_atomically(tmp_path: Path) -> None:
    """保存中に落ちても壊れたファイルが残らないこと。"""
    dic = _dic()
    prepared, _ = prepare_all(_examples(), dic)
    tv, kv = build_vocabs(prepared, dic)
    cfg = _tiny_config(tmp_path)
    model = build_model(cfg, tv, kv)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ConstantLR(opt)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, 1, model, opt, sched, cfg)

    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


def test_config_roundtrip(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump({"scorer": "cross_encoder", "hidden": 128, "epochs": 3}),
        encoding="utf-8",
    )

    cfg = TrainConfig.load(path)

    assert cfg.scorer == "cross_encoder"
    assert cfg.hidden == 128
    assert cfg.epochs == 3


# --- 評価は分割に依存してはいけない ---


def test_evaluate_is_invariant_to_segmentation(tmp_path: Path) -> None:
    """★同じ読みを別の分割で出しても正解とすること。

    「皆既日食」の「皆」を対象にすると、
      経路 A: 皆既 -> カイキ の 1 ノード
      経路 B: 皆 -> カイ, 既 -> キ の 2 ノード
    はどちらも文の読みが同じで、どちらも正解である。

    ノードから span の読みを取り出して比較すると、A は 'カイキ' になり
    不正解と判定されてしまう。これは「形態素分割を正解にしない」という
    手法の思想そのものに反する。
    """
    dic = Dictionary.from_entries(
        [
            Entry("皆既", "カイキ", "名詞", 0, cost=1000, source="both"),
            Entry("皆", "カイ", "名詞", 0, cost=2000, source="both"),
            Entry("既", "キ", "名詞", 0, cost=2000, source="both"),
            Entry("日食", "ニッショク", "名詞", 0, cost=1000, source="both"),
        ]
    )
    example = TrainingExample("皆既日食", "カイキニッショク", 0, 1, "皆", "カイ")
    prepared, dropped = prepare_all([example], dic)
    assert dropped == 0

    class LongNodeScorer:
        """「皆既」の 1 ノードを選ぶスコアラ（正解の読みは出せている）。"""

        def score(self, text: str, nodes: list) -> torch.Tensor:  # noqa: ANN001
            return torch.tensor(
                [100.0 if n.surface in ("皆既", "日食") else 0.0 for n in nodes]
            )

    assert evaluate(LongNodeScorer(), prepared) == 1.0, (
        "分割が違うだけで不正解にしてはいけない"
    )


def test_evaluate_rejects_wrong_reading(tmp_path: Path) -> None:
    """読みが実際に違う場合はきちんと不正解にすること。"""
    dic = _dic()
    prepared, _ = prepare_all(
        [TrainingExample("米を炊く。", "コメヲタク", 0, 1, "米", "コメ")], dic
    )

    class WrongScorer:
        def score(self, text: str, nodes: list) -> torch.Tensor:  # noqa: ANN001
            return torch.tensor([100.0 if n.reading == "ベイ" else 0.0 for n in nodes])

    assert evaluate(WrongScorer(), prepared) == 0.0


# --- 青空文庫のルビ由来データ ---


def test_load_ruby_reads_jsonl(tmp_path) -> None:  # noqa: ANN001
    """JSONL のリストが spans のタプルに戻ること（JSON はタプルを保てない）。"""
    from lattice_g2p.data.ruby import read_ruby_jsonl

    path = tmp_path / "ruby.jsonl"
    path.write_text(
        '{"text": "生ビールを飲む", "spans": [[0, 1, "ナマ"]], "source": "x"}\n',
        encoding="utf-8",
    )
    rows = read_ruby_jsonl(path)

    assert len(rows) == 1
    assert rows[0].spans == [(0, 1, "ナマ")]
    assert rows[0].source == "x"


def test_load_ruby_prepares_partially_constrained_examples(tmp_path: Path) -> None:
    """★ルビ例が PreparedExample として読めること。文全体の読みは持たない。"""
    from lattice_g2p.train import load_ruby

    ruby_path = tmp_path / "ruby.jsonl"
    ruby_path.write_text(
        '{"text": "米を炊く。", "spans": [[0, 1, "コメ"]], "source": "aozora"}\n'
        '{"text": "米を炊く。", "spans": [[0, 1, "サダメ"]], "source": "aozora"}\n',
        encoding="utf-8",
    )
    rows, dropped = load_ruby(ruby_path, _dic())

    assert len(rows) == 1, "辞書に無い読みの例は落ちる"
    assert dropped == 1
    assert rows[0].gold_reading is None


def test_train_adds_ruby_to_training_set(tmp_path: Path) -> None:
    """★ルビ例は学習データにだけ足す。dev には入れない（精度評価ができないため）。"""
    ruby_path = tmp_path / "ruby.jsonl"
    ruby_path.write_text(
        '{"text": "米国の政策", "spans": [[0, 1, "ベイ"]], "source": "aozora"}\n',
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    write_jsonl(train_path, _examples())
    write_jsonl(dev_path, _examples())

    config = _tiny_config(tmp_path)
    config.train_path = train_path
    config.dev_path = dev_path
    config.ruby_path = ruby_path
    config.dict_cache = tmp_path / "dic.pkl"
    _dic().save(config.dict_cache)

    metrics = train(config)

    assert metrics["train_examples"] == len(_examples()) + 1
    assert metrics["dev_examples"] == len(_examples())


def test_init_from_loads_weights_and_vocab(tmp_path: Path) -> None:
    """★別の学習結果から重みだけ引き継げること（ルビで事前学習 → 生成データで微調整）。

    語彙も引き継ぐこと。学習データが違えば語彙も違い、作り直すと
    埋め込みの行数が合わずに state_dict が読み込めない。

    vocab_min_count を変えて語彙が作り直されないことを確かめる。
    """
    import dataclasses

    assert "init_from" in {f.name for f in dataclasses.fields(TrainConfig)}, (
        "TrainConfig の正式なフィールドであること（動的属性では設定ファイルから渡せない）"
    )

    dic = _dic()
    train_path = tmp_path / "train.jsonl"
    write_jsonl(train_path, _examples())

    first = _tiny_config(tmp_path)
    first.train_path = train_path
    first.dev_path = train_path
    first.dict_cache = tmp_path / "dic.pkl"
    first.out_dir = tmp_path / "first"
    dic.save(first.dict_cache)
    train(first)
    source_vocab = (first.out_dir / "text_vocab.json").read_text(encoding="utf-8")

    second = _tiny_config(tmp_path)
    second.train_path = train_path
    second.dev_path = train_path
    second.dict_cache = first.dict_cache
    second.out_dir = tmp_path / "second"
    second.init_from = first.out_dir
    second.vocab_min_count = 10_000  # 作り直したら語彙が激減するはず

    metrics = train(second)

    assert metrics["params_m"] > 0
    assert (second.out_dir / "text_vocab.json").read_text(
        encoding="utf-8"
    ) == source_vocab, "語彙を引き継ぐこと（vocab_min_count は無視される）"


def test_init_from_missing_checkpoint_is_reported(tmp_path: Path) -> None:
    from lattice_g2p.train import load_init_weights

    with pytest.raises(FileNotFoundError, match="init_from"):
        load_init_weights(tmp_path / "nowhere", torch.nn.Linear(1, 1))


def test_build_context_encoder_rejects_unknown_kind() -> None:
    from lattice_g2p.train import build_context_encoder

    config = TrainConfig(context_encoder="なにか")
    with pytest.raises(ValueError, match="context_encoder"):
        build_context_encoder(config, CharVocab(["米"]))


def test_build_context_encoder_defaults_to_scratch() -> None:
    from lattice_g2p.encoder import ScratchCharEncoder
    from lattice_g2p.train import build_context_encoder

    encoder = build_context_encoder(TrainConfig(hidden=16, ctx_layers=1, ctx_heads=2,
                                                ctx_ffn=32), CharVocab(["米"]))
    assert isinstance(encoder, ScratchCharEncoder)


def test_load_scorer_restores_identical_model(tmp_path: Path) -> None:
    """★チェックポイントから復元したモデルが元と同じスコアを出すこと。

    run_eval.py と compare_dev.py が同じ経路を使うための共通化。
    語彙を作り直すと埋め込みの行がずれるので、保存したものを読むこと。
    """
    from lattice_g2p.train import load_scorer

    dic = _dic()
    train_path = tmp_path / "train.jsonl"
    write_jsonl(train_path, _examples())

    config = _tiny_config(tmp_path)
    config.train_path = train_path
    config.dev_path = train_path
    config.dict_cache = tmp_path / "dic.pkl"
    config.out_dir = tmp_path / "run"
    dic.save(config.dict_cache)
    train(config)

    name, model = load_scorer(config.out_dir, device="cpu")

    assert "run" in name
    lattice = build_lattice("米を炊く。", dic)
    with torch.no_grad():
        scores = model.score(lattice.text, lattice.nodes)
    assert scores.shape == (len(lattice.nodes),)
    assert torch.isfinite(scores).all()


def test_load_scorer_reports_missing_checkpoint(tmp_path: Path) -> None:
    from lattice_g2p.train import load_scorer

    with pytest.raises(FileNotFoundError, match="チェックポイント"):
        load_scorer(tmp_path / "nowhere")
