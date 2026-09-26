"""学習ループ.

★チェックポイントを定期保存し, resume 可能にすること.
vast.ai のインスタンスは予告なく停止しうる (docs/pitfalls.md R-14).
optimizer state と RNG state も保存する.

★評価は「文全体の読みが一致するか」で判定する.

  target span の読みをノードから取り出して比較してはいけない.
  「皆既日食」の「皆」を対象にすると,
    経路 A: 皆既 -> カイキ の 1 ノード
    経路 B: 皆 -> カイ, 既 -> キ の 2 ノード
  はどちらも文の読みが同じで, どちらも正解である. しかしノードから
  取り出すと A は 'カイキ' になり不正解と判定されてしまう.

  これは「形態素分割を正解にしない」という手法の思想そのものに反する
  (docs/pitfalls.md R-15). 実測では 100% 正解のモデルが 80.5% に見えた.

  benchmark の評価では tagged_yomi がカナ位置を示すので target 単位で
  測れるが, 生成データにはその情報がない. 文全体の一致は
  「target が正しい」ことの十分条件であり, 分割に依存しない.
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lattice_g2p.config import TrainConfig
from lattice_g2p.crf import crf_loss_from_prepared, viterbi
from lattice_g2p.data.prepare import PreparedExample, prepare_all, prepare_ruby_all
from lattice_g2p.data.ruby import read_ruby_jsonl
from lattice_g2p.data.schema import TrainingExample, read_jsonl
from lattice_g2p.decode import path_reading, span_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.encoder import PretrainedCharEncoder, ReadingEncoder, ScratchCharEncoder
from lattice_g2p.kana import phonetic_key
from lattice_g2p.losses import margin_loss
from lattice_g2p.scorer import BiEncoderScorer, CrossEncoderScorer
from lattice_g2p.vocab import CharVocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prepared(path: Path, dic: Dictionary) -> tuple[list[PreparedExample], int]:
    """学習データを前処理する. 正解経路がない例は捨てる."""
    rows = [TrainingExample(**r) for r in read_jsonl(path)]
    return prepare_all(rows, dic)


def load_ruby(path: Path, dic: Dictionary) -> tuple[list[PreparedExample], int]:
    """青空文庫のルビ由来データを前処理する.

    ★これらは学習専用. 文全体の読みを持たないので dev には使えない
    (evaluate が分母から外す).
    """
    return prepare_ruby_all(read_ruby_jsonl(path), dic)


def load_scorer(run_dir: Path, device: str = "cpu") -> tuple[str, nn.Module]:
    """学習結果のディレクトリからスコアラを復元する.

    ★語彙は学習時に保存したものを使う. 作り直してはいけない
    (学習データが違えば語彙も違い, 埋め込みの行がずれる).

    Returns:
        (表示用の名前, 評価モードのモデル)
    """
    for name in ("best.pt", "latest.pt"):
        path = run_dir / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(
            f"{run_dir} にチェックポイント (best.pt / latest.pt) がありません。"
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k != "extra"})
    model = build_model(
        config,
        CharVocab.load(run_dir / "text_vocab.json"),
        CharVocab.load(run_dir / "kana_vocab.json"),
    )
    # ★dict_prior_weight は M4 で追加した。それ以前のチェックポイントには無い
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"チェックポイントに未知のパラメータがあります: {unexpected}")
    if missing:
        print(f"★このチェックポイントには {missing} が無い（初期値を使う）")
    model.to(device).eval()
    return f"{run_dir.name} (step {ckpt['step']})", model


def load_init_weights(run_dir: Path, model: nn.Module, prefer: str = "best") -> int:
    """別の学習結果から重みだけ読み込む.

    optimizer / scheduler / RNG は引き継がない. 学習率スケジュールを
    最初からやり直すため (微調整では warmup を入れ直したい).

    ★prefer="best" は罠になりうる.
    事前学習の best.pt は「その時点の dev が最良」で選ばれており,
    事前学習の目的 (表現を作ること) とは別の基準である. 実際, ルビ事前学習で
    3,500 step 回しても best は step 1,400 で, 残りは捨てられていた.
    **事前学習からの初期化には latest を選ぶほうが筋が通る場合がある.**
    どちらが良いかは測って決めること.
    """
    order = ("best.pt", "latest.pt") if prefer == "best" else ("latest.pt", "best.pt")
    for name in order:
        path = run_dir / name
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            print(f"重みの由来: {name} (step {int(ckpt['step'])})")
            return int(ckpt["step"])
    raise FileNotFoundError(
        f"init_from に指定した {run_dir} に best.pt も latest.pt もありません。"
    )


def build_vocabs(
    examples: list[PreparedExample], dic: Dictionary, min_count: int = 1
) -> tuple[CharVocab, CharVocab]:
    """本文用と読み用の語彙を作る.

    本文の語彙は学習データと辞書の表記から構築する.
    読みの語彙はカタカナ固定 (文字種が限られるため).
    """
    text_vocab = CharVocab.build(
        [e.text for e in examples] + dic.surfaces(), min_count=min_count
    )
    return text_vocab, CharVocab.katakana()


def build_context_encoder(config: TrainConfig, text_vocab: CharVocab) -> nn.Module:
    """文脈エンコーダを作る.

    ★pretrained は研究用の参照。出荷経路は scratch のまま
    (docs/pitfalls.md R-08)。
    """
    if config.context_encoder == "pretrained":
        return PretrainedCharEncoder(config.pretrained_name, text_vocab, config.dropout)
    if config.context_encoder != "scratch":
        raise ValueError(
            f"未知の context_encoder: {config.context_encoder!r}（scratch | pretrained）"
        )
    return ScratchCharEncoder(
        vocab_size=len(text_vocab),
        hidden=config.hidden,
        layers=config.ctx_layers,
        heads=config.ctx_heads,
        ffn=config.ctx_ffn,
        max_len=config.max_len,
        dropout=config.dropout,
    )


def build_model(config: TrainConfig, text_vocab: CharVocab, kana_vocab: CharVocab) -> nn.Module:
    ctx = build_context_encoder(config, text_vocab)
    rd = ReadingEncoder(
        vocab_size=len(kana_vocab),
        hidden=config.hidden,
        layers=config.reading_layers,
        heads=config.ctx_heads,
        ffn=config.ctx_ffn,
        dropout=config.dropout,
    )
    kw = {"transition_dim": config.transition_dim, "transition_ids": config.transition_ids}
    if config.scorer == "bi_encoder":
        return BiEncoderScorer(ctx, rd, text_vocab, kana_vocab, **kw)
    if config.scorer == "cross_encoder":
        return CrossEncoderScorer(
            ctx,
            rd,
            text_vocab,
            kana_vocab,
            **kw,
            layers=config.pair_layers,
            heads=config.ctx_heads,
            ffn=config.ctx_ffn,
            dropout=config.dropout,
        )
    raise ValueError(f"未知の scorer: {config.scorer!r}（bi_encoder | cross_encoder）")


def _transitions(model, nodes):  # noqa: ANN001, ANN202
    """遷移スコアを取り出す. 遷移項を持たない構成 / UnigramScorer では None.

    ★評価と学習で同じ経路スコアを使うこと。片方だけ遷移項を入れると
    「損失は下がるのに精度が上がらない」という追いにくい不整合になる。
    """
    getter = getattr(model, "transitions", None)
    return None if getter is None else getter(nodes)


@torch.no_grad()
def evaluate(model, examples: list[PreparedExample]) -> float:  # noqa: ANN001
    """文全体の読みが一致した割合を返す.

    ★分割に依存しない. 同じ読みを別の分割で出しても正解とする
    (モジュール docstring 参照).

    ★文全体の読みを持たない例 (青空文庫のルビ由来) は分母に入れない.
    入れると必ず不正解扱いになり, 精度が理由もなく下がる.
    """
    evaluable = [ex for ex in examples if ex.gold_reading is not None]
    if not evaluable:
        return 0.0
    if isinstance(model, nn.Module):
        model.eval()

    correct = 0
    for ex in evaluable:
        lattice = ex.lattice()
        path = viterbi(lattice, model.score(ex.text, ex.nodes), _transitions(model, ex.nodes))
        correct += path_reading(lattice, path) == ex.gold_reading

    if isinstance(model, nn.Module):
        model.train()
    return correct / len(evaluable)


@torch.no_grad()
def evaluate_target(model, examples: list[PreparedExample]) -> float:  # noqa: ANN001
    """対象語の読みが一致した割合を返す (参考値).

    ★分割によっては対象語の読みを分離できないため, この値は過小評価になる.
    ゲートには使わない. evaluate() を使うこと.

    ★evaluate() と同じ母集団で測る (文全体の読みを持つ例だけ).
    ルビ由来の例を混ぜると, 母集団が実行ごとに変わって比較できなくなる.
    ルビを足すと学習データが 4 倍になるので, 実行時間も無駄に伸びる
    (実測で最終評価だけに 100 分近くかかる見込みだった).
    """
    evaluable = [ex for ex in examples if ex.gold_reading is not None]
    if not evaluable:
        return 0.0
    if isinstance(model, nn.Module):
        model.eval()

    correct = 0
    for ex in evaluable:
        lattice = ex.lattice()
        path = viterbi(lattice, model.score(ex.text, ex.nodes), _transitions(model, ex.nodes))
        pred = span_reading(lattice, path, ex.target_start, ex.target_end)
        correct += phonetic_key(pred) == phonetic_key(ex.gold_span_reading())

    if isinstance(model, nn.Module):
        model.train()
    return correct / len(evaluable)


def save_checkpoint(
    path: Path,
    step: int,
    model: nn.Module,
    optimizer,  # noqa: ANN001
    scheduler,  # noqa: ANN001
    config: TrainConfig,
) -> None:
    """★原子的に置き換える. 保存中に落ちても壊れたファイルが残らない."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "config": config.to_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scheduler) -> int:  # noqa: ANN001
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    rng = ckpt["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])

    return int(ckpt["step"])


def train(config: TrainConfig) -> dict[str, float]:
    """学習を実行し, 最終的な指標を返す."""
    set_seed(config.seed)
    device = torch.device(
        config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    print(f"device: {device}")

    dic = Dictionary.load(config.dict_cache)
    train_examples, dropped_train = load_prepared(config.train_path, dic)
    dev_examples, dropped_dev = load_prepared(config.dev_path, dic)
    if dropped_train or dropped_dev:
        print(f"★正解経路なしで除外: train {dropped_train} / dev {dropped_dev}")

    if config.ruby_path is not None:
        ruby_examples, dropped_ruby = load_ruby(config.ruby_path, dic)
        train_examples.extend(ruby_examples)
        print(f"ルビ由来を追加: {len(ruby_examples):,} 文（除外 {dropped_ruby}）")

    print(f"train {len(train_examples):,} / dev {len(dev_examples):,}")

    if config.init_from is not None:
        # ★語彙は引き継ぐ. 作り直すと行数が変わって state_dict が読めない
        text_vocab = CharVocab.load(config.init_from / "text_vocab.json")
        kana_vocab = CharVocab.load(config.init_from / "kana_vocab.json")
        print(f"語彙を引き継ぎ: {config.init_from}")
    else:
        text_vocab, kana_vocab = build_vocabs(train_examples, dic, config.vocab_min_count)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    text_vocab.save(config.out_dir / "text_vocab.json")
    kana_vocab.save(config.out_dir / "kana_vocab.json")
    print(f"語彙: 本文 {len(text_vocab):,} / 読み {len(kana_vocab):,}")

    model = build_model(config, text_vocab, kana_vocab)
    if config.init_from is not None:
        source_step = load_init_weights(
            config.init_from, model, config.init_from_prefer
        )
        print(f"重みを引き継ぎ: {config.init_from} (step {source_step})")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M  scorer={config.scorer}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps_per_epoch = max(1, (len(train_examples) + config.batch_size - 1) // config.batch_size)
    total_steps = max(1, steps_per_epoch * config.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=total_steps, pct_start=config.warmup_ratio
    )

    latest = config.out_dir / "latest.pt"
    step = load_checkpoint(latest, model, optimizer, scheduler) if latest.exists() else 0
    if step:
        print(f"チェックポイントから再開: step={step}")

    best_dev = 0.0
    start = time.time()
    model.train()

    for epoch in range(config.epochs):
        order = list(range(len(train_examples)))
        random.shuffle(order)

        for batch_start in range(0, len(order), config.batch_size):
            batch = [
                train_examples[i]
                for i in order[batch_start : batch_start + config.batch_size]
            ]

            # ★全ノードを 1 回の forward でスコアリングする
            all_scores = model.score_batch([e.text for e in batch], [e.nodes for e in batch])

            loss = torch.zeros((), device=device)
            for ex, scores in zip(batch, all_scores, strict=True):
                loss = loss + crf_loss_from_prepared(ex, scores, model.transitions(ex.nodes))
                if config.margin_weight:
                    loss = loss + config.margin_weight * margin_loss(
                        scores, ex.nodes, ex.gold_node_indices, config.margin
                    )
            loss = loss / len(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if step < total_steps - 1:
                scheduler.step()
            step += 1

            if config.log_every_steps and step % config.log_every_steps == 0:
                print(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {loss.detach().item():.4f} ({time.time() - start:.0f}s)"
                )

            if config.save_every_steps and step % config.save_every_steps == 0:
                save_checkpoint(latest, step, model, optimizer, scheduler, config)

            if config.eval_every_steps and step % config.eval_every_steps == 0 and dev_examples:
                acc = evaluate(model, dev_examples)
                print(f"  dev Accuracy: {acc * 100:.2f}%")
                if acc > best_dev:
                    best_dev = acc
                    save_checkpoint(
                        config.out_dir / "best.pt", step, model, optimizer, scheduler, config
                    )

        save_checkpoint(latest, step, model, optimizer, scheduler, config)

    train_acc = evaluate(model, train_examples)
    dev_acc = evaluate(model, dev_examples) if dev_examples else 0.0
    best_dev = max(best_dev, dev_acc)
    train_target = evaluate_target(model, train_examples)
    dev_target = evaluate_target(model, dev_examples) if dev_examples else 0.0

    print(f"\ntrain 文一致: {train_acc * 100:.2f}%   (対象語 {train_target * 100:.2f}%)")
    print(f"dev   文一致: {dev_acc * 100:.2f}%   (対象語 {dev_target * 100:.2f}%)")
    print(f"best dev    : {best_dev * 100:.2f}%")

    return {
        "train_accuracy": train_acc,
        "dev_accuracy": dev_acc,
        "best_dev_accuracy": best_dev,
        "train_target_accuracy": train_target,
        "dev_target_accuracy": dev_target,
        "params_m": n_params / 1e6,
        "steps": float(step),
        "train_examples": float(len(train_examples)),
        "dev_examples": float(len(dev_examples)),
    }
