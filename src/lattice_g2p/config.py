"""学習設定."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    # データ
    # ★現行の分割（M5 以降）。train と dev は必ず組で替えること。
    #   旧分割の data/train_split.jsonl には dev_r21 の語を含む行が 349 行あり、
    #   dev だけを替えると既定値の組み合わせで dev が漏れる（不変条件 12）。
    #   data/dev.jsonl（419 文）は M3d 以前の古い dev で、母集団が違う。
    train_path: Path = Path("data/train_m5_split.jsonl")
    dev_path: Path = Path("data/dev_r21.jsonl")
    dict_cache: Path = Path("data/dic_cache.pkl")

    init_from: Path | None = None
    """別の学習結果（out_dir）から重みと語彙を引き継ぐ.

    ★ルビで事前学習 -> 生成データで微調整、という使い方を想定する。
    optimizer と scheduler は引き継がない（新しい学習として始める）。
    語彙は必ず引き継ぐ。作り直すと埋め込みの行数が合わない。
    """

    transition_dim: int = 0
    """CRF の遷移項の埋め込み次元 (M6). 0 なら遷移項なし（M5 までと同じ）.

    ★MeCab の連接行列 480 MB に相当する部分を学習で圧縮する。
    16 次元 × 16,000 ID × 2 = 512K params ≈ INT8 0.5 MB。
    """

    transition_ids: int = 0
    """連接 ID の語彙サイズ. 0 なら TransitionScorer の既定値を使う."""

    init_from_prefer: str = "best"
    """init_from で best.pt と latest.pt のどちらを使うか（"best" | "latest"）.

    ★既定の "best" は罠になりうる。事前学習の best.pt は「その時点の dev が
    最良」で選ばれており、事前学習の目的（表現を作ること）とは別の基準である。
    実際、ルビ事前学習を 3,500 step 回しても best は step 1,400 だった。
    どちらが良いかは測って決めること。
    """

    ruby_path: Path | None = None
    """青空文庫のルビ由来データ (data/ruby.jsonl).

    ★学習データにだけ足す. 文全体の読みを持たないので dev には使えない.
    """

    # モデル
    scorer: str = "bi_encoder"
    """bi_encoder | cross_encoder"""

    context_encoder: str = "scratch"
    """scratch | pretrained

    ★pretrained は研究用の参照であり出荷経路ではない。
    ku-nlp の文字単位モデルは CC BY-SA 4.0 (copyleft) で,
    寛容なライセンスでの配布と両立しない (docs/pitfalls.md R-08)。
    「事前学習ありでどこまで出るか」を測るだけ。
    """

    pretrained_name: str = "ku-nlp/deberta-v2-tiny-japanese-char-wwm"
    hidden: int = 256
    ctx_layers: int = 4
    ctx_heads: int = 4
    ctx_ffn: int = 1024
    reading_layers: int = 2
    pair_layers: int = 2
    max_len: int = 512
    dropout: float = 0.1
    vocab_min_count: int = 1

    # 学習
    epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    margin: float = 1.0
    margin_weight: float = 0.5
    grad_clip: float = 1.0
    seed: int = 42

    # チェックポイント (★vast.ai の中断対策)
    out_dir: Path = Path("checkpoints")
    save_every_steps: int = 500
    eval_every_steps: int = 500
    log_every_steps: int = 50
    device: str = "cuda"

    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key in (
            "train_path",
            "dev_path",
            "dict_cache",
            "out_dir",
            "ruby_path",
            "init_from",
        ):
            if key in raw and raw[key] is not None:
                raw[key] = Path(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
