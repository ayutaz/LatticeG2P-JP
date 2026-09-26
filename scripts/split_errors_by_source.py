"""dev の誤りを生成元（opus / haiku）別に分解する（M3c Task 3）.

★dev の meta.model はどちらも "claude-code-subagent" で区別できない。
バッチの出力ファイルと本文の完全一致で突き合わせて復元する。

    data/batches/    Phase A = opus
    data/batches_b/  Phase B = haiku

haiku 由来の誤り率が明確に高ければ、天井の一部は生成品質で説明できる
（→ docs/pitfalls.md R-09）。差が無ければモデル側の問題に絞れる。
"""

import argparse
import json
from pathlib import Path

import torch

from lattice_g2p.config import TrainConfig
from lattice_g2p.crf import viterbi
from lattice_g2p.decode import span_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import phonetic_key
from lattice_g2p.train import build_model, load_prepared
from lattice_g2p.vocab import CharVocab


def texts_in(batch_dir: Path) -> set[str]:
    """バッチ出力に現れる本文の集合. 壊れた行は黙って飛ばす."""
    out: set[str] = set()
    for path in sorted(batch_dir.glob("*.out.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.add(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-path", type=Path, default=Path("data/dev_r21.jsonl"))
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument("--model-dir", type=Path, default=Path("checkpoints/finetune"))
    ap.add_argument("--opus-dir", type=Path, default=Path("data/batches"))
    ap.add_argument("--haiku-dir", type=Path, default=Path("data/batches_b"))
    args = ap.parse_args()

    opus = texts_in(args.opus_dir)
    haiku = texts_in(args.haiku_dir)
    print(f"opus  由来の本文 {len(opus):,} / haiku 由来 {len(haiku):,}")
    print(f"両方に現れる本文 {len(opus & haiku):,}（重複は判定不能として除く）\n")

    dic = Dictionary.load(args.dict_cache)
    examples, _ = load_prepared(args.dev_path, dic)
    rows = [json.loads(line) for line in args.dev_path.open(encoding="utf-8")]

    ckpt = torch.load(args.model_dir / "best.pt", map_location="cpu", weights_only=False)
    config = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k != "extra"})
    model = build_model(
        config,
        CharVocab.load(args.model_dir / "text_vocab.json"),
        CharVocab.load(args.model_dir / "kana_vocab.json"),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = {"opus": [0, 0], "haiku": [0, 0], "不明": [0, 0]}
    with torch.no_grad():
        for ex, row in zip(examples, rows, strict=True):
            both = ex.text in opus and ex.text in haiku
            if both:
                source = "不明"
            elif ex.text in opus:
                source = "opus"
            elif ex.text in haiku:
                source = "haiku"
            else:
                source = "不明"

            lattice = ex.lattice()
            path = viterbi(lattice, model.score(ex.text, ex.nodes))
            span = span_reading(lattice, path, row["target_start"], row["target_end"])
            ok = bool(span) and phonetic_key(row["target_reading"]) in phonetic_key(span)

            stats[source][0] += 1
            stats[source][1] += not ok

    print(f"{'生成元':<8} {'文数':>6} {'誤り':>6} {'誤り率':>8}")
    for name, (total, wrong) in stats.items():
        rate = wrong / total * 100 if total else 0.0
        print(f"{name:<8} {total:>6} {wrong:>6} {rate:>7.1f}%")

    o, h = stats["opus"], stats["haiku"]
    if o[0] and h[0]:
        diff = h[1] / h[0] * 100 - o[1] / o[0] * 100
        print(f"\n★haiku - opus の誤り率差: {diff:+.1f} ポイント")
        # ★符号を見ること。haiku の方が悪いときだけ「生成品質で説明できる」
        if abs(diff) < 5:
            print("  差は小さい。天井は生成品質では説明できない（R-09 の懸念は否定）")
        elif diff > 0:
            print("  haiku の方が誤りが多い。天井の一部は生成品質で説明できる")
        else:
            print("  ★opus の方が誤りが多い。生成品質では説明できない（R-09 の懸念は否定）")
            print("  Phase A（opus）は最頻の多音語を狙ったため、語自体が難しい可能性が高い")


if __name__ == "__main__":
    main()
