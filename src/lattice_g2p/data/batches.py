"""サブエージェントによる生成のためのバッチ分割と回収.

Claude Code のサブエージェントに生成させる場合, API クライアントの代わりに
「バッチファイルを渡して JSONL を書かせる」形をとる.

  1. make_batches()      多音語をバッチファイルに分割する
  2. (サブエージェントが各バッチを読んで <batch>.out.jsonl を書く)
  3. merge_batch_outputs() 出力を回収して GeneratedSentence にする
  4. filter_sentences()  以降は API 経由の場合とまったく同じ

★サブエージェントの出力は壊れうる. 1 行の欠損で全体を落とさないこと.
読みの正しさはラティス整合フィルタが担保するので, ここでは形式だけを見る.
"""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from lattice_g2p.data.schema import GeneratedSentence, read_jsonl, write_jsonl
from lattice_g2p.data.select import PolyphonicWord
from lattice_g2p.kana import normalize_kana

BATCH_PREFIX = "batch_"
OUTPUT_SUFFIX = ".out.jsonl"


def make_batches(
    words: list[PolyphonicWord], out_dir: Path, batch_size: int
) -> list[Path]:
    """多音語をバッチファイルに分割する.

    Returns:
        書き出したバッチファイルのパス (通し番号順)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i in range(0, len(words), batch_size):
        chunk = words[i : i + batch_size]
        path = out_dir / f"{BATCH_PREFIX}{i // batch_size:03d}.jsonl"
        write_jsonl(path, [asdict(w) for w in chunk])
        paths.append(path)

    return paths


def parse_batch_output(path: Path, model: str) -> list[GeneratedSentence]:
    """サブエージェントが書いた JSONL を GeneratedSentence にする.

    形式が壊れている行は黙って捨てる. 読みの正しさはこの先のフィルタが見る.
    """
    today = date.today().isoformat()
    rows: list[GeneratedSentence] = []

    if not path.exists():
        return rows

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            row = _to_sentence(obj, model, today)
            if row is not None:
                rows.append(row)

    return rows


def _to_sentence(obj: dict, model: str, today: str) -> GeneratedSentence | None:
    required = ("text", "reading", "target_surface", "target_reading")
    if not all(isinstance(obj.get(k), str) and obj[k] for k in required):
        return None

    text = obj["text"]
    surface = obj["target_surface"]

    # ★target_start は信用しない. 表記から復元できるならそちらを優先する.
    start = obj.get("target_start")
    if not (
        isinstance(start, int)
        and start >= 0
        and text[start : start + len(surface)] == surface
    ):
        start = text.find(surface)
    if start < 0:
        return None

    return GeneratedSentence(
        text=text,
        reading=normalize_kana(obj["reading"]),
        target_start=start,
        target_end=start + len(surface),
        target_surface=surface,
        target_reading=normalize_kana(obj["target_reading"]),
        prompt_version=str(obj.get("prompt_version", "subagent-v1")),
        model=model,
        generated_at=today,
    )


def merge_batch_outputs(out_dir: Path, model: str) -> list[GeneratedSentence]:
    """バッチ出力をすべて回収する."""
    rows: list[GeneratedSentence] = []
    for path in sorted(out_dir.glob(f"*{OUTPUT_SUFFIX}")):
        rows.extend(parse_batch_output(path, model))
    return rows


def pending_batches(out_dir: Path) -> list[Path]:
    """まだ出力が書かれていないバッチファイルを返す (途中再開用)."""
    pending: list[Path] = []
    for path in sorted(out_dir.glob(f"{BATCH_PREFIX}*.jsonl")):
        if path.name.endswith(OUTPUT_SUFFIX):
            continue
        if not _output_path(path).exists():
            pending.append(path)
    return pending


def _output_path(batch_path: Path) -> Path:
    return batch_path.with_suffix("") .with_suffix(OUTPUT_SUFFIX)


def batch_summary(out_dir: Path) -> str:
    """バッチの進捗を 1 行で返す."""
    batches = [
        p
        for p in sorted(out_dir.glob(f"{BATCH_PREFIX}*.jsonl"))
        if not p.name.endswith(OUTPUT_SUFFIX)
    ]
    done = [p for p in batches if _output_path(p).exists()]
    n_rows = sum(1 for p in done for _ in read_jsonl(p))
    return f"バッチ {len(done)}/{len(batches)} 完了  対象語 {n_rows}"
