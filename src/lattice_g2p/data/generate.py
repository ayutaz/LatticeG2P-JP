"""生成のオーケストレーション.

★1 エントリごとに JSONL へ追記する. 途中で落ちても再開できるようにする.
生成は高コストなので, メモリに溜めて最後に書き出す設計にしてはいけない.
"""

from datetime import date
from pathlib import Path

from lattice_g2p.data.client import ToolClient
from lattice_g2p.data.prompts import (
    PROMPT_VERSION,
    READING_GROUP_TOOL,
    SENTENCE_TOOL,
    build_reading_group_prompt,
    build_sentence_prompt,
)
from lattice_g2p.data.schema import (
    GeneratedSentence,
    ReadingGroup,
    append_jsonl,
    read_jsonl,
)
from lattice_g2p.data.select import PolyphonicWord
from lattice_g2p.kana import normalize_kana


def _done_keys(path: Path, fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    """既に処理済みのキーを集める (途中再開用)."""
    return {tuple(str(row.get(f, "")) for f in fields) for row in read_jsonl(path)}


def generate_reading_groups(
    client: ToolClient,
    words: list[PolyphonicWord],
    out_path: Path,
    progress_every: int = 20,
) -> None:
    """各多音語の読みを検証し, 意味ごとにグループ化する."""
    done = {k[0] for k in _done_keys(out_path, ("surface",))}

    for i, word in enumerate(words, start=1):
        if word.surface in done:
            continue
        result = client.call_tool(
            build_reading_group_prompt(word.surface, list(word.readings), list(word.pos_list)),
            READING_GROUP_TOOL,
        )
        for g in result.get("groups", []):
            if "reading" not in g:
                continue
            append_jsonl(
                out_path,
                ReadingGroup(
                    surface=word.surface,
                    reading=normalize_kana(str(g["reading"])),
                    sense=str(g.get("sense", "")),
                    pos=str(g.get("pos", "")),
                    valid=bool(g.get("valid", False)),
                    reason=str(g.get("reason", "")),
                ),
            )
        if progress_every and i % progress_every == 0:
            print(f"読み検証 {i}/{len(words)}")


def generate_sentences(
    client: ToolClient,
    groups: list[ReadingGroup],
    out_path: Path,
    n_per_group: int,
    progress_every: int = 20,
) -> None:
    """各 reading group について例文を生成する."""
    targets = [g for g in groups if g.valid]
    done = _done_keys(out_path, ("target_surface", "target_reading"))
    today = date.today().isoformat()

    for i, group in enumerate(targets, start=1):
        if (group.surface, group.reading) in done:
            continue
        result = client.call_tool(
            build_sentence_prompt(
                group.surface, group.reading, group.sense, group.pos, n_per_group
            ),
            SENTENCE_TOOL,
        )
        for s in result.get("sentences", []):
            if not all(k in s for k in ("text", "reading", "target_start")):
                continue
            text = str(s["text"])
            start = int(s["target_start"])
            append_jsonl(
                out_path,
                GeneratedSentence(
                    text=text,
                    reading=normalize_kana(str(s["reading"])),
                    target_start=start,
                    target_end=start + len(group.surface),
                    target_surface=group.surface,
                    target_reading=group.reading,
                    prompt_version=PROMPT_VERSION,
                    model=client.model,
                    generated_at=today,
                ),
            )
        if progress_every and i % progress_every == 0:
            print(f"文生成 {i}/{len(targets)}")
