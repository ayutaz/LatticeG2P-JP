"""生成データのスキーマと JSONL 入出力."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadingGroup:
    """多音語の 1 つの読みと, その意味.

    Claude に「この読みは実在するか」「意味ごとにグループ化せよ」と依頼した結果.
    """

    surface: str
    reading: str
    sense: str
    pos: str
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class GeneratedSentence:
    """生成された文 1 件 (フィルタ前)."""

    text: str
    reading: str
    target_start: int
    target_end: int
    target_surface: str
    target_reading: str
    prompt_version: str
    model: str
    generated_at: str


@dataclass(frozen=True)
class TrainingExample:
    """フィルタを通過した学習データ 1 件."""

    text: str
    reading: str
    target_start: int
    target_end: int
    target_surface: str
    target_reading: str
    source: str = "claude-generated"
    meta: dict[str, str] = field(default_factory=dict)


def _to_dict(obj: object) -> dict:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"JSONL に書けません: {type(obj)}")


def append_jsonl(path: Path, obj: object) -> None:
    """1 件を JSONL に追記する.

    ★生成は高コストなので, 必ず 1 件ずつ追記して途中再開できるようにする.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_dict(obj), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, objs: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(_to_dict(obj), ensure_ascii=False) + "\n")
