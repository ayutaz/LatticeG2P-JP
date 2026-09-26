"""文字語彙.

★語彙の順序を入力順に依存させないこと. 語彙 ID がずれると, 保存したモデルと
語彙の対応が壊れる.

本文用と読み用で別の語彙を使う.

  本文: 学習データと辞書から構築する. 文字種は数千
  読み: カタカナ + 長音符で固定. 文字種は 100 程度

★読みの表現を「(表記, 読み) ペアごとの埋め込みテーブル」で持ってはいけない
(docs/architecture.md §4-4). 151,000 entry × 256 次元で 155 MB になり,
軽量化目標を単独で破綻させる. カナ文字列を小さな文字単位エンコーダに通す.
"""

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from lattice_g2p.kana import KATAKANA_END, KATAKANA_START, PROLONGED


class CharVocab:
    """文字 -> ID の対応."""

    PAD = 0
    UNK = 1

    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self._index = {c: i + 2 for i, c in enumerate(chars)}

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 1) -> "CharVocab":
        """テキスト列から語彙を作る.

        Args:
            min_count: この回数未満しか出現しない文字を落とす

        ★ソートして順序を入力順に依存させない.
        """
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)
        return cls(sorted(c for c, n in counter.items() if n >= min_count))

    @classmethod
    def katakana(cls) -> "CharVocab":
        """読み用の固定語彙 (カタカナ + 長音符).

        正規化後の読みはこの範囲に収まる (kana.is_kana_only と同じ定義).
        """
        chars = [chr(c) for c in range(ord(KATAKANA_START), ord(KATAKANA_END) + 1)]
        chars.append(PROLONGED)
        return cls(sorted(set(chars)))

    def encode(self, s: str) -> list[int]:
        return [self._index.get(c, self.UNK) for c in s]

    def __len__(self) -> int:
        return len(self.chars) + 2

    def __contains__(self, ch: str) -> bool:
        return ch in self._index

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.chars, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        return cls(json.loads(path.read_text(encoding="utf-8")))
