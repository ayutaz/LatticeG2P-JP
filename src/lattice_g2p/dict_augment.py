"""複合語から単漢字の読みを抽出して辞書を補完する.

論文は mpaligner で「単独 entry として登録されていない漢字の読み」を
約 16,600 件抽出している (docs/architecture.md §3-5). ここではその代わりに,
既知の読みを使った制約伝播で決定的に抽出する.

  撮了 -> サツリョウ で「了 -> リョウ」が既知なら,「撮 -> サツ」が一意に決まる

★推測で辞書を汚さないこと. 割り当てが一意に決まらない場合は採用しない.
統計的な推定より件数は減るが, 誤った読みをラティスに入れる方が害が大きい.
"""

from collections.abc import Iterator

from lattice_g2p.dictionary import Dictionary, Entry

_KANJI_RANGES = (
    ("一", "鿿"),  # CJK 統合漢字
    ("㐀", "䶿"),  # CJK 拡張 A
    ("豈", "﫿"),  # CJK 互換漢字
)

DEFAULT_EXTRACTED_COST = 9000
"""抽出した読みの生起コスト. 実エントリより不利にする."""


def is_kanji(ch: str) -> bool:
    return any(lo <= ch <= hi for lo, hi in _KANJI_RANGES)


def split_reading(reading: str, n_parts: int, max_part_len: int) -> Iterator[tuple[str, ...]]:
    """読みを n_parts 個の空でない部分に分割する全パターンを列挙する."""
    if n_parts <= 0 or len(reading) < n_parts:
        return
    if n_parts == 1:
        if len(reading) <= max_part_len:
            yield (reading,)
        return
    for cut in range(1, min(max_part_len, len(reading) - n_parts + 1) + 1):
        for rest in split_reading(reading[cut:], n_parts - 1, max_part_len):
            yield (reading[:cut], *rest)


def extract_single_kanji_readings(
    dic: Dictionary,
    max_surface_len: int = 4,
    max_part_len: int = 6,
    extracted_cost: int = DEFAULT_EXTRACTED_COST,
) -> list[Entry]:
    """辞書から単漢字の新しい読みを抽出する.

    全漢字の複合語のうち, 既知の読みで説明できない文字がちょうど 1 つあり,
    かつその割り当てが一意に決まるものだけを採用する.

    Args:
        max_surface_len: 対象とする複合語の最大文字数
        max_part_len: 1 文字に割り当てる読みの最大長
        extracted_cost: 抽出したエントリに付ける生起コスト

    Returns:
        新規に判明した単漢字エントリ (既存の読みは含まない)
    """
    known: dict[str, set[str]] = {}
    for entry in dic.all_entries():
        if len(entry.surface) == 1 and is_kanji(entry.surface):
            known.setdefault(entry.surface, set()).add(entry.reading)

    discovered: dict[str, set[str]] = {}

    for entry in dic.all_entries():
        surface = entry.surface
        n = len(surface)
        if not (2 <= n <= max_surface_len):
            continue
        if not all(is_kanji(ch) for ch in surface):
            continue
        if not entry.reading:
            continue

        candidates: set[tuple[str, str]] = set()
        fully_explained = False

        for parts in split_reading(entry.reading, n, max_part_len):
            unknown: list[tuple[str, str]] = [
                (ch, part)
                for ch, part in zip(surface, parts, strict=True)
                if part not in known.get(ch, ())
            ]
            if not unknown:
                fully_explained = True
                break
            if len(unknown) == 1:
                candidates.add(unknown[0])

        # 既知の読みだけで説明できるなら, 新しい情報はない
        if fully_explained:
            continue
        # 割り当てが一意でないなら採用しない
        if len(candidates) != 1:
            continue

        ch, reading = next(iter(candidates))
        if not reading:
            continue
        discovered.setdefault(ch, set()).add(reading)

    return [
        Entry(
            surface=ch,
            reading=reading,
            pos="補完-単漢字",
            entry_id=0,
            cost=extracted_cost,
            source="aug",
        )
        for ch in sorted(discovered)
        for reading in sorted(discovered[ch])
    ]


def augment_dictionary(
    dic: Dictionary,
    max_surface_len: int = 4,
    max_part_len: int = 6,
    extracted_cost: int = DEFAULT_EXTRACTED_COST,
    rounds: int = 1,
) -> tuple[Dictionary, int]:
    """抽出した単漢字の読みを加えた辞書を返す.

    Args:
        rounds: 抽出を繰り返す回数. 新たに判明した読みが次の抽出の手掛かりになる.

    Returns:
        (補完後の辞書, 追加したエントリ数)
    """
    entries = dic.all_entries()
    total_added = 0
    keep_pos = dic.keep_pos  # ★補完でも重複排除の方針を引き継ぐ（R-27）

    for _ in range(rounds):
        new = extract_single_kanji_readings(dic, max_surface_len, max_part_len, extracted_cost)
        if not new:
            break
        entries = entries + new
        total_added += len(new)
        dic = Dictionary.from_entries(entries, keep_pos=keep_pos)

    return dic, total_added
