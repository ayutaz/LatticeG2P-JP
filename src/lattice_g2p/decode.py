"""経路から読み列を取り出す."""

from lattice_g2p.lattice import Lattice


def path_reading(lattice: Lattice, path: list[int]) -> str:
    """経路全体の読み列.

    句読点などの空読みノードは自然に何も寄与しない.
    """
    return "".join(lattice.nodes[i].reading for i in path)


def span_reading(lattice: Lattice, path: list[int], start: int, end: int) -> str:
    """target span の読み.

    規約: span と少しでも重なるノードの読みを, 経路上の順に連結する.
    span をまたぐノードは丸ごと含める.

    辞書の分割単位が評価対象の span と一致するとは限らないため
    (docs/evaluation.md §1-1). 「形態素分割を正解にしない」という
    手法の思想と一貫している.
    """
    return "".join(
        lattice.nodes[i].reading
        for i in path
        if lattice.nodes[i].start < end and lattice.nodes[i].end > start
    )
