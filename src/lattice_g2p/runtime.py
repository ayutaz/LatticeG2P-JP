"""ONNX Runtime での推論.

★CRF / Viterbi はホスト側 (この Python コード) に残る.
Rust / C++ へ移植するときも, この分担をそのまま持っていける
(docs/architecture.md §7-2).

★ノードスコアリングは必ず一括で行う (R-12). 1 文につき
  - 文脈エンコーダ 1 回
  - 読みエンコーダ 1 回 (★未キャッシュの読みだけ)
  - ノードスコアラ 1 回 (全ノードまとめて)

★読み表現をキャッシュする (docs/architecture.md §4-3).
読みは入力文に依存しないので, 一度計算すれば使い回せる.
キャッシュしない実装では 91 文字の文で読みエンコーダが 25 ms を占めていた.
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from lattice_g2p.crf import viterbi_numpy
from lattice_g2p.decode import path_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.lattice import Node, build_lattice
from lattice_g2p.scorer.neural_base import (
    DICT_COST_SCALE,
    EMPTY_READING_ID,
    SOURCE_IDS,
)
from lattice_g2p.vocab import CharVocab


class OnnxG2P:
    """ONNX の 2 グラフ + ホスト側 Viterbi による読み推定."""

    def __init__(
        self, model_dir: Path, dic: Dictionary, num_threads: int | None = None
    ) -> None:
        model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads is not None:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1

        self.context_session = ort.InferenceSession(
            str(model_dir / "context_encoder.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.scorer_session = ort.InferenceSession(
            str(model_dir / "node_scorer.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.reading_session = ort.InferenceSession(
            str(model_dir / "reading_encoder.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        transition_path = model_dir / "transition.onnx"
        self.transition_session = (
            ort.InferenceSession(
                str(transition_path), opts, providers=["CPUExecutionProvider"]
            )
            if transition_path.exists()
            else None
        )
        """CRF の遷移項 (M6). 持たないモデルでは None.

        ★キャッシュしない. 読みと違って「そのラティスのノードの組」に依存する.
        """
        self.text_vocab = CharVocab.load(model_dir / "text_vocab.json")
        self.kana_vocab = CharVocab.load(model_dir / "kana_vocab.json")
        self.dic = dic
        self._reading_cache: dict[str, np.ndarray] = {}
        """読み -> 表現. ★辞書の読みは有限なので, 使ううちに頭打ちになる."""

    @property
    def cache_size(self) -> int:
        return len(self._reading_cache)

    # --- 読み表現（キャッシュつき） ---

    def encode_readings(self, readings: list[str]) -> np.ndarray:
        """読みの並びを (N, H) にする. キャッシュにないものだけ計算する."""
        missing = sorted({r for r in readings if r not in self._reading_cache})
        if missing:
            ids = [self.kana_vocab.encode(r) or [EMPTY_READING_ID] for r in missing]
            width = max(len(x) for x in ids)
            batch = np.zeros((len(ids), width), dtype=np.int64)
            mask = np.zeros((len(ids), width), dtype=bool)
            for i, x in enumerate(ids):
                batch[i, : len(x)] = x
                mask[i, : len(x)] = True
            vecs = self.reading_session.run(
                ["reading_vec"], {"reading_ids": batch, "reading_mask": mask}
            )[0]
            for reading, vec in zip(missing, vecs, strict=True):
                self._reading_cache[reading] = vec

        return np.stack([self._reading_cache[r] for r in readings])

    # --- 特徴量 ---

    def _node_inputs(self, nodes: list[Node], length: int) -> dict[str, np.ndarray]:
        """NeuralScorer.build_features と同じものを numpy で作る.

        ★ここがずれると ONNX と PyTorch のスコアが一致しなくなる.
        tests/test_runtime.py が両者を突き合わせている.
        """
        del length  # span は境界だけで表す
        n = len(nodes)
        return {
            "span_starts": np.array([n_.start for n_ in nodes], dtype=np.int64),
            "span_ends": np.array([n_.end for n_ in nodes], dtype=np.int64),
            "reading_vec": self.encode_readings([n_.reading for n_ in nodes]),
            "node_lengths": np.array([n_.end - n_.start for n_ in nodes], dtype=np.float32),
            "source_ids": np.array(
                [SOURCE_IDS.get(n_.source, 0) for n_ in nodes], dtype=np.int64
            ),
            "is_unk": np.array(
                [1.0 if n_.entry_id < 0 else 0.0 for n_ in nodes], dtype=np.float32
            ),
            "batch_index": np.zeros(n, dtype=np.int64),
            "dict_prior": np.array(
                [-n_.cost / DICT_COST_SCALE for n_ in nodes], dtype=np.float32
            ),
        }

    # --- 推論 ---

    def encode_context(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self.text_vocab.encode(text)], dtype=np.int64)
        mask = np.ones(ids.shape, dtype=bool)
        context = self.context_session.run(["context"], {"ids": ids, "mask": mask})[0]
        return context, mask

    def score_nodes(self, text: str, nodes: list[Node]) -> np.ndarray:
        if not nodes:
            return np.zeros(0, dtype=np.float32)
        context, context_mask = self.encode_context(text)
        return self._run_scorer(context, context_mask, nodes, len(text))

    def _run_scorer(
        self,
        context: np.ndarray,
        context_mask: np.ndarray,
        nodes: list[Node],
        length: int,
    ) -> np.ndarray:
        inputs = self._node_inputs(nodes, length)
        inputs["context"] = context
        inputs["context_mask"] = context_mask
        return self.scorer_session.run(["scores"], inputs)[0]

    def transitions(self, nodes: list[Node]) -> np.ndarray | None:
        """(N, N) の遷移スコア. 遷移項を持たないモデルでは None."""
        if self.transition_session is None or not nodes:
            return None
        return self.transition_session.run(
            ["transitions"],
            {
                "right_ids": np.array([n.right_id for n in nodes], dtype=np.int64),
                "left_ids": np.array([n.left_id for n in nodes], dtype=np.int64),
            },
        )[0]

    def g2p(self, text: str) -> str:
        return self.g2p_with_timing(text)[0]

    def g2p_with_timing(self, text: str) -> tuple[str, dict[str, float]]:
        """読みと, 区間ごとの所要時間 (ミリ秒) を返す."""
        t0 = time.perf_counter()
        lattice = build_lattice(text, self.dic)
        t1 = time.perf_counter()

        context, context_mask = self.encode_context(text)
        t2 = time.perf_counter()

        if lattice.nodes:
            scores = self._run_scorer(context, context_mask, lattice.nodes, len(text))
            transitions = self.transitions(lattice.nodes)
        else:
            scores = np.zeros(0, dtype=np.float32)
            transitions = None
        t3 = time.perf_counter()

        reading = path_reading(lattice, viterbi_numpy(lattice, scores, transitions))
        t4 = time.perf_counter()

        return reading, {
            "lattice": (t1 - t0) * 1000,
            "context": (t2 - t1) * 1000,
            "scoring": (t3 - t2) * 1000,
            "viterbi": (t4 - t3) * 1000,
            "total": (t4 - t0) * 1000,
        }


class OnnxScorer:
    """OnnxG2P を NodeScorer として使うための薄いラッパ.

    ★評価スクリプトを PyTorch 版とまったく同じ経路で走らせるためにある.
    こうしないと「量子化で落ちたのか評価コードが違うのか」が分からなくなる.

    torch への変換が入るので, レイテンシの計測にはこれを使わないこと
    (計測は OnnxG2P.g2p_with_timing を使う).
    """

    def __init__(self, model_dir: Path, dic: Dictionary, num_threads: int | None = None) -> None:
        self.runtime = OnnxG2P(model_dir, dic, num_threads)

    def score(self, text: str, nodes: list[Node]):  # noqa: ANN201
        import torch

        return torch.from_numpy(self.runtime.score_nodes(text, nodes).copy())

    def transitions(self, nodes: list[Node]):  # noqa: ANN201
        import torch

        out = self.runtime.transitions(nodes)
        return None if out is None else torch.from_numpy(out.copy())
