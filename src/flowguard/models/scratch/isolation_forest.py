"""Isolation Forest (Liu, Ting and Zhou, ICDM 2008), written from scratch.

Idea: anomalies are few and different, so random axis-aligned cuts isolate them in fewer steps.

Training, per tree:   draw `max_samples` rows (psi). Split recursively: pick a random feature
                      that is not constant in the node, and a threshold uniformly between its min
                      and max. Stop at depth ceil(log2(psi)) or when one row is left.
Path length h(x):     edges from the root to x's leaf, plus c(leaf size) for the unbuilt subtree.
Normalizer c(n):      average path length of an unsuccessful search in a binary search tree,
                      c(n) = 2 H(n-1) - 2 (n-1) / n,  H(i) ~ ln(i) + 0.5772156649,  c(2) = 1.
Anomaly score:        s(x) = 2 ** (-mean_trees h(x) / c(psi)), in (0, 1].
                      ~1 means clearly anomalous, well below 0.5 means normal.

Unlike scikit-learn (which returns the negated score), higher means more anomalous here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EULER_GAMMA = 0.5772156649015329
SCORE_BATCH = 65_536


def average_path_length(n: Any) -> np.ndarray:
    """c(n) for an array of node sizes."""
    n = np.asarray(n, dtype=np.float64)
    c = np.zeros_like(n)
    big = n > 2
    c[big] = 2.0 * (np.log(n[big] - 1.0) + EULER_GAMMA) - 2.0 * (n[big] - 1.0) / n[big]
    c[n == 2] = 1.0
    return c


@dataclass
class IsolationTree:
    """Flat arrays, one entry per node. feature == -1 marks a leaf."""
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    size: np.ndarray

    @property
    def depth(self) -> int:
        depths = np.zeros(len(self.feature), dtype=int)
        for node in range(len(self.feature)):  # children always come after their parent
            if self.feature[node] >= 0:
                depths[self.left[node]] = depths[self.right[node]] = depths[node] + 1
        return int(depths.max())


class IsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, random_state: int = 0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"n_estimators": self.n_estimators, "max_samples": self.max_samples,
                "random_state": self.random_state}

    def fit(self, X: Any, y: Any = None) -> IsolationForest:
        X = np.asarray(X, dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError("IsolationForest needs complete data; impute first (features.pipeline does).")
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        psi = int(min(self.max_samples, n))
        if psi < 2:
            raise ValueError("Need at least 2 rows to build isolation trees")
        self.max_samples_ = psi
        self.height_limit_ = int(np.ceil(np.log2(psi)))
        self.n_features_in_ = X.shape[1]
        self.trees_ = [self._build_tree(X[rng.choice(n, psi, replace=False)], rng)
                       for _ in range(self.n_estimators)]
        return self

    def _build_tree(self, X: np.ndarray, rng: np.random.Generator) -> IsolationTree:
        feature: list[int] = []
        threshold: list[float] = []
        left: list[int] = []
        right: list[int] = []
        size: list[int] = []

        def new_node() -> int:
            for column, value in ((feature, -1), (threshold, 0.0), (left, -1), (right, -1), (size, 0)):
                column.append(value)
            return len(feature) - 1

        stack = [(new_node(), np.arange(len(X)), 0)]
        while stack:
            node, rows, depth = stack.pop()
            size[node] = len(rows)
            if depth >= self.height_limit_ or len(rows) <= 1:
                continue
            sub = X[rows]
            lo, hi = sub.min(axis=0), sub.max(axis=0)
            candidates = np.flatnonzero(hi > lo)
            if len(candidates) == 0:  # all rows identical: cannot be split further
                continue
            f = int(rng.choice(candidates))
            t = float(rng.uniform(lo[f], hi[f]))
            goes_left = sub[:, f] < t
            l_node, r_node = new_node(), new_node()
            feature[node], threshold[node], left[node], right[node] = f, t, l_node, r_node
            stack.append((l_node, rows[goes_left], depth + 1))
            stack.append((r_node, rows[~goes_left], depth + 1))

        return IsolationTree(np.array(feature), np.array(threshold), np.array(left),
                             np.array(right), np.array(size))

    @staticmethod
    def _path_length(tree: IsolationTree, X: np.ndarray) -> np.ndarray:
        """Vectorized descent: every row moves one level per loop iteration."""
        node = np.zeros(len(X), dtype=np.int64)
        depth = np.zeros(len(X))
        active = np.flatnonzero(tree.feature[node] >= 0)
        while len(active):
            nd = node[active]
            goes_left = X[active, tree.feature[nd]] < tree.threshold[nd]
            node[active] = np.where(goes_left, tree.left[nd], tree.right[nd])
            depth[active] += 1
            active = active[tree.feature[node[active]] >= 0]
        return depth + average_path_length(tree.size[node])

    def mean_path_length(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X))
        for start in range(0, len(X), SCORE_BATCH):
            batch = X[start:start + SCORE_BATCH]
            out[start:start + SCORE_BATCH] = np.mean([self._path_length(t, batch) for t in self.trees_], axis=0)
        return out

    def score_samples(self, X: Any) -> np.ndarray:
        """Anomaly score in (0, 1]; higher = more anomalous."""
        return 2.0 ** (-self.mean_path_length(X) / average_path_length([self.max_samples_])[0])
