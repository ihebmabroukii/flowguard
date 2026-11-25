"""Fully connected neural network classifier, with backpropagation written by hand.

Architecture (default hidden = [256, 128, 64]):
    x -> [Dense -> ReLU -> Dropout] x len(hidden) -> Dense -> softmax

Forward, for hidden layer l:   z_l = a_{l-1} W_l + b_l,   a_l = dropout(relu(z_l))
Backward, from G = dL/dlogits:
    dW_L = a_{L-1}^T G + alpha W_L,  db_L = sum(G),  da_{L-1} = G W_L^T
    for each hidden layer, going down:
        da_l <- da_l * mask_l            (inverted dropout: the same mask as the forward pass)
        dz_l  = da_l * 1{z_l > 0}        (ReLU derivative)
        dW_l  = a_{l-1}^T dz_l + alpha W_l,  db_l = sum(dz_l),  da_{l-1} = dz_l W_l^T
The last da is the gradient with respect to the input, which the adversarial lab uses for PGD.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from flowguard.models.scratch.core import GradientClassifier, weighted_cross_entropy


class MLPClassifier(GradientClassifier):
    def __init__(self, hidden: Sequence[int] = (256, 128, 64), dropout: float = 0.2, lr: float = 1e-3,
                 epochs: int = 20, batch_size: int = 2048, alpha: float = 1e-5, patience: int = 3,
                 tol: float = 1e-4, random_state: int = 0):
        super().__init__(lr=lr, epochs=epochs, batch_size=batch_size, alpha=alpha,
                         patience=patience, tol=tol, random_state=random_state)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden = tuple(int(h) for h in hidden)
        self.dropout = dropout

    def _init_params(self, n_features: int, n_classes: int, rng: np.random.Generator) -> None:
        sizes = [n_features, *self.hidden, n_classes]
        # He initialization: keeps the variance of ReLU activations stable from layer to layer.
        self.weights_ = [rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
                         for fan_in, fan_out in zip(sizes[:-1], sizes[1:])]
        self.biases_ = [np.zeros(fan_out) for fan_out in sizes[1:]]

    def _params(self) -> list[np.ndarray]:
        # Interleaved W0, b0, W1, b1, ... (gradients are returned in the same order)
        return [p for pair in zip(self.weights_, self.biases_) for p in pair]

    def _forward(self, X: np.ndarray, rng: np.random.Generator | None = None,
                 training: bool = False) -> tuple[np.ndarray, list[tuple[Any, Any, Any]]]:
        cache = []
        a = X
        for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
            z = a @ W + b
            h = np.maximum(z, 0.0)
            mask = None
            if training and self.dropout > 0:
                mask = (rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)
                h = h * mask
            cache.append((a, z, mask))
            a = h
        logits = a @ self.weights_[-1] + self.biases_[-1]
        cache.append((a, None, None))
        return logits, cache

    def _backward(self, G: np.ndarray, cache: list) -> tuple[list[np.ndarray], np.ndarray]:
        n_layers = len(self.weights_)
        dW: list[np.ndarray] = [None] * n_layers  # type: ignore[list-item]
        db: list[np.ndarray] = [None] * n_layers  # type: ignore[list-item]

        a_prev = cache[-1][0]
        dW[-1] = a_prev.T @ G + self.alpha * self.weights_[-1]
        db[-1] = G.sum(axis=0)
        da = G @ self.weights_[-1].T
        for layer in reversed(range(n_layers - 1)):
            a_prev, z, mask = cache[layer]
            if mask is not None:
                da = da * mask
            dz = da * (z > 0)
            dW[layer] = a_prev.T @ dz + self.alpha * self.weights_[layer]
            db[layer] = dz.sum(axis=0)
            da = dz @ self.weights_[layer].T
        grads = [g for pair in zip(dW, db) for g in pair]
        return grads, da

    def _logits(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X, training=False)[0]

    def _loss_and_grads(self, X, y, w, rng=None, training=False):
        logits, cache = self._forward(X, rng, training)
        loss, G = weighted_cross_entropy(logits, y, w)
        loss += 0.5 * self.alpha * float(sum((W ** 2).sum() for W in self.weights_))
        grads, _ = self._backward(G, cache)
        return loss, grads

    def input_gradient(self, X: Any, y: Any) -> np.ndarray:
        """d(loss of each flow) / d(its features), one row per flow, dropout off.

        Each row is the gradient of that flow's own cross-entropy, which is what a per-flow
        evasion attack (PGD) follows. Shape matches X.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        logits, cache = self._forward(X, training=False)
        _, G = weighted_cross_entropy(logits, y, np.ones(len(y)))
        _, dX = self._backward(G * len(y), cache)  # undo the 1/n of the mean loss
        return dX
