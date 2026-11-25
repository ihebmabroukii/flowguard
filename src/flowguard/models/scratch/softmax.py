"""Multi-class logistic regression (softmax regression), written from scratch.

Model:        p(y = k | x) = softmax(x W + b)_k
Loss:         weighted cross-entropy + (alpha / 2) * ||W||^2
Gradients:    G = dL/dz  (see core.weighted_cross_entropy)
              dL/dW = X^T G + alpha * W
              dL/db = sum over rows of G
Optimizer:    Adam on mini-batches, early stopping on validation loss.
"""

from __future__ import annotations

import numpy as np

from flowguard.models.scratch.core import GradientClassifier, weighted_cross_entropy


class SoftmaxRegression(GradientClassifier):
    def __init__(self, lr: float = 0.01, epochs: int = 30, batch_size: int = 4096, alpha: float = 1e-4,
                 patience: int = 3, tol: float = 1e-4, random_state: int = 0):
        super().__init__(lr=lr, epochs=epochs, batch_size=batch_size, alpha=alpha,
                         patience=patience, tol=tol, random_state=random_state)

    def _init_params(self, n_features: int, n_classes: int, rng: np.random.Generator) -> None:
        # The loss is convex, so zero initialization is fine (and fully reproducible).
        self.W_ = np.zeros((n_features, n_classes))
        self.b_ = np.zeros(n_classes)

    def _params(self) -> list[np.ndarray]:
        return [self.W_, self.b_]

    def _logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W_ + self.b_

    def _loss_and_grads(self, X, y, w, rng=None, training=False):
        loss, G = weighted_cross_entropy(self._logits(X), y, w)
        loss += 0.5 * self.alpha * float((self.W_ ** 2).sum())
        dW = X.T @ G + self.alpha * self.W_
        db = G.sum(axis=0)
        return loss, [dW, db]

    @property
    def coef_(self) -> np.ndarray:
        """(n_classes, n_features), the same layout as scikit-learn, for comparison and plots."""
        return self.W_.T
