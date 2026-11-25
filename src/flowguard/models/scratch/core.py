"""Building blocks shared by our gradient-trained models: softmax, loss, Adam, training loop."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

PREDICT_BATCH = 65_536


# --------------------------------------------------------------------------- softmax and loss

def log_softmax(z: np.ndarray) -> np.ndarray:
    """log(softmax(z)) row by row, computed stably: z - max - log(sum(exp(z - max)))."""
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def weighted_cross_entropy(logits: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted mean cross-entropy and its gradient with respect to the logits.

        L = sum_i w_i * (-log p[i, y_i]) / sum_i w_i
        dL/dz[i, k] = w_i * (p[i, k] - 1{k = y_i}) / sum_i w_i
    """
    n = len(y)
    rows = np.arange(n)
    logp = log_softmax(logits)
    total = w.sum()
    loss = float(-(w * logp[rows, y]).sum() / total)
    grad = np.exp(logp)
    grad[rows, y] -= 1.0
    grad *= (w / total)[:, None]
    return loss, grad


# --------------------------------------------------------------------------- optimizer

class Adam:
    """Adam (Kingma and Ba, 2015). Updates the parameter arrays in place."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            p -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)


# --------------------------------------------------------------------------- training loop

class GradientClassifier:
    """Mini-batch training loop with early stopping, shared by SoftmaxRegression and the MLP.

    Subclasses implement:
        _init_params(n_features, n_classes, rng)
        _params() -> list of parameter arrays (updated in place by Adam)
        _logits(X) -> raw class scores, no dropout
        _loss_and_grads(X, y, w, rng, training) -> (loss, list of gradients matching _params())

    Labels must be integers 0..k-1 (FlowClassifier encodes family names before calling fit).
    """

    def __init__(self, lr: float = 1e-3, epochs: int = 20, batch_size: int = 2048, alpha: float = 0.0,
                 patience: int = 3, tol: float = 1e-4, random_state: int = 0):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.patience = patience
        self.tol = tol
        self.random_state = random_state

    # sklearn-style parameter listing, used by FlowClassifier.summary() and MLflow
    def get_params(self, deep: bool = False) -> dict[str, Any]:
        names = [p for c in type(self).__mro__ if c is not object
                 for p in inspect.signature(c.__init__).parameters if p != "self"]
        return {n: getattr(self, n) for n in dict.fromkeys(names) if hasattr(self, n)}

    def fit(self, X: Any, y: Any, sample_weight: Any = None, eval_set: Any = None,
            verbose: bool = False) -> GradientClassifier:
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int64)
        n = len(X)
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self.n_classes_ = int(y.max()) + 1
        self.n_features_in_ = X.shape[1]
        self._init_params(X.shape[1], self.n_classes_, rng)
        params = self._params()
        optimizer = Adam(params, lr=self.lr)

        X_val = y_val = None
        if eval_set:
            X_val, y_val = np.asarray(eval_set[0][0]), np.asarray(eval_set[0][1], dtype=np.int64)

        self.history_: list[dict[str, float]] = []
        best, best_params, bad_epochs = np.inf, [p.copy() for p in params], 0
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n)
            weighted_loss = weight_seen = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                loss, grads = self._loss_and_grads(
                    X[idx].astype(np.float64), y[idx], w[idx], rng, training=True)
                optimizer.step(grads)
                weighted_loss += loss * w[idx].sum()
                weight_seen += w[idx].sum()
            record = {"epoch": epoch, "train_loss": weighted_loss / weight_seen}
            monitored = record["train_loss"]
            if X_val is not None:
                record["val_loss"] = self.loss(X_val, y_val)
                monitored = record["val_loss"]
            self.history_.append(record)
            if verbose:
                print(record)
            if monitored < best - self.tol:
                best, best_params, bad_epochs = monitored, [p.copy() for p in params], 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        for p, saved in zip(params, best_params):
            p[...] = saved  # restore the weights of the best epoch
        self.best_loss_ = float(best)
        self.n_epochs_ = len(self.history_)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        X = np.asarray(X)
        return np.vstack([self._logits(X[i:i + PREDICT_BATCH].astype(np.float64))
                          for i in range(0, len(X), PREDICT_BATCH)])

    def predict_proba(self, X: Any) -> np.ndarray:
        return softmax(self.decision_function(X))

    def predict(self, X: Any) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1)

    def loss(self, X: Any, y: Any) -> float:
        """Unweighted mean cross-entropy (no regularization term), in batches."""
        y = np.asarray(y, dtype=np.int64)
        logits = self.decision_function(X)
        return float(-log_softmax(logits)[np.arange(len(y)), y].mean())

    # subclass hooks -------------------------------------------------------
    def _init_params(self, n_features: int, n_classes: int, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def _params(self) -> list[np.ndarray]:
        raise NotImplementedError

    def _logits(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _loss_and_grads(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                        rng: np.random.Generator | None, training: bool) -> tuple[float, list[np.ndarray]]:
        raise NotImplementedError
