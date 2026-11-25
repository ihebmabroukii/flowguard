"""Tests for our NumPy models.

The gradient checks are the key ones: they compare our hand-written derivatives with numerical
derivatives (central differences). A relative error around 1e-7 means backpropagation is right.
"""

import numpy as np
import pytest

from flowguard.models.scratch import IsolationForest, MLPClassifier, SoftmaxRegression
from flowguard.models.scratch.core import Adam, log_softmax, softmax, weighted_cross_entropy
from flowguard.models.scratch.isolation_forest import average_path_length


def numerical_grads(model, X, y, w, eps=1e-6):
    grads = []
    for p in model._params():
        g = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            up, _ = model._loss_and_grads(X, y, w, training=False)
            p[idx] = old - eps
            down, _ = model._loss_and_grads(X, y, w, training=False)
            p[idx] = old
            g[idx] = (up - down) / (2 * eps)
        grads.append(g)
    return grads


def rel_error(a, b):
    return np.abs(a - b).max() / max(1e-12, np.abs(a).max() + np.abs(b).max())


def blobs(n_per_class=150, n_classes=3, n_features=4, seed=0, spread=4.0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(k * spread, 1.0, (n_per_class, n_features)) for k in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X, y


def xor(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


# ----------------------------------------------------------------------------- core

def test_softmax_is_stable_and_normalized():
    z = np.array([[1000.0, 1000.0], [-1000.0, 0.0]])
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1) and np.allclose(p[0], [0.5, 0.5])
    assert np.isfinite(log_softmax(z)).all()


def test_cross_entropy_matches_definition():
    logits = np.array([[2.0, 0.0], [0.0, 1.0]])
    y, w = np.array([0, 0]), np.array([1.0, 3.0])
    loss, _ = weighted_cross_entropy(logits, y, w)
    per_row = -np.log(softmax(logits)[[0, 1], [0, 0]])
    assert loss == pytest.approx((1 * per_row[0] + 3 * per_row[1]) / 4)


def test_adam_minimizes_a_quadratic():
    x = np.array([5.0, -3.0])
    opt = Adam([x], lr=0.1)
    for _ in range(500):
        opt.step([2 * (x - np.array([1.0, 2.0]))])
    assert np.allclose(x, [1.0, 2.0], atol=1e-3)


# ----------------------------------------------------------------------------- softmax regression

def test_softmax_regression_gradient_check():
    rng = np.random.default_rng(0)
    X, y, w = rng.normal(size=(12, 5)), rng.integers(0, 3, 12), rng.uniform(0.5, 2, 12)
    model = SoftmaxRegression(alpha=0.01)
    model._init_params(5, 3, rng)
    model.W_[...] = rng.normal(size=model.W_.shape)
    model.b_[...] = rng.normal(size=model.b_.shape)
    _, analytic = model._loss_and_grads(X, y, w)
    for a, n in zip(analytic, numerical_grads(model, X, y, w)):
        assert rel_error(a, n) < 1e-6


def test_softmax_regression_learns_blobs_and_matches_sklearn():
    X, y = blobs()
    X_val, y_val = blobs(seed=1)
    ours = SoftmaxRegression(lr=0.05, epochs=200, batch_size=64, alpha=1e-4, patience=10).fit(
        X, y, eval_set=[(X_val, y_val)])
    assert (ours.predict(X_val) == y_val).mean() > 0.97
    assert ours.history_[0]["val_loss"] > ours.best_loss_

    LogisticRegression = pytest.importorskip("sklearn.linear_model").LogisticRegression
    ref = LogisticRegression(C=1 / (1e-4 * len(X)), max_iter=5000).fit(X, y)
    agreement = (ours.predict(X_val) == ref.predict(X_val)).mean()
    assert agreement > 0.97


def test_softmax_regression_cannot_learn_xor():
    X, y = xor()
    acc = (SoftmaxRegression(lr=0.05, epochs=50, batch_size=64).fit(X, y).predict(X) == y).mean()
    assert acc < 0.7   # a linear model can't separate XOR; the MLP test below can


def test_sample_weights_are_scale_invariant():
    X, y = blobs()
    a = SoftmaxRegression(epochs=5, random_state=3).fit(X, y)
    b = SoftmaxRegression(epochs=5, random_state=3).fit(X, y, sample_weight=np.full(len(y), 7.0))
    assert np.allclose(a.W_, b.W_)


def test_class_weights_raise_minority_recall():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (950, 2)), rng.normal(1.2, 1, (50, 2))])
    y = np.r_[np.zeros(950, int), np.ones(50, int)]
    w = np.where(y == 1, 950 / 50, 1.0)
    plain = SoftmaxRegression(epochs=40, batch_size=100).fit(X, y)
    weighted = SoftmaxRegression(epochs=40, batch_size=100).fit(X, y, sample_weight=w)
    recall = lambda m: (m.predict(X)[y == 1] == 1).mean()
    assert recall(weighted) > recall(plain) + 0.3


# ----------------------------------------------------------------------------- MLP

def _small_mlp(alpha=0.01, seed=0):
    rng = np.random.default_rng(seed)
    model = MLPClassifier(hidden=(6, 4), dropout=0.0, alpha=alpha)
    model._init_params(5, 3, rng)
    # Random biases keep pre-activations away from exactly 0, where ReLU has no derivative and
    # a finite-difference check would compare against a meaningless number.
    for b in model.biases_:
        b[...] = rng.normal(0, 0.5, size=b.shape)
    return model, rng


def test_mlp_gradient_check():
    model, rng = _small_mlp()
    X, y, w = rng.normal(size=(10, 5)), rng.integers(0, 3, 10), rng.uniform(0.5, 2, 10)
    _, analytic = model._loss_and_grads(X, y, w)
    for a, n in zip(analytic, numerical_grads(model, X, y, w)):
        assert rel_error(a, n) < 1e-5


def test_mlp_input_gradient_check():
    model, rng = _small_mlp(alpha=0.0)
    X, y = rng.normal(size=(4, 5)), rng.integers(0, 3, 4)
    analytic = model.input_gradient(X, y)
    eps = 1e-6
    for i in range(len(X)):
        def flow_loss(x):
            return -log_softmax(model._logits(x[None, :]))[0, y[i]]
        for j in range(X.shape[1]):
            up, down = X[i].copy(), X[i].copy()
            up[j] += eps
            down[j] -= eps
            numeric = (flow_loss(up) - flow_loss(down)) / (2 * eps)
            assert analytic[i, j] == pytest.approx(numeric, rel=1e-4, abs=1e-8)


def test_mlp_learns_xor():
    X, y = xor()
    X_val, y_val = xor(seed=1)
    model = MLPClassifier(hidden=(32, 16), dropout=0.0, lr=0.01, epochs=200, batch_size=64,
                          patience=20).fit(X, y, eval_set=[(X_val, y_val)])
    assert (model.predict(X_val) == y_val).mean() > 0.95


def test_mlp_with_dropout_trains_and_is_deterministic_per_seed():
    X, y = blobs()
    a = MLPClassifier(hidden=(16,), dropout=0.3, lr=0.01, epochs=20, batch_size=64, random_state=1).fit(X, y)
    b = MLPClassifier(hidden=(16,), dropout=0.3, lr=0.01, epochs=20, batch_size=64, random_state=1).fit(X, y)
    assert all(np.array_equal(p, q) for p, q in zip(a._params(), b._params()))
    assert (a.predict(X) == y).mean() > 0.9
    assert np.allclose(a.predict_proba(X).sum(axis=1), 1)


def test_early_stopping_restores_best_epoch():
    X, y = blobs(n_per_class=40)
    X_val, y_val = blobs(n_per_class=40, seed=5)
    model = MLPClassifier(hidden=(64, 64), dropout=0.0, lr=0.05, epochs=100, batch_size=16,
                          patience=3).fit(X, y, eval_set=[(X_val, y_val)])
    best = min(r["val_loss"] for r in model.history_)
    assert model.loss(X_val, y_val) == pytest.approx(best, rel=1e-9)
    assert model.n_epochs_ < 100


def test_get_params_lists_constructor_arguments():
    params = MLPClassifier(hidden=(8,)).get_params()
    assert params["hidden"] == (8,) and params["dropout"] == 0.2 and "lr" in params


# ----------------------------------------------------------------------------- Isolation Forest

def test_average_path_length_values():
    c = average_path_length([0, 1, 2, 256])
    assert c[:3].tolist() == [0.0, 0.0, 1.0]
    assert c[3] == pytest.approx(2 * (np.log(255) + 0.5772156649015329) - 2 * 255 / 256)


def test_isolation_forest_scores_outliers_higher():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, (2000, 3))
    outliers = rng.normal(8, 0.5, (20, 3))
    forest = IsolationForest(n_estimators=100, max_samples=256, random_state=0).fit(normal)
    s_norm, s_out = forest.score_samples(normal), forest.score_samples(outliers)
    assert s_out.min() > np.quantile(s_norm, 0.99)
    assert s_norm.mean() < 0.5 < s_out.mean()
    assert all(t.depth <= forest.height_limit_ for t in forest.trees_)
    assert forest.height_limit_ == 8


def test_isolation_forest_agrees_with_sklearn_ranking():
    pd = pytest.importorskip("pandas")
    from sklearn.ensemble import IsolationForest as SkIsolationForest

    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (1500, 4)), rng.uniform(-6, 6, (100, 4))])
    ours = IsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X)
    ref = -SkIsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X)
    assert pd.Series(ours).corr(pd.Series(ref), method="spearman") > 0.9


def test_isolation_forest_edge_cases():
    forest = IsolationForest(n_estimators=5, random_state=0).fit(np.ones((50, 2)))  # nothing to split
    assert np.allclose(forest.score_samples(np.ones((3, 2))), forest.score_samples(np.zeros((3, 2))))
    with pytest.raises(ValueError):
        IsolationForest().fit(np.array([[1.0, np.nan], [2.0, 3.0]]))
