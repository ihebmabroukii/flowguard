"""Drift statistics, written from scratch (NumPy only)."""

import numpy as np
import pytest

from flowguard.eval.drift import ks_statistic, psi, roc_auc, spearman


def test_ks_statistic():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert ks_statistic(a, a) == 0.0
    assert ks_statistic(a, rng.normal(0, 1, 5000)) < 0.05
    assert ks_statistic(a, a + 100) == 1.0
    # N(0,1) vs N(1,1): D = 2 * Phi(0.5) - 1 ~ 0.383
    assert ks_statistic(rng.normal(0, 1, 50_000), rng.normal(1, 1, 50_000)) == pytest.approx(0.383, abs=0.02)
    assert ks_statistic([1, 2, np.nan], [1, 2, np.inf]) == 0.0
    assert np.isnan(ks_statistic([], [1.0]))


def test_ks_is_invariant_to_monotone_transforms():
    rng = np.random.default_rng(1)
    a, b = rng.exponential(1, 3000), rng.exponential(2, 3000)
    assert ks_statistic(a, b) == pytest.approx(ks_statistic(np.log1p(a), np.log1p(b)))


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20_000)
    assert psi(a, rng.normal(0, 1, 20_000)) < 0.01
    assert psi(a, rng.normal(1, 1, 20_000)) > 0.25
    constant = np.zeros(1000)
    assert psi(constant, constant) == pytest.approx(0.0)        # repeated quantiles merge into one bin
    assert psi(constant, np.ones(1000)) > 1.0


def test_roc_auc():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert roc_auc(y, [0.1, 0.4, 0.35, 0.8]) == 0.75
    rng = np.random.default_rng(0)
    assert roc_auc(rng.integers(0, 2, 10_000), rng.random(10_000)) == pytest.approx(0.5, abs=0.02)
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4], [1, 4, 9, 100]) == pytest.approx(1.0)   # rank-based: any monotone map
    assert np.isnan(spearman([1, 2], [1, 2]))
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
