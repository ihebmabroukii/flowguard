"""Random-search sampling, the trivial baselines, and tuned-parameter loading (NumPy + YAML only)."""

import json

import numpy as np
import pytest

from flowguard.config import Paths
from flowguard.models.scratch import TrivialClassifier
from flowguard.models.tune import (
    sample_value,
    searched_values,
    trial_configs,
    tuned_file,
    tuned_models_cfg,
)

SPACE = {
    "lr": {"log_uniform": [1e-4, 1e-1]},
    "dropout": {"uniform": [0.0, 0.5]},
    "layers": {"int_uniform": [1, 3]},
    "hidden": {"choice": [[64], [128, 64]]},
}


def test_sample_value_ranges():
    rng = np.random.default_rng(0)
    lrs = [sample_value(SPACE["lr"], rng) for _ in range(2000)]
    assert 1e-4 <= min(lrs) and max(lrs) <= 1e-1
    # log-uniform: about half the draws fall below the geometric midpoint 10^-2.5
    assert 0.45 < np.mean(np.array(lrs) < 10 ** -2.5) < 0.55
    assert {sample_value(SPACE["layers"], rng) for _ in range(200)} == {1, 2, 3}
    assert sample_value(SPACE["hidden"], rng) in ([64], [128, 64])
    assert sample_value(7, rng) == 7
    with pytest.raises(ValueError):
        sample_value({"normal": [0, 1]}, rng)


def test_trial_zero_is_the_config_and_draws_are_seeded():
    base = {"lr": 0.001, "epochs": 20, "hidden": [256]}
    trials = trial_configs(base, SPACE, n_trials=5, seed=1)
    assert len(trials) == 5 and trials[0] == base
    assert all(t["epochs"] == 20 for t in trials)               # unsearched values are kept
    assert trials == trial_configs(base, SPACE, n_trials=5, seed=1)
    assert trials != trial_configs(base, SPACE, n_trials=5, seed=2)
    assert searched_values(trials[0], {"lr": None, "hidden": None}) == {"lr": 0.001, "hidden": "[256]"}


def test_tuned_parameters_are_merged_only_when_enabled(tmp_path):
    paths = Paths(tmp_path)
    cfg = {"models": {"mlp": {"lr": 0.001, "epochs": 20}, "logreg": {"C": 1.0}}, "tuning": {"apply_best": False}}
    tuned_file(paths, "mlp").parent.mkdir(parents=True)
    tuned_file(paths, "mlp").write_text(json.dumps({"best_params": {"lr": 0.005}}))
    assert tuned_models_cfg(cfg, paths)["mlp"]["lr"] == 0.001
    cfg["tuning"]["apply_best"] = True
    merged = tuned_models_cfg(cfg, paths)
    assert merged["mlp"] == {"lr": 0.005, "epochs": 20}
    assert merged["logreg"] == {"C": 1.0}
    assert cfg["models"]["mlp"]["lr"] == 0.001                   # the input config is not mutated


def test_trivial_baselines():
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    X = np.zeros((len(y), 3))
    majority = TrivialClassifier("majority").fit(X, y)
    assert (majority.predict(X) == 0).all()
    assert np.allclose(majority.predict_proba(X).sum(1), 1)
    guess = TrivialClassifier("random_prior", random_state=0).fit(X, y)
    big = np.zeros((20_000, 3))
    picks = guess.predict(big)
    assert np.bincount(picks, minlength=3) / len(picks) == pytest.approx([0.8, 0.15, 0.05], abs=0.02)
    assert (guess.predict(big) == picks).all()                   # seeded
    assert guess.get_params() == {"strategy": "random_prior", "random_state": 0}
    with pytest.raises(ValueError):
        TrivialClassifier("smart")
