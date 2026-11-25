"""Trivial baselines (models/scratch/trivial.py): the floor every real model has to clear."""

from __future__ import annotations

from typing import Any

from flowguard.models.base import FlowClassifier
from flowguard.models.scratch import TrivialClassifier


def make_majority(cfg: dict[str, Any], seed: int = 42) -> FlowClassifier:
    return FlowClassifier("majority", TrivialClassifier("majority", random_state=seed), seed=seed)


def make_random_prior(cfg: dict[str, Any], seed: int = 42) -> FlowClassifier:
    return FlowClassifier("random_prior", TrivialClassifier("random_prior", random_state=seed), seed=seed)
