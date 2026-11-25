"""Our neural network classifier (models/scratch/mlp.py), configured from experiments.yaml."""

from __future__ import annotations

from typing import Any

from flowguard.models.base import FlowClassifier
from flowguard.models.scratch import MLPClassifier


def make_mlp(cfg: dict[str, Any], seed: int = 42) -> FlowClassifier:
    params = dict(cfg)
    max_rows = params.pop("max_train_rows", None)
    max_class_weight = params.pop("max_class_weight", 100)
    return FlowClassifier(
        "mlp",
        MLPClassifier(random_state=seed, **params),
        balanced_weights=True,
        max_class_weight=max_class_weight,
        max_train_rows=max_rows,
        uses_eval_set=True,
        seed=seed,
    )
