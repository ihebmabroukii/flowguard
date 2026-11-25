"""Anomaly detector for unseen attacks: our Isolation Forest, trained on benign flows only.

The alert threshold is the `benign_quantile` of the scores on benign validation flows, so by
construction about (1 - quantile) of benign traffic is flagged. The hybrid detector tunes it further.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flowguard.models.scratch import IsolationForest


def make_isolation_forest(cfg: dict[str, Any], seed: int = 42) -> IsolationForest:
    return IsolationForest(
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_samples=int(cfg.get("max_samples", 256)),
        random_state=seed,
    )


def benign_threshold(scores_on_benign: Any, quantile: float = 0.99) -> float:
    """Score above which a flow is called anomalous."""
    return float(np.quantile(np.asarray(scores_on_benign, dtype=float), quantile))
