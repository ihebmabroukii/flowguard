"""Optional MLflow experiment tracking. Everything works without MLflow installed."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Any

from flowguard.config import Paths

log = logging.getLogger(__name__)


def get_mlflow(enabled: bool, paths: Paths, experiment: str) -> Any | None:
    """The mlflow module pointed at <repo>/mlruns, or None when disabled or not installed."""
    if not enabled:
        return None
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow is not installed; skipping experiment logging")
        return None
    mlflow.set_tracking_uri((paths.root / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


@contextmanager
def run(mlflow: Any | None, name: str, tags: dict[str, str] | None = None):
    """A tagged MLflow run, or nothing when mlflow is None."""
    if mlflow is None:
        with nullcontext():
            yield None
        return
    with mlflow.start_run(run_name=name) as active:
        if tags:
            mlflow.set_tags(tags)
        yield active
