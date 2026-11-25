"""Project paths and YAML config loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DATASETS = ("cic-ids2017", "cse-cic-ids2018")


def project_root() -> Path:
    """Folder that holds pyproject.toml and configs/.

    FLOWGUARD_ROOT wins when set (the tests use it); otherwise search upwards from the
    current directory, then from this file.
    """
    env = os.environ.get("FLOWGUARD_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for candidate in (Path.cwd(), *Path.cwd().parents, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the FlowGuard project root. Run from inside the repo or set FLOWGUARD_ROOT."
    )


def load_config(name: str, root: Path | None = None) -> dict[str, Any]:
    """Read configs/<name>.yaml."""
    path = (root or project_root()) / "configs" / f"{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def default(cls) -> Paths:
        return cls(project_root())

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def tables(self) -> Path:
        return self.root / "reports" / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

    def processed_file(self, dataset: str) -> Path:
        return self.processed / f"{dataset}.parquet"

    def split_file(self, dataset: str) -> Path:
        return self.processed / f"{dataset}_splits.parquet"


def check_dataset(name: str) -> str:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Expected one of: {', '.join(DATASETS)}")
    return name
