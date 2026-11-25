"""Map raw dataset labels to the shared attack families defined in configs/labels.yaml."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from flowguard.data.schema import BENIGN, DROP

_ATTEMPTED = re.compile(r"\s*-\s*attempted\s*$", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


class UnknownLabelError(ValueError):
    """A label matched no rule in configs/labels.yaml."""


def is_attempted_label(raw: object) -> bool:
    return bool(_ATTEMPTED.search(str(raw)))


def normalize_label(raw: object) -> str:
    """'Web Attack \\x96 Brute Force - Attempted' -> 'web attack brute force'."""
    text = _ATTEMPTED.sub("", str(raw)).lower()
    return _NON_ALNUM.sub(" ", text).strip()


@dataclass(frozen=True)
class LabelMapper:
    rules: tuple[tuple[re.Pattern[str], str], ...]
    families: tuple[str, ...]
    attempted_policy: str = "benign"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> LabelMapper:
        families = tuple(cfg["families"])
        rules = []
        for rule in cfg["rules"]:
            family = rule["family"]
            if family != DROP and family not in families:
                raise ValueError(f"Rule {rule['pattern']!r} maps to {family!r}, not a listed family")
            rules.append((re.compile(rule["pattern"]), family))
        policy = cfg.get("attempted_policy", "benign")
        if policy not in ("benign", "drop"):
            raise ValueError(f"attempted_policy must be 'benign' or 'drop', got {policy!r}")
        return cls(tuple(rules), families, policy)

    def family_of(self, raw: object) -> str:
        """Family for one label, ignoring the attempted policy."""
        norm = normalize_label(raw)
        for pattern, family in self.rules:
            if pattern.search(norm):
                return family
        raise UnknownLabelError(f"No rule in configs/labels.yaml matches label {raw!r}")

    def lookup(self, labels: Iterable[object]) -> dict[str, str]:
        """Family for every distinct label. Raises once, listing every unknown label."""
        table: dict[str, str] = {}
        unknown: list[str] = []
        for raw in set(map(str, labels)):
            try:
                table[raw] = self.family_of(raw)
            except UnknownLabelError:
                unknown.append(raw)
        if unknown:
            raise UnknownLabelError(
                "Labels with no matching rule in configs/labels.yaml: " + ", ".join(sorted(unknown))
            )
        return table

    def map_series(self, labels: pd.Series, attempted: pd.Series | None = None) -> pd.Series:
        """Vectorized mapping, with the attempted policy applied.

        A flow counts as attempted when its label ends in '- Attempted' or its
        'Attempted Category' value is set (anything other than -1).
        """
        as_text = labels.astype(str)
        families = as_text.map(self.lookup(as_text.unique()))

        attempted_mask = as_text.str.contains(_ATTEMPTED)
        if attempted is not None:
            category = pd.to_numeric(attempted, errors="coerce").fillna(-1)
            attempted_mask |= category.ne(-1)
        attempted_mask &= families.ne(BENIGN) & families.ne(DROP)

        replacement = BENIGN if self.attempted_policy == "benign" else DROP
        return pd.Series(
            np.where(attempted_mask, replacement, families), index=labels.index, name="family"
        )
