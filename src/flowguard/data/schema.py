"""Column names shared by every stage of the data pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keeps this module importable with NumPy alone
    import pandas as pd

# Raw CSV columns (improved CICFlowMeter header).
RAW_LABEL = "Label"
RAW_ATTEMPTED = "Attempted Category"
RAW_TIMESTAMP = "Timestamp"

# Metadata columns added by load/clean. They travel with every row but are never model inputs.
ROW_ID = "row_id"
DATASET = "dataset"
DAY = "day"
TIMESTAMP = "timestamp"
LABEL_RAW = "label_raw"
FAMILY = "family"
META_COLUMNS = (ROW_ID, DATASET, DAY, TIMESTAMP, LABEL_RAW, FAMILY)
# Added when a split file is joined to the processed table.
SPLIT = "split"
NON_FEATURE_COLUMNS = frozenset((*META_COLUMNS, SPLIT))

BENIGN = "Benign"
DROP = "drop"
# Prediction of the hybrid detector for traffic that matches no known attack but looks anomalous.
UNKNOWN = "Unknown"


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column of a cleaned frame that the models may use."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
