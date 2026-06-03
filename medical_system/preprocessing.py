from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import FEATURE_COLUMNS, REFERENCE_UPPER_LIMITS


def clip_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    cols = columns or FEATURE_COLUMNS
    for col in cols:
        if col not in result.columns:
            continue
        series = pd.to_numeric(result[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        result[col] = series.clip(lower=lower, upper=upper)
    return result


def create_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
            ("normalize", MinMaxScaler()),
        ]
    )


def normalize_by_reference_ranges(
    df: pd.DataFrame,
    reference_upper_limits: dict[str, float] | None = None,
    *,
    log_transform: bool = True,
) -> pd.DataFrame:
    """Normalize tumor markers by their clinical reference upper limits.

    Each marker is converted to a multiple of its reference upper limit:
    ``normalized = raw_value / reference_upper_limit``.

    ``log1p`` is enabled by default to reduce the effect of extreme marker values
    while preserving ordering. The same function must be used during training and
    prediction, otherwise the model will see a different feature scale.
    """
    result = df.copy()
    limits = reference_upper_limits or REFERENCE_UPPER_LIMITS

    for col in FEATURE_COLUMNS:
        if col not in result.columns:
            continue

        upper_limit = float(limits.get(col, 0.0) or 0.0)
        if upper_limit <= 0:
            raise ValueError(f"{col} 的参考区间上限必须大于0。")

        values = pd.to_numeric(result[col], errors="coerce").clip(lower=0)
        values = values / upper_limit
        if log_transform:
            values = np.log1p(values.astype("float64"))
        result[col] = values

    return result
