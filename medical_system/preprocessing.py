from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import FEATURE_COLUMNS


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

