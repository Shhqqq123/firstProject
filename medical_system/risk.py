from __future__ import annotations

import pandas as pd

from .config import CLASS_NAME_MAP, FEATURE_COLUMNS


def to_cn_class(label: str) -> str:
    return CLASS_NAME_MAP.get(label, label)


def get_risk_level(predicted_class: str, malignant_prob: float, confidence: float) -> str:
    if predicted_class == "malignant":
        return "High" if malignant_prob >= 0.75 else "Medium-High"
    if predicted_class == "benign":
        return "Medium" if confidence >= 0.70 else "Medium-Low"
    return "Low" if confidence >= 0.70 else "Low-Medium"


def followup_warning_analysis(df: pd.DataFrame) -> tuple[bool, list[str]]:
    if df.empty or len(df) < 2:
        return False, ["Not enough follow-up records to evaluate trend."]

    data = df.copy()
    data["test_date"] = pd.to_datetime(data["test_date"], errors="coerce")
    data = data.sort_values("test_date")

    notes: list[str] = []
    warning = False

    latest = data.iloc[-1]
    baseline = data.iloc[0]

    for marker in FEATURE_COLUMNS:
        if marker not in data.columns:
            continue
        latest_value = pd.to_numeric(latest[marker], errors="coerce")
        base_value = pd.to_numeric(baseline[marker], errors="coerce")
        if pd.isna(latest_value) or pd.isna(base_value) or base_value == 0:
            continue
        change = (latest_value - base_value) / base_value
        if change >= 0.20:
            warning = True
            notes.append(f"{marker.upper()} increased by {change * 100:.1f}% from baseline.")

    malignant_prob = pd.to_numeric(latest.get("malignant_prob"), errors="coerce")
    if not pd.isna(malignant_prob) and malignant_prob >= 0.60:
        warning = True
        notes.append(f"Latest malignant probability is {malignant_prob:.2f}.")

    if not notes:
        notes.append("Trend appears stable with no obvious early warning signal.")

    return warning, notes

