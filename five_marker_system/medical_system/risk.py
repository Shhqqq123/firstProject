from __future__ import annotations

import pandas as pd

from .config import CLASS_NAME_MAP, FEATURE_COLUMNS


def to_cn_class(label: str) -> str:
    return CLASS_NAME_MAP.get(label, label)


def get_risk_level(predicted_class: str, malignant_prob: float, confidence: float) -> str:
    if predicted_class == "malignant":
        return "高风险" if malignant_prob >= 0.75 else "中高风险"
    if predicted_class == "benign":
        return "中风险" if confidence >= 0.70 else "中低风险"
    return "低风险" if confidence >= 0.70 else "低中风险"


def followup_warning_analysis(df: pd.DataFrame) -> tuple[bool, list[str]]:
    if df.empty or len(df) < 2:
        return False, ["随访记录不足，暂无法判断趋势。"]

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
            notes.append(f"{marker.upper()}较基线升高 {change * 100:.1f}%")

    malignant_prob = pd.to_numeric(latest.get("malignant_prob"), errors="coerce")
    if not pd.isna(malignant_prob) and malignant_prob >= 0.60:
        warning = True
        notes.append(f"最近一次恶性概率为 {malignant_prob:.2f}")

    if not notes:
        notes.append("当前趋势整体平稳，暂无明显早期预警信号。")

    return warning, notes

