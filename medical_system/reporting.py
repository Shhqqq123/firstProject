from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from .config import CLASS_NAME_MAP, FEATURE_COLUMNS


def generate_report_html(
    subject: dict[str, Any],
    test_row: dict[str, Any],
    evaluation_row: dict[str, Any],
    followup_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject_id = subject["id"]
    test_id = test_row["id"]
    report_path = output_dir / f"report_subject_{subject_id}_test_{test_id}.html"

    probs = {
        "normal": float(evaluation_row.get("normal_prob", 0.0)),
        "benign": float(evaluation_row.get("benign_prob", 0.0)),
        "malignant": float(evaluation_row.get("malignant_prob", 0.0)),
    }
    feature_importance = _parse_feature_importance(evaluation_row.get("feature_importance_json"))

    markers_html = "".join(
        f"<tr><td>{marker.upper()}</td><td>{test_row.get(marker, '')}</td></tr>" for marker in FEATURE_COLUMNS
    )
    prob_html = "".join(
        f"<tr><td>{CLASS_NAME_MAP.get(k, k)}</td><td>{v:.4f}</td></tr>" for k, v in probs.items()
    )
    importance_html = "".join(
        f"<tr><td>{k.upper()}</td><td>{v:.4f}</td></tr>"
        for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    followup_lines = []
    if not followup_df.empty:
        sorted_df = followup_df.sort_values("test_date")
        for _, row in sorted_df.iterrows():
            risk = row.get("risk_level", "") or "Not evaluated"
            followup_lines.append(
                f"<tr><td>{row.get('test_date', '')}</td><td>{row.get('ca153', '')}</td><td>{row.get('cea', '')}</td><td>{risk}</td></tr>"
            )
    followup_html = "".join(followup_lines) or "<tr><td colspan='4'>No follow-up records.</td></tr>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Breast Risk Assessment Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1a1a1a; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 18px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #f4f6f8; }}
    .warn {{ color: #b42318; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Breast Full-Process Risk Assessment System V1.0</h1>
  <div class="meta">Generated At: {now}</div>
  <h2>1. Subject Information</h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Subject ID</td><td>{subject.get("id")}</td></tr>
    <tr><td>Name</td><td>{subject.get("name", "")}</td></tr>
    <tr><td>Sex</td><td>{subject.get("sex", "")}</td></tr>
    <tr><td>Birth Date</td><td>{subject.get("birth_date", "")}</td></tr>
    <tr><td>Phone</td><td>{subject.get("phone", "")}</td></tr>
  </table>

  <h2>2. Marker Values</h2>
  <table>
    <tr><th>Marker</th><th>Value</th></tr>
    {markers_html}
  </table>

  <h2>3. Assessment Result</h2>
  <table>
    <tr><th>Class</th><th>Probability</th></tr>
    {prob_html}
  </table>
  <div>Predicted Class: <strong>{CLASS_NAME_MAP.get(evaluation_row.get("predicted_class", ""), evaluation_row.get("predicted_class", ""))}</strong></div>
  <div>Risk Level: <strong class="warn">{evaluation_row.get("risk_level", "")}</strong></div>

  <h2>4. Feature Contribution</h2>
  <table>
    <tr><th>Feature</th><th>Contribution</th></tr>
    {importance_html}
  </table>

  <h2>5. Follow-up Summary</h2>
  <table>
    <tr><th>Date</th><th>CA153</th><th>CEA</th><th>Risk Level</th></tr>
    {followup_html}
  </table>

  <h2>6. Disclaimer</h2>
  <p>1. This report is for risk assessment support only and is not a final diagnosis.</p>
  <p>2. Please combine imaging, pathology and clinical symptoms for final decisions.</p>
  <p>3. Medium-high risk or above should be reviewed by specialist clinicians promptly.</p>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def generate_report_pdf(
    subject: dict[str, Any],
    test_row: dict[str, Any],
    evaluation_row: dict[str, Any],
    followup_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    path = output_dir / f"report_subject_{subject['id']}_test_{test_row['id']}.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setAuthor("Breast Risk Assessment System")
    c.setTitle("Breast Risk Assessment Report")

    width, height = A4
    y = height - 18 * mm
    left = 16 * mm

    def write_line(text: str, size: int = 11, step_mm: float = 6.2) -> None:
        nonlocal y
        if y < 20 * mm:
            c.showPage()
            c.setFont("STSong-Light", 11)
            y = height - 18 * mm
        c.setFont("STSong-Light", size)
        c.drawString(left, y, text)
        y -= step_mm * mm

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    probs = {
        "normal": float(evaluation_row.get("normal_prob", 0.0)),
        "benign": float(evaluation_row.get("benign_prob", 0.0)),
        "malignant": float(evaluation_row.get("malignant_prob", 0.0)),
    }
    feature_importance = _parse_feature_importance(evaluation_row.get("feature_importance_json"))

    write_line("Breast Full-Process Risk Assessment Report", size=14, step_mm=8)
    write_line(f"Generated At: {now}")
    write_line("")
    write_line("1. Subject Information", size=12)
    write_line(f"ID: {subject.get('id')}   Name: {subject.get('name', '')}   Sex: {subject.get('sex', '')}")
    write_line(f"Birth Date: {subject.get('birth_date', '')}   Phone: {subject.get('phone', '')}")
    write_line("")
    write_line("2. Marker Values", size=12)
    for marker in FEATURE_COLUMNS:
        write_line(f"{marker.upper()}: {test_row.get(marker, '')}")
    write_line("")
    write_line("3. Assessment Result", size=12)
    write_line(f"Predicted Class: {CLASS_NAME_MAP.get(evaluation_row.get('predicted_class', ''), evaluation_row.get('predicted_class', ''))}")
    write_line(f"Risk Level: {evaluation_row.get('risk_level', '')}")
    for k, v in probs.items():
        write_line(f"{CLASS_NAME_MAP.get(k, k)} Probability: {v:.4f}")
    write_line("")
    write_line("4. Feature Contribution", size=12)
    for key, value in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        write_line(f"{key.upper()}: {value:.4f}")
    write_line("")
    write_line("5. Follow-up Summary", size=12)
    if followup_df.empty:
        write_line("No follow-up records.")
    else:
        for _, row in followup_df.sort_values("test_date").tail(8).iterrows():
            write_line(
                f"{row.get('test_date', '')}  CA153={row.get('ca153', '')}  CEA={row.get('cea', '')}  Risk={row.get('risk_level', '') or 'NA'}"
            )
    write_line("")
    write_line("6. Disclaimer", size=12)
    write_line("This report is for risk assessment support only and not a diagnosis.")
    write_line("Final decision must be made by licensed clinicians.")
    c.save()
    return path


def _parse_feature_importance(value: Any) -> dict[str, float]:
    if not value:
        return {c: 0.0 for c in FEATURE_COLUMNS}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    try:
        data = json.loads(str(value))
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except json.JSONDecodeError:
        pass
    return {c: 0.0 for c in FEATURE_COLUMNS}

