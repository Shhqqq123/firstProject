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
    report_path = output_dir / f"报告_受检者{subject_id}_检验{test_id}.html"

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
            risk = row.get("risk_level", "") or "未评估"
            followup_lines.append(
                f"<tr><td>{row.get('test_date', '')}</td><td>{row.get('ca153', '')}</td><td>{row.get('cea', '')}</td><td>{risk}</td></tr>"
            )
    followup_html = "".join(followup_lines) or "<tr><td colspan='4'>暂无随访记录</td></tr>"

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>乳腺风险评估报告</title>
  <style>
    body {{ font-family: "Microsoft YaHei", sans-serif; margin: 28px; color: #1a1a1a; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 18px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #f4f6f8; }}
    .warn {{ color: #b42318; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>基于6项肿瘤标志物与集成学习的乳腺全流程风险评估系统 V1.0</h1>
  <div class="meta">生成时间：{now}</div>
  <h2>1. 受检者信息</h2>
  <table>
    <tr><th>字段</th><th>内容</th></tr>
    <tr><td>受检者ID</td><td>{subject.get("id")}</td></tr>
    <tr><td>姓名</td><td>{subject.get("name", "")}</td></tr>
    <tr><td>性别</td><td>{subject.get("sex", "")}</td></tr>
    <tr><td>出生日期</td><td>{subject.get("birth_date", "")}</td></tr>
    <tr><td>联系电话</td><td>{subject.get("phone", "")}</td></tr>
  </table>

  <h2>2. 指标检测值</h2>
  <table>
    <tr><th>指标</th><th>数值</th></tr>
    {markers_html}
  </table>

  <h2>3. 智能评估结果</h2>
  <table>
    <tr><th>类别</th><th>概率</th></tr>
    {prob_html}
  </table>
  <div>预测类别：<strong>{CLASS_NAME_MAP.get(evaluation_row.get("predicted_class", ""), evaluation_row.get("predicted_class", ""))}</strong></div>
  <div>风险等级：<strong class="warn">{evaluation_row.get("risk_level", "")}</strong></div>

  <h2>4. 指标贡献度</h2>
  <table>
    <tr><th>指标</th><th>贡献度</th></tr>
    {importance_html}
  </table>

  <h2>5. 随访趋势摘要</h2>
  <table>
    <tr><th>日期</th><th>CA153</th><th>CEA</th><th>风险等级</th></tr>
    {followup_html}
  </table>

  <h2>6. 说明</h2>
  <p>1. 本报告仅用于风险评估与辅助参考，不作为最终诊断依据。</p>
  <p>2. 请结合影像、病理与临床症状进行综合判断。</p>
  <p>3. 当风险等级达到中高风险及以上时，建议尽快复查并专科评估。</p>
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
    path = output_dir / f"报告_受检者{subject['id']}_检验{test_row['id']}.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setAuthor("乳腺风险评估系统")
    c.setTitle("乳腺风险评估报告")

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

    write_line("基于6项肿瘤标志物与集成学习的乳腺全流程风险评估系统 V1.0", size=14, step_mm=8)
    write_line(f"生成时间: {now}")
    write_line("")
    write_line("1. 受检者信息", size=12)
    write_line(f"ID: {subject.get('id')}   姓名: {subject.get('name', '')}   性别: {subject.get('sex', '')}")
    write_line(f"出生日期: {subject.get('birth_date', '')}   电话: {subject.get('phone', '')}")
    write_line("")
    write_line("2. 指标检测值", size=12)
    for marker in FEATURE_COLUMNS:
        write_line(f"{marker.upper()}: {test_row.get(marker, '')}")
    write_line("")
    write_line("3. 智能评估结果", size=12)
    write_line(
        f"预测类别: {CLASS_NAME_MAP.get(evaluation_row.get('predicted_class', ''), evaluation_row.get('predicted_class', ''))}"
    )
    write_line(f"风险等级: {evaluation_row.get('risk_level', '')}")
    for k, v in probs.items():
        write_line(f"{CLASS_NAME_MAP.get(k, k)}概率: {v:.4f}")
    write_line("")
    write_line("4. 指标贡献度", size=12)
    for key, value in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        write_line(f"{key.upper()}: {value:.4f}")
    write_line("")
    write_line("5. 随访趋势摘要", size=12)
    if followup_df.empty:
        write_line("暂无随访记录。")
    else:
        for _, row in followup_df.sort_values("test_date").tail(8).iterrows():
            write_line(
                f"{row.get('test_date', '')}  CA153={row.get('ca153', '')}  CEA={row.get('cea', '')}  风险={row.get('risk_level', '') or '无'}"
            )
    write_line("")
    write_line("6. 说明", size=12)
    write_line("本报告仅用于风险评估与辅助参考，不作为最终诊断依据。")
    write_line("最终结论请由具备资质的临床医生结合完整资料给出。")
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

