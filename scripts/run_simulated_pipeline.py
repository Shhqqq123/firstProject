from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from medical_system.config import FEATURE_COLUMNS, MODEL_PATH, REPORT_DIR, ensure_directories
from medical_system.database import (
    add_subject,
    add_test,
    get_followup_dataframe,
    get_latest_evaluation_for_test,
    get_subject,
    get_test,
    init_db,
    save_evaluation,
)
from medical_system.modeling import BreastRiskModel
from medical_system.reporting import generate_report_html, generate_report_pdf
from medical_system.risk import get_risk_level


def make_synthetic_training_data() -> pd.DataFrame:
    rng = np.random.default_rng(2026)

    def make_group(n: int, center: tuple[float, ...], label: str) -> pd.DataFrame:
        scales = np.array([2.0, 4.5, 3.0, 5.0, 8.0, 1.0])
        x = rng.normal(loc=np.array(center), scale=scales, size=(n, 6))
        x = np.clip(x, a_min=0.1, a_max=None)
        return pd.DataFrame(
            {
                "test_date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
                "akr1b10": x[:, 0],
                "ca19_9": x[:, 1],
                "nse": x[:, 2],
                "ca125": x[:, 3],
                "ca153": x[:, 4],
                "cea": x[:, 5],
                "clinical_stage": "screening",
                "label": label,
            }
        )

    df = pd.concat(
        [
            make_group(300, (8, 12, 10, 16, 20, 2.8), "normal"),
            make_group(200, (11, 17, 14, 24, 30, 3.8), "benign"),
            make_group(500, (18, 35, 24, 40, 55, 8.2), "malignant"),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=42)
    return df.reset_index(drop=True)


def main() -> None:
    ensure_directories()
    init_db()

    df = make_synthetic_training_data()
    model = BreastRiskModel()
    result = model.train(df)
    model.save(MODEL_PATH)

    print("=== Training Metrics ===")
    print(f"AUC: {result.metrics['auc']:.4f}")
    print(f"Precision: {result.metrics['precision']:.4f}")
    print(f"Recall: {result.metrics['recall']:.4f}")
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")

    subject_id = add_subject(name="Simulated Subject", sex="Female", birth_date="1988-05-01")
    row = df.iloc[0]
    test_id = add_test(
        subject_id=subject_id,
        test_date=str(row["test_date"]),
        markers={k: float(row[k]) for k in FEATURE_COLUMNS},
        clinical_stage="screening",
        label=str(row["label"]),
        source="simulation",
    )

    sample_df = pd.DataFrame([{k: row[k] for k in FEATURE_COLUMNS}])
    pred = model.predict(sample_df)
    risk_level = get_risk_level(
        predicted_class=str(pred["predicted_class"]),
        malignant_prob=float(pred["probabilities"].get("malignant", 0.0)),
        confidence=float(pred["confidence"]),
    )
    save_evaluation(
        test_id=test_id,
        predicted_class=str(pred["predicted_class"]),
        probabilities={k: float(v) for k, v in pred["probabilities"].items()},
        risk_level=risk_level,
        warning_flag=False,
        feature_importance={k: float(v) for k, v in pred["feature_contribution"].items()},
    )

    subject = get_subject(subject_id)
    test = get_test(test_id)
    eval_row = get_latest_evaluation_for_test(test_id)
    followup_df = get_followup_dataframe(subject_id)
    assert subject and test and eval_row is not None

    html_path = generate_report_html(subject, test, eval_row, followup_df, Path(REPORT_DIR))
    pdf_path = generate_report_pdf(subject, test, eval_row, followup_df, Path(REPORT_DIR))
    print("=== Inference ===")
    print(f"Predicted class: {pred['predicted_class']}")
    print(f"Risk level: {risk_level}")
    print("=== Reports ===")
    print(f"HTML: {html_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()

