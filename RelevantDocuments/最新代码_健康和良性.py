from __future__ import annotations

from pathlib import Path
import platform

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ------------------------------
# 字体设置（兼容中文）
# ------------------------------
system = platform.system()
if system == "Windows":
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
elif system == "Darwin":
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ------------------------------
# 最终版参数（健康 vs 良性）
# ------------------------------
INPUT_FILE = "最终表-2603_良性和健康.xlsx"
LABEL_COL = "label"
FEATURES = ["AKR1B10", "CA19-9", "NSE", "CA125", "CA153", "CEA"]

RANDOM_STATE = 42
TEST_RATIO = 0.2

# 单阈值（用于常规对照）
BASELINE_THRESHOLD = 0.50

# 双阈值（最终部署建议）
DUAL_LOW_THRESHOLD = 0.20   # <= 0.20 判健康
DUAL_HIGH_THRESHOLD = 0.65  # >= 0.65 判良性

# RF + sigmoid 校准
CALIB_METHOD = "sigmoid"
CALIB_CV = 3
NEGATIVE_CLASS_WEIGHT = 1.0
POSITIVE_CLASS_WEIGHT = 2.0

MODEL_OUT = "最终版_健康良性_RF模型.joblib"
METRICS_OUT = "最终版_健康良性_评估结果.xlsx"
PR_CURVE_OUT = "最终版_健康良性_PR曲线.png"


def resolve_input_path(file_name: str) -> Path:
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    candidate = script_dir / file_name
    if candidate.exists():
        return candidate
    return Path(file_name)


def standardize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """统一标签：0=健康/正常，1=良性。恶性会被剔除。"""
    out = df.copy()
    raw = out[label_col].copy()

    text_map = {
        "正常": 0,
        "健康": 0,
        "normal": 0,
        "良性": 1,
        "benign": 1,
        "恶性": -1,
        "malignant": -1,
    }
    text_norm = raw.astype(str).str.strip().str.lower()
    label_text = text_norm.map(text_map)

    num = pd.to_numeric(raw, errors="coerce")
    has_code_2 = bool((num == 2).any())
    if has_code_2:
        # 0=健康, 1=恶性(剔除), 2=良性
        label_num = num.map({0: 0, 1: -1, 2: 1})
    else:
        # 二分类文件：0=健康, 1=良性
        label_num = num.map({0: 0, 1: 1})

    out[label_col] = label_text.fillna(label_num)
    out = out[out[label_col].isin([0, 1])].copy()
    out[label_col] = out[label_col].astype(int)
    return out


def split_xy_with_train_median(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    x_train = train_df[FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians)

    x_eval = eval_df[FEATURES].apply(pd.to_numeric, errors="coerce")
    x_eval = x_eval.fillna(medians)

    y_train = train_df[LABEL_COL].astype(int).values
    y_eval = eval_df[LABEL_COL].astype(int).values
    return x_train, x_eval, y_train, y_eval, medians


def build_model(seed: int) -> CalibratedClassifierCV:
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight={0: NEGATIVE_CLASS_WEIGHT, 1: POSITIVE_CLASS_WEIGHT},
        random_state=seed,
        n_jobs=-1,
    )
    return CalibratedClassifierCV(estimator=rf, method=CALIB_METHOD, cv=CALIB_CV)


def metric_pack(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob, pos_label=1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pred_positive": int(y_pred.sum()),
    }


def dual_threshold_stats(y_true: np.ndarray, y_prob: np.ndarray, low_thr: float, high_thr: float) -> dict:
    pred = np.full_like(y_true, fill_value=-1, dtype=int)  # -1=待复查
    pred[y_prob <= low_thr] = 0
    pred[y_prob >= high_thr] = 1

    decisive_mask = pred != -1
    decisive_cnt = int(decisive_mask.sum())
    coverage = float(decisive_cnt / len(y_true)) if len(y_true) else 0.0

    out = {
        "dual_low": float(low_thr),
        "dual_high": float(high_thr),
        "decisive_coverage": coverage,
    }

    if decisive_cnt == 0:
        out.update(
            {
                "decisive_accuracy": np.nan,
                "decisive_precision": np.nan,
                "decisive_recall": np.nan,
            }
        )
        return out

    y_d = y_true[decisive_mask]
    p_d = pred[decisive_mask]
    out.update(
        {
            "decisive_accuracy": float(accuracy_score(y_d, p_d)),
            "decisive_precision": float(precision_score(y_d, p_d, pos_label=1, zero_division=0)),
            "decisive_recall": float(recall_score(y_d, p_d, pos_label=1, zero_division=0)),
        }
    )
    return out


def main() -> None:
    input_path = resolve_input_path(INPUT_FILE)
    df_raw = pd.read_excel(input_path)

    raw_counts = df_raw[LABEL_COL].value_counts(dropna=False).to_dict() if LABEL_COL in df_raw.columns else {}
    df = standardize_labels(df_raw, LABEL_COL)

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"缺少特征列: {missing_features}")

    if len(df) < 80:
        raise ValueError("样本太少，建议至少 80 条。")

    print(f"特征: {FEATURES}")
    print(f"原始标签计数: {raw_counts}")
    print(f"总样本数(仅健康/良性0/1): {len(df)}")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\n分层切分完成:")
    print(f"  训练集: {len(train_df)}")
    print(f"  测试集: {len(test_df)}")

    x_train, x_test, y_train, y_test, _ = split_xy_with_train_median(train_df, test_df)

    model = build_model(RANDOM_STATE)
    model.fit(x_train, y_train)
    test_prob = model.predict_proba(x_test)[:, 1]

    baseline = metric_pack(y_test, test_prob, BASELINE_THRESHOLD)
    dual = dual_threshold_stats(y_test, test_prob, DUAL_LOW_THRESHOLD, DUAL_HIGH_THRESHOLD)

    print("\n=== 最终版评估（独立测试集）===")
    print(
        f"单阈值0.5 -> AUC={baseline['auc_roc']:.4f}, "
        f"Acc={baseline['accuracy']:.4f}, P={baseline['precision']:.4f}, "
        f"R={baseline['recall']:.4f}, F1={baseline['f1']:.4f}"
    )
    print(
        f"双阈值[{DUAL_LOW_THRESHOLD:.2f},{DUAL_HIGH_THRESHOLD:.2f}] -> "
        f"覆盖率={dual['decisive_coverage']:.4f}, "
        f"覆盖样本Acc={dual['decisive_accuracy']:.4f}, "
        f"覆盖样本P={dual['decisive_precision']:.4f}, "
        f"覆盖样本R={dual['decisive_recall']:.4f}"
    )

    pd.DataFrame([{"mode": "single_0.5", **baseline, **dual}]).to_excel(METRICS_OUT, index=False)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, test_prob, pos_label=1)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f"RF Sigmoid (AUC-PR={baseline['auc_pr']:.3f})")
    plt.axhline(
        y=(y_test == 1).mean(),
        color="gray",
        linestyle="--",
        label=f"随机基线={(y_test == 1).mean():.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("最终版PR曲线（良性=正类，独立测试集）")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.savefig(PR_CURVE_OUT, dpi=300, bbox_inches="tight")
    plt.show()

    # 全量数据重训，保存部署模型
    x_all = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    all_medians = x_all.median(numeric_only=True)
    x_all = x_all.fillna(all_medians)
    y_all = df[LABEL_COL].astype(int).values

    final_model = build_model(RANDOM_STATE)
    final_model.fit(x_all, y_all)

    payload = {
        "model_name": "final_benign_rf_sigmoid",
        "features": FEATURES,
        "label_col": LABEL_COL,
        "single_threshold": BASELINE_THRESHOLD,
        "dual_low_threshold": DUAL_LOW_THRESHOLD,
        "dual_high_threshold": DUAL_HIGH_THRESHOLD,
        "calib_method": CALIB_METHOD,
        "calib_cv": CALIB_CV,
        "train_medians": all_medians.to_dict(),
        "model": final_model,
        "test_metrics_single": baseline,
        "test_metrics_dual": dual,
    }
    joblib.dump(payload, MODEL_OUT)

    print(f"\n评估结果已保存: {METRICS_OUT}")
    print(f"PR曲线已保存: {PR_CURVE_OUT}")
    print(f"最终部署模型已保存: {MODEL_OUT}")


if __name__ == "__main__":
    main()
