from __future__ import annotations

from pathlib import Path
import platform

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
# 最终版参数（健康 vs 恶性）
# ------------------------------
INPUT_FILE = "最终表-2603_恶性和健康.xlsx"
LABEL_COL = "label"
FEATURES = ["CA19-9", "NSE", "CA125", "CA153", "CEA"]

RANDOM_STATE = 42
TEST_RATIO = 0.2
DEPLOY_THRESHOLD = 0.50

# 已验证较稳的融合配置
ET_WEIGHT = 0.70
RF_WEIGHT = 0.30
CALIB_METHOD = "isotonic"
CALIB_CV = 3

MODEL_OUT = "最终版_健康恶性_融合模型.joblib"
METRICS_OUT = "最终版_健康恶性_评估结果.xlsx"
PR_CURVE_OUT = "最终版_健康恶性_PR曲线.png"


def standardize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """统一标签：0=健康/正常，1=恶性；其余标签（如良性）自动过滤。"""
    out = df.copy()
    out[label_col] = out[label_col].replace(
        {
            "正常": 0,
            "健康": 0,
            "恶性": 1,
            0: 0,
            1: 1,
            "0": 0,
            "1": 1,
        }
    )
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
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


def build_calibrated_models(seed: int) -> tuple[CalibratedClassifierCV, CalibratedClassifierCV]:
    et = ExtraTreesClassifier(
        n_estimators=260,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=320,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed + 999,
        n_jobs=-1,
    )

    cal_et = CalibratedClassifierCV(estimator=et, method=CALIB_METHOD, cv=CALIB_CV)
    cal_rf = CalibratedClassifierCV(estimator=rf, method=CALIB_METHOD, cv=CALIB_CV)
    return cal_et, cal_rf


def predict_blend_prob(cal_et: CalibratedClassifierCV, cal_rf: CalibratedClassifierCV, x_eval: pd.DataFrame) -> np.ndarray:
    p_et = cal_et.predict_proba(x_eval)[:, 1]
    p_rf = cal_rf.predict_proba(x_eval)[:, 1]
    return ET_WEIGHT * p_et + RF_WEIGHT * p_rf


def metric_pack(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob, pos_label=1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pred_positive": int(y_pred.sum()),
    }


def resolve_input_path(file_name: str) -> Path:
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()

    candidate = script_dir / file_name
    if candidate.exists():
        return candidate
    return Path(file_name)


def main() -> None:
    input_path = resolve_input_path(INPUT_FILE)
    df = pd.read_excel(input_path)
    df = standardize_labels(df, LABEL_COL)

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"缺少特征列: {missing_features}")

    if len(df) < 100:
        raise ValueError("样本太少，建议至少 100 条（且正负类都有足够样本）。")

    print(f"特征: {FEATURES}")
    print(f"总样本数(仅健康/恶性0/1): {len(df)}")

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

    x_train, x_test, y_train, y_test, train_medians = split_xy_with_train_median(train_df, test_df)

    # 训练并评估（独立测试集）
    cal_et, cal_rf = build_calibrated_models(RANDOM_STATE)
    cal_et.fit(x_train, y_train)
    cal_rf.fit(x_train, y_train)

    test_prob = predict_blend_prob(cal_et, cal_rf, x_test)
    metrics = metric_pack(y_test, test_prob, DEPLOY_THRESHOLD)

    print("\n=== 最终版评估（独立测试集）===")
    print(f"AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:   {metrics['auc_pr']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision:{metrics['precision']:.4f}")
    print(f"Recall:   {metrics['recall']:.4f}")
    print(f"F1:       {metrics['f1']:.4f}")

    # 保存评估结果
    pd.DataFrame([metrics]).to_excel(METRICS_OUT, index=False)

    # 绘制 PR 曲线
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, test_prob, pos_label=1)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f"Final Isotonic Blend (AUC-PR={metrics['auc_pr']:.3f})")
    plt.axhline(
        y=(y_test == 1).mean(),
        color="gray",
        linestyle="--",
        label=f"随机基线={(y_test == 1).mean():.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("最终版PR曲线（恶性=正类，独立测试集）")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.savefig(PR_CURVE_OUT, dpi=300, bbox_inches="tight")
    plt.show()

    # 最终部署模型：用全量数据重训并保存
    x_all = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    all_medians = x_all.median(numeric_only=True)
    x_all = x_all.fillna(all_medians)
    y_all = df[LABEL_COL].astype(int).values

    final_et, final_rf = build_calibrated_models(RANDOM_STATE)
    final_et.fit(x_all, y_all)
    final_rf.fit(x_all, y_all)

    payload = {
        "model_name": "final_isotonic_et_rf_blend",
        "features": FEATURES,
        "label_col": LABEL_COL,
        "threshold": DEPLOY_THRESHOLD,
        "et_weight": ET_WEIGHT,
        "rf_weight": RF_WEIGHT,
        "calib_method": CALIB_METHOD,
        "calib_cv": CALIB_CV,
        "train_medians": all_medians.to_dict(),
        "et_model": final_et,
        "rf_model": final_rf,
        "test_metrics": metrics,
    }
    joblib.dump(payload, MODEL_OUT)

    print(f"\n评估结果已保存: {METRICS_OUT}")
    print(f"PR曲线已保存: {PR_CURVE_OUT}")
    print(f"最终部署模型已保存: {MODEL_OUT}")


if __name__ == "__main__":
    main()
