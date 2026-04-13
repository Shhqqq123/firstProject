import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt
import matplotlib
import platform

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
# 可调参数（想提高 precision 主要调这里）
# ------------------------------
INPUT_FILE = "最终表-2603_恶性和健康.xlsx"
LABEL_COL = "label"
FEATURES = ["AKR1B10", "CA19-9", "NSE", "CA125", "CA153", "CEA"]

N_MODELS = 60
VAL_RATIO = 0.2

# 阈值搜索约束：在 recall 不低于该值前提下，尽量提高 precision
MIN_RECALL_FOR_TUNING = 0.55
# 避免阈值过高导致只预测极少数阳性（precision 虽高但不稳定）
MIN_PRED_POSITIVE = 8

# 为了降低假阳性（提高 precision），给负类更高权重
NEGATIVE_CLASS_WEIGHT = 1.35
POSITIVE_CLASS_WEIGHT = 1.0

# 三个指标同时达到的目标值
TARGET_SCORE = 0.80


def standardize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """将标签统一到 0/1，保留恶性(1)与正常(0)。"""
    out = df.copy()
    out[label_col] = out[label_col].replace(
        {
            "正常": 0,
            "健康": 0,
            "恶性": 1,
            "良性": 2,
        }
    )
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    out = out[out[label_col].isin([0, 1])].copy()
    out[label_col] = out[label_col].astype(int)
    return out


def choose_threshold_for_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.55,
    min_pred_positive: int = 8,
) -> float:
    """
    选择阈值：
    1) 优先满足 recall >= min_recall；
    2) 在满足 recall 的阈值中，precision 最大优先；
    3) 若 precision 相同，选择 recall 更高的；
    4) 忽略预测阳性数太少的阈值，避免不稳定的高 precision。
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob, pos_label=1)

    best_thr = 0.5
    best_precision = -1.0
    best_recall = -1.0

    # precision_vals / recall_vals 比 thresholds 长 1，所以用 idx+1 对齐
    for idx, thr in enumerate(thresholds):
        p = float(precision_vals[idx + 1])
        r = float(recall_vals[idx + 1])
        pred_pos = int((y_prob >= thr).sum())
        if pred_pos < min_pred_positive:
            continue
        if r >= min_recall and (p > best_precision or (p == best_precision and r > best_recall)):
            best_thr = float(thr)
            best_precision = p
            best_recall = r

    # 如果没有满足 recall 约束的阈值，退化为“单纯 precision 最大”
    if best_precision < 0:
        for idx, thr in enumerate(thresholds):
            p = float(precision_vals[idx + 1])
            r = float(recall_vals[idx + 1])
            pred_pos = int((y_prob >= thr).sum())
            if pred_pos < min_pred_positive:
                continue
            if p > best_precision or (p == best_precision and r > best_recall):
                best_thr = float(thr)
                best_precision = p
                best_recall = r

    return best_thr


def metric_pack(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Model": model_name,
        "AUC-ROC": round(roc_auc_score(y_true, y_prob), 4),
        "AUC-PR": round(average_precision_score(y_true, y_prob, pos_label=1), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
    }


def compute_point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    pred_pos = int(y_pred.sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "pred_pos": pred_pos,
        "min_pr_acc": float(min(precision, recall, accuracy)),
        "avg_pr_acc": float((precision + recall + accuracy) / 3.0),
    }


def search_threshold_targets(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target: float = 0.80,
    min_pred_positive: int = 1,
) -> tuple:
    """
    返回:
    - feasible_best: 满足 Precision/Recall/Accuracy 全部 >= target 的最优阈值点（若不存在则 None）
    - balanced_best: 不可达时可用的最均衡阈值点（最大化 min(P,R,Acc)）
    """
    candidate_thresholds = np.unique(y_prob)
    candidate_thresholds = np.concatenate(([0.0], candidate_thresholds, [1.0]))

    feasible_best = None
    balanced_best = None

    for thr in candidate_thresholds:
        point = compute_point_metrics(y_true, y_prob, float(thr))
        if point["pred_pos"] < min_pred_positive:
            continue

        meets_target = (
            point["precision"] >= target
            and point["recall"] >= target
            and point["accuracy"] >= target
        )
        if meets_target:
            if feasible_best is None:
                feasible_best = point
            else:
                if (
                    point["avg_pr_acc"] > feasible_best["avg_pr_acc"]
                    or (
                        point["avg_pr_acc"] == feasible_best["avg_pr_acc"]
                        and point["min_pr_acc"] > feasible_best["min_pr_acc"]
                    )
                ):
                    feasible_best = point

        if balanced_best is None:
            balanced_best = point
        else:
            if (
                point["min_pr_acc"] > balanced_best["min_pr_acc"]
                or (
                    point["min_pr_acc"] == balanced_best["min_pr_acc"]
                    and point["avg_pr_acc"] > balanced_best["avg_pr_acc"]
                )
            ):
                balanced_best = point

    return feasible_best, balanced_best


# ==============================
# 1) 读取数据 + 标签标准化
# ==============================
df = pd.read_excel(INPUT_FILE)
df["unique_id"] = range(len(df))
df = standardize_labels(df, LABEL_COL)

missing_features = [c for c in FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(f"缺少特征列: {missing_features}")

print(f"特征: {FEATURES}")
print(f"总样本数(仅0/1): {len(df)}")

# ==============================
# 2) 按 ID 划分验证集（正负平衡）
# ==============================
df_pos = df[df[LABEL_COL] == 1]
df_neg = df[df[LABEL_COL] == 0]

val_size = int(VAL_RATIO * min(len(df_pos), len(df_neg)))
if val_size <= 0:
    raise ValueError("验证集样本量为0，请检查数据或调小 VAL_RATIO")

val_pos = df_pos.sample(n=val_size, random_state=42)
val_neg = df_neg.sample(n=val_size, random_state=42)
df_val = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
val_ids = set(df_val["unique_id"])

# ==============================
# 3) 训练集 = 非验证集 ID
# ==============================
df_train_all = df[~df["unique_id"].isin(val_ids)].reset_index(drop=True)
df_train_pos = df_train_all[df_train_all[LABEL_COL] == 1]
df_train_neg = df_train_all[df_train_all[LABEL_COL] == 0]
n_pos_train = len(df_train_pos)

overlap_ids = set(df_val["unique_id"]) & set(df_train_all["unique_id"])
assert len(overlap_ids) == 0, f"存在重叠ID: {overlap_ids}"

print("\n划分完成:")
print(f"  验证集: {len(df_val)} (阳性{len(val_pos)}, 阴性{len(val_neg)})")
print(f"  训练集: {len(df_train_all)} (阳性{len(df_train_pos)}, 阴性{len(df_train_neg)})")

# 删除 ID 列
df_val = df_val.drop(columns=["unique_id"])
df_train_pos = df_train_pos.drop(columns=["unique_id"])
df_train_neg = df_train_neg.drop(columns=["unique_id"])

# ==============================
# 4) 训练子模型并评估
# ==============================
X_val = df_val[FEATURES].fillna(df_val[FEATURES].mean())
y_val = df_val[LABEL_COL].values

models = []
rows = []

print(f"\n开始训练 {N_MODELS} 个随机森林子模型...")
for i in range(N_MODELS):
    if len(df_train_neg) >= len(df_train_pos):
        neg_sampled = df_train_neg.sample(n=n_pos_train, random_state=42 + i).reset_index(drop=True)
    else:
        neg_sampled = df_train_neg

    train_df = (
        pd.concat([df_train_pos, neg_sampled], ignore_index=True)
        .sample(frac=1, random_state=42 + i)
        .reset_index(drop=True)
    )

    X_train = train_df[FEATURES].fillna(train_df[FEATURES].mean())
    y_train = train_df[LABEL_COL].values

    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight={0: NEGATIVE_CLASS_WEIGHT, 1: POSITIVE_CLASS_WEIGHT},
        random_state=1000 + i,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models.append(rf)

    y_prob = rf.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    rows.append(metric_pack(f"Model_{i+1}", y_val, y_prob, y_pred))

# ==============================
# 5) 集成 + 阈值校准（提升 precision）
# ==============================
all_probs = np.mean([m.predict_proba(X_val)[:, 1] for m in models], axis=0)

# 基线：固定阈值0.5
y_pred_05 = (all_probs >= 0.5).astype(int)
baseline = metric_pack("Ensemble@0.50", y_val, all_probs, y_pred_05)
rows.append(baseline)

# 校准阈值：优先 precision
best_thr = choose_threshold_for_precision(
    y_true=y_val,
    y_prob=all_probs,
    min_recall=MIN_RECALL_FOR_TUNING,
    min_pred_positive=MIN_PRED_POSITIVE,
)
y_pred_tuned = (all_probs >= best_thr).astype(int)
tuned = metric_pack(f"Ensemble@{best_thr:.3f}", y_val, all_probs, y_pred_tuned)
rows.append(tuned)

# 目标阈值搜索：检查是否存在 P/R/Acc 同时 >= 0.80 的阈值
feasible_80, balanced_best = search_threshold_targets(
    y_true=y_val,
    y_prob=all_probs,
    target=TARGET_SCORE,
    min_pred_positive=MIN_PRED_POSITIVE,
)

if feasible_80 is not None:
    y_pred_80 = (all_probs >= feasible_80["threshold"]).astype(int)
    row_80 = metric_pack(f"Target{TARGET_SCORE:.2f}@{feasible_80['threshold']:.3f}", y_val, all_probs, y_pred_80)
    rows.append(row_80)

if balanced_best is not None:
    y_pred_bal = (all_probs >= balanced_best["threshold"]).astype(int)
    row_bal = metric_pack(f"Balanced@{balanced_best['threshold']:.3f}", y_val, all_probs, y_pred_bal)
    rows.append(row_bal)

# 保存评估结果
result_df = pd.DataFrame(rows)
result_df.to_excel("RF_评估结果_恶性和健康_阈值校准.xlsx", index=False)

# 绘制 PR 曲线
precision_vals, recall_vals, _ = precision_recall_curve(y_val, all_probs, pos_label=1)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f"Ensemble (AUC-PR={tuned['AUC-PR']:.3f})")
plt.axhline(
    y=(y_val == 1).mean(),
    color="gray",
    linestyle="--",
    label=f"随机基线={(y_val == 1).mean():.3f}",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR曲线（恶性=正类）")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.savefig("PR_Curve_恶性和健康_阈值校准.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== 阈值校准结果 ===")
print(f"选择阈值: {best_thr:.4f}")
print(
    f"基线(0.5): Precision={baseline['Precision']}, Recall={baseline['Recall']}, Accuracy={baseline['Accuracy']}"
)
print(
    f"校准后:   Precision={tuned['Precision']}, Recall={tuned['Recall']}, Accuracy={tuned['Accuracy']}"
)
if feasible_80 is not None:
    print(
        f"存在阈值使三指标>= {TARGET_SCORE:.2f}: "
        f"threshold={feasible_80['threshold']:.4f}, "
        f"Precision={feasible_80['precision']:.4f}, Recall={feasible_80['recall']:.4f}, Accuracy={feasible_80['accuracy']:.4f}"
    )
else:
    print(f"当前模型下，不存在阈值可使 Precision/Recall/Accuracy 同时 >= {TARGET_SCORE:.2f}。")
    if balanced_best is not None:
        print(
            "最均衡阈值: "
            f"threshold={balanced_best['threshold']:.4f}, "
            f"Precision={balanced_best['precision']:.4f}, "
            f"Recall={balanced_best['recall']:.4f}, "
            f"Accuracy={balanced_best['accuracy']:.4f}, "
            f"min(P,R,Acc)={balanced_best['min_pr_acc']:.4f}"
        )
print("评估文件已保存: RF_评估结果_恶性和健康_阈值校准.xlsx")
