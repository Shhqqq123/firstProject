from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from .config import CLASS_ORDER, FEATURE_COLUMNS, LABEL_COLUMN
from .preprocessing import clip_outliers_iqr


@dataclass
class TrainResult:
    metrics: dict[str, float]
    class_distribution: dict[str, int]


class BreastRiskModel:
    """按用户训练代码思路实现的后端模型。

    核心逻辑：
    1) 两个二分类任务
       - 任务A: 正常(normal) vs 恶性(malignant)
       - 任务B: 正常(normal) vs 良性(benign)
    2) 每个任务训练 60 个随机森林子模型并集成平均概率
    3) 推理时把两个任务概率合成为三分类概率:
       - malignant = p_m
       - benign = (1 - p_m) * p_b
       - normal = (1 - p_m) * (1 - p_b)
    """

    def __init__(self, random_state: int = 42, n_models: int = 60) -> None:
        self.random_state = random_state
        self.n_models = n_models
        self.malignant_models: list[RandomForestClassifier] = []
        self.benign_models: list[RandomForestClassifier] = []
        self.global_feature_importance: dict[str, float] = {}
        self.train_metrics: dict[str, float] = {}

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> TrainResult:
        _ = test_size  # 为兼容旧接口保留入参
        data = self._prepare_training_df(df)
        dist = data[LABEL_COLUMN].value_counts().to_dict()
        class_distribution = {k: int(v) for k, v in dist.items()}

        if class_distribution.get("normal", 0) < 20:
            raise ValueError("正常样本不足，至少需要20条。")
        if class_distribution.get("benign", 0) < 20:
            raise ValueError("良性样本不足，至少需要20条。")
        if class_distribution.get("malignant", 0) < 20:
            raise ValueError("恶性样本不足，至少需要20条。")

        malignant_result = self._train_binary_ensemble(
            data=data,
            positive_label="malignant",
            negative_label="normal",
            task_name="正常vs恶性",
        )
        benign_result = self._train_binary_ensemble(
            data=data,
            positive_label="benign",
            negative_label="normal",
            task_name="正常vs良性",
        )

        self.malignant_models = malignant_result["models"]
        self.benign_models = benign_result["models"]

        # 聚合两个任务特征重要性作为全局解释性权重
        self.global_feature_importance = self._merge_feature_importance(
            malignant_result["feature_importance"], benign_result["feature_importance"]
        )

        # 页面主指标：取两个任务的平均值，便于单卡片展示
        metrics = {
            "auc": float(np.mean([malignant_result["metrics"]["auc_roc"], benign_result["metrics"]["auc_roc"]])),
            "precision": float(
                np.mean([malignant_result["metrics"]["precision"], benign_result["metrics"]["precision"]])
            ),
            "recall": float(np.mean([malignant_result["metrics"]["recall"], benign_result["metrics"]["recall"]])),
            "accuracy": float(
                np.mean([malignant_result["metrics"]["accuracy"], benign_result["metrics"]["accuracy"]])
            ),
            "malignant_auc": float(malignant_result["metrics"]["auc_roc"]),
            "malignant_precision": float(malignant_result["metrics"]["precision"]),
            "malignant_recall": float(malignant_result["metrics"]["recall"]),
            "benign_auc": float(benign_result["metrics"]["auc_roc"]),
            "benign_precision": float(benign_result["metrics"]["precision"]),
            "benign_recall": float(benign_result["metrics"]["recall"]),
        }
        self.train_metrics = metrics
        return TrainResult(metrics=metrics, class_distribution=class_distribution)

    def predict(self, sample_df: pd.DataFrame) -> dict[str, Any]:
        if not self.malignant_models or not self.benign_models:
            raise ValueError("模型尚未训练或加载。")

        data = sample_df.copy()
        data = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        data = data.fillna(data.mean(numeric_only=True))

        p_malignant = float(np.mean([m.predict_proba(data)[:, 1][0] for m in self.malignant_models]))
        p_benign_binary = float(np.mean([m.predict_proba(data)[:, 1][0] for m in self.benign_models]))

        # 由两个二分类任务概率合成三分类概率（总和为1）
        malignant_prob = p_malignant
        benign_prob = (1.0 - p_malignant) * p_benign_binary
        normal_prob = (1.0 - p_malignant) * (1.0 - p_benign_binary)
        probs = {"normal": normal_prob, "benign": benign_prob, "malignant": malignant_prob}

        predicted_class = max(probs, key=probs.get)
        confidence = float(probs[predicted_class])
        contribution = self._sample_contribution(data.iloc[0])

        return {
            "predicted_class": predicted_class,
            "probabilities": probs,
            "confidence": confidence,
            "feature_contribution": contribution,
        }

    def save(self, path: Path) -> None:
        if not self.malignant_models or not self.benign_models:
            raise ValueError("模型尚未训练，无法保存。")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "random_state": self.random_state,
            "n_models": self.n_models,
            "malignant_models": self.malignant_models,
            "benign_models": self.benign_models,
            "global_feature_importance": self.global_feature_importance,
            "train_metrics": self.train_metrics,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "BreastRiskModel":
        payload = joblib.load(path)
        model = cls(
            random_state=int(payload.get("random_state", 42)),
            n_models=int(payload.get("n_models", 60)),
        )
        model.malignant_models = payload.get("malignant_models", [])
        model.benign_models = payload.get("benign_models", [])
        model.global_feature_importance = payload.get("global_feature_importance", {})
        model.train_metrics = payload.get("train_metrics", {})
        return model

    def _prepare_training_df(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data.columns = [str(c).strip().lower() for c in data.columns]
        data = _normalize_feature_columns(data)
        required = FEATURE_COLUMNS + [LABEL_COLUMN]
        missing = [col for col in required if col not in data.columns]
        if missing:
            readable_missing = ["ca19-9" if col == "ca19_9" else col for col in missing]
            raise ValueError(f"训练数据缺少字段: {', '.join(readable_missing)}")

        data = data[required]
        data[LABEL_COLUMN] = data[LABEL_COLUMN].map(_normalize_label)
        data = data.dropna(subset=[LABEL_COLUMN])
        data = clip_outliers_iqr(data, FEATURE_COLUMNS)
        return data

    def _train_binary_ensemble(
        self,
        data: pd.DataFrame,
        positive_label: str,
        negative_label: str,
        task_name: str,
    ) -> dict[str, Any]:
        subset = data[data[LABEL_COLUMN].isin([positive_label, negative_label])].copy().reset_index(drop=True)
        subset["unique_id"] = np.arange(len(subset))
        subset["binary_label"] = (subset[LABEL_COLUMN] == positive_label).astype(int)

        df_pos = subset[subset["binary_label"] == 1]
        df_neg = subset[subset["binary_label"] == 0]

        if len(df_pos) < 10 or len(df_neg) < 10:
            raise ValueError(f"{task_name}样本不足，至少每类10条。")

        val_size = int(0.2 * min(len(df_pos), len(df_neg)))
        val_size = max(val_size, 1)

        val_pos = df_pos.sample(n=val_size, random_state=self.random_state)
        val_neg = df_neg.sample(n=val_size, random_state=self.random_state)
        df_val = pd.concat([val_pos, val_neg], ignore_index=True).sample(frac=1.0, random_state=self.random_state)

        val_ids = set(df_val["unique_id"].tolist())
        df_train_pool = subset[~subset["unique_id"].isin(val_ids)].reset_index(drop=True)
        df_train_pos = df_train_pool[df_train_pool["binary_label"] == 1].copy().reset_index(drop=True)
        df_train_neg = df_train_pool[df_train_pool["binary_label"] == 0].copy().reset_index(drop=True)

        n_pos_train = len(df_train_pos)
        if n_pos_train < 5 or len(df_train_neg) < 5:
            raise ValueError(f"{task_name}训练集太小，无法稳定训练。")

        X_val = df_val[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        X_val = X_val.fillna(X_val.mean(numeric_only=True))
        y_val = df_val["binary_label"].astype(int)

        models: list[RandomForestClassifier] = []
        all_probs: list[np.ndarray] = []

        for i in range(self.n_models):
            # 按用户代码思路：负类下采样到与正类一样多
            if len(df_train_neg) >= n_pos_train:
                neg_sampled = df_train_neg.sample(n=n_pos_train, random_state=self.random_state + i).reset_index(drop=True)
            else:
                neg_sampled = df_train_neg.copy().reset_index(drop=True)

            df_train = (
                pd.concat([df_train_pos, neg_sampled], ignore_index=True)
                .sample(frac=1.0, random_state=self.random_state + i)
                .reset_index(drop=True)
            )

            X_train = df_train[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
            X_train = X_train.fillna(X_train.mean(numeric_only=True))
            y_train = df_train["binary_label"].astype(int)

            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=1000 + i,
                n_jobs=1,
            )
            rf.fit(X_train, y_train)
            models.append(rf)
            all_probs.append(rf.predict_proba(X_val)[:, 1])

        ensemble_prob = np.mean(all_probs, axis=0)
        y_pred = (ensemble_prob >= 0.5).astype(int)
        metrics = {
            "auc_roc": float(roc_auc_score(y_val, ensemble_prob)),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        }

        fi_vectors = [m.feature_importances_ for m in models if hasattr(m, "feature_importances_")]
        fi_avg = np.mean(fi_vectors, axis=0) if fi_vectors else np.ones(len(FEATURE_COLUMNS))
        fi_sum = float(np.sum(fi_avg))
        if fi_sum <= 0:
            feature_importance = {name: 1.0 / len(FEATURE_COLUMNS) for name in FEATURE_COLUMNS}
        else:
            feature_importance = {name: float(v / fi_sum) for name, v in zip(FEATURE_COLUMNS, fi_avg)}

        return {
            "models": models,
            "metrics": metrics,
            "feature_importance": feature_importance,
        }

    def _merge_feature_importance(
        self, fi_malignant: dict[str, float], fi_benign: dict[str, float]
    ) -> dict[str, float]:
        merged = {}
        for name in FEATURE_COLUMNS:
            merged[name] = float((fi_malignant.get(name, 0.0) + fi_benign.get(name, 0.0)) / 2.0)
        total = float(sum(merged.values()))
        if total <= 0:
            return {name: 1.0 / len(FEATURE_COLUMNS) for name in FEATURE_COLUMNS}
        return {name: float(v / total) for name, v in merged.items()}

    def _sample_contribution(self, row: pd.Series) -> dict[str, float]:
        if not self.global_feature_importance:
            return {f: 1.0 / len(FEATURE_COLUMNS) for f in FEATURE_COLUMNS}
        values = row[FEATURE_COLUMNS].astype(float).to_numpy()
        weights = np.array([self.global_feature_importance.get(c, 0.0) for c in FEATURE_COLUMNS], dtype=float)
        scores = np.abs(values * weights)
        total = float(np.sum(scores))
        if total <= 0:
            return self.global_feature_importance
        return {name: float(v / total) for name, v in zip(FEATURE_COLUMNS, scores)}


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    mapping = {
        "normal": "normal",
        "benign": "benign",
        "malignant": "malignant",
        "正常": "normal",
        "良性": "benign",
        "恶性": "malignant",
    }
    if v in mapping:
        return mapping[v]
    return None


def _normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一外部列名别名到系统内部列名。"""
    col_map = {
        "ca19-9": "ca19_9",
        "ca19 9": "ca19_9",
        "ca19_9": "ca19_9",
    }
    rename_dict: dict[str, str] = {}
    for col in df.columns:
        normalized = str(col).strip().lower()
        if normalized in col_map:
            rename_dict[col] = col_map[normalized]
    if rename_dict:
        return df.rename(columns=rename_dict)
    return df
