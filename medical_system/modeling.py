from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from .config import CLASS_ORDER, FEATURE_COLUMNS, FEATURE_PREPROCESSING_VERSION, LABEL_COLUMN
from .preprocessing import normalize_by_reference_ranges

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


@dataclass
class TrainResult:
    metrics: dict[str, float]
    class_distribution: dict[str, int]
    curve_data: dict[str, Any] | None = None


class BreastRiskModel:
    """Two-expert risk model.

    - Malignant task: normal vs malignant, uses calibrated ExtraTrees/RandomForest blend.
    - Benign task: normal vs benign, uses RandomForest ensemble.
    - Final prediction: combine two binary probabilities into three-class probabilities.
    """

    def __init__(self, random_state: int = 42, n_models: int = 60) -> None:
        self.random_state = random_state
        self.n_models = n_models
        self.malignant_models: list[Any] = []
        self.benign_models: list[Any] = []
        self.benign_malignant_models: list[Any] = []
        self.multiclass_models: list[Any] = []
        self.malignant_model_type: str = "catboost"
        self.benign_model_type: str = "random_forest"
        self.benign_malignant_model_type: str = "random_forest"
        self.multiclass_model_type: str = "balanced_multiclass_ensemble"
        self.malignant_threshold: float = 0.5
        self.benign_threshold: float = 0.5
        self.benign_malignant_threshold: float = 0.5
        self.malignant_guard_threshold: float = 0.34
        self.malignant_guard_margin: float = 0.08
        self.global_feature_importance: dict[str, float] = {}
        self.train_metrics: dict[str, float] = {}
        self.metric_target: float = 0.80
        # Benign task tuning parameters (aligned with your standalone script).
        self.benign_min_recall_for_tuning: float = 0.55
        self.benign_min_pred_positive: int = 8
        self.benign_negative_class_weight: float = 1.35
        self.benign_positive_class_weight: float = 1.0
        # Cap healthy samples for the benign expert so larger latest tables do not skew it.
        self.benign_negative_max_ratio: float = 3.0
        self.malignant_et_weight: float = 0.70
        self.malignant_rf_weight: float = 0.30
        self.malignant_calib_method: str = "isotonic"
        self.malignant_calib_cv: int = 3

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> TrainResult:
        _ = test_size  # keep signature compatible
        data = self._prepare_training_df(df)
        class_distribution = {k: int(v) for k, v in data[LABEL_COLUMN].value_counts().to_dict().items()}

        if class_distribution.get("normal", 0) < 20:
            raise ValueError("正常样本不足，至少需要20条。")
        if class_distribution.get("benign", 0) < 20:
            raise ValueError("良性样本不足，至少需要20条。")
        if class_distribution.get("malignant", 0) < 20:
            raise ValueError("恶性样本不足，至少需要20条。")

        malignant_result = self._train_binary_task(
            data=data,
            positive_label="malignant",
            negative_label="normal",
            task_name="正常vs恶性",
            preferred_model="et_rf_blend",
        )
        benign_result = self._train_binary_task(
            data=data,
            positive_label="benign",
            negative_label="normal",
            task_name="正常vs良性",
            preferred_model="random_forest",
        )
        benign_malignant_result = self._train_binary_task(
            data=data,
            positive_label="benign",
            negative_label="malignant",
            task_name="良性vs恶性",
            preferred_model="random_forest",
        )
        multiclass_result = self._train_multiclass_task(data)

        self.malignant_models = malignant_result["models"]
        self.benign_models = benign_result["models"]
        self.benign_malignant_models = benign_malignant_result["models"]
        self.multiclass_models = multiclass_result["models"]
        self.malignant_model_type = malignant_result["model_type"]
        self.benign_model_type = benign_result["model_type"]
        self.benign_malignant_model_type = benign_malignant_result["model_type"]
        self.multiclass_model_type = multiclass_result["model_type"]
        self.malignant_threshold = float(malignant_result["threshold"])
        self.benign_threshold = float(benign_result["threshold"])
        self.benign_malignant_threshold = float(benign_malignant_result["threshold"])
        self.global_feature_importance = self._merge_feature_importance(
            malignant_result["feature_importance"],
            benign_result["feature_importance"],
        )

        metrics = {
            "auc": float(np.mean([malignant_result["metrics"]["auc_roc"], benign_result["metrics"]["auc_roc"]])),
            "precision": float(
                np.mean([malignant_result["metrics"]["precision"], benign_result["metrics"]["precision"]])
            ),
            "recall": float(np.mean([malignant_result["metrics"]["recall"], benign_result["metrics"]["recall"]])),
            "accuracy": float(np.mean([malignant_result["metrics"]["accuracy"], benign_result["metrics"]["accuracy"]])),
            "malignant_auc": float(malignant_result["metrics"]["auc_roc"]),
            "malignant_auc_pr": float(malignant_result["metrics"]["auc_pr"]),
            "malignant_precision": float(malignant_result["metrics"]["precision"]),
            "malignant_recall": float(malignant_result["metrics"]["recall"]),
            "malignant_accuracy": float(malignant_result["metrics"]["accuracy"]),
            "malignant_val_n": float(malignant_result["metrics"].get("val_n", 0)),
            "malignant_val_positive_n": float(malignant_result["metrics"].get("val_positive_n", 0)),
            "malignant_val_negative_n": float(malignant_result["metrics"].get("val_negative_n", 0)),
            "benign_auc": float(benign_result["metrics"]["auc_roc"]),
            "benign_auc_pr": float(benign_result["metrics"]["auc_pr"]),
            "benign_precision": float(benign_result["metrics"]["precision"]),
            "benign_recall": float(benign_result["metrics"]["recall"]),
            "benign_accuracy": float(benign_result["metrics"]["accuracy"]),
            "benign_val_n": float(benign_result["metrics"].get("val_n", 0)),
            "benign_val_positive_n": float(benign_result["metrics"].get("val_positive_n", 0)),
            "benign_val_negative_n": float(benign_result["metrics"].get("val_negative_n", 0)),
            "benign_malignant_auc": float(benign_malignant_result["metrics"]["auc_roc"]),
            "benign_malignant_auc_pr": float(benign_malignant_result["metrics"]["auc_pr"]),
            "benign_malignant_precision": float(benign_malignant_result["metrics"]["precision"]),
            "benign_malignant_recall": float(benign_malignant_result["metrics"]["recall"]),
            "benign_malignant_accuracy": float(benign_malignant_result["metrics"]["accuracy"]),
            "benign_malignant_val_n": float(benign_malignant_result["metrics"].get("val_n", 0)),
            "benign_malignant_val_positive_n": float(
                benign_malignant_result["metrics"].get("val_positive_n", 0)
            ),
            "benign_malignant_val_negative_n": float(
                benign_malignant_result["metrics"].get("val_negative_n", 0)
            ),
            "multiclass_accuracy": float(multiclass_result["metrics"]["accuracy"]),
            "multiclass_balanced_accuracy": float(multiclass_result["metrics"]["balanced_accuracy"]),
            "multiclass_precision_macro": float(multiclass_result["metrics"]["precision_macro"]),
            "multiclass_recall_macro": float(multiclass_result["metrics"]["recall_macro"]),
            "multiclass_val_n": float(multiclass_result["metrics"].get("val_n", 0)),
            "multiclass_val_normal_n": float(multiclass_result["metrics"].get("val_normal_n", 0)),
            "multiclass_val_benign_n": float(multiclass_result["metrics"].get("val_benign_n", 0)),
            "multiclass_val_malignant_n": float(multiclass_result["metrics"].get("val_malignant_n", 0)),
        }
        self.train_metrics = metrics
        curve_data = {
            "malignant": malignant_result.get("curve_data", {}),
            "benign": benign_result.get("curve_data", {}),
            "benign_malignant": benign_malignant_result.get("curve_data", {}),
        }
        return TrainResult(metrics=metrics, class_distribution=class_distribution, curve_data=curve_data)

    def predict(self, sample_df: pd.DataFrame) -> dict[str, Any]:
        if not self.malignant_models or not self.benign_models:
            raise ValueError("模型尚未训练或加载。")

        data = _prepare_feature_frame(sample_df)
        p_malignant_raw = self._predict_binary_probability(self.malignant_models, data)
        p_benign_raw = self._predict_binary_probability(self.benign_models, data)
        p_benign_disease = self._predict_benign_vs_malignant_probability(data)
        multiclass_probs = self._predict_multiclass_probabilities(data)

        abnormal_score = max(float(p_malignant_raw), float(p_benign_raw))
        m = float(np.clip(p_malignant_raw, 0.0, 1.0))
        b = float(np.clip(p_benign_raw, 0.0, 1.0))
        if p_benign_disease is not None:
            disease_mass = float(np.clip(abnormal_score, 0.0, 1.0))
            benign_share = float(np.clip(p_benign_disease, 0.0, 1.0))
            probs = {
                "normal": 1.0 - disease_mass,
                "benign": disease_mass * benign_share,
                "malignant": disease_mass * (1.0 - benign_share),
            }
        else:
            disease_mass = float(np.clip(abnormal_score, 0.0, 1.0))
            disease_total = m + b
            if disease_total > 0:
                malignant_share = m / disease_total
                benign_share = b / disease_total
            else:
                malignant_share = benign_share = 0.5
            probs = {
                "normal": 1.0 - disease_mass,
                "benign": disease_mass * benign_share,
                "malignant": disease_mass * malignant_share,
            }

        if multiclass_probs:
            multi_weight = 0.60
            expert_weight = 1.0 - multi_weight
            probs = {
                key: float(multi_weight * multiclass_probs.get(key, 0.0) + expert_weight * probs.get(key, 0.0))
                for key in CLASS_ORDER
            }

        total = float(sum(probs.values()))
        if total <= 0:
            probs = {"normal": 1.0, "benign": 0.0, "malignant": 0.0}
        else:
            probs = {k: float(v / total) for k, v in probs.items()}

        predicted_class = max(probs, key=probs.get)
        if (
            probs.get("malignant", 0.0) >= self.malignant_guard_threshold
            and probs.get("malignant", 0.0) + self.malignant_guard_margin >= probs.get("benign", 0.0)
            and probs.get("malignant", 0.0) >= probs.get("normal", 0.0)
        ):
            predicted_class = "malignant"

        # Ensure the displayed probability ordering is consistent with gated decision.
        max_other = max(v for k, v in probs.items() if k != predicted_class)
        if probs[predicted_class] <= max_other:
            probs[predicted_class] = max_other + 1e-6
            total = float(sum(probs.values()))
            probs = {k: float(v / total) for k, v in probs.items()}

        confidence = float(probs[predicted_class])
        contribution = self._sample_contribution(data.iloc[0])

        return {
            "predicted_class": predicted_class,
            "probabilities": probs,
            "confidence": confidence,
            "feature_contribution": contribution,
        }

    def predict_disease_only(self, sample_df: pd.DataFrame) -> dict[str, Any]:
        """Predict only between benign and malignant for external diseased validation sets."""
        if not self.malignant_models or not self.benign_models:
            raise ValueError("模型尚未训练或加载。")

        data = _prepare_feature_frame(sample_df)
        p_benign = self._predict_benign_vs_malignant_probability(data)
        if p_benign is None:
            p_malignant_raw = self._predict_binary_probability(self.malignant_models, data)
            p_benign_raw = self._predict_binary_probability(self.benign_models, data)
            total = p_benign_raw + p_malignant_raw
            p_benign = 0.5 if total <= 0 else float(p_benign_raw / total)

        p_benign = float(np.clip(p_benign, 0.0, 1.0))
        predicted_class = "benign" if p_benign >= self.benign_malignant_threshold else "malignant"
        probabilities = {"benign": p_benign, "malignant": 1.0 - p_benign}
        confidence = float(probabilities[predicted_class])
        return {
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "confidence": confidence,
            "feature_contribution": self._sample_contribution(data.iloc[0]),
        }

    def save(self, path: Path) -> None:
        if not self.malignant_models or not self.benign_models:
            raise ValueError("模型尚未训练，无法保存。")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 5,
            "feature_preprocessing": FEATURE_PREPROCESSING_VERSION,
            "random_state": self.random_state,
            "n_models": self.n_models,
            "malignant_et_weight": self.malignant_et_weight,
            "malignant_rf_weight": self.malignant_rf_weight,
            "malignant_calib_method": self.malignant_calib_method,
            "malignant_calib_cv": self.malignant_calib_cv,
            "malignant_models": self.malignant_models,
            "benign_models": self.benign_models,
            "benign_malignant_models": self.benign_malignant_models,
            "multiclass_models": self.multiclass_models,
            "malignant_model_type": self.malignant_model_type,
            "benign_model_type": self.benign_model_type,
            "benign_malignant_model_type": self.benign_malignant_model_type,
            "multiclass_model_type": self.multiclass_model_type,
            "malignant_threshold": self.malignant_threshold,
            "benign_threshold": self.benign_threshold,
            "benign_malignant_threshold": self.benign_malignant_threshold,
            "malignant_guard_threshold": self.malignant_guard_threshold,
            "malignant_guard_margin": self.malignant_guard_margin,
            "global_feature_importance": self.global_feature_importance,
            "train_metrics": self.train_metrics,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "BreastRiskModel":
        payload = joblib.load(path)
        if isinstance(payload, cls):  # backward compatibility for direct dumps
            return payload
        if payload.get("feature_preprocessing") != FEATURE_PREPROCESSING_VERSION:
            raise ValueError("当前代码已启用参考区间标准化，请重新训练模型后再进行预测。")

        model = cls(
            random_state=int(payload.get("random_state", 42)),
            n_models=int(payload.get("n_models", 60)),
        )
        model.malignant_et_weight = float(payload.get("malignant_et_weight", 0.70))
        model.malignant_rf_weight = float(payload.get("malignant_rf_weight", 0.30))
        model.malignant_calib_method = str(payload.get("malignant_calib_method", "isotonic"))
        model.malignant_calib_cv = int(payload.get("malignant_calib_cv", 3))
        model.malignant_models = payload.get("malignant_models", [])
        model.benign_models = payload.get("benign_models", [])
        model.benign_malignant_models = payload.get("benign_malignant_models", [])
        model.multiclass_models = payload.get("multiclass_models", [])
        model.malignant_model_type = str(payload.get("malignant_model_type", "random_forest"))
        model.benign_model_type = str(payload.get("benign_model_type", "random_forest"))
        model.benign_malignant_model_type = str(payload.get("benign_malignant_model_type", "random_forest"))
        model.multiclass_model_type = str(payload.get("multiclass_model_type", "balanced_multiclass_ensemble"))
        model.malignant_threshold = float(payload.get("malignant_threshold", 0.5))
        model.benign_threshold = float(payload.get("benign_threshold", 0.5))
        model.benign_malignant_threshold = float(payload.get("benign_malignant_threshold", 0.5))
        model.malignant_guard_threshold = float(payload.get("malignant_guard_threshold", 0.34))
        model.malignant_guard_margin = float(payload.get("malignant_guard_margin", 0.08))
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

        data = data[required].copy()
        data[LABEL_COLUMN] = data[LABEL_COLUMN].map(_normalize_label)
        data = data.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)
        return data

    def _train_binary_task(
        self,
        data: pd.DataFrame,
        positive_label: str,
        negative_label: str,
        task_name: str,
        preferred_model: str,
    ) -> dict[str, Any]:
        if preferred_model == "et_rf_blend":
            return self._train_et_rf_blend_task(
                data=data,
                positive_label=positive_label,
                negative_label=negative_label,
                task_name=task_name,
            )
        if preferred_model == "random_forest":
            return self._train_random_forest_task(
                data=data,
                positive_label=positive_label,
                negative_label=negative_label,
                task_name=task_name,
            )

        subset = data[data[LABEL_COLUMN].isin([positive_label, negative_label])].copy().reset_index(drop=True)
        subset["binary_label"] = (subset[LABEL_COLUMN] == positive_label).astype(int)
        if len(subset) < 20:
            raise ValueError(f"{task_name}样本太少，无法训练。")

        train_df, val_df = train_test_split(
            subset,
            test_size=max(0.2, 1 / len(subset)),
            random_state=self.random_state,
            stratify=subset["binary_label"],
        )

        X_val = _prepare_feature_frame(val_df)
        y_val = val_df["binary_label"].astype(int).to_numpy()
        X_train_full = _prepare_feature_frame(train_df)
        y_train_full = train_df["binary_label"].astype(int).to_numpy()

        models: list[Any] = []
        all_probs: list[np.ndarray] = []
        train_rounds = 1 if preferred_model == "catboost" else self.n_models

        for i in range(train_rounds):
            X_train, y_train = self._bootstrap_training_set(
                X_train_full,
                y_train_full,
                random_seed=self.random_state + i,
            )
            model = self._build_model(preferred_model=preferred_model, seed=1000 + i)
            model.fit(X_train, y_train)
            models.append(model)
            all_probs.append(self._predict_model_proba(model, X_val))

        ensemble_prob = np.mean(all_probs, axis=0)
        threshold = 0.5
        y_pred = (ensemble_prob >= threshold).astype(int)

        metrics = {
            "auc_roc": float(roc_auc_score(y_val, ensemble_prob)),
            "auc_pr": float(average_precision_score(y_val, ensemble_prob)),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "val_n": float(len(y_val)),
            "val_positive_n": float(np.sum(y_val == 1)),
            "val_negative_n": float(np.sum(y_val == 0)),
        }

        feature_importance = self._average_feature_importance(models)
        model_type = "catboost" if preferred_model == "catboost" else "random_forest_ensemble"

        return {
            "models": models,
            "metrics": metrics,
            "curve_data": self._build_curve_data(y_val, ensemble_prob),
            "feature_importance": feature_importance,
            "model_type": model_type,
            "threshold": float(threshold),
        }

    def _train_et_rf_blend_task(
        self,
        data: pd.DataFrame,
        positive_label: str,
        negative_label: str,
        task_name: str,
    ) -> dict[str, Any]:
        """Train the malignant expert with the validated ET/RF isotonic blend."""
        subset = data[data[LABEL_COLUMN].isin([positive_label, negative_label])].copy().reset_index(drop=True)
        subset["binary_label"] = (subset[LABEL_COLUMN] == positive_label).astype(int)
        if len(subset) < 20:
            raise ValueError(f"{task_name}样本太少，无法训练。")

        train_df, val_df = train_test_split(
            subset,
            test_size=max(0.2, 1 / len(subset)),
            random_state=self.random_state,
            stratify=subset["binary_label"],
        )

        X_train = _prepare_feature_frame(train_df)
        y_train = train_df["binary_label"].astype(int).to_numpy()
        X_val = _prepare_feature_frame(val_df)
        y_val = val_df["binary_label"].astype(int).to_numpy()

        et_model, rf_model = self._build_malignant_blend_models(seed=self.random_state)
        et_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        blend_model = {
            "type": "et_rf_isotonic_blend",
            "et_model": et_model,
            "rf_model": rf_model,
            "et_weight": self.malignant_et_weight,
            "rf_weight": self.malignant_rf_weight,
        }
        ensemble_prob = self._predict_model_proba(blend_model, X_val)
        threshold = 0.5
        y_pred = (ensemble_prob >= threshold).astype(int)

        metrics = {
            "auc_roc": float(roc_auc_score(y_val, ensemble_prob)),
            "auc_pr": float(average_precision_score(y_val, ensemble_prob)),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "val_n": float(len(y_val)),
            "val_positive_n": float(np.sum(y_val == 1)),
            "val_negative_n": float(np.sum(y_val == 0)),
        }

        return {
            "models": [blend_model],
            "metrics": metrics,
            "curve_data": self._build_curve_data(y_val, ensemble_prob),
            "feature_importance": self._average_feature_importance([blend_model]),
            "model_type": "et_rf_isotonic_blend",
            "threshold": float(threshold),
        }

    def _train_random_forest_task(
        self,
        data: pd.DataFrame,
        positive_label: str,
        negative_label: str,
        task_name: str,
    ) -> dict[str, Any]:
        """RF ensemble training aligned with your standalone script style.

        - Balanced validation set (same number of pos/neg)
        - Majority-class downsampling for each ensemble round
        """
        subset = data[data[LABEL_COLUMN].isin([positive_label, negative_label])].copy().reset_index(drop=True)
        subset["unique_id"] = np.arange(len(subset))
        subset["binary_label"] = (subset[LABEL_COLUMN] == positive_label).astype(int)

        df_pos = subset[subset["binary_label"] == 1].copy().reset_index(drop=True)
        df_neg = subset[subset["binary_label"] == 0].copy().reset_index(drop=True)
        if len(df_pos) < 10 or len(df_neg) < 10:
            raise ValueError(f"{task_name}样本不足，至少每类10条。")

        max_neg = int(max(len(df_pos), len(df_pos) * self.benign_negative_max_ratio))
        if len(df_neg) > max_neg:
            df_neg = df_neg.sample(n=max_neg, random_state=self.random_state).reset_index(drop=True)

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

        X_val = _prepare_feature_frame(df_val)
        y_val = df_val["binary_label"].astype(int).to_numpy()

        models: list[Any] = []
        all_probs: list[np.ndarray] = []
        for i in range(self.n_models):
            if len(df_train_neg) >= n_pos_train:
                neg_sampled = df_train_neg.sample(n=n_pos_train, random_state=self.random_state + i).reset_index(
                    drop=True
                )
            else:
                neg_sampled = df_train_neg.copy().reset_index(drop=True)

            df_train = (
                pd.concat([df_train_pos, neg_sampled], ignore_index=True)
                .sample(frac=1.0, random_state=self.random_state + i)
                .reset_index(drop=True)
            )
            X_train = _prepare_feature_frame(df_train)
            y_train = df_train["binary_label"].astype(int).to_numpy()

            rf = RandomForestClassifier(
                n_estimators=120,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight={0: self.benign_negative_class_weight, 1: self.benign_positive_class_weight},
                random_state=1000 + i,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            models.append(rf)
            all_probs.append(self._predict_model_proba(rf, X_val))

        ensemble_prob = np.mean(all_probs, axis=0)
        threshold = 0.5
        y_pred = (ensemble_prob >= threshold).astype(int)

        metrics = {
            "auc_roc": float(roc_auc_score(y_val, ensemble_prob)),
            "auc_pr": float(average_precision_score(y_val, ensemble_prob)),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "val_n": float(len(y_val)),
            "val_positive_n": float(np.sum(y_val == 1)),
            "val_negative_n": float(np.sum(y_val == 0)),
        }
        feature_importance = self._average_feature_importance(models)
        return {
            "models": models,
            "metrics": metrics,
            "curve_data": self._build_curve_data(y_val, ensemble_prob),
            "feature_importance": feature_importance,
            "model_type": "random_forest_ensemble",
            "threshold": float(threshold),
        }

    def _find_best_threshold_benign(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Threshold strategy aligned with the standalone benign script.

        Priority:
        1) If there exists a threshold with Precision/Recall/Accuracy all >= target, use the best one.
        2) Otherwise, optimize precision under recall floor.
        3) Fallback to max precision if recall floor cannot be satisfied.
        """
        feasible_best, _ = self._search_threshold_targets(
            y_true=y_true,
            y_prob=y_prob,
            target=self.metric_target,
            min_pred_positive=self.benign_min_pred_positive,
        )
        if feasible_best is not None:
            return float(feasible_best["threshold"])
        return self._choose_threshold_for_precision(
            y_true=y_true,
            y_prob=y_prob,
            min_recall=self.benign_min_recall_for_tuning,
            min_pred_positive=self.benign_min_pred_positive,
        )

    def _choose_threshold_for_precision(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        min_recall: float = 0.55,
        min_pred_positive: int = 8,
    ) -> float:
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob, pos_label=1)
        best_thr = 0.5
        best_precision = -1.0
        best_recall = -1.0

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

    def _search_threshold_targets(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target: float = 0.80,
        min_pred_positive: int = 1,
    ) -> tuple[dict[str, float] | None, dict[str, float] | None]:
        candidate_thresholds = np.unique(y_prob)
        candidate_thresholds = np.concatenate(([0.0], candidate_thresholds, [1.0]))

        feasible_best = None
        balanced_best = None
        for thr in candidate_thresholds:
            point = self._compute_point_metrics(y_true, y_prob, float(thr))
            if point["pred_pos"] < min_pred_positive:
                continue

            meets_target = (
                point["precision"] >= target and point["recall"] >= target and point["accuracy"] >= target
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

    def _compute_point_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        return {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "pred_pos": float(int(y_pred.sum())),
            "min_pr_acc": float(min(precision, recall, accuracy)),
            "avg_pr_acc": float((precision + recall + accuracy) / 3.0),
        }

    def _build_curve_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        positive_rate = float(np.mean(y_true == 1))
        return {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "precision": [float(v) for v in precision_vals],
            "recall": [float(v) for v in recall_vals],
            "positive_rate": positive_rate,
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "auc_pr": float(average_precision_score(y_true, y_prob, pos_label=1)),
        }

    def _build_model(self, preferred_model: str, seed: int) -> Any:
        if preferred_model == "catboost":
            if CatBoostClassifier is None:
                raise RuntimeError("未安装catboost，无法训练“健康vs恶性”专家模型。请先安装 catboost。")
            return CatBoostClassifier(
                iterations=300,
                depth=4,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )

        return RandomForestClassifier(
            n_estimators=240,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )

    def _build_malignant_blend_models(self, seed: int) -> tuple[CalibratedClassifierCV, CalibratedClassifierCV]:
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
        return (
            CalibratedClassifierCV(estimator=et, method=self.malignant_calib_method, cv=self.malignant_calib_cv),
            CalibratedClassifierCV(estimator=rf, method=self.malignant_calib_method, cv=self.malignant_calib_cv),
        )

    def _bootstrap_training_set(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        random_seed: int,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(random_seed)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return X.copy(), y.copy()

        target_size = max(len(pos_idx), len(neg_idx))
        pos_sample = rng.choice(pos_idx, size=target_size, replace=True)
        neg_sample = rng.choice(neg_idx, size=target_size, replace=True)
        sampled_idx = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(sampled_idx)
        X_sampled = X.iloc[sampled_idx].reset_index(drop=True)
        y_sampled = y[sampled_idx]
        return X_sampled, y_sampled

    def _predict_model_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if isinstance(model, dict) and model.get("type") == "et_rf_isotonic_blend":
            et_weight = float(model.get("et_weight", 0.70))
            rf_weight = float(model.get("rf_weight", 0.30))
            total_weight = et_weight + rf_weight
            if total_weight <= 0:
                et_weight, rf_weight, total_weight = 0.70, 0.30, 1.0
            et_prob = self._predict_model_proba(model["et_model"], X)
            rf_prob = self._predict_model_proba(model["rf_model"], X)
            return (et_weight * et_prob + rf_weight * rf_prob) / total_weight

        proba = model.predict_proba(X)
        if isinstance(proba, list):
            proba = np.asarray(proba)
        return np.asarray(proba)[:, 1]

    def _predict_binary_probability(self, models: list[Any], X: pd.DataFrame) -> float:
        probs = [self._predict_model_proba(model, X)[0] for model in models]
        return float(np.mean(probs))

    def _predict_benign_vs_malignant_probability(self, X: pd.DataFrame) -> float | None:
        if not self.benign_malignant_models:
            return None
        return self._predict_binary_probability(self.benign_malignant_models, X)

    def _predict_multiclass_probabilities(self, X: pd.DataFrame) -> dict[str, float] | None:
        if not self.multiclass_models:
            return None

        vectors: list[np.ndarray] = []
        for model in self.multiclass_models:
            proba = np.asarray(model.predict_proba(X))[0]
            aligned = np.zeros(len(CLASS_ORDER), dtype=float)
            for idx, class_name in enumerate(getattr(model, "classes_", [])):
                if class_name in CLASS_ORDER:
                    aligned[CLASS_ORDER.index(str(class_name))] = float(proba[idx])
            total = float(aligned.sum())
            if total > 0:
                vectors.append(aligned / total)

        if not vectors:
            return None

        avg = np.mean(vectors, axis=0)
        total = float(avg.sum())
        if total <= 0:
            return None
        avg = avg / total
        return {class_name: float(avg[idx]) for idx, class_name in enumerate(CLASS_ORDER)}

    def _train_multiclass_task(self, data: pd.DataFrame) -> dict[str, Any]:
        subset = data[data[LABEL_COLUMN].isin(CLASS_ORDER)].copy().reset_index(drop=True)
        if subset[LABEL_COLUMN].nunique() < 3:
            raise ValueError("三分类训练至少需要正常、良性、恶性三类样本。")

        train_df, val_df = train_test_split(
            subset,
            test_size=max(0.2, 3 / len(subset)),
            random_state=self.random_state,
            stratify=subset[LABEL_COLUMN],
        )

        X_val = _prepare_feature_frame(val_df)
        y_val = val_df[LABEL_COLUMN].astype(str).to_numpy()

        models: list[Any] = []
        prob_vectors: list[np.ndarray] = []
        for i in range(self.n_models):
            sampled_train = self._balanced_multiclass_training_frame(train_df, seed=self.random_state + i)
            X_train = _prepare_feature_frame(sampled_train)
            y_train = sampled_train[LABEL_COLUMN].astype(str).to_numpy()

            if i % 2 == 0:
                clf = RandomForestClassifier(
                    n_estimators=160,
                    max_depth=6,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=3000 + i,
                    n_jobs=-1,
                )
            else:
                clf = ExtraTreesClassifier(
                    n_estimators=180,
                    max_depth=6,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=3000 + i,
                    n_jobs=-1,
                )

            clf.fit(X_train, y_train)
            models.append(clf)
            prob_vectors.append(self._predict_multiclass_model_proba(clf, X_val))

        ensemble_prob = np.mean(prob_vectors, axis=0)
        pred_idx = np.argmax(ensemble_prob, axis=1)
        y_pred = np.asarray([CLASS_ORDER[idx] for idx in pred_idx])

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
            "precision_macro": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
            "val_n": float(len(y_val)),
            "val_normal_n": float(np.sum(y_val == "normal")),
            "val_benign_n": float(np.sum(y_val == "benign")),
            "val_malignant_n": float(np.sum(y_val == "malignant")),
        }

        return {
            "models": models,
            "metrics": metrics,
            "feature_importance": self._average_feature_importance(models),
            "model_type": "balanced_multiclass_ensemble",
        }

    def _balanced_multiclass_training_frame(self, train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
        counts = train_df[LABEL_COLUMN].value_counts()
        min_count = int(counts.min())
        max_count = int(counts.max())
        target_count = int(min(max_count, max(80, min_count * 3)))

        parts: list[pd.DataFrame] = []
        for class_name in CLASS_ORDER:
            class_df = train_df[train_df[LABEL_COLUMN] == class_name]
            if class_df.empty:
                continue
            replace = len(class_df) < target_count
            parts.append(class_df.sample(n=target_count, replace=replace, random_state=seed + len(parts)))

        return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def _predict_multiclass_model_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(model.predict_proba(X))
        aligned = np.zeros((len(X), len(CLASS_ORDER)), dtype=float)
        for idx, class_name in enumerate(getattr(model, "classes_", [])):
            if class_name in CLASS_ORDER:
                aligned[:, CLASS_ORDER.index(str(class_name))] = raw[:, idx]
        totals = aligned.sum(axis=1)
        totals[totals <= 0] = 1.0
        return aligned / totals[:, None]

    def _find_best_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, target_min: float = 0.80) -> float:
        # Use probability-driven candidates instead of a coarse fixed grid,
        # so thresholds like 0.4858 can be discovered.
        prob = np.asarray(y_prob, dtype=float)
        unique_prob = np.unique(np.clip(prob, 1e-8, 1 - 1e-8))
        if unique_prob.size == 0:
            return 0.5

        mids = (unique_prob[:-1] + unique_prob[1:]) / 2.0 if unique_prob.size > 1 else np.array([], dtype=float)
        boundary = np.array([unique_prob[0], unique_prob[-1]], dtype=float)
        dense = np.linspace(max(0.01, float(unique_prob.min()) - 0.05), min(0.99, float(unique_prob.max()) + 0.05), 300)
        candidates = np.unique(np.concatenate([unique_prob, mids, boundary, dense, np.array([0.5])]))
        candidates = candidates[(candidates > 0) & (candidates < 1)]

        feasible: list[tuple[float, float, float, float]] = []
        for thr in candidates:
            y_pred = (y_prob >= thr).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            if precision >= target_min and recall >= target_min and accuracy >= target_min:
                feasible.append((float(thr), float(precision), float(recall), float(accuracy)))

        # 优先选三指标同时达标的阈值，按“最弱指标最大化”排序
        if feasible:
            feasible.sort(key=lambda x: (min(x[1], x[2], x[3]), (x[1] + x[2] + x[3]) / 3.0), reverse=True)
            return feasible[0][0]

        best_thr = 0.5
        best_score = -1.0
        for thr in candidates:
            y_pred = (y_prob >= thr).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            score = min(precision, recall, accuracy)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        return best_thr

    def _apply_threshold_calibration(self, prob: float, threshold: float) -> float:
        """Map task-specific threshold to decision midpoint 0.5.

        This keeps ranking monotonic while making tuned thresholds effective in fusion.
        """
        p = float(np.clip(prob, 1e-8, 1.0 - 1e-8))
        t = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
        if p <= t:
            return float(0.5 * (p / t))
        return float(0.5 + 0.5 * ((p - t) / (1.0 - t)))

    def _average_feature_importance(self, models: list[Any]) -> dict[str, float]:
        vectors: list[np.ndarray] = []
        for model in models:
            vectors.extend(self._feature_importance_vectors(model))

        if not vectors:
            return {name: 1.0 / len(FEATURE_COLUMNS) for name in FEATURE_COLUMNS}

        avg = np.mean(vectors, axis=0)
        total = float(np.sum(avg))
        if total <= 0:
            return {name: 1.0 / len(FEATURE_COLUMNS) for name in FEATURE_COLUMNS}
        return {name: float(v / total) for name, v in zip(FEATURE_COLUMNS, avg)}

    def _feature_importance_vectors(self, model: Any) -> list[np.ndarray]:
        if isinstance(model, dict) and model.get("type") == "et_rf_isotonic_blend":
            vectors = []
            vectors.extend(self._feature_importance_vectors(model.get("et_model")))
            vectors.extend(self._feature_importance_vectors(model.get("rf_model")))
            return vectors

        if hasattr(model, "feature_importances_"):
            return [np.asarray(model.feature_importances_, dtype=float)]

        if hasattr(model, "get_feature_importance"):
            try:
                return [np.asarray(model.get_feature_importance(), dtype=float)]
            except Exception:
                return []

        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated:
            vectors = []
            for item in calibrated:
                base_model = getattr(item, "estimator", None) or getattr(item, "base_estimator", None)
                if base_model is not None:
                    vectors.extend(self._feature_importance_vectors(base_model))
            return vectors

        return []

    def _merge_feature_importance(
        self,
        fi_malignant: dict[str, float],
        fi_benign: dict[str, float],
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
    return mapping.get(v)


def _normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "akr1b10": "akr1b10",
        "akr1b-10": "akr1b10",
        "akr1b 10": "akr1b10",
        "ca19-9": "ca19_9",
        "ca19 9": "ca19_9",
        "ca199": "ca19_9",
        "ca19.9": "ca19_9",
        "ca19_9": "ca19_9",
        "nse": "nse",
        "ca125": "ca125",
        "ca-125": "ca125",
        "ca 125": "ca125",
        "ca153": "ca153",
        "ca15-3": "ca153",
        "ca15 3": "ca153",
        "ca15_3": "ca153",
        "cea": "cea",
    }
    rename_dict: dict[str, str] = {}
    for col in df.columns:
        normalized = str(col).strip().lower()
        if normalized in col_map:
            rename_dict[col] = col_map[normalized]
    if rename_dict:
        return df.rename(columns=rename_dict)
    return df


def _prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    data = _normalize_feature_columns(data)
    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing:
        readable_missing = ["ca19-9" if col == "ca19_9" else col for col in missing]
        raise ValueError(f"缺少必要特征: {', '.join(readable_missing)}")

    data = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data = normalize_by_reference_ranges(data)
    data = data.fillna(data.mean(numeric_only=True)).fillna(0.0)
    return data
