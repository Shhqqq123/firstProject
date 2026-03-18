from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import CLASS_ORDER, FEATURE_COLUMNS, LABEL_COLUMN
from .preprocessing import clip_outliers_iqr, create_preprocessor


@dataclass
class TrainResult:
    metrics: dict[str, float]
    class_distribution: dict[str, int]


class BreastRiskModel:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.global_feature_importance: dict[str, float] = {}
        self.train_metrics: dict[str, float] = {}

    def build_pipeline(self) -> Pipeline:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=self.random_state,
        )
        et = ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=self.random_state,
        )
        gb = GradientBoostingClassifier(random_state=self.random_state)
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("et", et), ("gb", gb)],
            voting="soft",
        )
        return Pipeline(
            steps=[
                ("preprocess", create_preprocessor()),
                ("classifier", ensemble),
            ]
        )

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> TrainResult:
        data = self._prepare_training_df(df)
        if len(data) < 30:
            raise ValueError("At least 30 labeled samples are required.")

        X = data[FEATURE_COLUMNS]
        y = data[LABEL_COLUMN].astype(str)

        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state,
        )

        self.pipeline = self.build_pipeline()
        self.pipeline.fit(x_train, y_train)

        y_pred = self.pipeline.predict(x_test)
        y_proba = self.pipeline.predict_proba(x_test)
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        y_test_one_hot = pd.get_dummies(y_test).reindex(columns=classes, fill_value=0).to_numpy()

        auc = roc_auc_score(y_test_one_hot, y_proba, multi_class="ovr", average="macro")
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "auc": float(auc),
        }
        self.train_metrics = metrics
        self.global_feature_importance = self._extract_feature_importance()

        distribution = y.value_counts().to_dict()
        return TrainResult(metrics=metrics, class_distribution={k: int(v) for k, v in distribution.items()})

    def predict(self, sample_df: pd.DataFrame) -> dict[str, Any]:
        if self.pipeline is None:
            raise ValueError("Model is not loaded or trained.")
        data = sample_df.copy()[FEATURE_COLUMNS]
        pred_class = str(self.pipeline.predict(data)[0])
        probs_raw = self.pipeline.predict_proba(data)[0]
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        probs = {cls: float(prob) for cls, prob in zip(classes, probs_raw)}
        for cls in CLASS_ORDER:
            probs.setdefault(cls, 0.0)

        contribution = self._sample_contribution(data.iloc[0])
        confidence = max(probs.values())
        return {
            "predicted_class": pred_class,
            "probabilities": probs,
            "confidence": confidence,
            "feature_contribution": contribution,
        }

    def save(self, path: Path) -> None:
        if self.pipeline is None:
            raise ValueError("Model is not trained.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "global_feature_importance": self.global_feature_importance,
            "train_metrics": self.train_metrics,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "BreastRiskModel":
        payload = joblib.load(path)
        model = cls()
        model.pipeline = payload["pipeline"]
        model.global_feature_importance = payload.get("global_feature_importance", {})
        model.train_metrics = payload.get("train_metrics", {})
        return model

    def _prepare_training_df(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data.columns = [str(c).strip().lower() for c in data.columns]
        required = FEATURE_COLUMNS + [LABEL_COLUMN]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing training columns: {', '.join(missing)}")
        data = data[required]
        data = clip_outliers_iqr(data, FEATURE_COLUMNS)
        data = data[data[LABEL_COLUMN].isin(CLASS_ORDER)]
        return data.dropna(subset=[LABEL_COLUMN])

    def _extract_feature_importance(self) -> dict[str, float]:
        if self.pipeline is None:
            return {}
        clf: VotingClassifier = self.pipeline.named_steps["classifier"]
        vectors = []
        if hasattr(clf, "estimators_"):
            for estimator in clf.estimators_:
                if hasattr(estimator, "feature_importances_"):
                    vectors.append(np.array(estimator.feature_importances_, dtype=float))
        if not vectors:
            return {f: 1.0 / len(FEATURE_COLUMNS) for f in FEATURE_COLUMNS}
        avg = np.mean(vectors, axis=0)
        total = float(np.sum(avg))
        if total <= 0:
            return {f: 1.0 / len(FEATURE_COLUMNS) for f in FEATURE_COLUMNS}
        return {name: float(v / total) for name, v in zip(FEATURE_COLUMNS, avg)}

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

