from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_training_data.csv"


def main() -> None:
    rng = np.random.default_rng(42)

    n_normal = 300
    n_benign = 200
    n_malignant = 500

    normal = _make_group(rng, n_normal, (8, 12, 10, 16, 20, 2.8), "normal")
    benign = _make_group(rng, n_benign, (11, 17, 14, 24, 30, 3.8), "benign")
    malignant = _make_group(rng, n_malignant, (18, 35, 24, 40, 55, 8.2), "malignant")

    df = pd.concat([normal, benign, malignant], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"saved: {OUT_PATH}")


def _make_group(rng: np.random.Generator, n: int, center: tuple[float, ...], label: str) -> pd.DataFrame:
    scales = np.array([2.0, 4.5, 3.0, 5.0, 8.0, 1.0])
    x = rng.normal(loc=np.array(center), scale=scales, size=(n, 6))
    x = np.clip(x, a_min=0.1, a_max=None)
    dates = pd.date_range("2025-01-01", periods=n, freq="D").astype(str)
    return pd.DataFrame(
        {
            "test_date": dates,
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


if __name__ == "__main__":
    main()

