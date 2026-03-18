from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

DB_PATH = DATA_DIR / "medical_system.db"
MODEL_PATH = MODEL_DIR / "breast_risk_model.joblib"

FEATURE_COLUMNS = ["akr1b10", "ca19_9", "nse", "ca125", "ca153", "cea"]
LABEL_COLUMN = "label"

CLASS_ORDER = ["normal", "benign", "malignant"]
CLASS_NAME_MAP = {
    "normal": "Normal",
    "benign": "Benign",
    "malignant": "Malignant",
}


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

