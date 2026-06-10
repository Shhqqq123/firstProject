import os
import sys
from pathlib import Path

# 项目根目录（app.py 所在目录）
BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_runtime_dir() -> Path:
    """Return a writable runtime directory for database, models and reports."""
    override = os.environ.get("BREAST_RISK_DATA_DIR")
    if override:
        return Path(override).expanduser()

    base_text = str(BASE_DIR).lower()
    is_installed_app = getattr(sys, "frozen", False) or "\\program files\\" in base_text or "/program files/" in base_text
    if is_installed_app:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "BreastRiskDesktop"
        return Path.home() / "AppData" / "Local" / "BreastRiskDesktop"

    return BASE_DIR


RUNTIME_DIR = _resolve_runtime_dir()
# 运行数据目录
DATA_DIR = RUNTIME_DIR / "data"
# 模型目录
MODEL_DIR = RUNTIME_DIR / "models"
# 报告目录
REPORT_DIR = RUNTIME_DIR / "reports"

# SQLite 数据库文件路径
DB_PATH = DATA_DIR / "medical_system.db"
# 模型保存路径
MODEL_PATH = MODEL_DIR / "breast_risk_model.joblib"

# 六项肿瘤标志物特征
FEATURE_COLUMNS = ["akr1b10", "ca19_9", "nse", "ca125", "ca153", "cea"]
LABEL_COLUMN = "label"

# 默认参考区间上限，用于消除不同检测设备/医院量纲差异。
# 变换方式见 medical_system.preprocessing.normalize_by_reference_ranges。
REFERENCE_UPPER_LIMITS = {
    "akr1b10": 373.5,
    "ca19_9": 25.8,
    "nse": 15.8,
    "ca125": 36.0,
    "ca153": 26.2,
    "cea": 5.77,
}
FEATURE_PREPROCESSING_VERSION = "reference_log1p_uln_v1"

FEATURE_DISPLAY_NAMES = {
    "akr1b10": "AKR1B10",
    "ca19_9": "CA19-9",
    "nse": "NSE",
    "ca125": "CA125",
    "ca153": "CA15-3",
    "cea": "CEA",
}

# 固定类别顺序
CLASS_ORDER = ["normal", "benign", "malignant"]
CLASS_NAME_MAP = {
    "normal": "正常",
    "benign": "良性",
    "malignant": "恶性",
}


def ensure_directories() -> None:
    """确保运行时所需目录存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
