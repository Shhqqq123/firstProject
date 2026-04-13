from pathlib import Path

# 项目根目录（app.py 所在目录）
BASE_DIR = Path(__file__).resolve().parent.parent
# 运行数据目录
DATA_DIR = BASE_DIR / "data"
# 模型目录
MODEL_DIR = BASE_DIR / "models"
# 报告目录
REPORT_DIR = BASE_DIR / "reports"

# SQLite 数据库文件路径
DB_PATH = DATA_DIR / "medical_system.db"
# 模型保存路径
MODEL_PATH = MODEL_DIR / "breast_risk_model.joblib"

# 六项肿瘤标志物特征
FEATURE_COLUMNS = ["akr1b10", "ca19_9", "nse", "ca125", "ca153", "cea"]
LABEL_COLUMN = "label"

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

