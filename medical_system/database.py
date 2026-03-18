from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd

from .auth import hash_password, verify_password
from .config import DB_PATH, FEATURE_COLUMNS


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            sex TEXT,
            birth_date TEXT,
            phone TEXT,
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL,
            test_date TEXT NOT NULL,
            akr1b10 REAL,
            ca19_9 REAL,
            nse REAL,
            ca125 REAL,
            ca153 REAL,
            cea REAL,
            clinical_stage TEXT DEFAULT 'screening',
            label TEXT,
            source TEXT DEFAULT 'manual',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id INTEGER NOT NULL,
            predicted_class TEXT NOT NULL,
            normal_prob REAL,
            benign_prob REAL,
            malignant_prob REAL,
            risk_level TEXT,
            warning_flag INTEGER DEFAULT 0,
            feature_importance_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(test_id) REFERENCES tests(id) ON DELETE CASCADE
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'doctor',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            action TEXT NOT NULL,
            module TEXT NOT NULL,
            target_type TEXT,
            target_id TEXT,
            details_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
        );
        """
    )
    conn.commit()
    conn.close()
    ensure_default_admin()


def ensure_default_admin() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS c FROM users")
    has_user = int(cursor.fetchone()["c"]) > 0
    if not has_user:
        cursor.execute(
            """
            INSERT INTO users(username, password_hash, role, is_active)
            VALUES (?, ?, ?, 1)
            """,
            ("admin", hash_password("Admin@123456"), "admin"),
        )
        conn.commit()
    conn.close()


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, password_hash, role, is_active, created_at FROM users WHERE username = ?",
        (username.strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    data = dict(row)
    if not int(data.get("is_active", 0)):
        return None
    if not verify_password(password, str(data["password_hash"])):
        return None
    data.pop("password_hash", None)
    return data


def list_users() -> list[dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, is_active, created_at FROM users ORDER BY id ASC")
    rows = _rows_to_dicts(cursor.fetchall())
    conn.close()
    return rows


def create_user(username: str, password: str, role: str) -> int:
    normalized = username.strip().lower()
    if not normalized:
        raise ValueError("username is required")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    if role not in {"admin", "doctor", "viewer"}:
        raise ValueError("invalid role")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO users(username, password_hash, role, is_active)
        VALUES (?, ?, ?, 1)
        """,
        (normalized, hash_password(password), role),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return int(new_id)


def update_user_role(user_id: int, role: str) -> None:
    if role not in {"admin", "doctor", "viewer"}:
        raise ValueError("invalid role")
    conn = get_connection()
    conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()
    conn.close()


def update_user_password(user_id: int, new_password: str) -> None:
    if len(new_password) < 8:
        raise ValueError("password must be at least 8 characters")
    conn = get_connection()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new_password), user_id))
    conn.commit()
    conn.close()


def set_user_active(user_id: int, is_active: bool) -> None:
    conn = get_connection()
    conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
    conn.commit()
    conn.close()


def log_audit_event(
    *,
    user_id: int | None,
    username: str | None,
    action: str,
    module: str,
    target_type: str | None = None,
    target_id: str | int | None = None,
    details: dict[str, Any] | None = None,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO audit_logs(
            user_id, username, action, module, target_type, target_id, details_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            username,
            action,
            module,
            target_type,
            str(target_id) if target_id is not None else None,
            json.dumps(details or {}, ensure_ascii=False),
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return int(new_id)


def list_audit_logs(limit: int = 300, username: str = "", action: str = "") -> list[dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM audit_logs WHERE 1=1"
    params: list[Any] = []
    if username.strip():
        query += " AND username LIKE ?"
        params.append(f"%{username.strip()}%")
    if action.strip():
        query += " AND action LIKE ?"
        params.append(f"%{action.strip()}%")
    query += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    cursor.execute(query, tuple(params))
    rows = _rows_to_dicts(cursor.fetchall())
    conn.close()
    return rows


def add_subject(
    name: str,
    sex: str | None = None,
    birth_date: str | None = None,
    phone: str | None = None,
    note: str | None = None,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO subjects(name, sex, birth_date, phone, note)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, sex, birth_date, phone, note),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return int(new_id)


def list_subjects(keyword: str = "") -> list[dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if keyword.strip():
        like_keyword = f"%{keyword.strip()}%"
        cursor.execute(
            """
            SELECT * FROM subjects
            WHERE name LIKE ? OR phone LIKE ?
            ORDER BY id DESC
            """,
            (like_keyword, like_keyword),
        )
    else:
        cursor.execute("SELECT * FROM subjects ORDER BY id DESC")
    rows = _rows_to_dicts(cursor.fetchall())
    conn.close()
    return rows


def get_subject(subject_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_subject(
    subject_id: int,
    name: str,
    sex: str | None = None,
    birth_date: str | None = None,
    phone: str | None = None,
    note: str | None = None,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        UPDATE subjects
        SET name = ?, sex = ?, birth_date = ?, phone = ?, note = ?
        WHERE id = ?
        """,
        (name, sex, birth_date, phone, note, subject_id),
    )
    conn.commit()
    conn.close()


def delete_subject(subject_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
    conn.commit()
    conn.close()


def add_test(
    subject_id: int,
    test_date: str,
    markers: dict[str, float | None],
    clinical_stage: str = "screening",
    label: str | None = None,
    source: str = "manual",
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO tests(
            subject_id, test_date, akr1b10, ca19_9, nse, ca125, ca153, cea,
            clinical_stage, label, source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            subject_id,
            test_date,
            markers.get("akr1b10"),
            markers.get("ca19_9"),
            markers.get("nse"),
            markers.get("ca125"),
            markers.get("ca153"),
            markers.get("cea"),
            clinical_stage,
            label,
            source,
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return int(new_id)


def update_test(
    test_id: int,
    test_date: str,
    markers: dict[str, float | None],
    clinical_stage: str,
    label: str | None,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        UPDATE tests
        SET test_date = ?, akr1b10 = ?, ca19_9 = ?, nse = ?, ca125 = ?, ca153 = ?, cea = ?,
            clinical_stage = ?, label = ?
        WHERE id = ?
        """,
        (
            test_date,
            markers.get("akr1b10"),
            markers.get("ca19_9"),
            markers.get("nse"),
            markers.get("ca125"),
            markers.get("ca153"),
            markers.get("cea"),
            clinical_stage,
            label,
            test_id,
        ),
    )
    conn.commit()
    conn.close()


def delete_test(test_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM tests WHERE id = ?", (test_id,))
    conn.commit()
    conn.close()


def list_tests(subject_id: int | None = None) -> list[dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if subject_id is None:
        cursor.execute("SELECT * FROM tests ORDER BY test_date DESC, id DESC")
    else:
        cursor.execute(
            "SELECT * FROM tests WHERE subject_id = ? ORDER BY test_date DESC, id DESC",
            (subject_id,),
        )
    rows = _rows_to_dicts(cursor.fetchall())
    conn.close()
    return rows


def get_test(test_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tests WHERE id = ?", (test_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_labeled_tests() -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT * FROM tests
    WHERE label IN ('normal', 'benign', 'malignant')
    ORDER BY id ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def import_tests_from_dataframe(
    df: pd.DataFrame,
    default_subject_id: int | None = None,
    source: str = "batch_import",
) -> int:
    normalized = df.copy()
    normalized.columns = [c.strip().lower() for c in normalized.columns]
    required = set(FEATURE_COLUMNS + ["test_date"])
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    if "subject_id" not in normalized.columns:
        if default_subject_id is None:
            raise ValueError("subject_id is missing and no default subject_id was provided")
        normalized["subject_id"] = default_subject_id

    if "clinical_stage" not in normalized.columns:
        normalized["clinical_stage"] = "screening"
    if "label" not in normalized.columns:
        normalized["label"] = None

    inserted = 0
    conn = get_connection()
    cursor = conn.cursor()
    for _, row in normalized.iterrows():
        cursor.execute(
            """
            INSERT INTO tests(
                subject_id, test_date, akr1b10, ca19_9, nse, ca125, ca153, cea,
                clinical_stage, label, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["subject_id"]),
                str(row["test_date"]),
                _to_float(row.get("akr1b10")),
                _to_float(row.get("ca19_9")),
                _to_float(row.get("nse")),
                _to_float(row.get("ca125")),
                _to_float(row.get("ca153")),
                _to_float(row.get("cea")),
                str(row.get("clinical_stage") or "screening"),
                _normalize_label(row.get("label")),
                source,
            ),
        )
        inserted += 1
    conn.commit()
    conn.close()
    return inserted


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _normalize_label(label: Any) -> str | None:
    if label is None:
        return None
    v = str(label).strip().lower()
    if v in {"normal", "benign", "malignant"}:
        return v
    return None


def save_evaluation(
    test_id: int,
    predicted_class: str,
    probabilities: dict[str, float],
    risk_level: str,
    warning_flag: bool,
    feature_importance: dict[str, float],
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO evaluations(
            test_id, predicted_class, normal_prob, benign_prob, malignant_prob,
            risk_level, warning_flag, feature_importance_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            test_id,
            predicted_class,
            probabilities.get("normal", 0.0),
            probabilities.get("benign", 0.0),
            probabilities.get("malignant", 0.0),
            risk_level,
            1 if warning_flag else 0,
            json.dumps(feature_importance, ensure_ascii=False),
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return int(new_id)


def get_latest_evaluation_for_test(test_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM evaluations
        WHERE test_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (test_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_followup_dataframe(subject_id: int) -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT t.*, e.predicted_class, e.normal_prob, e.benign_prob, e.malignant_prob,
           e.risk_level, e.warning_flag, e.created_at AS eval_created_at
    FROM tests t
    LEFT JOIN evaluations e ON e.test_id = t.id
    WHERE t.subject_id = ?
    ORDER BY t.test_date ASC, t.id ASC
    """
    df = pd.read_sql_query(query, conn, params=(subject_id,))
    conn.close()
    return df


def dashboard_stats() -> dict[str, int]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS c FROM subjects")
    subjects = int(cursor.fetchone()["c"])
    cursor.execute("SELECT COUNT(*) AS c FROM tests")
    tests = int(cursor.fetchone()["c"])
    cursor.execute("SELECT COUNT(*) AS c FROM evaluations")
    evaluations = int(cursor.fetchone()["c"])
    cursor.execute("SELECT COUNT(*) AS c FROM users")
    users = int(cursor.fetchone()["c"])
    cursor.execute("SELECT COUNT(*) AS c FROM audit_logs")
    audits = int(cursor.fetchone()["c"])
    conn.close()
    return {"subjects": subjects, "tests": tests, "evaluations": evaluations, "users": users, "audits": audits}

