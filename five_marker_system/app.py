from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from medical_system.config import FEATURE_COLUMNS, MODEL_PATH, REPORT_DIR, ensure_directories
from medical_system.database import (
    add_subject,
    add_test,
    authenticate_user,
    create_user,
    dashboard_stats,
    delete_subject,
    delete_test,
    get_followup_dataframe,
    get_latest_evaluation_for_test,
    get_subject,
    get_test,
    import_tests_from_dataframe,
    init_db,
    list_audit_logs,
    list_labeled_tests,
    list_subjects,
    list_tests,
    list_users,
    log_audit_event,
    save_evaluation,
    set_user_active,
    update_subject,
    update_test,
    update_user_password,
    update_user_role,
)
from medical_system.modeling import BreastRiskModel
from medical_system.reporting import generate_report_html, generate_report_pdf
from medical_system.risk import followup_warning_analysis, get_risk_level, to_cn_class


APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "app_icon.png"


def _load_page_icon() -> Image.Image | str:
    if APP_ICON_PATH.exists():
        return Image.open(APP_ICON_PATH)
    return ":hospital:"


st.set_page_config(
    page_title="乳腺健康智能筛查系统",
    page_icon=_load_page_icon(),
    layout="wide",
)
ensure_directories()
init_db()


ROLE_NAME_MAP = {"admin": "管理员", "doctor": "医生", "viewer": "只读用户"}
STAGE_NAME_MAP = {"screening": "体检筛查", "benign_followup": "良性随访", "cancer_followup": "肿瘤随访"}
LABEL_NAME_MAP = {"": "未标注", "normal": "正常", "benign": "良性", "malignant": "恶性"}


def apply_global_style() -> None:
    """Inject a professional medical visual layer without changing business logic."""
    st.markdown(
        """
        <style>
        :root {
            --medical-blue: #1a5f7a;
            --medical-blue-dark: #124559;
            --medical-green: #00a896;
            --medical-bg: #f6f8fa;
            --medical-card: #ffffff;
            --medical-border: #e2e8f0;
            --medical-text: #1f2937;
            --medical-muted: #64748b;
            --medical-red: #c0392b;
            --medical-orange: #d97706;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fbfc 0%, var(--medical-bg) 100%) !important;
            color: var(--medical-text);
            font-family: "Microsoft YaHei", "DengXian", "Noto Sans CJK SC", sans-serif !important;
        }

        h1, h2, h3, h4 {
            color: var(--medical-blue) !important;
            font-weight: 700 !important;
            letter-spacing: .01em;
        }

        [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid var(--medical-border);
            box-shadow: 8px 0 24px rgba(15, 23, 42, .03);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 10px;
            padding: 6px 8px;
            transition: background-color .18s ease, color .18s ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: #eef7fa;
        }

        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: var(--medical-blue);
            margin: 24px 0 14px;
            border-left: 5px solid var(--medical-blue);
            padding-left: 12px;
            line-height: 1.35;
        }

        .sub-section-title {
            font-size: 16px;
            font-weight: 700;
            color: #334155;
            margin: 20px 0 10px;
            border-left: 4px solid var(--medical-green);
            padding-left: 10px;
            line-height: 1.35;
        }

        .med-card {
            background: var(--medical-card);
            border: 1px solid var(--medical-border);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, .05);
            margin-bottom: 16px;
        }

        .med-note {
            background: #edf7fa;
            border: 1px solid #d7edf3;
            border-left: 5px solid var(--medical-blue);
            border-radius: 12px;
            padding: 14px 16px;
            color: #245064;
            font-size: 14px;
            margin-bottom: 18px;
        }

        .task-card table {
            width: 100%;
            border-collapse: collapse;
        }

        .task-card td {
            padding: 7px 0;
            font-size: 14px;
            border-bottom: 1px solid #f1f5f9;
        }

        .task-card td:last-child {
            text-align: right;
            font-weight: 700;
            color: var(--medical-blue);
        }

        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }

        .badge-red {
            background: #fee2e2;
            color: #991b1b;
        }

        .badge-blue {
            background: #e0f2fe;
            color: #075985;
        }

        .result-panel {
            border-radius: 16px;
            padding: 20px 22px;
            border: 1px solid var(--medical-border);
            box-shadow: 0 14px 34px rgba(15, 23, 42, .06);
            margin: 18px 0;
        }

        .prob-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border: 1px solid var(--medical-border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(15, 23, 42, .04);
        }

        .prob-table td {
            padding: 12px;
            border-bottom: 1px solid #edf2f7;
            font-size: 14px;
        }

        .prob-track {
            width: 100%;
            height: 12px;
            border-radius: 999px;
            background: #edf2f7;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
        }

        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid var(--medical-border) !important;
            border-radius: 14px !important;
            padding: 16px 14px !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, .05) !important;
            transition: transform .18s ease, box-shadow .18s ease !important;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(15, 23, 42, .08) !important;
        }

        div[data-testid="stMetricLabel"] > div {
            color: var(--medical-muted) !important;
            font-size: 13px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetricValue"] > div {
            color: var(--medical-blue) !important;
            font-size: 27px !important;
            font-weight: 800 !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--medical-border) !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, .04) !important;
            overflow: hidden;
        }

        .stForm {
            background: #ffffff !important;
            border: 1px solid var(--medical-border) !important;
            border-radius: 14px !important;
            padding: 20px !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, .04) !important;
        }

        button {
            border-radius: 9px !important;
            font-weight: 700 !important;
            transition: all .18s ease !important;
        }

        button[kind="primary"] {
            background: var(--medical-blue) !important;
            border: 1px solid var(--medical-blue) !important;
            color: #ffffff !important;
            box-shadow: 0 8px 18px rgba(26, 95, 122, .22) !important;
        }

        button[kind="primary"]:hover {
            background: var(--medical-blue-dark) !important;
            transform: translateY(-1px);
        }

        section[data-testid="stFileUploadDropzone"] {
            border: 2px dashed #cbd5e1 !important;
            background: #f8fafc !important;
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()


def current_user() -> dict[str, Any] | None:
    user = st.session_state.get("auth_user")
    return user if isinstance(user, dict) else None


def is_admin() -> bool:
    user = current_user()
    return bool(user and user.get("role") == "admin")


def can_write() -> bool:
    user = current_user()
    if not user:
        return False
    return user.get("role") in {"admin", "doctor"}


def role_to_cn(role: str) -> str:
    return ROLE_NAME_MAP.get(role, role)


def stage_to_cn(stage: str) -> str:
    return STAGE_NAME_MAP.get(stage, stage)


def label_to_cn(label: str) -> str:
    return LABEL_NAME_MAP.get(label, label)


def _format_metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _render_training_result(metrics: dict[str, Any], class_distribution: dict[str, int] | None = None) -> None:
    st.markdown("<div class='section-title'>交叉验证多维评估指标</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("平均AUC", _format_metric(metrics, "auc"))
    c2.metric("平均Precision", _format_metric(metrics, "precision"))
    c3.metric("平均Recall", _format_metric(metrics, "recall"))
    c4.metric("平均Accuracy", _format_metric(metrics, "accuracy"))

    if "malignant_auc" in metrics or "benign_auc" in metrics:
        st.markdown("<div class='sub-section-title'>各预测子任务性能详情</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        if "malignant_auc" in metrics:
            with left:
                st.markdown(
                    f"""
                    <div class="med-card task-card">
                        <h4 style="color:#c0392b; margin-top:0; border-bottom:1px solid #fee2e2; padding-bottom:8px;">
                            恶性病变预测子任务
                        </h4>
                        <table>
                            <tr><td>AUC 检测效能</td><td><span class="badge badge-red">{_format_metric(metrics, "malignant_auc")}</span></td></tr>
                            <tr><td>Precision 精确率</td><td>{_format_metric(metrics, "malignant_precision")}</td></tr>
                            <tr><td>Recall 召回率</td><td>{_format_metric(metrics, "malignant_recall")}</td></tr>
                            <tr><td>Accuracy 准确率</td><td>{_format_metric(metrics, "malignant_accuracy")}</td></tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if "benign_auc" in metrics:
            with right:
                st.markdown(
                    f"""
                    <div class="med-card task-card">
                        <h4 style="color:#1a5f7a; margin-top:0; border-bottom:1px solid #e0f2fe; padding-bottom:8px;">
                            良性病变预测子任务
                        </h4>
                        <table>
                            <tr><td>AUC 检测效能</td><td><span class="badge badge-blue">{_format_metric(metrics, "benign_auc")}</span></td></tr>
                            <tr><td>Precision 精确率</td><td>{_format_metric(metrics, "benign_precision")}</td></tr>
                            <tr><td>Recall 召回率</td><td>{_format_metric(metrics, "benign_recall")}</td></tr>
                            <tr><td>Accuracy 准确率</td><td>{_format_metric(metrics, "benign_accuracy")}</td></tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if class_distribution:
        st.markdown("<div class='sub-section-title'>样本类别分布</div>", unsafe_allow_html=True)
        dist_df = pd.DataFrame([{"类别": to_cn_class(k), "样本数": v} for k, v in class_distribution.items()])
        st.dataframe(dist_df, use_container_width=True)


def _risk_visual_style(risk_level: str) -> tuple[str, str, str]:
    text = str(risk_level)
    if "高" in text or "High" in text:
        return "#c0392b", "#fff1f2", "需重点关注"
    if "中" in text or "Medium" in text:
        return "#d97706", "#fffbeb", "建议复核随访"
    return "#15803d", "#f0fdf4", "当前风险较低"


def _render_probability_table(probabilities: dict[str, float], predicted_class: str, accent_color: str) -> None:
    rows = []
    for key, value in probabilities.items():
        percent = max(0.0, min(100.0, float(value) * 100.0))
        fill_color = accent_color if key == predicted_class else "#94a3b8"
        rows.append(
            f"""
            <tr>
                <td><strong>{to_cn_class(key)}</strong></td>
                <td>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{percent:.2f}%; background:{fill_color};"></div>
                    </div>
                </td>
                <td style="text-align:right;"><strong>{percent:.2f}%</strong></td>
            </tr>
            """
        )
    st.markdown(f"<table class='prob-table'>{''.join(rows)}</table>", unsafe_allow_html=True)


def _render_inference_result(
    predicted_class: str,
    probabilities: dict[str, float],
    confidence: float,
    risk_level: str,
    feature_contribution: dict[str, float],
    warning: bool,
    notes: list[str],
) -> None:
    accent_color, bg_color, risk_hint = _risk_visual_style(risk_level)
    st.markdown("<div class='section-title'>诊断评估结果报告</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-panel" style="background:{bg_color}; border-left:6px solid {accent_color};">
            <div style="display:flex; justify-content:space-between; gap:18px; align-items:flex-start;">
                <div>
                    <div style="font-size:14px; color:#64748b; font-weight:700;">风险等级</div>
                    <div style="font-size:30px; color:{accent_color}; font-weight:800; margin-top:2px;">{risk_level}</div>
                    <div style="font-size:14px; color:#334155; margin-top:8px;">{risk_hint}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:14px; color:#64748b; font-weight:700;">预测类别</div>
                    <div style="font-size:24px; color:#1a5f7a; font-weight:800; margin-top:2px;">{to_cn_class(predicted_class)}</div>
                    <div style="font-size:14px; color:#334155; margin-top:8px;">可信度 {float(confidence) * 100:.2f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='sub-section-title'>三分类概率分布</div>", unsafe_allow_html=True)
        _render_probability_table(probabilities, predicted_class, accent_color)
    with right:
        st.markdown("<div class='sub-section-title'>标志物贡献度</div>", unsafe_allow_html=True)
        contrib_df = pd.DataFrame(
            [{"指标": k.upper() if k != "ca19_9" else "CA19-9", "贡献度": v} for k, v in sorted(feature_contribution.items(), key=lambda x: x[1], reverse=True)]
        ).set_index("指标")
        st.bar_chart(contrib_df)

    st.markdown("<div class='sub-section-title'>随访趋势提示</div>", unsafe_allow_html=True)
    if warning:
        st.error("预警：随访趋势提示复发风险可能升高，建议结合临床资料进一步复核。")
    else:
        st.info("当前未触发复发风险预警。")
    for note in notes:
        st.write(f"- {note}")


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    if "clinical_stage" in out.columns:
        out["clinical_stage"] = out["clinical_stage"].map(lambda x: stage_to_cn(str(x)) if pd.notna(x) else x)
    if "label" in out.columns:
        out["label"] = out["label"].map(lambda x: label_to_cn(str(x)) if pd.notna(x) else x)
    if "predicted_class" in out.columns:
        out["predicted_class"] = out["predicted_class"].map(lambda x: to_cn_class(str(x)) if pd.notna(x) else x)
    if "role" in out.columns:
        out["role"] = out["role"].map(lambda x: role_to_cn(str(x)) if pd.notna(x) else x)
    if "is_active" in out.columns:
        out["is_active"] = out["is_active"].map(lambda x: "启用" if int(x) else "停用")
    if "warning_flag" in out.columns:
        out["warning_flag"] = out["warning_flag"].map(lambda x: "是" if int(x) else "否")
    if "sex" in out.columns:
        out["sex"] = out["sex"].replace({"Female": "女", "Male": "男"})

    rename_map = {
        "id": "ID",
        "subject_id": "受检者ID",
        "test_id": "检验ID",
        "name": "姓名",
        "sex": "性别",
        "birth_date": "出生日期",
        "phone": "电话",
        "note": "备注",
        "test_date": "检验日期",
        "clinical_stage": "临床场景",
        "label": "真实标签",
        "source": "数据来源",
        "predicted_class": "预测类别",
        "normal_prob": "正常概率",
        "benign_prob": "良性概率",
        "malignant_prob": "恶性概率",
        "risk_level": "风险等级",
        "warning_flag": "预警",
        "created_at": "创建时间",
        "username": "用户名",
        "role": "角色",
        "is_active": "状态",
        "action": "动作",
        "module": "模块",
        "target_type": "对象类型",
        "target_id": "对象ID",
        "details_json": "详情",
        "eval_created_at": "评估时间",
    }
    return out.rename(columns=rename_map)


def _read_uploaded_table(uploaded_file: Any) -> pd.DataFrame:
    """读取上传文件（支持 csv/xlsx）。"""
    file_name = str(getattr(uploaded_file, "name", "")).lower()
    if file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def audit(
    action: str,
    module: str,
    target_type: str | None = None,
    target_id: str | int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    user = current_user()
    log_audit_event(
        user_id=int(user["id"]) if user else None,
        username=str(user["username"]) if user else "anonymous",
        action=action,
        module=module,
        target_type=target_type,
        target_id=target_id,
        details=details or {},
    )


def load_model() -> BreastRiskModel | None:
    if not MODEL_PATH.exists():
        return None
    return BreastRiskModel.load(MODEL_PATH)


def fetch_subject_options() -> dict[str, int]:
    subjects = list_subjects()
    return {f"{item['id']} - {item['name']}": item["id"] for item in subjects}


def make_synthetic_training_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_normal, n_benign, n_malignant = 300, 200, 500

    def make_group(n: int, center: tuple[float, ...], label: str) -> pd.DataFrame:
        scales = np.array([4.5, 3.0, 5.0, 8.0, 1.0])
        x = rng.normal(loc=np.array(center), scale=scales, size=(n, 5))
        x = np.clip(x, a_min=0.1, a_max=None)
        dates = pd.date_range("2025-01-01", periods=n, freq="D").astype(str)
        return pd.DataFrame(
            {
                "test_date": dates,
                "ca19_9": x[:, 0],
                "nse": x[:, 1],
                "ca125": x[:, 2],
                "ca153": x[:, 3],
                "cea": x[:, 4],
                "clinical_stage": "screening",
                "label": label,
            }
        )

    df = pd.concat(
        [
            make_group(n_normal, (12, 10, 16, 20, 2.8), "normal"),
            make_group(n_benign, (17, 14, 24, 30, 3.8), "benign"),
            make_group(n_malignant, (35, 24, 40, 55, 8.2), "malignant"),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=42)
    return df.reset_index(drop=True)


def login_page() -> None:
    st.title("乳腺健康智能筛查系统")
    st.caption("请先登录")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")

    if submitted:
        user = authenticate_user(username=username, password=password)
        if user is None:
            st.error("用户名或密码错误，或账号已被停用。")
            log_audit_event(
                user_id=None,
                username=username.strip() or "unknown",
                action="login_failed",
                module="auth",
                details={"username": username.strip()},
            )
            return
        st.session_state["auth_user"] = user
        audit("login_success", "auth")
        _rerun()

    st.info("默认管理员账号：admin / Admin@123456")


def main() -> None:
    apply_global_style()

    if current_user() is None:
        login_page()
        return

    user = current_user()
    assert user is not None

    if APP_ICON_PATH.exists():
        st.sidebar.image(str(APP_ICON_PATH), width=74)
        st.sidebar.markdown(
            """
            <div style="font-size:16px; font-weight:800; color:#1a5f7a; margin:-4px 0 14px;">
                乳腺健康智能筛查系统
            </div>
            """,
            unsafe_allow_html=True,
        )

    username_display = html.escape(str(user["username"]))
    role_display = html.escape(role_to_cn(str(user["role"])))
    st.sidebar.markdown(
        f"""
        <div class="med-card" style="padding:14px 16px; margin-bottom:14px;">
            <div style="font-size:13px; color:#64748b; font-weight:700;">当前登录用户</div>
            <div style="font-size:18px; color:#1a5f7a; font-weight:800; margin-top:4px;">{username_display}</div>
            <div style="font-size:13px; color:#334155; margin-top:6px;">角色：{role_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("退出登录"):
        audit("logout", "auth")
        st.session_state.pop("auth_user", None)
        _rerun()

    base_menu = [
        "系统概览",
        "受检者管理",
        "检验数据管理",
        "模型训练",
        "风险评估",
        "随访监测",
        "报告导出",
    ]
    admin_menu = ["审计日志", "用户管理"]
    menus = base_menu + admin_menu if is_admin() else base_menu

    st.sidebar.markdown("<div style='font-size:13px; color:#64748b; font-weight:800; margin:12px 0 8px;'>功能菜单</div>", unsafe_allow_html=True)
    menu = st.sidebar.radio("功能菜单", menus, label_visibility="collapsed")
    st.title("基于五项常规肿瘤标志物的乳腺健康智能筛查系统 V1.0")
    st.caption("本系统用于风险筛查与辅助参考，不作为最终诊断依据。")

    if menu == "系统概览":
        page_dashboard()
    elif menu == "受检者管理":
        page_subjects()
    elif menu == "检验数据管理":
        page_tests()
    elif menu == "模型训练":
        page_training()
    elif menu == "风险评估":
        page_inference()
    elif menu == "随访监测":
        page_followup()
    elif menu == "报告导出":
        page_report()
    elif menu == "审计日志":
        page_audit_logs()
    elif menu == "用户管理":
        page_user_admin()


def page_dashboard() -> None:
    stats = dashboard_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("受检者数量", stats["subjects"])
    c2.metric("检验记录数量", stats["tests"])
    c3.metric("评估记录数量", stats["evaluations"])
    c4.metric("系统用户数量", stats["users"])
    c5.metric("审计日志数量", stats["audits"])

    st.subheader("最近检验记录")
    tests = list_tests()[:20]
    if tests:
        st.dataframe(_display_df(pd.DataFrame(tests)), use_container_width=True)
    else:
        st.info("暂无检验记录。")


def page_subjects() -> None:
    if not can_write():
        st.warning("当前角色为只读，不能编辑受检者信息。")
        return

    st.subheader("新增受检者")
    with st.form("add_subject_form", clear_on_submit=True):
        name = st.text_input("姓名*", max_chars=64)
        sex = st.selectbox("性别", ["", "女", "男"])
        birth_date = st.text_input("出生日期（YYYY-MM-DD）")
        phone = st.text_input("联系电话")
        note = st.text_area("备注")
        submitted = st.form_submit_button("保存受检者")
    if submitted:
        if not name.strip():
            st.error("姓名不能为空。")
        else:
            subject_id = add_subject(
                name=name.strip(),
                sex=sex or None,
                birth_date=birth_date or None,
                phone=phone or None,
                note=note or None,
            )
            audit("create", "subjects", "subject", subject_id, {"name": name.strip()})
            st.success(f"受检者已创建，ID={subject_id}")

    st.divider()
    st.subheader("受检者列表")
    keyword = st.text_input("按姓名或电话搜索")
    rows = list_subjects(keyword=keyword)
    if not rows:
        st.info("未找到受检者。")
        return
    st.dataframe(_display_df(pd.DataFrame(rows)), use_container_width=True)

    options = {f"{r['id']} - {r['name']}": r["id"] for r in rows}
    selected = st.selectbox("选择受检者", list(options.keys()))
    subject_id = options[selected]
    current = get_subject(subject_id)
    if not current:
        return

    current_sex = current.get("sex") or ""
    sex_options = ["", "女", "男", "Female", "Male"]
    sex_index = sex_options.index(current_sex) if current_sex in sex_options else 0

    with st.form("edit_subject_form"):
        name = st.text_input("姓名", value=current.get("name") or "")
        sex = st.selectbox("性别", sex_options, index=sex_index)
        birth_date = st.text_input("出生日期", value=current.get("birth_date") or "")
        phone = st.text_input("联系电话", value=current.get("phone") or "")
        note = st.text_area("备注", value=current.get("note") or "")
        do_update = st.form_submit_button("更新受检者")
    if do_update:
        update_subject(
            subject_id,
            name=name,
            sex=sex or None,
            birth_date=birth_date or None,
            phone=phone or None,
            note=note or None,
        )
        audit("update", "subjects", "subject", subject_id, {"name": name})
        st.success("受检者信息已更新。")

    if st.checkbox("启用删除受检者"):
        if st.button("删除受检者", type="primary"):
            delete_subject(subject_id)
            audit("delete", "subjects", "subject", subject_id)
            st.success("已删除。")
            _rerun()


def _marker_form(prefix: str, defaults: dict[str, float] | None = None) -> dict[str, float]:
    values: dict[str, float] = {}
    defaults = defaults or {}
    cols = st.columns(3)
    for idx, name in enumerate(FEATURE_COLUMNS):
        with cols[idx % 3]:
            marker_display = "CA19-9" if name == "ca19_9" else name.upper()
            values[name] = st.number_input(
                marker_display,
                min_value=0.0,
                value=float(defaults.get(name, 0.0)),
                key=f"{prefix}_{name}",
                format="%.4f",
            )
    return values


def page_tests() -> None:
    if not can_write():
        st.warning("当前角色为只读，不能编辑检验数据。")
        return

    subject_options = fetch_subject_options()
    if not subject_options:
        st.warning("请先新增受检者。")
        return

    selected_label = st.selectbox("选择受检者", list(subject_options.keys()))
    subject_id = subject_options[selected_label]

    st.subheader("手动录入检验数据")
    with st.form("add_test_form", clear_on_submit=True):
        test_date = st.date_input("检验日期")
        stage = st.selectbox(
            "临床场景",
            ["screening", "benign_followup", "cancer_followup"],
            format_func=stage_to_cn,
        )
        label = st.selectbox(
            "真实标签（训练用）",
            ["", "normal", "benign", "malignant"],
            format_func=label_to_cn,
        )
        markers = _marker_form("add")
        submitted = st.form_submit_button("保存检验记录")
    if submitted:
        new_test_id = add_test(
            subject_id=subject_id,
            test_date=str(test_date),
            markers=markers,
            clinical_stage=stage,
            label=label or None,
        )
        audit("create", "tests", "test", new_test_id, {"subject_id": subject_id})
        st.success(f"检验记录已保存，ID={new_test_id}")

    st.divider()
    st.subheader("批量导入 CSV/XLSX")
    st.caption("必需列：test_date, ca19-9, nse, ca125, ca153, cea（也兼容 ca19_9、ca15-3）")
    upload = st.file_uploader("上传数据文件", type=["csv", "xlsx"])
    if upload is not None:
        try:
            preview_df = _read_uploaded_table(upload)
            st.dataframe(preview_df.head(20), use_container_width=True)
        except Exception as exc:
            st.error(f"文件读取失败：{exc}")
            return
        if st.button("执行导入"):
            try:
                count = import_tests_from_dataframe(preview_df, default_subject_id=subject_id)
                audit("batch_import", "tests", "subject", subject_id, {"count": count})
                st.success(f"导入完成，共 {count} 条。")
            except Exception as exc:
                st.error(f"导入失败：{exc}")

    st.divider()
    st.subheader("历史检验记录")
    rows = list_tests(subject_id=subject_id)
    if not rows:
        st.info("该受检者暂无检验记录。")
        return
    st.dataframe(_display_df(pd.DataFrame(rows)), use_container_width=True)

    selected_test = st.selectbox("选择要编辑/删除的检验记录", [f"{r['id']} - {r['test_date']}" for r in rows])
    test_id = int(selected_test.split(" - ")[0])
    current = get_test(test_id)
    if not current:
        return

    with st.form("edit_test_form"):
        test_date = st.text_input("检验日期", value=current.get("test_date") or "")
        stage = st.selectbox(
            "临床场景",
            ["screening", "benign_followup", "cancer_followup"],
            index=["screening", "benign_followup", "cancer_followup"].index(current.get("clinical_stage") or "screening"),
            format_func=stage_to_cn,
        )
        label = st.selectbox(
            "真实标签",
            ["", "normal", "benign", "malignant"],
            index=["", "normal", "benign", "malignant"].index(current.get("label") or ""),
            format_func=label_to_cn,
        )
        markers = _marker_form("edit", defaults=current)
        do_update = st.form_submit_button("更新检验记录")
    if do_update:
        update_test(test_id=test_id, test_date=test_date, markers=markers, clinical_stage=stage, label=label or None)
        audit("update", "tests", "test", test_id)
        st.success("检验记录已更新。")

    if st.checkbox("启用删除检验记录"):
        if st.button("删除检验记录", type="primary"):
            delete_test(test_id)
            audit("delete", "tests", "test", test_id)
            st.success("已删除。")
            _rerun()


def page_training() -> None:
    if not can_write():
        st.warning("当前角色为只读，不能执行模型训练。")
        return

    st.markdown("<div class='section-title'>诊断评估模型训练与部署</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="med-note">
            系统将基于已标注的五项常规肿瘤标志物样本训练双专家模型。训练完成后，模型权重会保存到本地并用于后续风险评估。
        </div>
        """,
        unsafe_allow_html=True,
    )
    source = st.radio(
        "训练数据来源",
        ["数据库已标注数据", "上传 CSV/XLSX", "合成样本（1000条）"],
        horizontal=True,
    )
    train_df = pd.DataFrame()

    if source == "数据库已标注数据":
        train_df = list_labeled_tests()
        st.warning("提示：数据库样本可能包含历史导入/演示数据。若要复现你单独脚本结果，建议使用“上传 CSV/XLSX”并上传同一份训练文件。")
        st.write(f"可用标注样本：{len(train_df)} 条")
        if not train_df.empty:
            st.dataframe(train_df.head(20), use_container_width=True)
    elif source == "上传 CSV/XLSX":
        file = st.file_uploader("上传训练文件", type=["csv", "xlsx"], key="train_csv")
        if file is not None:
            try:
                train_df = _read_uploaded_table(file)
                st.dataframe(train_df.head(20), use_container_width=True)
            except Exception as exc:
                st.error(f"文件读取失败：{exc}")
    else:
        train_df = make_synthetic_training_data()
        st.write(f"合成样本：{len(train_df)} 条")
        st.dataframe(train_df.head(20), use_container_width=True)
        if st.button("导入前200条合成样本到数据库"):
            options = fetch_subject_options()
            if not options:
                subject_id = add_subject(name="演示受检者", sex="女")
                audit("create_demo_subject", "training", "subject", subject_id)
            else:
                subject_id = next(iter(options.values()))
            imported = import_tests_from_dataframe(train_df.head(200), default_subject_id=subject_id, source="synthetic_demo")
            audit("import_synthetic_demo", "training", "subject", subject_id, {"count": imported})
            st.success(f"导入完成，共 {imported} 条，受检者ID={subject_id}")

    if st.button("开始训练", type="primary", use_container_width=True):
        if train_df.empty:
            st.error("训练数据为空。")
            return
        with st.spinner("模型训练中，请稍候..."):
            try:
                model = BreastRiskModel()
                result = model.train(train_df)
                model.save(MODEL_PATH)
                audit("train_model", "ml", "model", "breast_risk_model.joblib", result.metrics)

                st.session_state["last_training_message"] = f"训练完成，模型已保存：{MODEL_PATH}"
                st.session_state["last_training_result"] = {
                    "metrics": result.metrics,
                    "class_distribution": result.class_distribution,
                }
            except Exception as exc:
                st.error(f"训练失败：{exc}")

    last_training = st.session_state.get("last_training_result")
    if isinstance(last_training, dict):
        message = st.session_state.get("last_training_message")
        if message:
            st.success(str(message))
        _render_training_result(
            metrics=last_training.get("metrics", {}),
            class_distribution=last_training.get("class_distribution", {}),
        )


def page_inference() -> None:
    if not can_write():
        st.warning("当前角色为只读，不能写入评估结果。")
        return

    st.markdown("<div class='section-title'>五项常规肿瘤标志物智能风险评估</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="med-note">
            请选择受检者及对应检验记录。系统会读取本地已训练模型，输出正常、良性、恶性的三分类概率及随访风险提示。
        </div>
        """,
        unsafe_allow_html=True,
    )
    model = load_model()
    if model is None:
        st.warning("未检测到模型，请先训练模型。")
        return

    subject_options = fetch_subject_options()
    if not subject_options:
        st.warning("暂无受检者。")
        return

    selected_label = st.selectbox("选择受检者", list(subject_options.keys()), key="pred_subject")
    subject_id = subject_options[selected_label]
    rows = list_tests(subject_id=subject_id)
    if not rows:
        st.info("该受检者暂无检验记录。")
        return

    selected_test = st.selectbox("选择检验记录", [f"{r['id']} - {r['test_date']}" for r in rows])
    test_id = int(selected_test.split(" - ")[0])
    current = get_test(test_id)
    if not current:
        return

    st.markdown("<div class='sub-section-title'>当前送检样本特征明细</div>", unsafe_allow_html=True)
    marker_cols = st.columns(len(FEATURE_COLUMNS))
    for idx, key in enumerate(FEATURE_COLUMNS):
        marker_display = "CA19-9" if key == "ca19_9" else key.upper()
        value = current.get(key)
        marker_cols[idx].metric(marker_display, f"{float(value):.4f}" if value is not None else "-")

    if st.button("开始评估", type="primary", use_container_width=True):
        sample_df = pd.DataFrame([{k: current.get(k) for k in FEATURE_COLUMNS}])
        try:
            with st.spinner("模型评估中，请稍候..."):
                pred = model.predict(sample_df)
                predicted_class = pred["predicted_class"]
                probabilities = pred["probabilities"]
                risk_level = get_risk_level(
                    predicted_class=predicted_class,
                    malignant_prob=probabilities.get("malignant", 0.0),
                    confidence=float(pred["confidence"]),
                )
                follow_df = get_followup_dataframe(subject_id)
                warning, notes = followup_warning_analysis(follow_df)
                eval_id = save_evaluation(
                    test_id=test_id,
                    predicted_class=predicted_class,
                    probabilities=probabilities,
                    risk_level=risk_level,
                    warning_flag=warning,
                    feature_importance=pred["feature_contribution"],
                )
                audit("inference", "ml", "evaluation", eval_id, {"test_id": test_id, "risk_level": risk_level})

            st.success("评估完成，结果已保存。")
            _render_inference_result(
                predicted_class=predicted_class,
                probabilities=probabilities,
                confidence=float(pred["confidence"]),
                risk_level=risk_level,
                feature_contribution=pred["feature_contribution"],
                warning=warning,
                notes=notes,
            )
        except Exception as exc:
            st.error(f"评估失败：{exc}")


def page_followup() -> None:
    st.subheader("随访监测")
    subject_options = fetch_subject_options()
    if not subject_options:
        st.warning("暂无受检者。")
        return

    selected_label = st.selectbox("选择受检者", list(subject_options.keys()), key="follow_subject")
    subject_id = subject_options[selected_label]
    df = get_followup_dataframe(subject_id)
    if df.empty:
        st.info("暂无随访数据。")
        return

    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    st.dataframe(_display_df(df), use_container_width=True)
    st.write("五项指标趋势")
    st.line_chart(df.set_index("test_date")[FEATURE_COLUMNS])

    if {"normal_prob", "benign_prob", "malignant_prob"}.issubset(df.columns):
        st.write("分类概率趋势")
        st.line_chart(df.set_index("test_date")[["normal_prob", "benign_prob", "malignant_prob"]].fillna(0.0))

    warning, notes = followup_warning_analysis(df)
    if warning:
        st.error("随访提示：存在需要重点关注的风险信号。")
    else:
        st.success("随访提示：当前趋势相对平稳。")
    for note in notes:
        st.write(f"- {note}")


def page_report() -> None:
    st.subheader("报告导出")
    subject_options = fetch_subject_options()
    if not subject_options:
        st.warning("暂无受检者。")
        return

    selected_label = st.selectbox("选择受检者", list(subject_options.keys()), key="report_subject")
    subject_id = subject_options[selected_label]
    subject = get_subject(subject_id)
    if not subject:
        return
    tests = list_tests(subject_id=subject_id)
    if not tests:
        st.info("该受检者暂无检验记录。")
        return

    selected_test = st.selectbox("选择检验记录", [f"{r['id']} - {r['test_date']}" for r in tests], key="report_test")
    test_id = int(selected_test.split(" - ")[0])
    test_row = get_test(test_id)
    if not test_row:
        return

    evaluation = get_latest_evaluation_for_test(test_id)
    if not evaluation:
        st.warning("该记录尚未评估，请先在“风险评估”页面执行分析。")
        return

    followup_df = get_followup_dataframe(subject_id)
    if st.button("生成 HTML + PDF 报告", type="primary"):
        html_path = generate_report_html(
            subject=subject,
            test_row=test_row,
            evaluation_row=evaluation,
            followup_df=followup_df,
            output_dir=Path(REPORT_DIR),
        )
        pdf_path = generate_report_pdf(
            subject=subject,
            test_row=test_row,
            evaluation_row=evaluation,
            followup_df=followup_df,
            output_dir=Path(REPORT_DIR),
        )
        audit("generate_report", "reports", "test", test_id, {"html": html_path.name, "pdf": pdf_path.name})
        st.success(f"报告生成完成：{html_path.name}，{pdf_path.name}")

        html_content = html_path.read_text(encoding="utf-8")
        pdf_content = pdf_path.read_bytes()

        st.download_button("下载 HTML 报告", data=html_content.encode("utf-8"), file_name=html_path.name, mime="text/html")
        st.download_button("下载 PDF 报告", data=pdf_content, file_name=pdf_path.name, mime="application/pdf")
        st.components.v1.html(html_content, height=700, scrolling=True)


def page_audit_logs() -> None:
    if not is_admin():
        st.warning("仅管理员可查看审计日志。")
        return

    st.subheader("审计日志")
    c1, c2, c3 = st.columns(3)
    username = c1.text_input("按用户名筛选")
    action = c2.text_input("按动作筛选")
    limit = c3.number_input("显示条数", min_value=50, max_value=2000, value=300, step=50)
    rows = list_audit_logs(limit=int(limit), username=username, action=action)
    if not rows:
        st.info("暂无日志记录。")
        return
    st.dataframe(_display_df(pd.DataFrame(rows)), use_container_width=True)


def page_user_admin() -> None:
    if not is_admin():
        st.warning("仅管理员可管理用户。")
        return

    st.subheader("用户管理")
    with st.form("create_user_form", clear_on_submit=True):
        username = st.text_input("新用户名")
        password = st.text_input("新密码", type="password")
        role = st.selectbox("角色", ["doctor", "viewer", "admin"], format_func=role_to_cn)
        submitted = st.form_submit_button("创建用户")
    if submitted:
        try:
            user_id = create_user(username=username, password=password, role=role)
            audit("create_user", "users", "user", user_id, {"username": username, "role": role})
            st.success(f"用户创建成功，ID={user_id}")
        except Exception as exc:
            st.error(f"创建失败：{exc}")

    rows = list_users()
    if not rows:
        return

    st.dataframe(_display_df(pd.DataFrame(rows)), use_container_width=True)

    options = {f"{r['id']} - {r['username']}": r["id"] for r in rows}
    selected = st.selectbox("选择用户", list(options.keys()))
    user_id = options[selected]
    target = next((r for r in rows if r["id"] == user_id), None)
    if target is None:
        return

    c1, c2 = st.columns(2)
    with c1:
        new_role = st.selectbox(
            "修改角色",
            ["admin", "doctor", "viewer"],
            index=["admin", "doctor", "viewer"].index(target["role"]),
            format_func=role_to_cn,
        )
        if st.button("更新角色"):
            update_user_role(user_id, new_role)
            audit("update_user_role", "users", "user", user_id, {"role": new_role})
            st.success("角色已更新。")
            _rerun()
    with c2:
        active_label = "停用账号" if int(target["is_active"]) else "启用账号"
        if st.button(active_label):
            set_user_active(user_id, not bool(int(target["is_active"])))
            audit("toggle_user_active", "users", "user", user_id, {"is_active": int(not bool(int(target["is_active"])))})
            st.success("状态已更新。")
            _rerun()

    with st.form("reset_password_form"):
        new_password = st.text_input("重置密码", type="password")
        do_reset = st.form_submit_button("确认重置")
    if do_reset:
        try:
            update_user_password(user_id, new_password)
            audit("reset_user_password", "users", "user", user_id)
            st.success("密码重置完成。")
        except Exception as exc:
            st.error(f"重置失败：{exc}")


if __name__ == "__main__":
    main()
