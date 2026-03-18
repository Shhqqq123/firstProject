from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

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


st.set_page_config(page_title="Breast Risk Assessment", page_icon=":hospital:", layout="wide")
ensure_directories()
init_db()


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

    df = pd.concat(
        [
            make_group(n_normal, (8, 12, 10, 16, 20, 2.8), "normal"),
            make_group(n_benign, (11, 17, 14, 24, 30, 3.8), "benign"),
            make_group(n_malignant, (18, 35, 24, 40, 55, 8.2), "malignant"),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=42)
    return df.reset_index(drop=True)


def login_page() -> None:
    st.title("Breast Risk Assessment System")
    st.caption("Login required")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")

    if submitted:
        user = authenticate_user(username=username, password=password)
        if user is None:
            st.error("Invalid username/password or inactive account.")
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

    st.info("Default admin account: admin / Admin@123456")


def main() -> None:
    if current_user() is None:
        login_page()
        return

    user = current_user()
    assert user is not None

    st.sidebar.write(f"User: `{user['username']}`")
    st.sidebar.write(f"Role: `{user['role']}`")
    if st.sidebar.button("Sign Out"):
        audit("logout", "auth")
        st.session_state.pop("auth_user", None)
        _rerun()

    base_menu = [
        "Dashboard",
        "Subjects",
        "Tests",
        "Model Training",
        "Risk Inference",
        "Follow-up",
        "Reports",
    ]
    admin_menu = ["Audit Logs", "User Admin"]
    menus = base_menu + admin_menu if is_admin() else base_menu

    menu = st.sidebar.radio("Navigation", menus)
    st.title("Breast Full-Process Risk Assessment System V1.0")
    st.caption("For risk screening/support only. Not a final diagnosis.")

    if menu == "Dashboard":
        page_dashboard()
    elif menu == "Subjects":
        page_subjects()
    elif menu == "Tests":
        page_tests()
    elif menu == "Model Training":
        page_training()
    elif menu == "Risk Inference":
        page_inference()
    elif menu == "Follow-up":
        page_followup()
    elif menu == "Reports":
        page_report()
    elif menu == "Audit Logs":
        page_audit_logs()
    elif menu == "User Admin":
        page_user_admin()


def page_dashboard() -> None:
    stats = dashboard_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Subjects", stats["subjects"])
    c2.metric("Tests", stats["tests"])
    c3.metric("Evaluations", stats["evaluations"])
    c4.metric("Users", stats["users"])
    c5.metric("Audit Logs", stats["audits"])

    st.subheader("Recent Tests")
    tests = list_tests()[:20]
    if tests:
        st.dataframe(pd.DataFrame(tests), use_container_width=True)
    else:
        st.info("No test records yet.")


def page_subjects() -> None:
    if not can_write():
        st.warning("Read-only role. Subject editing is disabled.")
        return

    st.subheader("Add Subject")
    with st.form("add_subject_form", clear_on_submit=True):
        name = st.text_input("Name*", max_chars=64)
        sex = st.selectbox("Sex", ["", "Female", "Male"])
        birth_date = st.text_input("Birth Date (YYYY-MM-DD)")
        phone = st.text_input("Phone")
        note = st.text_area("Note")
        submitted = st.form_submit_button("Save Subject")
    if submitted:
        if not name.strip():
            st.error("Name is required.")
        else:
            subject_id = add_subject(name=name.strip(), sex=sex or None, birth_date=birth_date or None, phone=phone or None, note=note or None)
            audit("create", "subjects", "subject", subject_id, {"name": name.strip()})
            st.success(f"Subject created: ID={subject_id}")

    st.divider()
    st.subheader("Subject List")
    keyword = st.text_input("Search by name/phone")
    rows = list_subjects(keyword=keyword)
    if not rows:
        st.info("No subjects found.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    options = {f"{r['id']} - {r['name']}": r["id"] for r in rows}
    selected = st.selectbox("Select subject", list(options.keys()))
    subject_id = options[selected]
    current = get_subject(subject_id)
    if not current:
        return

    with st.form("edit_subject_form"):
        name = st.text_input("Name", value=current.get("name") or "")
        sex = st.selectbox("Sex", ["", "Female", "Male"], index=["", "Female", "Male"].index(current.get("sex") or ""))
        birth_date = st.text_input("Birth Date", value=current.get("birth_date") or "")
        phone = st.text_input("Phone", value=current.get("phone") or "")
        note = st.text_area("Note", value=current.get("note") or "")
        do_update = st.form_submit_button("Update Subject")
    if do_update:
        update_subject(subject_id, name=name, sex=sex or None, birth_date=birth_date or None, phone=phone or None, note=note or None)
        audit("update", "subjects", "subject", subject_id, {"name": name})
        st.success("Updated.")

    if st.checkbox("Enable delete subject"):
        if st.button("Delete Subject", type="primary"):
            delete_subject(subject_id)
            audit("delete", "subjects", "subject", subject_id)
            st.success("Deleted.")
            _rerun()


def _marker_form(prefix: str, defaults: dict[str, float] | None = None) -> dict[str, float]:
    values: dict[str, float] = {}
    defaults = defaults or {}
    cols = st.columns(3)
    for idx, name in enumerate(FEATURE_COLUMNS):
        with cols[idx % 3]:
            values[name] = st.number_input(
                f"{name.upper()}",
                min_value=0.0,
                value=float(defaults.get(name, 0.0)),
                key=f"{prefix}_{name}",
                format="%.4f",
            )
    return values


def page_tests() -> None:
    if not can_write():
        st.warning("Read-only role. Test editing is disabled.")
        return

    subject_options = fetch_subject_options()
    if not subject_options:
        st.warning("Please create a subject first.")
        return

    selected_label = st.selectbox("Select Subject", list(subject_options.keys()))
    subject_id = subject_options[selected_label]

    st.subheader("Manual Input")
    with st.form("add_test_form", clear_on_submit=True):
        test_date = st.date_input("Test Date")
        stage = st.selectbox("Clinical Stage", ["screening", "benign_followup", "cancer_followup"])
        label = st.selectbox("True Label (for training)", ["", "normal", "benign", "malignant"])
        markers = _marker_form("add")
        submitted = st.form_submit_button("Save Test")
    if submitted:
        new_test_id = add_test(
            subject_id=subject_id,
            test_date=str(test_date),
            markers=markers,
            clinical_stage=stage,
            label=label or None,
        )
        audit("create", "tests", "test", new_test_id, {"subject_id": subject_id})
        st.success(f"Test saved: ID={new_test_id}")

    st.divider()
    st.subheader("Batch Import CSV")
    st.caption("Required columns: test_date, akr1b10, ca19_9, nse, ca125, ca153, cea")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload is not None:
        preview_df = pd.read_csv(upload)
        st.dataframe(preview_df.head(20), use_container_width=True)
        if st.button("Import CSV"):
            try:
                count = import_tests_from_dataframe(preview_df, default_subject_id=subject_id)
                audit("batch_import", "tests", "subject", subject_id, {"count": count})
                st.success(f"Imported {count} rows.")
            except Exception as exc:
                st.error(f"Import failed: {exc}")

    st.divider()
    st.subheader("History")
    rows = list_tests(subject_id=subject_id)
    if not rows:
        st.info("No tests for this subject.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    selected_test = st.selectbox("Select test to edit/delete", [f"{r['id']} - {r['test_date']}" for r in rows])
    test_id = int(selected_test.split(" - ")[0])
    current = get_test(test_id)
    if not current:
        return

    with st.form("edit_test_form"):
        test_date = st.text_input("Test Date", value=current.get("test_date") or "")
        stage = st.selectbox(
            "Clinical Stage",
            ["screening", "benign_followup", "cancer_followup"],
            index=["screening", "benign_followup", "cancer_followup"].index(current.get("clinical_stage") or "screening"),
        )
        label = st.selectbox(
            "True Label",
            ["", "normal", "benign", "malignant"],
            index=["", "normal", "benign", "malignant"].index(current.get("label") or ""),
        )
        markers = _marker_form("edit", defaults=current)
        do_update = st.form_submit_button("Update Test")
    if do_update:
        update_test(test_id=test_id, test_date=test_date, markers=markers, clinical_stage=stage, label=label or None)
        audit("update", "tests", "test", test_id)
        st.success("Test updated.")

    if st.checkbox("Enable delete test"):
        if st.button("Delete Test", type="primary"):
            delete_test(test_id)
            audit("delete", "tests", "test", test_id)
            st.success("Deleted.")
            _rerun()


def page_training() -> None:
    if not can_write():
        st.warning("Read-only role. Training is disabled.")
        return

    st.subheader("Model Training")
    source = st.radio(
        "Training Data Source",
        ["Database labeled data", "Upload CSV", "Synthetic data (1000 samples)"],
        horizontal=True,
    )
    train_df = pd.DataFrame()

    if source == "Database labeled data":
        train_df = list_labeled_tests()
        st.write(f"Labeled rows: {len(train_df)}")
        if not train_df.empty:
            st.dataframe(train_df.head(20), use_container_width=True)
    elif source == "Upload CSV":
        file = st.file_uploader("Upload training CSV", type=["csv"], key="train_csv")
        if file is not None:
            train_df = pd.read_csv(file)
            st.dataframe(train_df.head(20), use_container_width=True)
    else:
        train_df = make_synthetic_training_data()
        st.write(f"Synthetic rows: {len(train_df)}")
        st.dataframe(train_df.head(20), use_container_width=True)
        if st.button("Import synthetic rows into DB as demo data"):
            options = fetch_subject_options()
            if not options:
                subject_id = add_subject(name="Demo Subject", sex="Female")
                audit("create_demo_subject", "training", "subject", subject_id)
            else:
                subject_id = next(iter(options.values()))
            imported = import_tests_from_dataframe(train_df.head(200), default_subject_id=subject_id, source="synthetic_demo")
            audit("import_synthetic_demo", "training", "subject", subject_id, {"count": imported})
            st.success(f"Imported {imported} demo rows into subject {subject_id}.")

    if st.button("Train Model", type="primary"):
        if train_df.empty:
            st.error("Training data is empty.")
            return
        try:
            model = BreastRiskModel()
            result = model.train(train_df)
            model.save(MODEL_PATH)
            audit("train_model", "ml", "model", "breast_risk_model.joblib", result.metrics)

            st.success(f"Training completed. Model saved: {MODEL_PATH}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AUC", f"{result.metrics['auc']:.4f}")
            c2.metric("Precision", f"{result.metrics['precision']:.4f}")
            c3.metric("Recall", f"{result.metrics['recall']:.4f}")
            c4.metric("Accuracy", f"{result.metrics['accuracy']:.4f}")

            dist_df = pd.DataFrame(
                [{"Class": to_cn_class(k), "Samples": v} for k, v in result.class_distribution.items()]
            )
            st.dataframe(dist_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Training failed: {exc}")


def page_inference() -> None:
    if not can_write():
        st.warning("Read-only role. Inference write-back is disabled.")
        return

    st.subheader("Risk Inference")
    model = load_model()
    if model is None:
        st.warning("No model found. Train model first.")
        return

    subject_options = fetch_subject_options()
    if not subject_options:
        st.warning("No subjects found.")
        return

    selected_label = st.selectbox("Select Subject", list(subject_options.keys()), key="pred_subject")
    subject_id = subject_options[selected_label]
    rows = list_tests(subject_id=subject_id)
    if not rows:
        st.info("No tests for this subject.")
        return

    selected_test = st.selectbox("Select Test", [f"{r['id']} - {r['test_date']}" for r in rows])
    test_id = int(selected_test.split(" - ")[0])
    current = get_test(test_id)
    if not current:
        return

    if st.button("Run Inference", type="primary"):
        sample_df = pd.DataFrame([{k: current.get(k) for k in FEATURE_COLUMNS}])
        try:
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

            st.success("Inference completed and saved.")
            st.write(f"Predicted class: **{to_cn_class(predicted_class)}**")
            st.write(f"Risk level: **{risk_level}**")

            prob_df = pd.DataFrame([{"Class": to_cn_class(k), "Probability": float(v)} for k, v in probabilities.items()])
            st.dataframe(prob_df, use_container_width=True)

            contrib_df = pd.DataFrame(
                [{"Feature": k.upper(), "Contribution": v} for k, v in sorted(pred["feature_contribution"].items(), key=lambda x: x[1], reverse=True)]
            ).set_index("Feature")
            st.bar_chart(contrib_df)

            if warning:
                st.error("Warning: follow-up trend indicates possible elevated recurrence risk.")
            else:
                st.info("No follow-up warning triggered.")
            for note in notes:
                st.write(f"- {note}")
        except Exception as exc:
            st.error(f"Inference failed: {exc}")


def page_followup() -> None:
    st.subheader("Follow-up Monitoring")
    subject_options = fetch_subject_options()
    if not subject_options:
        st.warning("No subjects found.")
        return

    selected_label = st.selectbox("Select Subject", list(subject_options.keys()), key="follow_subject")
    subject_id = subject_options[selected_label]
    df = get_followup_dataframe(subject_id)
    if df.empty:
        st.info("No follow-up data.")
        return

    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    st.dataframe(df, use_container_width=True)
    st.write("Marker Trends")
    st.line_chart(df.set_index("test_date")[FEATURE_COLUMNS])

    if {"normal_prob", "benign_prob", "malignant_prob"}.issubset(df.columns):
        st.write("Probability Trends")
        st.line_chart(df.set_index("test_date")[["normal_prob", "benign_prob", "malignant_prob"]].fillna(0.0))

    warning, notes = followup_warning_analysis(df)
    if warning:
        st.error("Warning signal found in follow-up trend.")
    else:
        st.success("Follow-up trend currently stable.")
    for note in notes:
        st.write(f"- {note}")


def page_report() -> None:
    st.subheader("Report Export")
    subject_options = fetch_subject_options()
    if not subject_options:
        st.warning("No subjects found.")
        return

    selected_label = st.selectbox("Select Subject", list(subject_options.keys()), key="report_subject")
    subject_id = subject_options[selected_label]
    subject = get_subject(subject_id)
    if not subject:
        return
    tests = list_tests(subject_id=subject_id)
    if not tests:
        st.info("No tests for this subject.")
        return

    selected_test = st.selectbox("Select Test", [f"{r['id']} - {r['test_date']}" for r in tests], key="report_test")
    test_id = int(selected_test.split(" - ")[0])
    test_row = get_test(test_id)
    if not test_row:
        return

    evaluation = get_latest_evaluation_for_test(test_id)
    if not evaluation:
        st.warning("No evaluation found. Run inference first.")
        return

    followup_df = get_followup_dataframe(subject_id)
    if st.button("Generate HTML + PDF Report", type="primary"):
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
        st.success(f"Reports generated: {html_path.name}, {pdf_path.name}")

        html_content = html_path.read_text(encoding="utf-8")
        pdf_content = pdf_path.read_bytes()

        st.download_button("Download HTML", data=html_content.encode("utf-8"), file_name=html_path.name, mime="text/html")
        st.download_button("Download PDF", data=pdf_content, file_name=pdf_path.name, mime="application/pdf")
        st.components.v1.html(html_content, height=700, scrolling=True)


def page_audit_logs() -> None:
    if not is_admin():
        st.warning("Admin only.")
        return

    st.subheader("Audit Logs")
    c1, c2, c3 = st.columns(3)
    username = c1.text_input("Filter by username")
    action = c2.text_input("Filter by action")
    limit = c3.number_input("Rows", min_value=50, max_value=2000, value=300, step=50)
    rows = list_audit_logs(limit=int(limit), username=username, action=action)
    if not rows:
        st.info("No logs found.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_user_admin() -> None:
    if not is_admin():
        st.warning("Admin only.")
        return

    st.subheader("User Administration")
    with st.form("create_user_form", clear_on_submit=True):
        username = st.text_input("New username")
        password = st.text_input("New password", type="password")
        role = st.selectbox("Role", ["doctor", "viewer", "admin"])
        submitted = st.form_submit_button("Create User")
    if submitted:
        try:
            user_id = create_user(username=username, password=password, role=role)
            audit("create_user", "users", "user", user_id, {"username": username, "role": role})
            st.success(f"User created: {user_id}")
        except Exception as exc:
            st.error(f"Create failed: {exc}")

    rows = list_users()
    if not rows:
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    options = {f"{r['id']} - {r['username']}": r["id"] for r in rows}
    selected = st.selectbox("Select user", list(options.keys()))
    user_id = options[selected]
    target = next((r for r in rows if r["id"] == user_id), None)
    if target is None:
        return

    c1, c2 = st.columns(2)
    with c1:
        new_role = st.selectbox("Change role", ["admin", "doctor", "viewer"], index=["admin", "doctor", "viewer"].index(target["role"]))
        if st.button("Update Role"):
            update_user_role(user_id, new_role)
            audit("update_user_role", "users", "user", user_id, {"role": new_role})
            st.success("Role updated.")
            _rerun()
    with c2:
        active_label = "Deactivate User" if int(target["is_active"]) else "Activate User"
        if st.button(active_label):
            set_user_active(user_id, not bool(int(target["is_active"])))
            audit("toggle_user_active", "users", "user", user_id, {"is_active": int(not bool(int(target["is_active"])))})
            st.success("Status updated.")
            _rerun()

    with st.form("reset_password_form"):
        new_password = st.text_input("Reset password", type="password")
        do_reset = st.form_submit_button("Reset Password")
    if do_reset:
        try:
            update_user_password(user_id, new_password)
            audit("reset_user_password", "users", "user", user_id)
            st.success("Password reset completed.")
        except Exception as exc:
            st.error(f"Reset failed: {exc}")


if __name__ == "__main__":
    main()

