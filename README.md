# Breast Full-Process Risk Assessment System V1.0

## Features

- Subject and test data management (CRUD, CSV import)
- Data preprocessing (outlier clipping, missing imputation, scaling)
- Ensemble model training (AUC / Precision / Recall / Accuracy)
- Risk inference with class probability and feature contribution
- Follow-up trend monitoring and warning rules
- Report export (HTML + PDF)
- Authentication and role-based access (`admin`, `doctor`, `viewer`)
- Full audit trail (login, CRUD, training, inference, report export)

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Default admin account:

- Username: `admin`
- Password: `Admin@123456`

## Data fields

Required model fields:

- `akr1b10`
- `ca19_9`
- `nse`
- `ca125`
- `ca153`
- `cea`
- `label` (`normal` / `benign` / `malignant`)

Optional:

- `test_date`
- `subject_id`
- `clinical_stage`

## Structure

```text
.
├── app.py
├── medical_system/
│   ├── auth.py
│   ├── config.py
│   ├── database.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── risk.py
│   └── reporting.py
├── scripts/
│   └── generate_sample_data.py
├── data/
│   └── sample_input_template.csv
└── requirements.txt
```

