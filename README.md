# CCZG506 — Assignment I: Cloud-Native ML Pipeline
### Stack: Prefect Cloud · MLflow · FastAPI · AWS S3 · AWS CloudShell

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [What Runs Where](#2-what-runs-where)
3. [Project Structure](#3-project-structure)
4. [Step 1 — Get AWS Credentials from CloudShell](#4-step-1--get-aws-credentials-from-cloudshell)
5. [Step 2 — Create S3 Bucket](#5-step-2--create-s3-bucket)
6. [Step 3 — Upload Titanic Dataset](#6-step-3--upload-titanic-dataset)
7. [Step 4 — Local Python Setup](#7-step-4--local-python-setup)
8. [Step 5 — Prefect Cloud Setup](#8-step-5--prefect-cloud-setup)
9. [Sub-Objective 1 — Data Pipeline Code](#9-sub-objective-1--data-pipeline-code)
10. [Sub-Objective 2 — ML Pipeline Code](#10-sub-objective-2--ml-pipeline-code)
11. [Sub-Objective 3 — API Code](#11-sub-objective-3--api-code)
12. [Run Everything — Step by Step](#12-run-everything--step-by-step)
13. [Refresh Expired Credentials](#13-refresh-expired-credentials)
14. [Screenshots Guide](#14-screenshots-guide)
15. [Mark Breakdown](#15-mark-breakdown)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                                   │
│                                                                          │
│  ┌─────────────────────────┐      ┌─────────────────────────────────┐   │
│  │      Amazon S3          │      │        Prefect Cloud            │   │
│  │                         │      │     app.prefect.cloud           │   │
│  │  data/titanic.csv       │      │                                 │   │
│  │  data/titanic_proc.csv  │      │  ✓ Schedule  (*/3 * * * *)      │   │
│  │  plots/*.png (5 charts) │      │  ✓ Flow run dashboard           │   │
│  │  mlflow-artifacts/      │      │  ✓ Live task logs               │   │
│  └────────────▲────────────┘      │  ✓ REST API                     │   │
│               │ reads/writes      └──────────────▲──────────────────┘   │
│  ┌────────────┴────────────┐                     │ schedules & logs      │
│  │    AWS CloudShell       │                     │                       │
│  │  (credential source)    │                     │                       │
│  │  aws sts get-session-   │                     │                       │
│  │  token  →  copy creds   │                     │                       │
│  └─────────────────────────┘                     │                       │
└─────────────────────────────────────────────────-│───────────────────────┘
                                                   │
┌──────────────────────────────────────────────────│───────────────────────┐
│                     Your Local Machine            │                       │
│                                                   │                       │
│   ┌───────────────────────────────────────────────▼──────────────────┐   │
│   │  Terminal 1: mlflow server  (localhost:5000)                      │   │
│   │  Terminal 2: python ml/ml_pipeline.py   (runs once)              │   │
│   │  Terminal 3: prefect worker  ←──────────────── polls Prefect     │   │
│   │             + python flows/data_pipeline.py  (registers schedule)│   │
│   │  Terminal 4: uvicorn api/main  (localhost:8000)                  │   │
│   └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. What Runs Where

| Component | Where | Notes |
|---|---|---|
| **Prefect Cloud dashboard** | Prefect's hosted servers | `app.prefect.cloud` — real cloud SaaS |
| **Flow schedule (*/3 min)** | Prefect Cloud | Hosted externally |
| **Flow run logs** | Prefect Cloud | Streamed from local worker, stored in cloud |
| **Prefect REST API** | Prefect Cloud | Used by FastAPI for Sub-Objective 3 |
| **S3 — data, plots, models** | AWS S3 | Fully cloud storage |
| **AWS credentials** | AWS CloudShell | Temporary session credentials, ~12hr expiry |
| **Prefect worker** | Local machine | Executes flows, reports to Prefect Cloud |
| **MLflow UI + tracking** | Local machine | localhost:5000 |
| **FastAPI Swagger UI** | Local machine | localhost:8000 |

> **Why local execution:** The BITS Virtual Lab AWS account has an SCP (Service Control Policy)
> that blocks `ec2:RunInstances` and federated sessions cannot generate permanent IAM keys.
> All cloud components (Prefect Cloud, S3, schedule, dashboard) remain genuinely cloud-hosted.

---

## 3. Project Structure

```
mlpipeline/
├── flows/
│   └── data_pipeline.py       # Prefect @flow — 3 @tasks + deployment
├── ml/
│   └── ml_pipeline.py         # MLflow training pipeline
├── api/
│   └── main.py                # FastAPI application
├── data/
│   └── titanic.csv            # Downloaded from Kaggle
└── requirements.txt
```

---

## 4. Step 1 — Get AWS Credentials from CloudShell

Your BITS Virtual Lab account uses **federated sessions** — you cannot create permanent
IAM access keys. Instead, you get **temporary credentials** from AWS CloudShell.
These last approximately 12 hours per session.

### Open CloudShell

1. Log in to **https://console.aws.amazon.com**
2. In the top navigation bar, click the **CloudShell icon** (looks like `>_`)
   — OR — search for **CloudShell** in the services search bar
3. Wait ~30 seconds for the shell to initialize

### Get temporary credentials

Run this in CloudShell:

```bash
aws sts get-session-token --duration-seconds 43200
```

You will get output like this:

```json
{
    "Credentials": {
        "AccessKeyId":     "ASIA3XXXXXXXXXXXXXXXXX",
        "SecretAccessKey": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "SessionToken":    "FwoGZXIvYXdzEJ...very long string...==",
        "Expiration":      "2026-04-01T10:00:00+00:00"
    }
}
```

### Set credentials on your local machine

Copy the three values and set them as environment variables locally.

**Mac / Linux** — paste in your terminal:
```bash
export AWS_ACCESS_KEY_ID="ASIA3XXXXXXXXXXXXXXXXX"
export AWS_SECRET_ACCESS_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export AWS_SESSION_TOKEN="FwoGZXIvYXdz...paste full token here..."
export AWS_DEFAULT_REGION="ap-south-1"
```

Also add your other project variables:
```bash
export S3_BUCKET="mlpipeline-titanic-<yourname>"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export PREFECT_API_KEY=""       # fill in after Step 5
export PREFECT_API_URL=""       # fill in after Step 5
```

**Windows CMD** — paste in Command Prompt:
```cmd
set AWS_ACCESS_KEY_ID=ASIA3XXXXXXXXXXXXXXXXX
set AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
set AWS_SESSION_TOKEN=FwoGZXIvYXdz...paste full token here...
set AWS_DEFAULT_REGION=ap-south-1
set S3_BUCKET=mlpipeline-titanic-<yourname>
set MLFLOW_TRACKING_URI=http://localhost:5000
set PREFECT_API_KEY=
set PREFECT_API_URL=
```

**Windows PowerShell:**
```powershell
$env:AWS_ACCESS_KEY_ID     = "ASIA3XXXXXXXXXXXXXXXXX"
$env:AWS_SECRET_ACCESS_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:AWS_SESSION_TOKEN     = "FwoGZXIvYXdz...paste full token here..."
$env:AWS_DEFAULT_REGION    = "ap-south-1"
$env:S3_BUCKET             = "mlpipeline-titanic-<yourname>"
$env:MLFLOW_TRACKING_URI   = "http://localhost:5000"
$env:PREFECT_API_KEY       = ""
$env:PREFECT_API_URL       = ""
```

### Verify credentials work

```bash
aws sts get-caller-identity
# Should return your account ID and role without error
```

> **Important:** These credentials expire after ~12 hours. If you see
> `ExpiredTokenException` at any point, go to Section 13 to refresh them.
> You do NOT need to redo any other steps — just refresh the credentials.

---

## 5. Step 2 — Create S3 Bucket

Run these commands **in CloudShell** (credentials work automatically there):

```bash
# Choose a globally unique bucket name
BUCKET="mlpipeline-titanic-<yourname>"    # CHANGE <yourname>

# Create bucket
aws s3 mb s3://$BUCKET --region ap-south-1

# Create the three folders
aws s3api put-object --bucket $BUCKET --key data/
aws s3api put-object --bucket $BUCKET --key plots/
aws s3api put-object --bucket $BUCKET --key mlflow-artifacts/

# Verify
aws s3 ls s3://$BUCKET/
```

If `aws s3 mb` is restricted, create the bucket manually instead:
1. AWS Console → **S3** → **Create Bucket**
2. Name: `mlpipeline-titanic-<yourname>` | Region: `ap-south-1`
3. All other settings default → **Create Bucket**
4. Open bucket → **Create folder** → create: `data/` `plots/` `mlflow-artifacts/`

---

## 6. Step 3 — Upload Titanic Dataset

1. Download `titanic.csv` from:
   **https://www.kaggle.com/datasets/yasserh/titanic-dataset**

2. Upload to S3 — two options:

**Option A — AWS Console (easiest):**
- AWS Console → S3 → your bucket → `data/` folder → **Upload** → select `titanic.csv`

**Option B — from your local terminal** (after credentials are set):
```bash
aws s3 cp data/titanic.csv s3://$S3_BUCKET/data/titanic.csv

# Verify
aws s3 ls s3://$S3_BUCKET/data/
# Should show: titanic.csv
```

---

## 7. Step 4 — Local Python Setup

### Install Python

Make sure Python 3.9+ is installed:
```bash
python --version    # or python3 --version
```

Download from https://python.org if needed.

### Create project folder and virtual environment

```bash
mkdir mlpipeline
cd mlpipeline
mkdir flows ml api data

# Create virtual environment
python -m venv venv          # Windows
python3 -m venv venv         # Mac/Linux

# Activate
venv\Scripts\activate        # Windows CMD
venv\Scripts\Activate.ps1   # Windows PowerShell
source venv/bin/activate     # Mac/Linux

# Prompt should now show (venv)
```

### Install all dependencies

```bash
pip install --upgrade pip

pip install \
  "prefect==2.19.9" \
  "mlflow==2.13.0" \
  "fastapi==0.111.0" \
  "uvicorn==0.29.0" \
  "pandas==2.2.2" \
  "scikit-learn==1.4.2" \
  "matplotlib==3.9.0" \
  "seaborn==0.13.2" \
  "boto3==1.34.0" \
  "requests==2.32.0" \
  "python-multipart==0.0.9"
```

Verify:
```bash
python -c "import prefect, mlflow, pandas, sklearn, boto3; print('All packages OK')"
```

---

## 8. Step 5 — Prefect Cloud Setup

### Create account

1. Go to **https://app.prefect.cloud** → Sign Up (free, no credit card)
2. You will land in the **default workspace** — this is all you need

### Get API key and URL

1. Click **Settings** (gear icon, bottom-left) → **API Keys** → **+ Add API Key**
2. Name: `local-agent` → **Create** → **copy the key immediately** (shown once only)
3. Your API URL is shown on the same Settings page. It looks like:
   ```
   https://api.prefect.cloud/api/accounts/abc12345-xxxx-xxxx-xxxx-xxxxxxxxxxxx/workspaces/def67890-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   ```

### Set environment variables

Go back to your local terminal and fill in these two values:

**Mac/Linux:**
```bash
export PREFECT_API_KEY="pnu_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export PREFECT_API_URL="https://api.prefect.cloud/api/accounts/YOUR_ACCOUNT_ID/workspaces/YOUR_WORKSPACE_ID"
```

**Windows CMD:**
```cmd
set PREFECT_API_KEY=pnu_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
set PREFECT_API_URL=https://api.prefect.cloud/api/accounts/YOUR_ACCOUNT_ID/workspaces/YOUR_WORKSPACE_ID
```

**Windows PowerShell:**
```powershell
$env:PREFECT_API_KEY = "pnu_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:PREFECT_API_URL = "https://api.prefect.cloud/api/accounts/YOUR_ACCOUNT_ID/workspaces/YOUR_WORKSPACE_ID"
```

### Connect local machine to Prefect Cloud

```bash
source venv/bin/activate    # or venv\Scripts\activate on Windows

prefect cloud login --key $PREFECT_API_KEY
# Select the default workspace when prompted

prefect cloud workspace ls
# Should show your default workspace with a * marking it active
```

### Create work pool

```bash
prefect work-pool create "local-work-pool" --type process

prefect work-pool ls
# Should show local-work-pool as READY
```

---

## 9. Sub-Objective 1 — Data Pipeline Code

Create `flows/data_pipeline.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 1
Prefect Flow: Data Ingestion → Preprocessing → EDA
Scheduled every 3 minutes via Prefect Cloud.
Data and plots stored on AWS S3.
Logs stream live to app.prefect.cloud dashboard.
"""

from prefect import flow, task, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import boto3
import io
import os

# ─── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET       = os.environ.get("S3_BUCKET", "mlpipeline-titanic-yourname")
S3_DATA_KEY     = "data/titanic.csv"
S3_PROC_KEY     = "data/titanic_processed.csv"
S3_PLOTS_PREFIX = "plots/"
AWS_REGION      = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

# ─── S3 Helpers ───────────────────────────────────────────────────────────────
def _s3():
    return boto3.client("s3", region_name=AWS_REGION)

def read_csv_from_s3(key: str) -> pd.DataFrame:
    obj = _s3().get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def write_csv_to_s3(df: pd.DataFrame, key: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())

def upload_plot_to_s3(fig, key: str):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    _s3().put_object(
        Bucket=S3_BUCKET, Key=key,
        Body=buf.getvalue(), ContentType="image/png"
    )

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.2 — Data Ingestion
# ─────────────────────────────────────────────────────────────────────────────
@task(
    name="Data Ingestion",
    description="Load Titanic dataset from S3 and log dataset statistics.",
    retries=2,
    retry_delay_seconds=30,
    tags=["ingestion", "s3"],
)
def ingest_data() -> dict:
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("TASK 1.2: DATA INGESTION — STARTED")
    logger.info("=" * 60)

    df = read_csv_from_s3(S3_DATA_KEY)

    logger.info(f"Source         : s3://{S3_BUCKET}/{S3_DATA_KEY}")
    logger.info(f"Rows           : {df.shape[0]}")
    logger.info(f"Columns        : {df.shape[1]}")
    logger.info(f"Column names   : {list(df.columns)}")
    logger.info(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    logger.info(f"Data Types:\n{df.dtypes.to_string()}")
    logger.info(f"First 3 rows:\n{df.head(3).to_string()}")

    total_missing = int(df.isnull().sum().sum())
    logger.info(f"Total missing values across dataset: {total_missing}")
    logger.info("TASK 1.2: DATA INGESTION — COMPLETED SUCCESSFULLY")

    return {
        "row_count":     int(df.shape[0]),
        "col_count":     int(df.shape[1]),
        "columns":       list(df.columns),
        "missing_total": total_missing,
    }

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.3 — Data Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
@task(
    name="Data Preprocessing",
    description="Summary stats, impute missing values, normalize, save to S3.",
    retries=2,
    retry_delay_seconds=30,
    tags=["preprocessing", "s3"],
)
def preprocess_data(ingestion_stats: dict) -> dict:
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("TASK 1.3: DATA PREPROCESSING — STARTED")
    logger.info(f"Upstream ingestion stats: {ingestion_stats}")
    logger.info("=" * 60)

    df = read_csv_from_s3(S3_DATA_KEY)

    # ── 1. Summary Statistics ─────────────────────────────────────────────────
    logger.info("--- Summary Statistics ---")
    logger.info(f"\n{df.describe().round(4).to_string()}")

    # ── 2. Data Types ─────────────────────────────────────────────────────────
    logger.info("--- Data Types ---")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col:<15} : {dtype}")

    # ── 3. Check Missing Values ───────────────────────────────────────────────
    missing = df.isnull().sum()
    logger.info("--- Missing Values (Before Imputation) ---")
    for col, cnt in missing[missing > 0].items():
        pct = (cnt / len(df)) * 100
        logger.info(f"  {col:<15} : {cnt} missing ({pct:.1f}% of rows)")

    # ── 4. Impute Numeric Columns with Median ─────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputed = {}
    for col in numeric_cols:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            imputed[col] = float(median_val)
            logger.info(f"  Imputed '{col}' ({n_null} nulls) → median = {median_val:.4f}")

    # ── 5. Impute Categorical Columns ─────────────────────────────────────────
    if df["Embarked"].isnull().sum() > 0:
        mode_val = df["Embarked"].mode()[0]
        df["Embarked"].fillna(mode_val, inplace=True)
        logger.info(f"  Imputed 'Embarked' → mode = '{mode_val}'")

    if "Cabin" in df.columns:
        df["Cabin"].fillna("Unknown", inplace=True)
        logger.info("  Imputed 'Cabin' → 'Unknown'")

    remaining_nulls = int(df.isnull().sum().sum())
    logger.info(f"  Remaining null values after imputation: {remaining_nulls}")

    # ── 6. Normalize Numeric Columns ─────────────────────────────────────────
    cols_to_scale = ["Age", "Fare", "SibSp", "Parch"]
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    logger.info(f"  Normalized (MinMaxScaler, 0–1): {cols_to_scale}")
    logger.info(f"  Post-normalization stats:\n{df[cols_to_scale].describe().round(4).to_string()}")

    # ── 7. Save to S3 ─────────────────────────────────────────────────────────
    write_csv_to_s3(df, S3_PROC_KEY)
    logger.info(f"  Saved → s3://{S3_BUCKET}/{S3_PROC_KEY}")

    logger.info("TASK 1.3: DATA PREPROCESSING — COMPLETED SUCCESSFULLY")
    return {
        "processed_rows": int(df.shape[0]),
        "processed_cols": int(df.shape[1]),
        "imputed_medians": imputed,
        "normalized_cols": cols_to_scale,
        "remaining_nulls": remaining_nulls,
    }

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.4 — Exploratory Data Analysis
# ─────────────────────────────────────────────────────────────────────────────
@task(
    name="Exploratory Data Analysis",
    description="Correlation, encoding, binning, feature importance, 5 plots → S3.",
    retries=2,
    retry_delay_seconds=30,
    tags=["eda", "visualization", "s3"],
)
def perform_eda(preprocessing_stats: dict) -> dict:
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("TASK 1.4: EXPLORATORY DATA ANALYSIS — STARTED")
    logger.info(f"Upstream preprocessing stats: {preprocessing_stats}")
    logger.info("=" * 60)

    df         = read_csv_from_s3(S3_PROC_KEY)
    numeric_df = df.select_dtypes(include=[np.number])

    # ── 1. Correlation Matrix ─────────────────────────────────────────────────
    corr = numeric_df.corr()
    logger.info("--- Full Correlation Matrix ---")
    logger.info(f"\n{corr.round(3).to_string()}")

    # ── 2. Correlations with Target ───────────────────────────────────────────
    target_corr = corr["Survived"].abs().sort_values(ascending=False)
    logger.info("--- Absolute Correlation with Target (Survived) ---")
    for feat, val in target_corr.items():
        bar = "█" * int(val * 30)
        logger.info(f"  {feat:<25} : {val:.4f}  {bar}")

    # ── 3. Encoding Categorical Features ─────────────────────────────────────
    df["Sex_encoded"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked_encoded"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(-1).astype(int)
    logger.info("Encoding: Sex → {male:0, female:1}  |  Embarked → {S:0, C:1, Q:2}")

    # ── 4. Binning Age ────────────────────────────────────────────────────────
    df["Age_bin"] = pd.cut(
        df["Age"], bins=5,
        labels=["Very Young", "Young", "Middle-aged", "Senior", "Elderly"]
    )
    logger.info("--- Age Binning Distribution ---")
    for label, count in df["Age_bin"].value_counts().sort_index().items():
        logger.info(f"  {str(label):<15} : {count} passengers")

    # ── 5. Feature Importance ─────────────────────────────────────────────────
    numeric_df2 = df.select_dtypes(include=[np.number])
    feature_imp = (
        numeric_df2.corr()["Survived"].abs()
        .drop("Survived")
        .sort_values(ascending=False)
    )
    logger.info("--- Feature Importance (|Pearson r| with Survived) ---")
    for feat, score in feature_imp.items():
        logger.info(f"  {feat:<25} : {score:.4f}")

    # ── 6. Survival Stats ─────────────────────────────────────────────────────
    survival_rate = df["Survived"].mean() * 100
    logger.info(f"Overall Survival Rate : {survival_rate:.1f}%")
    logger.info(f"By Sex:\n{df.groupby('Sex')['Survived'].mean().mul(100).round(1).to_string()}%")
    logger.info(f"By Pclass:\n{df.groupby('Pclass')['Survived'].mean().mul(100).round(1).to_string()}%")

    plots_saved = []

    # ── Plot 1: Correlation Heatmap ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Correlation Heatmap — Titanic Features",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    key = f"{S3_PLOTS_PREFIX}1_correlation_heatmap.png"
    upload_plot_to_s3(fig, key)
    plt.close(fig)
    plots_saved.append(key)
    logger.info(f"Plot 1 saved → s3://{S3_BUCKET}/{key}")

    # ── Plot 2: Univariate Analysis ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df["Survived"].value_counts().plot(
        kind="bar", ax=axes[0],
        color=["#e74c3c", "#2ecc71"], edgecolor="black", width=0.5
    )
    axes[0].set_title("Survival Count", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Survived (0=No, 1=Yes)")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(["Did Not Survive", "Survived"], rotation=0)
    for p in axes[0].patches:
        axes[0].annotate(
            str(int(p.get_height())),
            (p.get_x() + p.get_width() / 2, p.get_height() + 3),
            ha="center", fontsize=11
        )
    df["Pclass"].value_counts().sort_index().plot(
        kind="bar", ax=axes[1],
        color=["#3498db", "#f39c12", "#9b59b6"], edgecolor="black", width=0.5
    )
    axes[1].set_title("Passenger Class Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Passenger Class")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(["1st Class", "2nd Class", "3rd Class"], rotation=0)
    fig.suptitle("Univariate Analysis — Titanic Dataset", fontsize=15, fontweight="bold")
    plt.tight_layout()
    key = f"{S3_PLOTS_PREFIX}2_univariate_analysis.png"
    upload_plot_to_s3(fig, key)
    plt.close(fig)
    plots_saved.append(key)
    logger.info(f"Plot 2 saved → s3://{S3_BUCKET}/{key}")

    # ── Plot 3: Bivariate — Survival by Sex and Pclass ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    surv_label = df["Survived"].map({0: "No", 1: "Yes"})
    sns.countplot(data=df.assign(Survived_Label=surv_label),
                  x="Sex", hue="Survived_Label",
                  palette={"No": "#e74c3c", "Yes": "#2ecc71"},
                  ax=axes[0], edgecolor="black")
    axes[0].set_title("Survival by Gender", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Survived")
    surv_class = df.groupby("Pclass")["Survived"].mean() * 100
    bars = axes[1].bar(
        surv_class.index.astype(str), surv_class.values,
        color=["#3498db", "#f39c12", "#9b59b6"], edgecolor="black", width=0.5
    )
    axes[1].set_title("Survival Rate by Passenger Class", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Passenger Class")
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars, surv_class.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     val + 1.5, f"{val:.1f}%", ha="center", fontsize=11)
    fig.suptitle("Bivariate Analysis — Survival by Demographics",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    key = f"{S3_PLOTS_PREFIX}3_bivariate_sex_pclass.png"
    upload_plot_to_s3(fig, key)
    plt.close(fig)
    plots_saved.append(key)
    logger.info(f"Plot 3 saved → s3://{S3_BUCKET}/{key}")

    # ── Plot 4: Age KDE by Survival ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for val, color, label in [
        (0, "#e74c3c", "Did Not Survive"),
        (1, "#2ecc71", "Survived"),
    ]:
        sns.kdeplot(df[df["Survived"] == val]["Age"],
                    ax=ax, fill=True, color=color, alpha=0.5, label=label)
    ax.set_title("Age Distribution by Survival Outcome",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (Normalized 0–1)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=11)
    plt.tight_layout()
    key = f"{S3_PLOTS_PREFIX}4_age_distribution_survival.png"
    upload_plot_to_s3(fig, key)
    plt.close(fig)
    plots_saved.append(key)
    logger.info(f"Plot 4 saved → s3://{S3_BUCKET}/{key}")

    # ── Plot 5: Feature Importance ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    feature_imp.plot(kind="bar", ax=ax, color="#3498db", edgecolor="black", width=0.6)
    ax.set_title("Feature Importance — |Correlation| with Survived",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Absolute Pearson Correlation")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    key = f"{S3_PLOTS_PREFIX}5_feature_importance.png"
    upload_plot_to_s3(fig, key)
    plt.close(fig)
    plots_saved.append(key)
    logger.info(f"Plot 5 saved → s3://{S3_BUCKET}/{key}")

    logger.info(f"All {len(plots_saved)} plots uploaded to S3.")
    logger.info("TASK 1.4: EDA — COMPLETED SUCCESSFULLY")
    return {"plots_saved": plots_saved, "survival_rate_pct": round(survival_rate, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# FLOW (Sub-Objective 1.5 — DataOps)
# ─────────────────────────────────────────────────────────────────────────────
@flow(
    name="titanic-data-pipeline",
    description=(
        "End-to-end Titanic data pipeline: ingestion → preprocessing → EDA. "
        "Scheduled every 3 minutes via Prefect Cloud. Artifacts on AWS S3."
    ),
    version="1.0",
    log_prints=True,
)
def data_pipeline():
    ingestion_stats     = ingest_data()
    preprocessing_stats = preprocess_data(ingestion_stats)
    eda_stats           = perform_eda(preprocessing_stats)
    return {
        "ingestion":     ingestion_stats,
        "preprocessing": preprocessing_stats,
        "eda":           eda_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEPLOYMENT — run once: python flows/data_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=data_pipeline,
        name="titanic-pipeline-every-3min",
        work_pool_name="local-work-pool",
        schedule=CronSchedule(
            cron="*/3 * * * *",       # Every 3 minutes ✅
            timezone="Asia/Kolkata",
        ),
        tags=["DataOps", "Titanic", "CCZG506"],
        description="Automated Titanic data pipeline. Runs every 3 minutes.",
    )
    deployment_id = deployment.apply()
    print(f"\n✅ Deployment registered on Prefect Cloud!")
    print(f"   ID       : {deployment_id}")
    print(f"   Schedule : Every 3 minutes  (*/3 * * * *)")
    print(f"   Pool     : local-work-pool")
    print(f"   View at  : https://app.prefect.cloud → Deployments")
```

---

## 10. Sub-Objective 2 — ML Pipeline Code

Create `ml/ml_pipeline.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 2
MLflow Training Pipeline.
Algorithms: Random Forest + Logistic Regression
Metrics logged to MLflow. Model artifacts stored on S3.
UI: http://localhost:5000
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import boto3
import logging
import io
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

# ─── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET   = os.environ.get("S3_BUCKET", "mlpipeline-titanic-yourname")
S3_DATA_KEY = "data/titanic.csv"
AWS_REGION  = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
MLFLOW_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT  = "Titanic_Survival_Prediction"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLPipeline")

mlflow.set_tracking_uri(MLFLOW_URI)
os.environ["MLFLOW_ARTIFACT_ROOT"] = f"s3://{S3_BUCKET}/mlflow-artifacts"
mlflow.set_experiment(EXPERIMENT)


def load_and_prepare_data():
    logger.info("=" * 60)
    logger.info("STEP 2.1: MODEL PREPARATION")
    logger.info("=" * 60)

    s3  = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY)
    df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
    logger.info(f"Loaded {df.shape[0]} rows from s3://{S3_BUCKET}/{S3_DATA_KEY}")

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"], inplace=True)

    le = LabelEncoder()
    df["Sex"]      = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])

    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[feature_cols]
    y = df["Survived"]

    logger.info(f"Features      : {feature_cols}")
    logger.info(f"Class balance : {y.value_counts().to_dict()}")
    logger.info(f"Survival rate : {y.mean() * 100:.1f}%")
    return X, y, feature_cols


def train_and_log(model, model_name, X_train, X_test,
                  y_train, y_test, feature_cols):
    logger.info(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")

    with mlflow.start_run(run_name=model_name) as run:

        mlflow.log_param("model_name",    model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples",  len(X_test))
        mlflow.log_param("train_pct",     "70%")
        mlflow.log_param("test_pct",      "30%")
        mlflow.log_param("features",      str(feature_cols))
        mlflow.log_param("n_features",    len(feature_cols))
        mlflow.log_param("random_state",  42)

        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth",    model.max_depth)
        if hasattr(model, "max_iter"):
            mlflow.log_param("max_iter", model.max_iter)
            mlflow.log_param("solver",   model.solver)
            mlflow.log_param("C",        model.C)

        # Train
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy    = accuracy_score(y_test, y_pred)
        precision   = precision_score(y_test, y_pred, zero_division=0)
        recall      = recall_score(y_test, y_pred, zero_division=0)
        f1          = f1_score(y_test, y_pred, zero_division=0)
        roc_auc     = roc_auc_score(y_test, y_pred_prob)
        cv_scores   = cross_val_score(model, X_train, y_train,
                                      cv=5, scoring="accuracy")
        cv_mean     = cv_scores.mean()
        cv_std      = cv_scores.std()
        cm          = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Log all metrics ✅ (satisfies 4+ metric requirement)
        mlflow.log_metric("accuracy",         round(accuracy,    4))
        mlflow.log_metric("precision",        round(precision,   4))
        mlflow.log_metric("recall",           round(recall,      4))
        mlflow.log_metric("f1_score",         round(f1,          4))
        mlflow.log_metric("roc_auc",          round(roc_auc,     4))
        mlflow.log_metric("cv_mean_accuracy", round(cv_mean,     4))
        mlflow.log_metric("cv_std_accuracy",  round(cv_std,      4))
        mlflow.log_metric("specificity",      round(specificity, 4))
        mlflow.log_metric("true_positives",   int(tp))
        mlflow.log_metric("true_negatives",   int(tn))
        mlflow.log_metric("false_positives",  int(fp))
        mlflow.log_metric("false_negatives",  int(fn))

        # Save model to S3
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        logger.info(f"  Accuracy          : {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision         : {precision:.4f}")
        logger.info(f"  Recall            : {recall:.4f}")
        logger.info(f"  F1 Score          : {f1:.4f}")
        logger.info(f"  ROC-AUC           : {roc_auc:.4f}")
        logger.info(f"  CV (5-fold) Acc   : {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"  Specificity       : {specificity:.4f}")
        logger.info(
            f"\n  Confusion Matrix:\n"
            f"                    Predicted No   Predicted Yes\n"
            f"  Actual No           TN={tn:<6}     FP={fp}\n"
            f"  Actual Yes          FN={fn:<6}     TP={tp}"
        )
        logger.info(
            f"\n{classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived'])}"
        )
        logger.info(f"  MLflow Run ID : {run.info.run_id}")

        return run.info.run_id, {
            "accuracy": accuracy, "precision": precision,
            "recall": recall,    "f1_score": f1,
            "roc_auc": roc_auc,  "run_id": run.info.run_id,
        }


def main():
    logger.info("\n" + "=" * 60)
    logger.info("CCZG506 — ML PIPELINE STARTING")
    logger.info("=" * 60)

    X, y, feature_cols = load_and_prepare_data()

    # 70/30 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    logger.info(f"\nSplit → Train: {len(X_train)} | Test: {len(X_test)}")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # Algorithm 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        min_samples_split=5, random_state=42, n_jobs=-1,
    )
    _, results["RandomForest"] = train_and_log(
        rf, "RandomForest", X_train, X_test, y_train, y_test, feature_cols
    )

    # Algorithm 2: Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    _, results["LogisticRegression"] = train_and_log(
        lr, "LogisticRegression",
        X_train_sc, X_test_sc, y_train, y_test, feature_cols
    )

    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<22} {'Random Forest':>16} {'Logistic Reg':>16}")
    logger.info("-" * 56)
    for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        logger.info(
            f"  {m:<20} "
            f"{results['RandomForest'][m]:>16.4f} "
            f"{results['LogisticRegression'][m]:>16.4f}"
        )
    best = max(results, key=lambda k: results[k]["f1_score"])
    logger.info(f"\nBest Model (by F1) : {best}")
    logger.info(f"MLflow UI          : http://localhost:5000")


if __name__ == "__main__":
    main()
```

---

## 11. Sub-Objective 3 — API Code

Create `api/main.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 3
FastAPI: exposes Prefect Cloud pipeline details + MLflow model details.
Satisfies 3.1 (Built-in APIs) and 3.2 (Display Application Details).
Swagger UI: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
import mlflow
import boto3
import requests
import os

# ─── Config ───────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "Titanic_Survival_Prediction"
S3_BUCKET       = os.environ.get("S3_BUCKET", "mlpipeline-titanic-yourname")
AWS_REGION      = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
PREFECT_API_URL = os.environ.get("PREFECT_API_URL", "")
PREFECT_API_KEY = os.environ.get("PREFECT_API_KEY", "")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow_client = MlflowClient(MLFLOW_URI)

app = FastAPI(
    title="CCZG506 Cloud ML Application API",
    description=(
        "REST APIs exposing Prefect Cloud pipeline details and MLflow "
        "model deployment details.  Assignment I — BITS Pilani WILP."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

def _ph():
    return {"Authorization": f"Bearer {PREFECT_API_KEY}",
            "Content-Type":  "application/json"}


@app.get("/health", tags=["System"],
         summary="Status of all connected services")
def health_check():
    status = {"api": "ok", "mlflow": "unknown",
              "prefect_cloud": "unknown", "s3": "unknown"}
    try:
        mlflow_client.search_experiments()
        status["mlflow"] = "ok"
    except Exception:
        status["mlflow"] = "unreachable"
    try:
        r = requests.get(f"{PREFECT_API_URL}/health", headers=_ph(), timeout=5)
        status["prefect_cloud"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        status["prefect_cloud"] = "unreachable"
    try:
        boto3.client("s3", region_name=AWS_REGION).head_bucket(Bucket=S3_BUCKET)
        status["s3"] = "ok"
    except Exception:
        status["s3"] = "unreachable"
    return status


@app.get("/prefect/flows", tags=["Prefect Cloud — Flow Details"],
         summary="App Detail 1: All registered Prefect flows")
def list_flows():
    """All flows registered on Prefect Cloud — name, ID, tags."""
    if not PREFECT_API_URL:
        raise HTTPException(503, "PREFECT_API_URL env var not set.")
    try:
        r = requests.post(f"{PREFECT_API_URL}/flows/filter",
                          headers=_ph(), json={"limit": 25}, timeout=10)
        r.raise_for_status()
        return {
            "total_flows": len(r.json()),
            "flows": [{"id": f.get("id"), "name": f.get("name"),
                        "created": f.get("created"), "tags": f.get("tags", [])}
                      for f in r.json()],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/prefect/deployments", tags=["Prefect Cloud — Flow Details"],
         summary="App Detail 2: Deployments with schedule info")
def list_deployments():
    """
    All deployments from Prefect Cloud.
    Shows schedule (*/3 * * * *), work pool, active status.
    Directly satisfies 'flow/deployment details via Built-in APIs'.
    """
    if not PREFECT_API_URL:
        raise HTTPException(503, "PREFECT_API_URL env var not set.")
    try:
        r = requests.post(f"{PREFECT_API_URL}/deployments/filter",
                          headers=_ph(), json={"limit": 25}, timeout=10)
        r.raise_for_status()
        return {
            "total_deployments": len(r.json()),
            "deployments": [
                {
                    "id":                 d.get("id"),
                    "name":               d.get("name"),
                    "status":             d.get("status"),
                    "is_schedule_active": d.get("is_schedule_active"),
                    "schedule": {
                        "cron":     d.get("schedule", {}).get("cron"),
                        "timezone": d.get("schedule", {}).get("timezone"),
                    } if d.get("schedule") else None,
                    "work_pool_name": d.get("work_pool_name"),
                    "tags":           d.get("tags", []),
                    "created":        d.get("created"),
                }
                for d in r.json()
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/prefect/flow-runs", tags=["Prefect Cloud — Flow Details"],
         summary="App Detail 3: Recent flow run history")
def get_flow_runs(limit: int = 10):
    """Recent pipeline executions — state, timing, duration."""
    if not PREFECT_API_URL:
        raise HTTPException(503, "PREFECT_API_URL env var not set.")
    try:
        r = requests.post(f"{PREFECT_API_URL}/flow_runs/filter",
                          headers=_ph(),
                          json={"limit": limit, "sort": "START_TIME_DESC"},
                          timeout=10)
        r.raise_for_status()
        return {
            "total_runs": len(r.json()),
            "runs": [
                {
                    "id":   run.get("id"),
                    "name": run.get("name"),
                    "state": {"type": run.get("state", {}).get("type"),
                               "name": run.get("state", {}).get("name")},
                    "start_time":     run.get("start_time"),
                    "end_time":       run.get("end_time"),
                    "total_run_time": run.get("total_run_time"),
                    "tags":           run.get("tags", []),
                }
                for run in r.json()
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/mlflow/experiment", tags=["MLflow — Model Details"],
         summary="App Detail 4: ML experiment with all model metrics")
def get_experiment():
    """Both model runs with metrics: accuracy, precision, recall, F1, ROC-AUC."""
    try:
        exp = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
        if not exp:
            raise HTTPException(404, "Run ml/ml_pipeline.py first.")
        runs = mlflow_client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"], max_results=20,
        )
        return {
            "experiment": {"id": exp.experiment_id, "name": exp.name,
                           "artifact_location": exp.artifact_location},
            "total_runs": len(runs),
            "model_runs": [
                {
                    "run_id":     r.info.run_id,
                    "model_name": r.data.params.get("model_name", "N/A"),
                    "status":     r.info.status,
                    "metrics": {
                        k: round(r.data.metrics.get(k, 0), 4)
                        for k in ["accuracy", "precision", "recall",
                                  "f1_score", "roc_auc",
                                  "cv_mean_accuracy", "specificity"]
                    },
                    "params": {"train_pct":  r.data.params.get("train_pct"),
                               "test_pct":   r.data.params.get("test_pct"),
                               "n_features": r.data.params.get("n_features")},
                }
                for r in runs
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/mlflow/models", tags=["MLflow — Model Details"],
         summary="App Detail 5: Registered models and versions")
def list_registered_models():
    """Model Registry — name, version, stage (None/Staging/Production), S3 path."""
    try:
        models = mlflow_client.search_registered_models()
        return {
            "total_models": len(models),
            "models": [
                {
                    "name":    m.name,
                    "updated": m.last_updated_timestamp,
                    "versions": [
                        {"version": v.version, "stage": v.current_stage,
                         "status": v.status,   "run_id": v.run_id,
                         "source": v.source}
                        for v in m.latest_versions
                    ],
                }
                for m in models
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/s3/artifacts", tags=["S3 — Storage"],
         summary="All files stored in S3 (data, plots, model artifacts)")
def list_s3_artifacts():
    """Lists all objects in the S3 bucket."""
    try:
        s3   = boto3.client("s3", region_name=AWS_REGION)
        resp = s3.list_objects_v2(Bucket=S3_BUCKET)
        objs = resp.get("Contents", [])
        return {
            "bucket":        S3_BUCKET,
            "total_objects": len(objs),
            "objects": [
                {"key":  o["Key"], "size_bytes": o["Size"],
                 "last_modified": o["LastModified"].isoformat()}
                for o in sorted(objs, key=lambda x: x["Key"])
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))
```

---

## 12. Run Everything — Step by Step

Open **4 terminal windows**. In every terminal, navigate to the project folder
and activate the virtual environment first:

```bash
cd mlpipeline
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows CMD
```

Also make sure all environment variables are set in **each terminal**
(AWS credentials + S3_BUCKET + PREFECT_API_KEY + PREFECT_API_URL).

---

### Terminal 1 — Start MLflow server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://$S3_BUCKET/mlflow-artifacts \
  --host 127.0.0.1 \
  --port 5000
```

**Windows CMD:**
```cmd
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://%S3_BUCKET%/mlflow-artifacts --host 127.0.0.1 --port 5000
```

Keep this running. Open **http://localhost:5000** — MLflow UI loads.

---

### Terminal 2 — Run the ML pipeline (one time only)

```bash
python ml/ml_pipeline.py
```

Wait until it finishes. Both models train and all metrics are logged.
Then open **http://localhost:5000** → Experiments → `Titanic_Survival_Prediction`
— both runs appear. **Take your MLflow screenshots now.**

---

### Terminal 3 — Register deployment then start Prefect worker

Run these two commands **in sequence** in the same terminal:

```bash
# A: Register the deployment and schedule on Prefect Cloud (run once)
python flows/data_pipeline.py

# Expected output:
# ✅ Deployment registered on Prefect Cloud!
#    Schedule : Every 3 minutes  (*/3 * * * *)
#    View at  : https://app.prefect.cloud → Deployments

# B: Start the worker — leave this running permanently
prefect worker start --pool "local-work-pool"

# Expected output:
# Worker 'ProcessWorker ...' started!
# Looking for scheduled flow runs...
```

Keep the worker running. Go to **https://app.prefect.cloud** → Deployments
— you will see `titanic-pipeline-every-3min` with an active cron schedule.

---

### Terminal 4 — Start FastAPI

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://localhost:8000/docs** — Swagger UI loads with all endpoints.

---

### Trigger a manual run immediately

Open a 5th terminal (venv active, env vars set):

```bash
prefect deployment run "titanic-data-pipeline/titanic-pipeline-every-3min"
```

Watch it execute live at **https://app.prefect.cloud → Flow Runs**.
The run will show all 3 tasks completing in sequence.
Click into any task to see the detailed logs.

---

### Download EDA plots from S3 for your report

After the first pipeline run completes:

```bash
# Mac/Linux
aws s3 cp s3://$S3_BUCKET/plots/ ./plots/ --recursive
ls plots/

# Windows CMD
aws s3 cp s3://%S3_BUCKET%/plots/ .\plots\ --recursive
dir plots\
```

You will get 5 PNG files — paste all of them into your Word doc.

---

### All services at a glance

| Terminal | What's running | URL |
|---|---|---|
| 1 | MLflow server | http://localhost:5000 |
| 2 | ML pipeline (done) | — |
| 3 | Prefect worker | https://app.prefect.cloud |
| 4 | FastAPI | http://localhost:8000/docs |

---

### Quick reference commands

```bash
# Trigger a manual Prefect run
prefect deployment run "titanic-data-pipeline/titanic-pipeline-every-3min"

# List deployments
prefect deployment ls

# List recent flow runs
prefect flow-run ls

# Check S3 contents
aws s3 ls s3://$S3_BUCKET/ --recursive

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/prefect/deployments
curl http://localhost:8000/mlflow/experiment
```

---

## 13. Refresh Expired Credentials

When credentials expire (~12 hours) you will see `ExpiredTokenException` errors.
**Do not redo any other steps** — just refresh the credentials.

1. Open **AWS CloudShell** in your browser
2. Run:
   ```bash
   aws sts get-session-token --duration-seconds 43200
   ```
3. Copy the new `AccessKeyId`, `SecretAccessKey`, `SessionToken`
4. In **every open terminal** on your local machine, paste:

   **Mac/Linux:**
   ```bash
   export AWS_ACCESS_KEY_ID="ASIA3...new key..."
   export AWS_SECRET_ACCESS_KEY="...new secret..."
   export AWS_SESSION_TOKEN="...new token..."
   ```

   **Windows CMD:**
   ```cmd
   set AWS_ACCESS_KEY_ID=ASIA3...new key...
   set AWS_SECRET_ACCESS_KEY=...new secret...
   set AWS_SESSION_TOKEN=...new token...
   ```

5. Restart the MLflow server and FastAPI (Ctrl+C then rerun the commands)
6. The Prefect worker reconnects automatically

---

## 14. Screenshots Guide

| # | Screenshot | Where |
|---|---|---|
| 1 | **Prefect Cloud — Deployments** showing `titanic-pipeline-every-3min` with `*/3 * * * *` | `app.prefect.cloud` → Deployments |
| 2 | **Prefect Cloud — Deployment detail** showing schedule, work pool, tags | Click the deployment name |
| 3 | **Prefect Cloud — Flow Runs list** multiple completed runs (green ticks) | Prefect Cloud → Flow Runs |
| 4 | **Prefect Cloud — Single flow run** showing 3 task boxes in sequence | Click any completed run |
| 5 | **Prefect Cloud — Ingestion task logs** rows, columns, dtypes | Click run → Ingestion task → Logs |
| 6 | **Prefect Cloud — Preprocessing task logs** imputation and normalization | Click Preprocessing task → Logs |
| 7 | **Prefect Cloud — EDA task logs** correlation matrix, feature importance | Click EDA task → Logs |
| 8 | **Prefect Cloud — Work Pools** showing `local-work-pool` READY | Prefect Cloud → Work Pools |
| 9 | **AWS S3 Bucket** showing all 3 folders | AWS Console → S3 |
| 10 | **S3 plots folder** showing 5 PNG files | S3 → plots/ |
| 11 | **All 5 EDA plots** (downloaded, pasted into doc) | Your local `plots/` folder |
| 12 | **MLflow Experiments page** | http://localhost:5000 |
| 13 | **MLflow Runs table** both models visible | MLflow → click experiment |
| 14 | **MLflow Compare view** both models side by side | Select both runs → Compare |
| 15 | **MLflow — Random Forest run** all 8+ metrics | Click RF run |
| 16 | **MLflow — Logistic Regression run** all 8+ metrics | Click LR run |
| 17 | **MLflow — Artifacts panel** model files in S3 | Any run → Artifacts tab |
| 18 | **FastAPI Swagger UI** all endpoints grouped by tag | http://localhost:8000/docs |
| 19 | **`GET /prefect/deployments`** JSON response in Swagger showing cron | Swagger → Try it out → Execute |
| 20 | **`GET /mlflow/experiment`** JSON showing both models and metrics | Swagger → Try it out → Execute |

---

## 15. Mark Breakdown

| Requirement | Marks | Delivered By |
|---|---|---|
| **1.1 Business Understanding** | ✅ | Titanic survival prediction — binary classification problem |
| **1.2 Data Ingestion** | ✅ | `ingest_data()` — S3 read, logs shape, dtypes, columns, missing count |
| **1.3 Data Preprocessing** | ✅ | `preprocess_data()` — summary stats, missing check, median imputation, MinMax normalization |
| **1.4 EDA** | ✅ | `perform_eda()` — correlation matrix, encoding, binning, feature importance, 5 plots → S3 |
| **1.5 DataOps** | ✅ | Prefect Cloud deployment, `*/3 * * * *` cron, real hosted cloud dashboard |
| **2.1 Model Preparation** | ✅ | Random Forest + Logistic Regression |
| **2.2 Model Training** | ✅ | Stratified 70/30 split + 5-fold cross-validation |
| **2.3 Model Evaluation** | ✅ | Accuracy, confusion matrix, classification report |
| **2.4 MLOps Monitoring** | ✅ | MLflow: 8 metrics logged (accuracy, precision, recall, F1, ROC-AUC, CV, specificity, TP/FP/FN/TN) |
| **3.1 Built-in APIs** | ✅ | Prefect Cloud REST API + MLflow tracking API |
| **3.2 Display Details** | ✅ | 6 FastAPI endpoints with complete JSON responses |
| **Submission doc + video** | ✅ | 20 screenshots + screen-record all UIs |
| **Cloud deployment** | ✅ | Prefect Cloud (hosted) + AWS S3 (storage) — both genuinely cloud |

---

## 16. Troubleshooting

**`ExpiredTokenException` on any AWS call:**
→ Go to Section 13 — refresh credentials from CloudShell. Takes 2 minutes.

**`NoCredentialsError` or `Unable to locate credentials`:**
```bash
# Check env vars are actually set
echo $AWS_ACCESS_KEY_ID        # Mac/Linux
echo %AWS_ACCESS_KEY_ID%       # Windows CMD
echo $env:AWS_ACCESS_KEY_ID    # Windows PowerShell
# If blank → paste the export/set commands again from Section 4
```

**`prefect cloud login` fails:**
```bash
echo $PREFECT_API_KEY    # Must not be blank
prefect cloud login --key "pnu_yourkey"
```

**MLflow can't write model artifacts to S3:**
```bash
# Test S3 write from your local terminal
aws s3 cp README.md s3://$S3_BUCKET/test.txt
# If this fails, credentials are expired — go to Section 13
```

**Prefect worker shows no runs after 3 minutes:**
```bash
# Check deployment schedule is active
prefect deployment ls
# If PAUSED:
prefect deployment resume "titanic-data-pipeline/titanic-pipeline-every-3min"
# Or trigger manually:
prefect deployment run "titanic-data-pipeline/titanic-pipeline-every-3min"
```

**FastAPI returns 503 on Prefect endpoints:**
```bash
echo $PREFECT_API_URL    # Must not be blank
# If blank — set it again and restart uvicorn
```

**`ModuleNotFoundError` for any package:**
```bash
# Make sure (venv) is showing in your terminal prompt
# If not: source venv/bin/activate  (or venv\Scripts\activate)
pip install <missing-package>
```

**Plots not appearing in S3 after EDA task runs:**
```bash
aws s3 ls s3://$S3_BUCKET/plots/
# If empty — check Prefect Cloud task logs for the error
# Most likely cause: S3 credentials expired mid-run
```

---

*CCZG506 — API-driven Cloud Native Solutions | Assignment I | BITS Pilani WILP*
