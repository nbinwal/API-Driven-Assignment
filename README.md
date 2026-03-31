# CCZG506 — Assignment I: Cloud-Native ML Pipeline
### Stack: Prefect Cloud · MLflow · FastAPI · AWS S3 · Local Execution

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [What Runs Where](#2-what-runs-where)
3. [Prerequisites](#3-prerequisites)
4. [Project Structure](#4-project-structure)
5. [AWS Setup](#5-aws-setup)
6. [Local Environment Setup](#6-local-environment-setup)
7. [Prefect Cloud Setup](#7-prefect-cloud-setup)
8. [Sub-Objective 1 — Data Pipeline](#8-sub-objective-1--data-pipeline)
9. [Sub-Objective 2 — ML Pipeline](#9-sub-objective-2--ml-pipeline)
10. [Sub-Objective 3 — API Access](#10-sub-objective-3--api-access)
11. [Run Everything — Step by Step](#11-run-everything--step-by-step)
12. [Screenshots Guide](#12-screenshots-guide)
13. [Mark Breakdown](#13-mark-breakdown)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AWS Cloud                                     │
│                                                                         │
│   ┌──────────────────────────────────┐   ┌───────────────────────────┐ │
│   │           Amazon S3              │   │      Prefect Cloud        │ │
│   │   mlpipeline-titanic-<name>/     │   │   app.prefect.cloud       │ │
│   │                                  │   │                           │ │
│   │   data/titanic.csv               │   │  ✓ Schedule (*/3 * * * *) │ │
│   │   data/titanic_processed.csv     │   │  ✓ Flow run dashboard     │ │
│   │   plots/*.png  (5 EDA charts)    │   │  ✓ Live task logs         │ │
│   │   mlflow-artifacts/  (models)    │   │  ✓ REST API               │ │
│   └──────────────────────────────────┘   └───────────────────────────┘ │
│              ▲  reads / writes                    ▲  schedules          │
└──────────────│───────────────────────────────────│─────────────────────┘
               │                                   │ polls every ~5s
┌──────────────│───────────────────────────────────│─────────────────────┐
│              │        Your Local Machine          │                     │
│              │                                   │                     │
│   ┌──────────▼──────────────────────────────────▼──────────────────┐  │
│   │                   Python Processes                              │  │
│   │                                                                 │  │
│   │   prefect worker  ──── executes flows on schedule              │  │
│   │   flows/data_pipeline.py                                        │  │
│   │       @task: ingest_data()        → reads from S3              │  │
│   │       @task: preprocess_data()    → cleans + saves to S3       │  │
│   │       @task: perform_eda()        → plots + saves to S3        │  │
│   │                                                                 │  │
│   │   mlflow server  (localhost:5000)                               │  │
│   │   ml/ml_pipeline.py                                             │  │
│   │       RandomForest + LogisticRegression                         │  │
│   │       metrics logged → MLflow  │  model artifacts → S3         │  │
│   │                                                                 │  │
│   │   fastapi (localhost:8000)                                      │  │
│   │   api/main.py                                                   │  │
│   │       /prefect/deployments  → calls Prefect Cloud REST API     │  │
│   │       /prefect/flow-runs    → calls Prefect Cloud REST API     │  │
│   │       /mlflow/experiment    → calls local MLflow               │  │
│   │       /mlflow/models        → calls local MLflow               │  │
│   └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. What Runs Where

| Component | Where | Why it counts as cloud |
|---|---|---|
| **Prefect Cloud dashboard** | Prefect's hosted servers | Real cloud SaaS — `app.prefect.cloud` |
| **Prefect schedule** | Prefect Cloud | Hosted externally, not on your machine |
| **Flow run logs** | Prefect Cloud | Streamed from local worker, stored in cloud |
| **Prefect REST API** | Prefect Cloud | Used by FastAPI for Sub-Objective 3 |
| **S3 — data & plots** | AWS S3 | Fully cloud storage |
| **S3 — model artifacts** | AWS S3 | MLflow artifacts stored in S3 |
| **Prefect worker** | Local machine | Executes flows, reports to Prefect Cloud |
| **MLflow server** | Local machine | UI + tracking DB on localhost |
| **FastAPI** | Local machine | Calls cloud APIs (Prefect + S3) |

> **Note on Virtual Lab:** The BITS AWS account has an SCP (Service Control Policy) that explicitly blocks `ec2:RunInstances`. This is an organization-level restriction that cannot be overridden. All cloud components (Prefect Cloud, S3) remain fully in the cloud. The assignment requirement for a "cloud dashboard" is satisfied by Prefect Cloud.

---

## 3. Prerequisites

### Accounts needed
- **AWS Account** — your BITS Virtual Lab account (already have it)
- **Prefect Cloud** — free account at https://app.prefect.cloud (sign up now)
- **Kaggle** — to download the Titanic dataset

### Software needed on your machine
- Python 3.9 or higher — check with `python --version`
- pip — check with `pip --version`
- AWS CLI — install from https://aws.amazon.com/cli/

Check Python version:
```bash
python --version    # Need 3.9+
# If you have python3 instead of python:
python3 --version
```

---

## 4. Project Structure

```
mlpipeline/
├── flows/
│   └── data_pipeline.py       # Prefect @flow — 3 @tasks + deployment registration
├── ml/
│   └── ml_pipeline.py         # MLflow training pipeline
├── api/
│   └── main.py                # FastAPI application
├── data/
│   └── titanic.csv            # Downloaded from Kaggle (local copy)
├── requirements.txt
└── README.md
```

---

## 5. AWS Setup

### Step 1 — Configure AWS CLI

```bash
aws configure
```

Enter these when prompted:
```
AWS Access Key ID:     <from Virtual Lab — see below>
AWS Secret Access Key: <from Virtual Lab — see below>
Default region name:   ap-south-1
Default output format: json
```

**To find your credentials:**
1. AWS Console → top right corner → your username → **Security Credentials**
2. **Access Keys** → **Create Access Key**
3. Copy both the key ID and secret

Verify it works:
```bash
aws sts get-caller-identity
# Should return your account ID and role ARN
```

### Step 2 — Create S3 Bucket

```bash
# Replace <yourname> with something unique e.g. mlpipeline-titanic-john2024
aws s3 mb s3://mlpipeline-titanic-<yourname> --region ap-south-1

# Create the three folders inside
aws s3api put-object --bucket mlpipeline-titanic-<yourname> --key data/
aws s3api put-object --bucket mlpipeline-titanic-<yourname> --key plots/
aws s3api put-object --bucket mlpipeline-titanic-<yourname> --key mlflow-artifacts/

# Verify
aws s3 ls s3://mlpipeline-titanic-<yourname>/
```

> If `aws s3 mb` is also restricted, create the bucket manually:
> AWS Console → S3 → Create Bucket → name it → region ap-south-1 → Create.
> Then create folders `data/`, `plots/`, `mlflow-artifacts/` using the **Create folder** button.

### Step 3 — Upload Titanic dataset

1. Download `titanic.csv` from https://www.kaggle.com/datasets/yasserh/titanic-dataset
2. Place it in your `mlpipeline/data/` folder
3. Upload to S3:

```bash
# Set your bucket name as a variable — use this throughout
export S3_BUCKET="mlpipeline-titanic-<yourname>"    # Mac/Linux
# On Windows CMD:  set S3_BUCKET=mlpipeline-titanic-<yourname>
# On Windows PS:   $env:S3_BUCKET="mlpipeline-titanic-<yourname>"

aws s3 cp data/titanic.csv s3://$S3_BUCKET/data/titanic.csv

# Verify
aws s3 ls s3://$S3_BUCKET/data/
# Should show: titanic.csv
```

---

## 6. Local Environment Setup

### Step 4 — Create project folder and virtual environment

```bash
# Create and enter project folder
mkdir mlpipeline
cd mlpipeline
mkdir flows ml api data plots

# Create virtual environment
python -m venv venv        # Windows
# OR
python3 -m venv venv       # Mac/Linux

# Activate virtual environment
# Windows CMD:
venv\Scripts\activate
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

# Your terminal prompt should now show (venv)
```

### Step 5 — Install all dependencies

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

Verify key installs:
```bash
python -c "import prefect; print('prefect', prefect.__version__)"
python -c "import mlflow; print('mlflow', mlflow.__version__)"
python -c "import pandas; print('pandas', pandas.__version__)"
```

### Step 6 — Set environment variables

**Mac/Linux** — add to `~/.bashrc` or `~/.zshrc` then run `source ~/.bashrc`:
```bash
export S3_BUCKET="mlpipeline-titanic-<yourname>"
export AWS_DEFAULT_REGION="ap-south-1"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export PREFECT_API_KEY=""      # Fill in after Step 8
export PREFECT_API_URL=""      # Fill in after Step 8
```

**Windows CMD** — run each line, then restart terminal:
```cmd
setx S3_BUCKET "mlpipeline-titanic-<yourname>"
setx AWS_DEFAULT_REGION "ap-south-1"
setx MLFLOW_TRACKING_URI "http://localhost:5000"
setx PREFECT_API_KEY ""
setx PREFECT_API_URL ""
```

**Windows PowerShell** — add to your PowerShell profile:
```powershell
$env:S3_BUCKET = "mlpipeline-titanic-<yourname>"
$env:AWS_DEFAULT_REGION = "ap-south-1"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PREFECT_API_KEY = ""
$env:PREFECT_API_URL = ""
```

---

## 7. Prefect Cloud Setup

### Step 7 — Create account and workspace

1. Go to **https://app.prefect.cloud** → Sign Up (free, no credit card)
2. Create a **Workspace** → name it `mlpipeline`
3. Go to **Settings** (bottom left) → **API Keys** → **+ Add API Key**
   - Name: `local-agent`
   - Expiry: leave as never
   - Click **Create** → **copy the key immediately** (shown once only)

### Step 8 — Get your Prefect API URL

1. In Prefect Cloud → **Settings** → **API Keys**
2. Your workspace API URL is shown on this page. It looks like:
   ```
   https://api.prefect.cloud/api/accounts/abc12345-xxxx-xxxx-xxxx-xxxxxxxxxxxx/workspaces/def67890-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   ```
3. Copy this full URL

Now go back and fill in the environment variables from Step 6:
```bash
export PREFECT_API_KEY="pnu_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export PREFECT_API_URL="https://api.prefect.cloud/api/accounts/YOUR_ACCOUNT_ID/workspaces/YOUR_WORKSPACE_ID"
```

### Step 9 — Connect your machine to Prefect Cloud

```bash
# Make sure venv is active
source venv/bin/activate    # Mac/Linux
# venv\Scripts\activate      # Windows

prefect cloud login --key $PREFECT_API_KEY
# When prompted: select your mlpipeline workspace

# Verify connection
prefect cloud workspace ls
# Should show your workspace marked as (active)
```

### Step 10 — Create Work Pool

```bash
prefect work-pool create "local-work-pool" --type process

# Verify
prefect work-pool ls
# Should show local-work-pool with status READY
```

Also verify in Prefect Cloud UI: **Work Pools** → `local-work-pool` should appear.

---

## 8. Sub-Objective 1 — Data Pipeline

Create the file `flows/data_pipeline.py` — copy this entire block:

```python
"""
CCZG506 Assignment I — Sub-Objective 1
Prefect Flow: Data Ingestion → Preprocessing → EDA
Scheduled every 3 minutes via Prefect Cloud.
All data and plots stored on AWS S3.
Logs stream live to app.prefect.cloud dashboard.
"""

from prefect import flow, task, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — required for server/headless environments
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
    logger.info(f"Total missing values across all columns: {total_missing}")
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
    description="Display stats, check/impute missing values, normalize, save to S3.",
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
    logger.info("--- Summary Statistics (describe) ---")
    logger.info(f"\n{df.describe().round(4).to_string()}")

    # ── 2. Data Types ─────────────────────────────────────────────────────────
    logger.info("--- Data Types ---")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col:<15} : {dtype}")

    # ── 3. Missing Values Before Imputation ───────────────────────────────────
    missing = df.isnull().sum()
    logger.info("--- Missing Values (Before Imputation) ---")
    for col, cnt in missing[missing > 0].items():
        pct = (cnt / len(df)) * 100
        logger.info(f"  {col:<15} : {cnt} missing  ({pct:.1f}% of rows)")

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

    # ── 6. Normalize Numeric Columns (MinMaxScaler → 0 to 1) ─────────────────
    cols_to_scale = ["Age", "Fare", "SibSp", "Parch"]
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    logger.info(f"  Normalized columns (MinMaxScaler, range 0–1): {cols_to_scale}")
    logger.info(f"  Post-normalization stats:\n{df[cols_to_scale].describe().round(4).to_string()}")

    # ── 7. Save processed data to S3 ──────────────────────────────────────────
    write_csv_to_s3(df, S3_PROC_KEY)
    logger.info(f"  Processed dataset saved → s3://{S3_BUCKET}/{S3_PROC_KEY}")

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

    # ── 2. Correlation with Target ────────────────────────────────────────────
    target_corr = corr["Survived"].abs().sort_values(ascending=False)
    logger.info("--- Absolute Correlation with Target (Survived) ---")
    for feat, val in target_corr.items():
        bar = "█" * int(val * 30)
        logger.info(f"  {feat:<25} : {val:.4f}  {bar}")

    # ── 3. Encoding Categorical Features ─────────────────────────────────────
    df["Sex_encoded"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked_encoded"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(-1).astype(int)
    logger.info("Encoding complete:")
    logger.info("  Sex      → Sex_encoded      {male: 0, female: 1}")
    logger.info("  Embarked → Embarked_encoded  {S: 0, C: 1, Q: 2}")

    # ── 4. Binning Age ────────────────────────────────────────────────────────
    df["Age_bin"] = pd.cut(
        df["Age"], bins=5,
        labels=["Very Young", "Young", "Middle-aged", "Senior", "Elderly"]
    )
    logger.info("--- Age Binning Distribution ---")
    for label, count in df["Age_bin"].value_counts().sort_index().items():
        logger.info(f"  {str(label):<15} : {count} passengers")

    # ── 5. Feature Importance (correlation with target) ───────────────────────
    numeric_df2 = df.select_dtypes(include=[np.number])
    feature_imp = (
        numeric_df2.corr()["Survived"].abs()
        .drop("Survived")
        .sort_values(ascending=False)
    )
    logger.info("--- Feature Importance (|Pearson r| with Survived) ---")
    for feat, score in feature_imp.items():
        logger.info(f"  {feat:<25} : {score:.4f}")

    # ── 6. Summary Stats ──────────────────────────────────────────────────────
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
    axes[1].set_title("Passenger Class Distribution",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Passenger Class")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(["1st Class", "2nd Class", "3rd Class"], rotation=0)
    fig.suptitle("Univariate Analysis — Titanic Dataset",
                 fontsize=15, fontweight="bold")
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
    axes[1].set_title("Survival Rate by Passenger Class",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Passenger Class")
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars, surv_class.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5, f"{val:.1f}%", ha="center", fontsize=11
        )
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

    # ── Plot 5: Feature Importance Bar Chart ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    feature_imp.plot(kind="bar", ax=ax, color="#3498db",
                     edgecolor="black", width=0.6)
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

    logger.info(f"All {len(plots_saved)} EDA plots uploaded to S3 successfully.")
    logger.info("TASK 1.4: EDA — COMPLETED SUCCESSFULLY")
    return {
        "plots_saved":        plots_saved,
        "survival_rate_pct":  round(survival_rate, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FLOW — Ties all 3 tasks together (Sub-Objective 1.5: DataOps)
# ─────────────────────────────────────────────────────────────────────────────
@flow(
    name="titanic-data-pipeline",
    description=(
        "End-to-end Titanic data pipeline: ingestion → preprocessing → EDA. "
        "Scheduled every 3 minutes via Prefect Cloud. Artifacts on S3."
    ),
    version="1.0",
    log_prints=True,
)
def data_pipeline():
    """
    Orchestrates all 3 tasks in sequence.
    Each task's return value is passed to the next so Prefect Cloud
    can display data lineage and dependency graph.
    """
    ingestion_stats     = ingest_data()
    preprocessing_stats = preprocess_data(ingestion_stats)
    eda_stats           = perform_eda(preprocessing_stats)
    return {
        "ingestion":     ingestion_stats,
        "preprocessing": preprocessing_stats,
        "eda":           eda_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEPLOYMENT — Run this script once to register the schedule on Prefect Cloud
# python flows/data_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=data_pipeline,
        name="titanic-pipeline-every-3min",
        work_pool_name="local-work-pool",      # runs on your local machine
        schedule=CronSchedule(
            cron="*/3 * * * *",                # Every 3 minutes ✅
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

## 9. Sub-Objective 2 — ML Pipeline

Create the file `ml/ml_pipeline.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 2
MLflow Training Pipeline.
Algorithms: Random Forest + Logistic Regression
Metrics logged to MLflow. Model artifacts stored on S3.
UI available at http://localhost:5000
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MLPipeline")

# ─── MLflow Setup ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
os.environ["MLFLOW_ARTIFACT_ROOT"] = f"s3://{S3_BUCKET}/mlflow-artifacts"
mlflow.set_experiment(EXPERIMENT)


# ─────────────────────────────────────────────────────────────────────────────
# 2.1 — MODEL PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare_data():
    logger.info("=" * 60)
    logger.info("STEP 2.1: MODEL PREPARATION")
    logger.info("=" * 60)

    # Load from S3
    s3  = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY)
    df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
    logger.info(f"Loaded {df.shape[0]} rows from s3://{S3_BUCKET}/{S3_DATA_KEY}")

    # Preprocessing
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"], inplace=True)

    # Encode categorical
    le = LabelEncoder()
    df["Sex"]      = le.fit_transform(df["Sex"])       # male=1, female=0
    df["Embarked"] = le.fit_transform(df["Embarked"])  # S/C/Q → integers

    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[feature_cols]
    y = df["Survived"]

    logger.info(f"Features      : {feature_cols}")
    logger.info(f"Class balance : {y.value_counts().to_dict()}")
    logger.info(f"Survival rate : {y.mean() * 100:.1f}%")
    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 + 2.3 + 2.4 — TRAIN, EVALUATE, LOG TO MLFLOW
# ─────────────────────────────────────────────────────────────────────────────
def train_and_log(model, model_name, X_train, X_test,
                  y_train, y_test, feature_cols):
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name}")
    logger.info(f"{'='*60}")

    with mlflow.start_run(run_name=model_name) as run:

        # ── Log hyperparameters ───────────────────────────────────────────────
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

        # ── 2.2 Train ─────────────────────────────────────────────────────────
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # ── 2.3 + 2.4 Compute and log metrics ────────────────────────────────
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

        # Log all metrics (satisfies 4+ metric requirement) ✅
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

        # Save model artifact to S3
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        # Print results
        logger.info(f"  Accuracy          : {accuracy:.4f}  ({accuracy*100:.2f}%)")
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
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1,
            "roc_auc":   roc_auc,
            "run_id":    run.info.run_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("\n" + "=" * 60)
    logger.info("CCZG506 — ML PIPELINE STARTING")
    logger.info("=" * 60)

    X, y, feature_cols = load_and_prepare_data()

    # 2.2 — Stratified 70% train / 30% test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    logger.info(f"\nData Split → Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Scale features for Logistic Regression
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Algorithm 1: Random Forest ────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    _, results["RandomForest"] = train_and_log(
        rf, "RandomForest",
        X_train, X_test, y_train, y_test, feature_cols
    )

    # ── Algorithm 2: Logistic Regression ──────────────────────────────────────
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    _, results["LogisticRegression"] = train_and_log(
        lr, "LogisticRegression",
        X_train_sc, X_test_sc, y_train, y_test, feature_cols
    )

    # ── Comparison Summary ────────────────────────────────────────────────────
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
    logger.info(f"View results at    : http://localhost:5000")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 10. Sub-Objective 3 — API Access

Create the file `api/main.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 3
FastAPI: exposes Prefect Cloud pipeline details + MLflow model details.
Satisfies 3.1 (Built-in APIs) and 3.2 (Display Application Details).
Swagger UI at http://localhost:8000/docs
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
        "REST APIs exposing Prefect Cloud pipeline details and "
        "MLflow model deployment details.\n\n"
        "Assignment I — BITS Pilani WILP  |  CCZG506"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

def _ph():
    """Prefect Cloud auth headers."""
    return {
        "Authorization": f"Bearer {PREFECT_API_KEY}",
        "Content-Type":  "application/json",
    }


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"],
         summary="Check status of all connected services")
def health_check():
    """Returns status of API, MLflow, Prefect Cloud, and S3."""
    status = {
        "api":           "ok",
        "mlflow":        "unknown",
        "prefect_cloud": "unknown",
        "s3":            "unknown",
    }
    try:
        mlflow_client.search_experiments()
        status["mlflow"] = "ok"
    except Exception:
        status["mlflow"] = "unreachable — is MLflow server running on port 5000?"

    try:
        r = requests.get(
            f"{PREFECT_API_URL}/health",
            headers=_ph(), timeout=5
        )
        status["prefect_cloud"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        status["prefect_cloud"] = "unreachable — check PREFECT_API_URL env var"

    try:
        boto3.client("s3", region_name=AWS_REGION).head_bucket(Bucket=S3_BUCKET)
        status["s3"] = "ok"
    except Exception:
        status["s3"] = "unreachable — check AWS credentials"

    return status


# ─────────────────────────────────────────────────────────────────────────────
# 3.1 + 3.2 — PREFECT CLOUD: Flow & Deployment Details
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/prefect/flows",
    tags=["Prefect Cloud — Flow Details"],
    summary="App Detail 1: All registered Prefect flows",
)
def list_flows():
    """
    Retrieves all flows registered on Prefect Cloud workspace.
    Shows flow name, ID, creation time, and tags.
    """
    if not PREFECT_API_URL:
        raise HTTPException(503, "PREFECT_API_URL env var not set.")
    try:
        r = requests.post(
            f"{PREFECT_API_URL}/flows/filter",
            headers=_ph(),
            json={"limit": 25},
            timeout=10,
        )
        r.raise_for_status()
        flows = r.json()
        return {
            "total_flows": len(flows),
            "flows": [
                {
                    "id":      f.get("id"),
                    "name":    f.get("name"),
                    "created": f.get("created"),
                    "updated": f.get("updated"),
                    "tags":    f.get("tags", []),
                }
                for f in flows
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(
    "/prefect/deployments",
    tags=["Prefect Cloud — Flow Details"],
    summary="App Detail 2: Deployments with schedule and work pool info",
)
def list_deployments():
    """
    Retrieves all deployments from Prefect Cloud.
    Shows schedule (cron = */3 * * * *), work pool, and active status.
    This directly satisfies 'flow/deployment details via Built-in APIs'.
    """
    if not PREFECT_API_URL:
        raise HTTPException(503, "PREFECT_API_URL env var not set.")
    try:
        r = requests.post(
            f"{PREFECT_API_URL}/deployments/filter",
            headers=_ph(),
            json={"limit": 25},
            timeout=10,
        )
        r.raise_for_status()
        deployments = r.json()
        return {
            "total_deployments": len(deployments),
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
                for d in deployments
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(
    "/prefect/flow-runs",
    tags=["Prefect Cloud — Flow Details"],
    summary="App Detail 3: Recent flow run history",
)
def get_flow_runs(limit: int = 10):
    """
    Retrieves recent pipeline execution history from Prefect Cloud.
    Shows state (Completed/Failed/Running), start time, and duration.
    """
    if not PREFECT_API_URL:
        raise HTTPException(503, "PREFECT_API_URL env var not set.")
    try:
        r = requests.post(
            f"{PREFECT_API_URL}/flow_runs/filter",
            headers=_ph(),
            json={"limit": limit, "sort": "START_TIME_DESC"},
            timeout=10,
        )
        r.raise_for_status()
        runs = r.json()
        return {
            "total_runs": len(runs),
            "runs": [
                {
                    "id":             run.get("id"),
                    "name":           run.get("name"),
                    "state": {
                        "type": run.get("state", {}).get("type"),
                        "name": run.get("state", {}).get("name"),
                    },
                    "start_time":     run.get("start_time"),
                    "end_time":       run.get("end_time"),
                    "total_run_time": run.get("total_run_time"),
                    "tags":           run.get("tags", []),
                }
                for run in runs
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3.1 + 3.2 — MLFLOW: Model Deployment Details
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/mlflow/experiment",
    tags=["MLflow — Model Details"],
    summary="App Detail 4: ML experiment with all model metrics",
)
def get_experiment():
    """
    Retrieves the ML experiment from MLflow showing both models
    and all logged metrics (accuracy, precision, recall, F1, ROC-AUC etc.)
    """
    try:
        exp = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
        if not exp:
            raise HTTPException(
                404,
                f"Experiment '{EXPERIMENT_NAME}' not found. "
                "Run ml/ml_pipeline.py first."
            )
        runs = mlflow_client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
        return {
            "experiment": {
                "id":                exp.experiment_id,
                "name":              exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage":   exp.lifecycle_stage,
            },
            "total_runs": len(runs),
            "model_runs": [
                {
                    "run_id":     r.info.run_id,
                    "model_name": r.data.params.get("model_name", "N/A"),
                    "status":     r.info.status,
                    "metrics": {
                        k: round(r.data.metrics.get(k, 0), 4)
                        for k in [
                            "accuracy", "precision", "recall",
                            "f1_score", "roc_auc",
                            "cv_mean_accuracy", "specificity",
                        ]
                    },
                    "params": {
                        "train_pct":  r.data.params.get("train_pct"),
                        "test_pct":   r.data.params.get("test_pct"),
                        "features":   r.data.params.get("features"),
                        "n_features": r.data.params.get("n_features"),
                    },
                }
                for r in runs
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(
    "/mlflow/models",
    tags=["MLflow — Model Details"],
    summary="App Detail 5: Registered models and deployment versions",
)
def list_registered_models():
    """
    Lists models from MLflow Model Registry.
    Shows name, version, stage (None/Staging/Production), and S3 artifact path.
    """
    try:
        models = mlflow_client.search_registered_models()
        return {
            "total_models": len(models),
            "models": [
                {
                    "name":    m.name,
                    "updated": m.last_updated_timestamp,
                    "versions": [
                        {
                            "version": v.version,
                            "stage":   v.current_stage,
                            "status":  v.status,
                            "run_id":  v.run_id,
                            "source":  v.source,
                        }
                        for v in m.latest_versions
                    ],
                }
                for m in models
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(
    "/s3/artifacts",
    tags=["S3 — Storage"],
    summary="All files in S3 bucket (data, plots, model artifacts)",
)
def list_s3_artifacts():
    """Lists everything stored in S3 — dataset, processed data, plots, models."""
    try:
        s3   = boto3.client("s3", region_name=AWS_REGION)
        resp = s3.list_objects_v2(Bucket=S3_BUCKET)
        objs = resp.get("Contents", [])
        return {
            "bucket":        S3_BUCKET,
            "total_objects": len(objs),
            "objects": [
                {
                    "key":           o["Key"],
                    "size_bytes":    o["Size"],
                    "last_modified": o["LastModified"].isoformat(),
                }
                for o in sorted(objs, key=lambda x: x["Key"])
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))
```

---

## 11. Run Everything — Step by Step

You need **4 terminal windows** open simultaneously. Make sure your virtual environment is active in every terminal.

```bash
# Activate venv in every new terminal you open:
# Mac/Linux:  source venv/bin/activate
# Windows:    venv\Scripts\activate
```

---

### Terminal 1 — Start MLflow server

```bash
cd mlpipeline
source venv/bin/activate    # (or venv\Scripts\activate on Windows)

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://$S3_BUCKET/mlflow-artifacts \
  --host 127.0.0.1 \
  --port 5000
```

Leave this running. Open http://localhost:5000 in your browser — you should see the MLflow UI.

---

### Terminal 2 — Run the ML pipeline (one time)

```bash
cd mlpipeline
source venv/bin/activate

python ml/ml_pipeline.py
```

Wait for it to finish. You will see both models train with all metrics printed.
Then go to http://localhost:5000 → **Experiments** → `Titanic_Survival_Prediction` — both model runs will appear.

---

### Terminal 3 — Register deployment + start Prefect worker

Run these two commands in order in the same terminal:

```bash
cd mlpipeline
source venv/bin/activate

# Step A: Register the deployment and schedule on Prefect Cloud
python flows/data_pipeline.py
# Expected output:
# ✅ Deployment registered on Prefect Cloud!
#    ID       : xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#    Schedule : Every 3 minutes  (*/3 * * * *)
#    Pool     : local-work-pool

# Step B: Start the worker (keep this running — it executes the scheduled flows)
prefect worker start --pool "local-work-pool"
# Expected output:
# Worker 'ProcessWorker ...' started!
# Looking for scheduled flow runs...
```

Leave the worker running. Every 3 minutes it will pick up a scheduled run from Prefect Cloud and execute the flow.

Go to **https://app.prefect.cloud** → **Deployments** — you should see `titanic-pipeline-every-3min` with schedule active.

---

### Terminal 4 — Start FastAPI

```bash
cd mlpipeline
source venv/bin/activate

uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open http://localhost:8000/docs in your browser — the Swagger UI shows all API endpoints.

---

### Trigger a manual run immediately (don't wait 3 minutes)

Open a 5th terminal:
```bash
cd mlpipeline
source venv/bin/activate

prefect deployment run "titanic-data-pipeline/titanic-pipeline-every-3min"
```

Watch it run live at https://app.prefect.cloud → **Flow Runs**.

---

### All 4 services running — summary

| Terminal | Command | URL |
|---|---|---|
| 1 | MLflow server | http://localhost:5000 |
| 2 | ML pipeline (run once) | — |
| 3 | Prefect worker | https://app.prefect.cloud |
| 4 | FastAPI | http://localhost:8000/docs |

---

### Download EDA plots from S3

After the first pipeline run completes, download the plots to include in your report:

```bash
# Mac/Linux
aws s3 cp s3://$S3_BUCKET/plots/ ./plots/ --recursive
ls plots/
# 1_correlation_heatmap.png
# 2_univariate_analysis.png
# 3_bivariate_sex_pclass.png
# 4_age_distribution_survival.png
# 5_feature_importance.png

# Windows (PowerShell)
aws s3 cp s3://$env:S3_BUCKET/plots/ .\plots\ --recursive
```

---

### Useful commands

```bash
# ── Prefect CLI ───────────────────────────────────────────────────────────────
prefect deployment ls                  # List all deployments
prefect flow-run ls                    # List recent flow runs
prefect work-pool ls                   # List work pools

# Manually trigger a run
prefect deployment run "titanic-data-pipeline/titanic-pipeline-every-3min"

# ── S3 ────────────────────────────────────────────────────────────────────────
aws s3 ls s3://$S3_BUCKET/ --recursive          # List all files
aws s3 ls s3://$S3_BUCKET/plots/                # List EDA plots
aws s3 ls s3://$S3_BUCKET/mlflow-artifacts/     # List model artifacts

# ── Check env vars are set correctly ─────────────────────────────────────────
echo $S3_BUCKET
echo $PREFECT_API_KEY
echo $MLFLOW_TRACKING_URI
```

---

## 12. Screenshots Guide

Take these 20 screenshots for your submission document.

| # | Screenshot | Where to capture it |
|---|---|---|
| 1 | **Prefect Cloud — Deployments page** showing `titanic-pipeline-every-3min` with `*/3 * * * *` | `app.prefect.cloud` → Deployments |
| 2 | **Prefect Cloud — Deployment detail** showing schedule, work pool, tags | Click the deployment name |
| 3 | **Prefect Cloud — Flow Runs list** showing multiple completed runs (green) | Prefect Cloud → Flow Runs |
| 4 | **Prefect Cloud — Single flow run** showing 3 task bubbles in sequence | Click any completed run |
| 5 | **Prefect Cloud — Ingestion task logs** showing rows, columns, dtypes | Click run → click Ingestion task → Logs tab |
| 6 | **Prefect Cloud — Preprocessing task logs** showing imputation and normalization | Click Preprocessing task → Logs |
| 7 | **Prefect Cloud — EDA task logs** showing correlation matrix and feature importance | Click EDA task → Logs |
| 8 | **Prefect Cloud — Work Pools** showing `local-work-pool` as READY | Prefect Cloud → Work Pools |
| 9 | **AWS S3 Bucket** showing `data/`, `plots/`, `mlflow-artifacts/` folders | AWS Console → S3 → your bucket |
| 10 | **S3 plots folder** showing all 5 PNG files | S3 → click plots/ folder |
| 11 | **All 5 EDA plots** (download from S3, paste all 5 into doc) | Downloaded to your `plots/` folder |
| 12 | **MLflow Experiments page** showing `Titanic_Survival_Prediction` | http://localhost:5000 |
| 13 | **MLflow Runs table** showing both model runs | MLflow → Experiments → click experiment |
| 14 | **MLflow Compare view** — both models side by side | Select both runs → Compare |
| 15 | **MLflow — Random Forest run detail** showing all 8+ metrics | Click RF run |
| 16 | **MLflow — Logistic Regression run detail** showing all 8+ metrics | Click LR run |
| 17 | **MLflow — Artifacts panel** showing model files saved to S3 | Any run → Artifacts tab |
| 18 | **FastAPI Swagger UI** (`/docs`) showing all endpoints grouped by tag | http://localhost:8000/docs |
| 19 | **`GET /prefect/deployments`** — executed in Swagger showing cron schedule in JSON response | Swagger → expand → Try it out → Execute |
| 20 | **`GET /mlflow/experiment`** — executed in Swagger showing both models and metrics | Swagger → expand → Try it out → Execute |

---

## 13. Mark Breakdown

| Requirement | Marks | Delivered By |
|---|---|---|
| **1.1 Business Understanding** | ✅ | Titanic survival prediction — binary classification |
| **1.2 Data Ingestion** | ✅ | `ingest_data()` @task — reads from S3, logs shape, dtypes, columns, missing count |
| **1.3 Data Preprocessing** | ✅ | `preprocess_data()` @task — summary stats, missing values, median imputation, MinMax normalization |
| **1.4 EDA** | ✅ | `perform_eda()` @task — correlation matrix, encoding, binning, feature importance, 5 plots → S3 |
| **1.5 DataOps** | ✅ | Prefect Cloud deployment with `*/3 * * * *` cron, real hosted cloud dashboard, live logs |
| **2.1 Model Preparation** | ✅ | Random Forest + Logistic Regression identified and configured |
| **2.2 Model Training** | ✅ | Stratified 70/30 split, 5-fold cross-validation |
| **2.3 Model Evaluation** | ✅ | Accuracy, confusion matrix, full classification report |
| **2.4 MLOps Monitoring** | ✅ | MLflow: 8 metrics (accuracy, precision, recall, F1, ROC-AUC, CV, specificity + TP/FP/FN/TN) |
| **3.1 Built-in APIs** | ✅ | Prefect Cloud REST API (flows, deployments, flow-runs) + MLflow tracking API |
| **3.2 Display Details** | ✅ | 6 FastAPI endpoints with complete JSON responses visible in Swagger |
| **Submission doc + video** | ✅ | 20 screenshots above + screen-record all 4 UIs |
| **Cloud deployment** | ✅ | Prefect Cloud (hosted), S3 (AWS) — cloud components verified |

---

## 14. Troubleshooting

**`prefect cloud login` fails:**
```bash
# Make sure PREFECT_API_KEY is set
echo $PREFECT_API_KEY    # Should print your key
# Try passing key directly
prefect cloud login --key "pnu_yourkey"
```

**S3 access denied / no credentials:**
```bash
# Verify credentials are configured
aws sts get-caller-identity
# If this fails, re-run: aws configure
```

**MLflow can't write to S3:**
```bash
# Test S3 write access
aws s3 cp README.md s3://$S3_BUCKET/test.txt
aws s3 rm s3://$S3_BUCKET/test.txt
# If this works, MLflow will too
```

**`ModuleNotFoundError` for any package:**
```bash
# Make sure venv is active (you should see (venv) in terminal prompt)
pip install --upgrade <package-name>
```

**Prefect worker shows no runs after 3 minutes:**
```bash
# Make sure the deployment schedule is active
prefect deployment ls
# If PAUSED, resume it:
prefect deployment resume "titanic-data-pipeline/titanic-pipeline-every-3min"

# Or just trigger manually
prefect deployment run "titanic-data-pipeline/titanic-pipeline-every-3min"
```

**FastAPI returns 503 for Prefect endpoints:**
```bash
# PREFECT_API_URL is not set — verify in terminal
echo $PREFECT_API_URL
# Should print the full URL. If blank, set it again and restart uvicorn.
```

**Plots not showing in S3 after EDA task completes:**
```bash
aws s3 ls s3://$S3_BUCKET/plots/
# If empty, check Prefect Cloud task logs for any Python errors
```

---

*CCZG506 — API-driven Cloud Native Solutions | Assignment I | BITS Pilani WILP*
