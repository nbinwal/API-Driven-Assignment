# CCZG506 — Assignment I: Cloud-Native ML Pipeline
### Stack: Prefect Cloud · MLflow · FastAPI · AWS EC2 · S3

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [AWS Infrastructure Setup](#aws-infrastructure-setup)
3. [Project Structure](#project-structure)
4. [EC2 Environment Setup](#ec2-environment-setup)
5. [Prefect Cloud Setup](#prefect-cloud-setup)
6. [Sub-Objective 1 — Data Pipeline (Prefect)](#sub-objective-1--data-pipeline)
7. [Sub-Objective 2 — ML Pipeline (MLflow)](#sub-objective-2--ml-pipeline)
8. [Sub-Objective 3 — API Access (FastAPI)](#sub-objective-3--api-access)
9. [Deploy & Run Everything](#deploy--run-everything)
10. [Screenshots Guide](#screenshots-guide)
11. [Mark Breakdown](#mark-breakdown)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          AWS Cloud (EC2)                             │
│                                                                      │
│   ┌─────────────┐    reads/writes  ┌──────────────────────────────┐ │
│   │  Amazon S3  │◄────────────────►│   Prefect Worker (EC2)       │ │
│   │             │                  │                              │ │
│   │  data/      │                  │  flows/data_pipeline.py      │ │
│   │  plots/     │                  │    @task: ingest             │ │
│   │  mlflow-    │                  │    @task: preprocess          │ │
│   │  artifacts/ │                  │    @task: eda                 │ │
│   └─────────────┘                  │                              │ │
│                                    │  ml/ml_pipeline.py           │ │
│                                    │    RandomForest + LogReg      │ │
│                                    │    → MLflow (port 5000)       │ │
│                                    │                              │ │
│                                    │  api/main.py                 │ │
│                                    │    FastAPI (port 8000)        │ │
│                                    └──────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
              ▲  schedules / logs / monitors
              │
┌─────────────▼────────────────┐
│      Prefect Cloud           │
│   app.prefect.cloud          │
│                              │
│  ✓ Flow run dashboard        │
│  ✓ Schedule (*/3 * * * *)    │
│  ✓ Live logs per task        │
│  ✓ REST API for details      │
│  ✓ Retries & alerting        │
└──────────────────────────────┘
```

**How it works:**
- Your code runs on **AWS EC2**
- **Prefect Cloud** is the brain — it holds the schedule, shows the cloud dashboard, streams live logs, and exposes a REST API
- The Prefect **worker** on EC2 polls Prefect Cloud every few seconds, picks up scheduled runs, and executes them locally
- Data, plots, and model artifacts are all stored in **AWS S3**
- No manual scheduler management — Prefect Cloud handles everything

---

## AWS Infrastructure Setup

### Step 1 — Create S3 Bucket

1. AWS Console → **S3** → **Create Bucket**
2. Bucket name: `mlpipeline-titanic-<yourname>` (must be globally unique)
3. Region: `ap-south-1` (Mumbai) or your preferred region
4. Block all public access: **ON**
5. Click **Create Bucket**
6. Open the bucket → **Create folder** — create these three:
   - `data/`
   - `plots/`
   - `mlflow-artifacts/`

### Step 2 — Create IAM Role for EC2

1. AWS Console → **IAM** → **Roles** → **Create Role**
2. Trusted entity: **AWS Service** → **EC2**
3. Attach these policies:
   - `AmazonS3FullAccess`
   - `CloudWatchLogsFullAccess`
4. Role name: `EC2-MLPipeline-Role`
5. **Create Role**

### Step 3 — Launch EC2 Instance

1. AWS Console → **EC2** → **Launch Instance**
2. Configuration:

| Setting | Value |
|---|---|
| AMI | Ubuntu Server 22.04 LTS (64-bit x86) |
| Instance type | `t3.medium` (2 vCPU, 4 GB RAM) |
| Key pair | Create new → `mlpipeline-key` → download `.pem` |
| IAM Instance Profile | `EC2-MLPipeline-Role` |
| Storage | 20 GB gp3 |

3. Security Group inbound rules:

| Port | Source | Purpose |
|---|---|---|
| 22 | My IP | SSH access |
| 5000 | 0.0.0.0/0 | MLflow UI |
| 8000 | 0.0.0.0/0 | FastAPI |

> Prefect Cloud dashboard is at `app.prefect.cloud` — no port needed on EC2.

4. **Launch Instance**

### Step 4 — Connect to EC2

```bash
# On your local machine
chmod 400 mlpipeline-key.pem
ssh -i mlpipeline-key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## Project Structure

```
~/mlpipeline/
├── flows/
│   └── data_pipeline.py       # Prefect @flow with 3 @tasks + deployment
├── ml/
│   └── ml_pipeline.py         # MLflow training pipeline
├── api/
│   └── main.py                # FastAPI — Prefect Cloud API + MLflow API
├── requirements.txt
└── README.md
```

---

## EC2 Environment Setup

### Step 5 — Install dependencies

```bash
# Update system
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv wget git screen

# Create project directory
mkdir -p ~/mlpipeline/{flows,ml,api,data,plots}
cd ~/mlpipeline

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all packages
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

### Step 6 — Upload Titanic dataset to S3

Download `titanic.csv` from https://www.kaggle.com/datasets/yasserh/titanic-dataset

```bash
# From your LOCAL machine — copy csv to EC2
scp -i mlpipeline-key.pem titanic.csv ubuntu@<EC2_PUBLIC_IP>:~/mlpipeline/data/

# On EC2 — push to S3 (replace bucket name first)
export S3_BUCKET="mlpipeline-titanic-<yourname>"
aws s3 cp ~/mlpipeline/data/titanic.csv s3://$S3_BUCKET/data/titanic.csv
aws s3 ls s3://$S3_BUCKET/data/    # verify upload
```

### Step 7 — Set environment variables

```bash
cat >> ~/.bashrc << 'EOF'

# ── ML Pipeline Config ────────────────────────────────────────────────────────
export S3_BUCKET="mlpipeline-titanic-<yourname>"    # CHANGE THIS
export AWS_DEFAULT_REGION="ap-south-1"              # CHANGE IF DIFFERENT
export MLFLOW_TRACKING_URI="http://localhost:5000"
export PROJECT_DIR=~/mlpipeline
export PREFECT_API_KEY=""         # Fill in Step 11
export PREFECT_API_URL=""         # Fill in Step 11
EOF

source ~/.bashrc
```

---

## Prefect Cloud Setup

### Step 8 — Create Prefect Cloud account

1. Go to **https://app.prefect.cloud** → Sign up for free (no credit card needed)
2. Create a **Workspace** — name it `mlpipeline`
3. Go to **Settings** → **API Keys** → **+ Add API Key**
   - Name: `ec2-agent`
   - Copy the key immediately — you cannot see it again

### Step 9 — Connect EC2 to Prefect Cloud

```bash
source ~/mlpipeline/venv/bin/activate

# Paste your API key here
prefect cloud login --key <YOUR_API_KEY>
# When prompted, select your workspace

# Verify connection
prefect cloud workspace ls
# Your workspace should show as (active)
```

### Step 10 — Create a Work Pool

```bash
# Create a process-based work pool
prefect work-pool create "ec2-work-pool" --type process

# Verify it appears
prefect work-pool ls
```

You will also see `ec2-work-pool` in Prefect Cloud UI under **Work Pools**.

---

## Sub-Objective 1 — Data Pipeline

Create `~/mlpipeline/flows/data_pipeline.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 1
Prefect Flow: Data Ingestion → Preprocessing → EDA
Scheduled every 3 minutes via Prefect Cloud.
Data stored on AWS S3. Logs stream to app.prefect.cloud dashboard.
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
    _s3().put_object(Bucket=S3_BUCKET, Key=key,
                     Body=buf.getvalue(), ContentType="image/png")

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

    logger.info(f"Source        : s3://{S3_BUCKET}/{S3_DATA_KEY}")
    logger.info(f"Rows          : {df.shape[0]}")
    logger.info(f"Columns       : {df.shape[1]}")
    logger.info(f"Column names  : {list(df.columns)}")
    logger.info(f"Memory usage  : {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    logger.info(f"Data Types:\n{df.dtypes.to_string()}")
    logger.info(f"First 3 rows:\n{df.head(3).to_string()}")

    total_missing = int(df.isnull().sum().sum())
    logger.info(f"Total missing values in dataset: {total_missing}")
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
    description="Clean, impute missing values, normalize, and save processed data to S3.",
    retries=2,
    retry_delay_seconds=30,
    tags=["preprocessing", "s3"],
)
def preprocess_data(ingestion_stats: dict) -> dict:
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("TASK 1.3: DATA PREPROCESSING — STARTED")
    logger.info(f"Input from ingestion task: {ingestion_stats}")
    logger.info("=" * 60)

    df = read_csv_from_s3(S3_DATA_KEY)

    # ── Summary Statistics ────────────────────────────────────────────────────
    logger.info("--- Summary Statistics ---")
    logger.info(f"\n{df.describe().round(4).to_string()}")

    # ── Data Types ────────────────────────────────────────────────────────────
    logger.info("--- Data Types ---")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col:<15} : {dtype}")

    # ── Missing Values Before Imputation ─────────────────────────────────────
    missing = df.isnull().sum()
    logger.info("--- Missing Values (Before Imputation) ---")
    for col, cnt in missing[missing > 0].items():
        pct = (cnt / len(df)) * 100
        logger.info(f"  {col:<15} : {cnt} missing ({pct:.1f}%)")

    # ── Impute Numeric Columns with Median ────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputed = {}
    for col in numeric_cols:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            imputed[col] = float(median_val)
            logger.info(f"  Imputed '{col}' ({n_null} nulls) → median = {median_val:.4f}")

    # ── Impute Categorical Columns ────────────────────────────────────────────
    if df["Embarked"].isnull().sum() > 0:
        mode_val = df["Embarked"].mode()[0]
        df["Embarked"].fillna(mode_val, inplace=True)
        logger.info(f"  Imputed 'Embarked' → mode = '{mode_val}'")

    if "Cabin" in df.columns:
        df["Cabin"].fillna("Unknown", inplace=True)
        logger.info("  Imputed 'Cabin' → 'Unknown'")

    remaining_nulls = int(df.isnull().sum().sum())
    logger.info(f"  Null values remaining after imputation: {remaining_nulls}")

    # ── Normalize Numeric Columns ─────────────────────────────────────────────
    cols_to_scale = ["Age", "Fare", "SibSp", "Parch"]
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    logger.info(f"  Normalized (0–1 scale): {cols_to_scale}")
    logger.info(f"  Post-normalization stats:\n{df[cols_to_scale].describe().round(4).to_string()}")

    # ── Save to S3 ────────────────────────────────────────────────────────────
    write_csv_to_s3(df, S3_PROC_KEY)
    logger.info(f"  Processed data saved → s3://{S3_BUCKET}/{S3_PROC_KEY}")

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
    logger.info(f"Input from preprocessing task: {preprocessing_stats}")
    logger.info("=" * 60)

    df = read_csv_from_s3(S3_PROC_KEY)
    numeric_df = df.select_dtypes(include=[np.number])

    # ── Correlation Matrix ────────────────────────────────────────────────────
    corr = numeric_df.corr()
    logger.info("--- Full Correlation Matrix ---")
    logger.info(f"\n{corr.round(3).to_string()}")

    # ── Correlation with Target ───────────────────────────────────────────────
    target_corr = corr["Survived"].abs().sort_values(ascending=False)
    logger.info("--- Correlation with Target (Survived) ---")
    for feat, val in target_corr.items():
        bar = "█" * int(val * 30)
        logger.info(f"  {feat:<25} : {val:.4f}  {bar}")

    # ── Encoding Categorical Features ────────────────────────────────────────
    df["Sex_encoded"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked_encoded"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(-1).astype(int)
    logger.info("Encoding: Sex → {male:0, female:1}, Embarked → {S:0, C:1, Q:2}")

    # ── Binning Age ───────────────────────────────────────────────────────────
    df["Age_bin"] = pd.cut(
        df["Age"], bins=5,
        labels=["Very Young", "Young", "Middle-aged", "Senior", "Elderly"]
    )
    logger.info("--- Age Binning Distribution ---")
    for label, count in df["Age_bin"].value_counts().sort_index().items():
        logger.info(f"  {str(label):<15} : {count} passengers")

    # ── Feature Importance ────────────────────────────────────────────────────
    numeric_df2  = df.select_dtypes(include=[np.number])
    feature_imp  = (
        numeric_df2.corr()["Survived"].abs()
        .drop("Survived").sort_values(ascending=False)
    )
    logger.info("--- Feature Importance (|Pearson r| with Survived) ---")
    for feat, score in feature_imp.items():
        logger.info(f"  {feat:<25} : {score:.4f}")

    # ── Survival Stats ────────────────────────────────────────────────────────
    survival_rate = df["Survived"].mean() * 100
    logger.info(f"Overall Survival Rate : {survival_rate:.1f}%")
    logger.info(f"By Sex   :\n{df.groupby('Sex')['Survived'].mean().mul(100).round(1).to_string()}%")
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
        axes[0].annotate(str(int(p.get_height())),
                         (p.get_x() + p.get_width() / 2, p.get_height() + 3),
                         ha="center", fontsize=11)

    df["Pclass"].value_counts().sort_index().plot(
        kind="bar", ax=axes[1],
        color=["#3498db", "#f39c12", "#9b59b6"], edgecolor="black", width=0.5
    )
    axes[1].set_title("Passenger Class Distribution", fontsize=13, fontweight="bold")
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
    bars = axes[1].bar(surv_class.index.astype(str), surv_class.values,
                       color=["#3498db", "#f39c12", "#9b59b6"],
                       edgecolor="black", width=0.5)
    axes[1].set_title("Survival Rate by Passenger Class",
                      fontsize=13, fontweight="bold")
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
    for val, color, label in [(0, "#e74c3c", "Did Not Survive"),
                               (1, "#2ecc71", "Survived")]:
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

    logger.info(f"All {len(plots_saved)} EDA plots uploaded to S3.")
    logger.info("TASK 1.4: EDA — COMPLETED SUCCESSFULLY")
    return {"plots_saved": plots_saved, "survival_rate_pct": round(survival_rate, 2)}

# ─────────────────────────────────────────────────────────────────────────────
# FLOW — Orchestrates all 3 tasks (Sub-Objective 1.5: DataOps)
# ─────────────────────────────────────────────────────────────────────────────
@flow(
    name="titanic-data-pipeline",
    description=(
        "End-to-end Titanic data pipeline: ingestion → preprocessing → EDA. "
        "Scheduled every 3 minutes via Prefect Cloud."
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
# DEPLOYMENT — Run this script to register the schedule on Prefect Cloud
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=data_pipeline,
        name="titanic-pipeline-every-3min",
        work_pool_name="ec2-work-pool",
        schedule=CronSchedule(
            cron="*/3 * * * *",        # Every 3 minutes ✅
            timezone="Asia/Kolkata",
        ),
        tags=["DataOps", "Titanic", "CCZG506"],
        description="Automated Titanic data pipeline. Runs every 3 minutes.",
    )
    deployment_id = deployment.apply()
    print(f"\nDeployment created!")
    print(f"ID       : {deployment_id}")
    print(f"Schedule : Every 3 minutes  (*/3 * * * *)")
    print(f"Pool     : ec2-work-pool")
    print(f"View at  : https://app.prefect.cloud  → Deployments")
```

---

## Sub-Objective 2 — ML Pipeline

Create `~/mlpipeline/ml/ml_pipeline.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 2
MLflow training pipeline.
Algorithms: Random Forest + Logistic Regression
Artifacts stored on S3. UI at http://<EC2_IP>:5000
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import boto3
import logging
import io
import os

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import (
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

# ─────────────────────────────────────────────────────────────────────────────
# 2.1 MODEL PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
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

    logger.info(f"Features     : {feature_cols}")
    logger.info(f"Class balance: {y.value_counts().to_dict()}")
    logger.info(f"Survival rate: {y.mean() * 100:.1f}%")
    return X, y, feature_cols

# ─────────────────────────────────────────────────────────────────────────────
# 2.2 + 2.3 + 2.4 — TRAIN, EVALUATE, LOG
# ─────────────────────────────────────────────────────────────────────────────
def train_and_log(model, model_name, X_train, X_test, y_train, y_test, feature_cols):
    logger.info(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")

    with mlflow.start_run(run_name=model_name) as run:

        # Log params
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

        # 2.2 Train
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 2.3 + 2.4 Compute all metrics
        accuracy    = accuracy_score(y_test, y_pred)
        precision   = precision_score(y_test, y_pred, zero_division=0)
        recall      = recall_score(y_test, y_pred, zero_division=0)
        f1          = f1_score(y_test, y_pred, zero_division=0)
        roc_auc     = roc_auc_score(y_test, y_pred_prob)
        cv_scores   = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_mean     = cv_scores.mean()
        cv_std      = cv_scores.std()
        cm          = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Log metrics to MLflow (satisfies 4+ metric requirement) ✅
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

        # Save model artifact to S3 via MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        logger.info(f"  Accuracy          : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        logger.info(f"  Precision         : {precision:.4f}")
        logger.info(f"  Recall            : {recall:.4f}")
        logger.info(f"  F1 Score          : {f1:.4f}")
        logger.info(f"  ROC-AUC           : {roc_auc:.4f}")
        logger.info(f"  CV (5-fold) Acc   : {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"  Specificity       : {specificity:.4f}")
        logger.info(f"\n  Confusion Matrix:\n"
                    f"                    Predicted No  Predicted Yes\n"
                    f"  Actual No           TN={tn:<6}   FP={fp}\n"
                    f"  Actual Yes          FN={fn:<6}   TP={tp}")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Did Not Survive','Survived'])}")
        logger.info(f"  MLflow Run ID: {run.info.run_id}")

        return run.info.run_id, {
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1_score": f1,
            "roc_auc": roc_auc, "run_id": run.info.run_id,
        }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("\n" + "="*60)
    logger.info("CCZG506 — ML PIPELINE STARTING")
    logger.info("="*60)

    X, y, feature_cols = load_and_prepare_data()

    # 2.2 — 70/30 stratified split
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

    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Metric':<22} {'Random Forest':>16} {'Logistic Reg':>16}")
    logger.info("-" * 56)
    for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        logger.info(
            f"  {m:<20} "
            f"{results['RandomForest'][m]:>16.4f} "
            f"{results['LogisticRegression'][m]:>16.4f}"
        )
    best = max(results, key=lambda k: results[k]["f1_score"])
    logger.info(f"\nBest Model (by F1): {best}")
    logger.info(f"MLflow UI         : http://localhost:5000")

if __name__ == "__main__":
    main()
```

---

## Sub-Objective 3 — API Access

Create `~/mlpipeline/api/main.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 3
FastAPI: exposes Prefect Cloud pipeline details + MLflow model details.
Satisfies 3.1 (Built-in APIs) and 3.2 (Display Application Details).
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
        "REST APIs exposing Prefect Cloud pipeline details and MLflow model "
        "deployment details. Assignment I — BITS Pilani WILP."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

def _prefect_headers():
    return {"Authorization": f"Bearer {PREFECT_API_KEY}",
            "Content-Type":  "application/json"}

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    status = {"api": "ok", "mlflow": "unknown",
              "prefect_cloud": "unknown", "s3": "unknown"}
    try:
        mlflow_client.search_experiments()
        status["mlflow"] = "ok"
    except Exception:
        status["mlflow"] = "unreachable"
    try:
        r = requests.get(f"{PREFECT_API_URL}/health",
                         headers=_prefect_headers(), timeout=5)
        status["prefect_cloud"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        status["prefect_cloud"] = "unreachable — check PREFECT_API_URL"
    try:
        boto3.client("s3", region_name=AWS_REGION).head_bucket(Bucket=S3_BUCKET)
        status["s3"] = "ok"
    except Exception:
        status["s3"] = "unreachable"
    return status

# ─────────────────────────────────────────────────────────────────────────────
# 3.1 + 3.2 — PREFECT CLOUD: Flow & Deployment Details
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/prefect/flows", tags=["Prefect Cloud — Flow Details"],
         summary="App Detail 1: List all registered Prefect flows")
def list_flows():
    """Retrieve all flows registered on Prefect Cloud. Shows name, ID, tags."""
    if not PREFECT_API_URL:
        raise HTTPException(503, "PREFECT_API_URL env var not set.")
    try:
        r = requests.post(f"{PREFECT_API_URL}/flows/filter",
                          headers=_prefect_headers(),
                          json={"limit": 25}, timeout=10)
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
         summary="App Detail 2: List deployments with schedule info")
def list_deployments():
    """
    Retrieve all deployments from Prefect Cloud.
    Shows schedule (cron), work pool, active status.
    Directly answers 'flow/deployment details via Built-in APIs'.
    """
    if not PREFECT_API_URL:
        raise HTTPException(503, "PREFECT_API_URL env var not set.")
    try:
        r = requests.post(f"{PREFECT_API_URL}/deployments/filter",
                          headers=_prefect_headers(),
                          json={"limit": 25}, timeout=10)
        r.raise_for_status()
        return {
            "total_deployments": len(r.json()),
            "deployments": [
                {
                    "id":                  d.get("id"),
                    "name":                d.get("name"),
                    "status":              d.get("status"),
                    "is_schedule_active":  d.get("is_schedule_active"),
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
    """Retrieve recent pipeline execution history — state, timing, duration."""
    if not PREFECT_API_URL:
        raise HTTPException(503, "PREFECT_API_URL env var not set.")
    try:
        r = requests.post(f"{PREFECT_API_URL}/flow_runs/filter",
                          headers=_prefect_headers(),
                          json={"limit": limit, "sort": "START_TIME_DESC"},
                          timeout=10)
        r.raise_for_status()
        return {
            "total_runs": len(r.json()),
            "runs": [
                {
                    "id":             run.get("id"),
                    "name":           run.get("name"),
                    "state":          {"type": run.get("state", {}).get("type"),
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


# ─────────────────────────────────────────────────────────────────────────────
# 3.1 + 3.2 — MLFLOW: Model Deployment Details
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/mlflow/experiment", tags=["MLflow — Model Details"],
         summary="App Detail 4: ML experiment with all model metrics")
def get_experiment():
    """All model runs with metrics: accuracy, precision, recall, F1, ROC-AUC."""
    try:
        exp = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
        if not exp:
            raise HTTPException(404, f"Run ml_pipeline.py first.")
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
                                  "f1_score", "roc_auc", "cv_mean_accuracy",
                                  "specificity"]
                    },
                    "params": {"train_pct": r.data.params.get("train_pct"),
                               "test_pct":  r.data.params.get("test_pct"),
                               "features":  r.data.params.get("features")},
                }
                for r in runs
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/mlflow/models", tags=["MLflow — Model Details"],
         summary="App Detail 5: Registered models and deployment versions")
def list_registered_models():
    """Model Registry — name, version, stage, S3 artifact location."""
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
         summary="All files in S3 bucket (data, plots, model artifacts)")
def list_s3_artifacts():
    try:
        s3   = boto3.client("s3", region_name=AWS_REGION)
        resp = s3.list_objects_v2(Bucket=S3_BUCKET)
        objs = resp.get("Contents", [])
        return {
            "bucket": S3_BUCKET, "total_objects": len(objs),
            "objects": [
                {"key": o["Key"], "size_bytes": o["Size"],
                 "last_modified": o["LastModified"].isoformat()}
                for o in sorted(objs, key=lambda x: x["Key"])
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))
```

---

## Deploy & Run Everything

### Step 11 — Set Prefect Cloud environment variables

```bash
# Find your API URL at: app.prefect.cloud → Settings → API Keys
# It looks like: https://api.prefect.cloud/api/accounts/XXXX/workspaces/YYYY

# Edit .bashrc and fill in both values
nano ~/.bashrc
# Set PREFECT_API_KEY and PREFECT_API_URL, then save

source ~/.bashrc

# Login to Prefect Cloud
source ~/mlpipeline/venv/bin/activate
prefect cloud login --key $PREFECT_API_KEY
prefect cloud workspace ls    # Confirm active workspace
```

### Step 12 — Start MLflow

```bash
screen -S mlflow
source ~/mlpipeline/venv/bin/activate
mlflow server \
  --backend-store-uri sqlite:///$HOME/mlpipeline/mlflow.db \
  --default-artifact-root s3://$S3_BUCKET/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
# Ctrl+A then D to detach
```

### Step 13 — Run the ML pipeline

```bash
source ~/mlpipeline/venv/bin/activate
cd ~/mlpipeline/ml
python ml_pipeline.py
# Both models train and all metrics log to MLflow
```

### Step 14 — Register the Prefect deployment

```bash
source ~/mlpipeline/venv/bin/activate
cd ~/mlpipeline/flows
python data_pipeline.py
# Output: Deployment created! Schedule: */3 * * * *
```

Go to `app.prefect.cloud` → **Deployments** — you will see `titanic-pipeline-every-3min` with the cron schedule active.

### Step 15 — Start the Prefect worker

```bash
screen -S prefect-worker
source ~/mlpipeline/venv/bin/activate
prefect worker start --pool "ec2-work-pool"
# Output: Worker started! Looking for scheduled flow runs...
# Ctrl+A then D to detach
```

Every 3 minutes, Prefect Cloud tells the worker to run the flow. Logs stream live to `app.prefect.cloud`.

### Step 16 — Start FastAPI

```bash
screen -S fastapi
source ~/mlpipeline/venv/bin/activate
cd ~/mlpipeline/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Ctrl+A then D to detach
```

### Step 17 — Trigger a manual run immediately

```bash
source ~/mlpipeline/venv/bin/activate
prefect deployment run "titanic-data-pipeline/titanic-pipeline-every-3min"
# Watch it run live at app.prefect.cloud → Flow Runs
```

---

### All services at a glance

| Service | Where it runs | How to access |
|---|---|---|
| Prefect Cloud Dashboard | Prefect's servers | `https://app.prefect.cloud` |
| Prefect Worker | EC2 (screen session) | — |
| MLflow UI | EC2 port 5000 | `http://<EC2_IP>:5000` |
| FastAPI Swagger | EC2 port 8000 | `http://<EC2_IP>:8000/docs` |
| Data + Plots + Models | AWS S3 | AWS Console → S3 |

---

### Useful commands reference

```bash
# ── Screen sessions ───────────────────────────────────────────────────────────
screen -ls                      # List all
screen -r mlflow                # Reattach MLflow
screen -r prefect-worker        # Reattach worker
screen -r fastapi               # Reattach FastAPI

# ── Prefect CLI ───────────────────────────────────────────────────────────────
prefect deployment ls           # List deployments
prefect flow-run ls             # Recent flow runs
prefect work-pool ls            # Work pools
prefect deployment run "titanic-data-pipeline/titanic-pipeline-every-3min"

# ── Test API ──────────────────────────────────────────────────────────────────
curl http://localhost:8000/health
curl http://localhost:8000/prefect/deployments
curl http://localhost:8000/mlflow/experiment

# ── S3 ────────────────────────────────────────────────────────────────────────
aws s3 ls s3://$S3_BUCKET/ --recursive
aws s3 cp s3://$S3_BUCKET/plots/ ./plots/ --recursive   # Download all EDA plots

# ── Restart all after EC2 reboot ─────────────────────────────────────────────
source ~/.bashrc && source ~/mlpipeline/venv/bin/activate
screen -S mlflow         -dm bash -c "source ~/mlpipeline/venv/bin/activate && mlflow server --backend-store-uri sqlite:///$HOME/mlpipeline/mlflow.db --default-artifact-root s3://$S3_BUCKET/mlflow-artifacts --host 0.0.0.0 --port 5000"
screen -S prefect-worker -dm bash -c "source ~/mlpipeline/venv/bin/activate && prefect worker start --pool ec2-work-pool"
screen -S fastapi        -dm bash -c "source ~/mlpipeline/venv/bin/activate && cd ~/mlpipeline/api && uvicorn main:app --host 0.0.0.0 --port 8000"
```

---

## Screenshots Guide

| # | Screenshot | Where to find it |
|---|---|---|
| 1 | **Prefect Cloud — Deployments page** showing `titanic-pipeline-every-3min` with `*/3 * * * *` schedule | `app.prefect.cloud` → Deployments |
| 2 | **Prefect Cloud — Flow Runs list** showing multiple completed runs (green ticks) | Prefect Cloud → Flow Runs |
| 3 | **Prefect Cloud — Single flow run** showing 3 task boxes in sequence | Click any run → Task Runs tab |
| 4 | **Prefect Cloud — Ingestion task logs** showing row count, columns, dtypes | Click Ingestion task → Logs |
| 5 | **Prefect Cloud — Preprocessing task logs** showing imputation and normalization | Click Preprocessing task → Logs |
| 6 | **Prefect Cloud — EDA task logs** showing correlation matrix and feature importance | Click EDA task → Logs |
| 7 | **Prefect Cloud — Work Pools** showing `ec2-work-pool` READY | Prefect Cloud → Work Pools |
| 8 | **AWS S3 Bucket** showing all 3 folders | AWS Console → S3 → your bucket |
| 9 | **5 EDA plots** (download from S3 → paste into Word doc) | S3 → plots/ folder |
| 10 | **MLflow Experiments page** showing `Titanic_Survival_Prediction` | `http://<EC2_IP>:5000` |
| 11 | **MLflow — Both runs** side by side (Compare view) | MLflow → select both → Compare |
| 12 | **MLflow — Random Forest run** showing all 8+ metrics | MLflow → click RF run |
| 13 | **MLflow — Logistic Regression run** showing all 8+ metrics | MLflow → click LR run |
| 14 | **MLflow — Artifacts panel** showing model files stored in S3 | MLflow → any run → Artifacts |
| 15 | **FastAPI Swagger UI** (`/docs`) showing all endpoints | `http://<EC2_IP>:8000/docs` |
| 16 | **`GET /prefect/deployments`** — JSON response showing cron schedule | Swagger → try it out → Execute |
| 17 | **`GET /prefect/flow-runs`** — JSON response showing run states | Swagger → try it out → Execute |
| 18 | **`GET /mlflow/experiment`** — JSON showing both models + metrics | Swagger → try it out → Execute |
| 19 | **`GET /s3/artifacts`** — JSON listing all stored files | Swagger → try it out → Execute |
| 20 | **EC2 instance running** (instance state = running) | AWS Console → EC2 → Instances |

---

## Mark Breakdown

| Requirement | Marks | Delivered By |
|---|---|---|
| **1.1 Business Understanding** | ✅ | Titanic survival — binary classification problem |
| **1.2 Data Ingestion** | ✅ | `ingest_data()` @task — S3 read, logs shape/columns/dtypes/missing |
| **1.3 Data Preprocessing** | ✅ | `preprocess_data()` @task — summary stats, missing check, median imputation, MinMax normalization |
| **1.4 EDA** | ✅ | `perform_eda()` @task — correlation matrix, encoding, binning, feature importance, 5 plots → S3 |
| **1.5 DataOps** | ✅ | Prefect Cloud deployment, `*/3 * * * *` cron, real hosted cloud dashboard |
| **2.1 Model Preparation** | ✅ | Random Forest + Logistic Regression, feature engineering |
| **2.2 Model Training** | ✅ | Stratified 70/30 split + 5-fold cross-validation |
| **2.3 Model Evaluation** | ✅ | Accuracy, confusion matrix, classification report |
| **2.4 MLOps** | ✅ | MLflow: 8 metrics (accuracy, precision, recall, F1, ROC-AUC, CV, specificity, TP/FP/FN/TN) |
| **3.1 Built-in APIs** | ✅ | Prefect Cloud REST API (flows, deployments, runs) + MLflow tracking API |
| **3.2 Display Details** | ✅ | 5 FastAPI endpoints with complete JSON responses |
| **Submission doc + video** | ✅ | 20 screenshots + screen-record all UIs |
| **Virtual Lab (AWS EC2)** | ✅ | Deployed on EC2 with S3 storage = bonus mark |

---

*CCZG506 — API-driven Cloud Native Solutions | Assignment I | BITS Pilani WILP*
