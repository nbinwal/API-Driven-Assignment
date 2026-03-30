# CCZG506 — Assignment I: Cloud-Native ML Pipeline on AWS

> **Dataset**: Titanic (Kaggle) | **Stack**: Apache Airflow · MLflow · FastAPI · AWS EC2 · S3 · CloudWatch

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [AWS Infrastructure Setup](#aws-infrastructure-setup)
3. [Project Structure](#project-structure)
4. [EC2 Environment Setup](#ec2-environment-setup)
5. [Sub-Objective 1 — Data Pipeline](#sub-objective-1--data-pipeline)
6. [Sub-Objective 2 — ML Pipeline](#sub-objective-2--ml-pipeline)
7. [Sub-Objective 3 — API Access](#sub-objective-3--api-access)
8. [Running Everything](#running-everything)
9. [Screenshots Guide](#screenshots-guide)
10. [Mark Breakdown](#mark-breakdown)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                            │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Amazon S3  │    │  Amazon EC2  │    │  CloudWatch  │  │
│  │              │    │  t3.medium   │    │   Logging    │  │
│  │ - Titanic    │◄───│              │───►│  Dashboard   │  │
│  │   dataset    │    │ - Airflow    │    │              │  │
│  │ - MLflow     │    │ - MLflow     │    └──────────────┘  │
│  │   artifacts  │    │ - FastAPI    │                       │
│  │ - EDA plots  │    │             │                       │
│  └──────────────┘    └──────────────┘                      │
│                             │                              │
│                    ┌────────▼────────┐                     │
│                    │  IAM Role       │                     │
│                    │  S3 + CW Access │                     │
│                    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## AWS Infrastructure Setup

### Step 1 — Create an S3 Bucket

1. Go to **AWS Console → S3 → Create Bucket**
2. Bucket name: `mlpipeline-titanic-<yourname>` (must be globally unique)
3. Region: `ap-south-1` (Mumbai) or any preferred region
4. Leave all other settings as default → **Create Bucket**
5. Inside the bucket, create these folders:
   - `data/`
   - `plots/`
   - `mlflow-artifacts/`

### Step 2 — Create an IAM Role for EC2

1. Go to **AWS Console → IAM → Roles → Create Role**
2. Trusted entity: **EC2**
3. Attach these policies:
   - `AmazonS3FullAccess`
   - `CloudWatchLogsFullAccess`
4. Role name: `EC2-MLPipeline-Role`
5. **Create Role**

### Step 3 — Launch EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Settings:
   - **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
   - **Instance type**: `t3.medium` (2 vCPU, 4 GB RAM) — minimum; use `t3.large` for comfort
   - **Key pair**: Create new → name it `mlpipeline-key` → download `.pem`
   - **IAM Instance Profile**: Select `EC2-MLPipeline-Role`
   - **Storage**: 20 GB gp3
3. **Security Group** — Add these inbound rules:

| Type       | Protocol | Port  | Source    | Purpose         |
|------------|----------|-------|-----------|-----------------|
| SSH        | TCP      | 22    | My IP     | Terminal access |
| Custom TCP | TCP      | 8080  | 0.0.0.0/0 | Airflow UI      |
| Custom TCP | TCP      | 5000  | 0.0.0.0/0 | MLflow UI       |
| Custom TCP | TCP      | 8000  | 0.0.0.0/0 | FastAPI         |

4. **Launch Instance**

### Step 4 — Connect to EC2

```bash
# On your local machine:
chmod 400 mlpipeline-key.pem
ssh -i mlpipeline-key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## Project Structure

```
~/mlpipeline/
├── dags/
│   └── data_pipeline_dag.py       # Airflow DAG (runs every 3 min)
├── ml/
│   └── ml_pipeline.py             # MLflow training pipeline
├── api/
│   └── main.py                    # FastAPI application
├── scripts/
│   ├── setup.sh                   # One-time environment setup
│   ├── start_all.sh               # Start all services
│   └── upload_data.sh             # Upload dataset to S3
├── requirements.txt
└── README.md
```

---

## EC2 Environment Setup

### Step 5 — Run setup on EC2

SSH into your EC2 instance and run the following commands one by one.

```bash
# ── Update system ────────────────────────────────────────────────────────────
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv wget unzip git screen

# ── Create project directory ─────────────────────────────────────────────────
mkdir -p ~/mlpipeline/{dags,ml,api,scripts,data,plots,logs}
cd ~/mlpipeline

# ── Create and activate virtual environment ───────────────────────────────────
python3 -m venv venv
source venv/bin/activate

# ── Install all dependencies ──────────────────────────────────────────────────
pip install --upgrade pip
pip install \
  apache-airflow==2.8.4 \
  mlflow==2.13.0 \
  fastapi==0.111.0 \
  uvicorn==0.29.0 \
  pandas==2.2.2 \
  scikit-learn==1.4.2 \
  matplotlib==3.9.0 \
  seaborn==0.13.2 \
  boto3==1.34.0 \
  watchtower==3.2.0 \
  requests==2.32.0 \
  python-multipart==0.0.9

# ── Configure AWS CLI (confirm region matches your EC2) ───────────────────────
# Since we attached an IAM role to EC2, no credentials needed.
# Just verify it works:
aws s3 ls   # Should list your buckets without error
```

### Step 6 — Upload Titanic dataset to S3

Download Titanic CSV from https://www.kaggle.com/datasets/yasserh/titanic-dataset
Upload it to your EC2 instance, then push to S3:

```bash
# From your LOCAL machine — upload the csv to EC2 first:
scp -i mlpipeline-key.pem titanic.csv ubuntu@<EC2_PUBLIC_IP>:~/mlpipeline/data/

# Then on EC2 — push to S3 (replace bucket name):
export S3_BUCKET="mlpipeline-titanic-<yourname>"
aws s3 cp ~/mlpipeline/data/titanic.csv s3://$S3_BUCKET/data/titanic.csv
echo "Upload complete"
```

### Step 7 — Set environment variables

```bash
# Add to ~/.bashrc so they persist across sessions
cat >> ~/.bashrc << 'EOF'

# ── ML Pipeline Config ─────────────────────────────────────────────────────
export S3_BUCKET="mlpipeline-titanic-<yourname>"       # CHANGE THIS
export AWS_DEFAULT_REGION="ap-south-1"                 # CHANGE IF DIFFERENT
export AIRFLOW_HOME=~/airflow
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL=""
export PROJECT_DIR=~/mlpipeline
export CW_LOG_GROUP="/mlpipeline/airflow"
EOF

source ~/.bashrc
```

---

## Sub-Objective 1 — Data Pipeline

Create the file `~/mlpipeline/dags/data_pipeline_dag.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 1
Data Pipeline DAG: Ingestion → Preprocessing → EDA
Runs every 3 minutes. Logs to AWS CloudWatch.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import boto3
import logging
import watchtower
import os
import io

# ─── Constants ────────────────────────────────────────────────────────────────
S3_BUCKET       = os.environ.get("S3_BUCKET", "mlpipeline-titanic-yourname")
S3_DATA_KEY     = "data/titanic.csv"
S3_PROC_KEY     = "data/titanic_processed.csv"
S3_PLOTS_PREFIX = "plots/"
CW_LOG_GROUP    = os.environ.get("CW_LOG_GROUP", "/mlpipeline/airflow")
AWS_REGION      = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

# ─── CloudWatch Logger Factory ────────────────────────────────────────────────
def get_logger(task_name: str) -> logging.Logger:
    """Return a logger that writes to both CloudWatch and local stdout."""
    logger = logging.getLogger(task_name)
    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers

    logger.setLevel(logging.INFO)

    # stdout handler
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(sh)

    # CloudWatch handler
    try:
        cw_client = boto3.client("logs", region_name=AWS_REGION)
        cw_handler = watchtower.CloudWatchLogHandler(
            boto3_client=cw_client,
            log_group=CW_LOG_GROUP,
            stream_name=task_name,
            create_log_group=True,
        )
        cw_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(cw_handler)
    except Exception as e:
        logger.warning(f"CloudWatch handler setup failed (running locally?): {e}")

    return logger

# ─── S3 Helper ────────────────────────────────────────────────────────────────
def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def write_csv_to_s3(df: pd.DataFrame, bucket: str, key: str):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

def upload_plot_to_s3(fig, bucket: str, key: str):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType="image/png")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.2 — Data Ingestion
# ─────────────────────────────────────────────────────────────────────────────
def ingest_data(**context):
    logger = get_logger("DataIngestion")
    logger.info("=" * 60)
    logger.info("TASK 1.2: DATA INGESTION — STARTED")
    logger.info("=" * 60)

    try:
        df = read_csv_from_s3(S3_BUCKET, S3_DATA_KEY)
        logger.info(f"Source       : s3://{S3_BUCKET}/{S3_DATA_KEY}")
        logger.info(f"Rows         : {df.shape[0]}")
        logger.info(f"Columns      : {df.shape[1]}")
        logger.info(f"Column names : {list(df.columns)}")
        logger.info(f"Memory usage : {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        logger.info(f"Sample (head):\n{df.head(3).to_string()}")

        # Push metadata to XCom for downstream tasks
        context["ti"].xcom_push(key="row_count",    value=int(df.shape[0]))
        context["ti"].xcom_push(key="col_count",    value=int(df.shape[1]))
        context["ti"].xcom_push(key="column_names", value=list(df.columns))

        logger.info("TASK 1.2: DATA INGESTION — COMPLETED SUCCESSFULLY")
        return "Ingestion Done"

    except Exception as e:
        logger.error(f"TASK 1.2 FAILED: {e}", exc_info=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.3 — Data Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(**context):
    logger = get_logger("DataPreprocessing")
    logger.info("=" * 60)
    logger.info("TASK 1.3: DATA PREPROCESSING — STARTED")
    logger.info("=" * 60)

    try:
        df = read_csv_from_s3(S3_BUCKET, S3_DATA_KEY)

        # ── Summary Statistics ────────────────────────────────────────────────
        logger.info("--- Summary Statistics ---")
        logger.info(f"\n{df.describe().to_string()}")

        # ── Data Types ────────────────────────────────────────────────────────
        logger.info("--- Data Types ---")
        for col, dtype in df.dtypes.items():
            logger.info(f"  {col:<15} : {dtype}")

        # ── Missing Values ────────────────────────────────────────────────────
        missing = df.isnull().sum()
        logger.info("--- Missing Values ---")
        for col, cnt in missing[missing > 0].items():
            pct = (cnt / len(df)) * 100
            logger.info(f"  {col:<15} : {cnt} ({pct:.1f}%)")

        # ── Impute Numeric Columns with Median ────────────────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  Imputed '{col}' ({n_missing} nulls) with median={median_val:.4f}")

        # ── Impute Categorical Columns ────────────────────────────────────────
        if df["Embarked"].isnull().sum() > 0:
            mode_val = df["Embarked"].mode()[0]
            df["Embarked"].fillna(mode_val, inplace=True)
            logger.info(f"  Imputed 'Embarked' with mode='{mode_val}'")

        if "Cabin" in df.columns:
            df["Cabin"].fillna("Unknown", inplace=True)
            logger.info("  Imputed 'Cabin' with 'Unknown'")

        # ── Normalize Numeric Columns ─────────────────────────────────────────
        cols_to_scale = ["Age", "Fare", "SibSp", "Parch"]
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        logger.info(f"  Normalized (MinMaxScaler): {cols_to_scale}")

        # Verify no missing values remain
        remaining_nulls = df.isnull().sum().sum()
        logger.info(f"  Remaining null values after preprocessing: {remaining_nulls}")

        # ── Save processed data to S3 ─────────────────────────────────────────
        write_csv_to_s3(df, S3_BUCKET, S3_PROC_KEY)
        logger.info(f"  Processed data saved to s3://{S3_BUCKET}/{S3_PROC_KEY}")

        context["ti"].xcom_push(key="processed_shape", value=str(df.shape))
        logger.info("TASK 1.3: DATA PREPROCESSING — COMPLETED SUCCESSFULLY")
        return "Preprocessing Done"

    except Exception as e:
        logger.error(f"TASK 1.3 FAILED: {e}", exc_info=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.4 — Exploratory Data Analysis
# ─────────────────────────────────────────────────────────────────────────────
def perform_eda(**context):
    logger = get_logger("EDA")
    logger.info("=" * 60)
    logger.info("TASK 1.4: EXPLORATORY DATA ANALYSIS — STARTED")
    logger.info("=" * 60)

    try:
        df = read_csv_from_s3(S3_BUCKET, S3_PROC_KEY)
        numeric_df = df.select_dtypes(include=[np.number])

        # ── Correlation Coefficients ──────────────────────────────────────────
        corr = numeric_df.corr()
        logger.info("--- Correlation Matrix ---")
        logger.info(f"\n{corr.round(3).to_string()}")

        logger.info("--- Top Correlations with 'Survived' ---")
        survived_corr = corr["Survived"].abs().sort_values(ascending=False)
        for feat, val in survived_corr.items():
            logger.info(f"  {feat:<15} : {val:.4f}")

        # ── Encoding Categorical Columns ──────────────────────────────────────
        df["Sex_encoded"]      = df["Sex"].map({"male": 0, "female": 1})
        df["Embarked_encoded"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(-1)
        logger.info("Encoded: Sex → Sex_encoded, Embarked → Embarked_encoded")

        # ── Binning Age ───────────────────────────────────────────────────────
        df["Age_bin"] = pd.cut(
            df["Age"],
            bins=5,
            labels=["Very Young", "Young", "Middle-aged", "Senior", "Elderly"]
        )
        logger.info("--- Age Binning Distribution ---")
        for label, count in df["Age_bin"].value_counts().sort_index().items():
            logger.info(f"  {str(label):<15} : {count}")

        # ── Feature Importance (Correlation with Target) ──────────────────────
        numeric_df2 = df.select_dtypes(include=[np.number])
        feature_imp = numeric_df2.corr()["Survived"].abs().drop("Survived").sort_values(ascending=False)
        logger.info("--- Feature Importance (|Correlation| with Survived) ---")
        for feat, score in feature_imp.items():
            logger.info(f"  {feat:<20} : {score:.4f}")

        # ── Survival Rate Statistics ──────────────────────────────────────────
        survival_rate = df["Survived"].mean() * 100
        logger.info(f"Overall Survival Rate: {survival_rate:.1f}%")
        logger.info(f"Survival by Sex:\n{df.groupby('Sex')['Survived'].mean().round(3).to_string()}")
        logger.info(f"Survival by Pclass:\n{df.groupby('Pclass')['Survived'].mean().round(3).to_string()}")

        # ─────────────────────────────────────────────────────────────────────
        # PLOTS
        # ─────────────────────────────────────────────────────────────────────

        # Plot 1: Correlation Heatmap
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, annot_kws={"size": 9})
        ax.set_title("Correlation Heatmap — Titanic Features", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        upload_plot_to_s3(fig, S3_BUCKET, f"{S3_PLOTS_PREFIX}1_correlation_heatmap.png")
        plt.close(fig)
        logger.info("Plot saved: 1_correlation_heatmap.png")

        # Plot 2: Univariate — Survival Count
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df["Survived"].value_counts().plot(
            kind="bar", ax=axes[0], color=["#e74c3c", "#2ecc71"], edgecolor="black", width=0.5
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
            kind="bar", ax=axes[1], color=["#3498db", "#f39c12", "#9b59b6"],
            edgecolor="black", width=0.5
        )
        axes[1].set_title("Passenger Class Distribution", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Passenger Class")
        axes[1].set_ylabel("Count")
        axes[1].set_xticklabels(["1st Class", "2nd Class", "3rd Class"], rotation=0)
        fig.suptitle("Univariate Analysis — Titanic Dataset", fontsize=15, fontweight="bold")
        plt.tight_layout()
        upload_plot_to_s3(fig, S3_BUCKET, f"{S3_PLOTS_PREFIX}2_univariate_analysis.png")
        plt.close(fig)
        logger.info("Plot saved: 2_univariate_analysis.png")

        # Plot 3: Bivariate — Survival by Sex and Pclass
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.countplot(data=df, x="Sex", hue="Survived", palette={"0": "#e74c3c", "1": "#2ecc71"},
                      ax=axes[0], edgecolor="black")
        axes[0].set_title("Survival by Gender", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Gender")
        axes[0].set_ylabel("Count")
        axes[0].legend(title="Survived", labels=["No", "Yes"])

        survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100
        axes[1].bar(survival_by_class.index.astype(str),
                    survival_by_class.values,
                    color=["#3498db", "#f39c12", "#9b59b6"],
                    edgecolor="black", width=0.5)
        axes[1].set_title("Survival Rate by Passenger Class", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Passenger Class")
        axes[1].set_ylabel("Survival Rate (%)")
        for i, val in enumerate(survival_by_class.values):
            axes[1].text(i, val + 1, f"{val:.1f}%", ha="center", fontsize=11)
        fig.suptitle("Bivariate Analysis — Survival Relationships", fontsize=15, fontweight="bold")
        plt.tight_layout()
        upload_plot_to_s3(fig, S3_BUCKET, f"{S3_PLOTS_PREFIX}3_bivariate_analysis.png")
        plt.close(fig)
        logger.info("Plot saved: 3_bivariate_analysis.png")

        # Plot 4: Age Distribution by Survival (KDE)
        fig, ax = plt.subplots(figsize=(10, 5))
        for survived_val, color, label in [(0, "#e74c3c", "Did Not Survive"), (1, "#2ecc71", "Survived")]:
            subset = df[df["Survived"] == survived_val]["Age"]
            sns.kdeplot(subset, ax=ax, fill=True, color=color, alpha=0.5, label=label)
        ax.set_title("Age Distribution by Survival Outcome", fontsize=14, fontweight="bold")
        ax.set_xlabel("Age (Normalized 0–1)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=11)
        plt.tight_layout()
        upload_plot_to_s3(fig, S3_BUCKET, f"{S3_PLOTS_PREFIX}4_age_distribution_survival.png")
        plt.close(fig)
        logger.info("Plot saved: 4_age_distribution_survival.png")

        # Plot 5: Feature Importance Bar Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        feature_imp.plot(kind="bar", ax=ax, color="#3498db", edgecolor="black", width=0.6)
        ax.set_title("Feature Importance (|Correlation| with Survived)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Absolute Correlation")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        upload_plot_to_s3(fig, S3_BUCKET, f"{S3_PLOTS_PREFIX}5_feature_importance.png")
        plt.close(fig)
        logger.info("Plot saved: 5_feature_importance.png")

        logger.info(f"All 5 EDA plots uploaded to s3://{S3_BUCKET}/{S3_PLOTS_PREFIX}")
        logger.info("TASK 1.4: EDA — COMPLETED SUCCESSFULLY")
        return "EDA Done"

    except Exception as e:
        logger.error(f"TASK 1.4 FAILED: {e}", exc_info=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION (Task 1.5 — DataOps / Scheduling)
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    "owner":          "mlpipeline",
    "retries":        1,
    "retry_delay":    timedelta(minutes=1),
    "start_date":     datetime(2026, 3, 1),
    "email_on_failure": False,
    "email_on_retry":   False,
}

with DAG(
    dag_id="titanic_data_pipeline",
    default_args=default_args,
    description="Automated data pipeline: ingestion, preprocessing, EDA — logged to CloudWatch",
    schedule_interval="*/3 * * * *",   # Every 3 minutes ✅
    catchup=False,
    max_active_runs=1,
    tags=["DataOps", "Titanic", "CCZG506"],
) as dag:

    task_ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data,
        provide_context=True,
    )

    task_preprocess = PythonOperator(
        task_id="data_preprocessing",
        python_callable=preprocess_data,
        provide_context=True,
    )

    task_eda = PythonOperator(
        task_id="exploratory_data_analysis",
        python_callable=perform_eda,
        provide_context=True,
    )

    # Pipeline execution order
    task_ingest >> task_preprocess >> task_eda
```

---

## Sub-Objective 2 — ML Pipeline

Create the file `~/mlpipeline/ml/ml_pipeline.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 2
ML Pipeline: Training, Evaluation, MLOps with MLflow (artifacts stored on S3)
Algorithms: Random Forest + Logistic Regression
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import boto3
import logging
import io
import os

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# ─── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET    = os.environ.get("S3_BUCKET", "mlpipeline-titanic-yourname")
S3_DATA_KEY  = "data/titanic.csv"
AWS_REGION   = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT   = "Titanic_Survival_Prediction"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLPipeline")

# ─── MLflow Setup ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)

# Store MLflow artifacts on S3
os.environ["MLFLOW_ARTIFACT_ROOT"] = f"s3://{S3_BUCKET}/mlflow-artifacts"

mlflow.set_experiment(EXPERIMENT)

# ─────────────────────────────────────────────────────────────────────────────
# 2.1 MODEL PREPARATION — Data loading and feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare_data():
    logger.info("=" * 60)
    logger.info("STEP 2.1: MODEL PREPARATION")
    logger.info("=" * 60)

    # Read from S3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    logger.info(f"Loaded {df.shape[0]} rows from s3://{S3_BUCKET}/{S3_DATA_KEY}")

    # ── Feature Engineering ───────────────────────────────────────────────────
    # Fill missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Drop columns with too many missing values or irrelevant to model
    df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"], inplace=True)

    # Encode categorical features
    le = LabelEncoder()
    df["Sex"]      = le.fit_transform(df["Sex"])        # male=1, female=0
    df["Embarked"] = le.fit_transform(df["Embarked"])   # S/C/Q → 0/1/2

    # Feature set
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[feature_cols]
    y = df["Survived"]

    logger.info(f"Features used : {feature_cols}")
    logger.info(f"Target        : Survived (0=No, 1=Yes)")
    logger.info(f"Class balance : {y.value_counts().to_dict()}")

    return X, y, feature_cols

# ─────────────────────────────────────────────────────────────────────────────
# 2.2 & 2.3 & 2.4 — Train, Evaluate, Log
# ─────────────────────────────────────────────────────────────────────────────
def train_and_log(model, model_name: str, X_train, X_test, y_train, y_test, feature_cols):
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name}")
    logger.info(f"{'='*60}")

    with mlflow.start_run(run_name=model_name) as run:

        # ── Log Parameters ────────────────────────────────────────────────────
        mlflow.log_param("model_name",    model_name)
        mlflow.log_param("train_size",    "70%")
        mlflow.log_param("test_size",     "30%")
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples",  len(X_test))
        mlflow.log_param("features",      str(feature_cols))
        mlflow.log_param("n_features",    len(feature_cols))

        # Log model-specific hyperparameters
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth",    model.max_depth)
        if hasattr(model, "max_iter"):
            mlflow.log_param("max_iter",     model.max_iter)
            mlflow.log_param("C",            model.C)

        # ── Train Model ───────────────────────────────────────────────────────
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # ── Compute All Metrics ───────────────────────────────────────────────
        accuracy   = accuracy_score(y_test, y_pred)
        precision  = precision_score(y_test, y_pred, zero_division=0)
        recall     = recall_score(y_test, y_pred, zero_division=0)
        f1         = f1_score(y_test, y_pred, zero_division=0)
        roc_auc    = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else 0.0
        cv_scores  = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_mean    = cv_scores.mean()
        cv_std     = cv_scores.std()
        cm         = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # ── Log Metrics (MLOps — 4+ metrics required) ─────────────────────────
        mlflow.log_metric("accuracy",          round(accuracy,    4))
        mlflow.log_metric("precision",         round(precision,   4))
        mlflow.log_metric("recall",            round(recall,      4))
        mlflow.log_metric("f1_score",          round(f1,          4))
        mlflow.log_metric("roc_auc",           round(roc_auc,     4))
        mlflow.log_metric("cv_mean_accuracy",  round(cv_mean,     4))
        mlflow.log_metric("cv_std_accuracy",   round(cv_std,      4))
        mlflow.log_metric("specificity",       round(specificity, 4))
        mlflow.log_metric("true_positives",    int(tp))
        mlflow.log_metric("true_negatives",    int(tn))
        mlflow.log_metric("false_positives",   int(fp))
        mlflow.log_metric("false_negatives",   int(fn))

        # ── Log the Model Artifact (saved to S3) ──────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        # ── Print Results ─────────────────────────────────────────────────────
        logger.info(f"  Accuracy          : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        logger.info(f"  Precision         : {precision:.4f}")
        logger.info(f"  Recall            : {recall:.4f}")
        logger.info(f"  F1 Score          : {f1:.4f}")
        logger.info(f"  ROC-AUC           : {roc_auc:.4f}")
        logger.info(f"  CV Accuracy (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"  Specificity       : {specificity:.4f}")
        logger.info(f"\nConfusion Matrix:\n"
                    f"  TP={tp}  FP={fp}\n"
                    f"  FN={fn}  TN={tn}")
        logger.info(f"\nClassification Report:\n"
                    f"{classification_report(y_test, y_pred, target_names=['Did Not Survive','Survived'])}")
        logger.info(f"  MLflow Run ID     : {run.info.run_id}")

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

    # Prepare data
    X, y, feature_cols = load_and_prepare_data()

    # 2.2 Train/Test Split — 70% / 30%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    logger.info(f"\nSPLIT: Train={len(X_train)} | Test={len(X_test)}")

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

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
        rf, "RandomForest", X_train, X_test, y_train, y_test, feature_cols
    )

    # ── Algorithm 2: Logistic Regression ──────────────────────────────────────
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    _, results["LogisticRegression"] = train_and_log(
        lr, "LogisticRegression", X_train_scaled, X_test_scaled,
        y_train, y_test, feature_cols
    )

    # ── Model Comparison Summary ──────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    header = f"{'Metric':<22} {'Random Forest':>15} {'Logistic Reg':>15}"
    logger.info(header)
    logger.info("-" * len(header))
    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        rf_val = results["RandomForest"][metric]
        lr_val = results["LogisticRegression"][metric]
        logger.info(f"  {metric:<20} {rf_val:>15.4f} {lr_val:>15.4f}")

    best = max(results, key=lambda k: results[k]["f1_score"])
    logger.info(f"\nBest Model (by F1): {best}")
    logger.info(f"View results at  : http://localhost:5000")
    logger.info("="*60)

if __name__ == "__main__":
    main()
```

---

## Sub-Objective 3 — API Access

Create the file `~/mlpipeline/api/main.py`:

```python
"""
CCZG506 Assignment I — Sub-Objective 3
FastAPI application exposing Airflow and MLflow details via REST APIs
"""

from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
import mlflow
import boto3
import requests
import logging
import os

# ─── Config ───────────────────────────────────────────────────────────────────
AIRFLOW_BASE    = "http://localhost:8080/api/v1"
AIRFLOW_CREDS   = ("admin", "admin")    # Update if you changed Airflow password
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "Titanic_Survival_Prediction"
S3_BUCKET       = os.environ.get("S3_BUCKET", "mlpipeline-titanic-yourname")
AWS_REGION      = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("API")

# ─── App Init ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CCZG506 Cloud ML Application API",
    description=(
        "REST APIs to retrieve Airflow pipeline details and MLflow model details. "
        "Assignment I — BITS Pilani WILP."
    ),
    version="1.0.0",
    contact={"name": "CCZG506 Group"},
    docs_url="/docs",
    redoc_url="/redoc",
)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow_client = MlflowClient(MLFLOW_URI)

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], summary="API health check")
def health_check():
    """Verify the API and downstream services are reachable."""
    status = {"api": "ok", "mlflow": "unknown", "airflow": "unknown"}
    try:
        mlflow_client.search_experiments()
        status["mlflow"] = "ok"
    except Exception:
        status["mlflow"] = "unreachable"
    try:
        resp = requests.get(f"{AIRFLOW_BASE}/health", auth=AIRFLOW_CREDS, timeout=4)
        status["airflow"] = "ok" if resp.status_code == 200 else "degraded"
    except Exception:
        status["airflow"] = "unreachable"
    return status

# ─────────────────────────────────────────────────────────────────────────────
# 3.1 + 3.2 — AIRFLOW: Pipeline Flow Details
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/airflow/dags",
    tags=["Airflow — Flow Details"],
    summary="List all DAGs (pipeline flows)",
)
def list_dags():
    """
    **Application Detail 1**: Retrieve all pipeline flow configurations from Airflow.
    Shows DAG ID, active status, schedule, tags and file location.
    """
    try:
        resp = requests.get(f"{AIRFLOW_BASE}/dags", auth=AIRFLOW_CREDS, timeout=10)
        resp.raise_for_status()
        dags_raw = resp.json().get("dags", [])
        return {
            "total_dags": len(dags_raw),
            "dags": [
                {
                    "dag_id":          d.get("dag_id"),
                    "description":     d.get("description"),
                    "is_active":       d.get("is_active"),
                    "is_paused":       d.get("is_paused"),
                    "schedule":        d.get("schedule_interval"),
                    "last_parsed_time": d.get("last_parsed_time"),
                    "file_location":   d.get("fileloc"),
                    "owners":          d.get("owners", []),
                    "tags":            [t.get("name") for t in d.get("tags", [])],
                }
                for d in dags_raw
            ],
        }
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Airflow is not reachable at localhost:8080")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/airflow/dags/{dag_id}/runs",
    tags=["Airflow — Flow Details"],
    summary="Get recent DAG run history",
)
def get_dag_runs(
    dag_id: str = Path(default="titanic_data_pipeline", description="DAG identifier")
):
    """
    **Application Detail 2**: Retrieve recent execution/run history for a pipeline.
    Shows run state, execution time, duration, and trigger type.
    """
    try:
        resp = requests.get(
            f"{AIRFLOW_BASE}/dags/{dag_id}/dagRuns",
            auth=AIRFLOW_CREDS,
            params={"limit": 10, "order_by": "-execution_date"},
            timeout=10,
        )
        resp.raise_for_status()
        runs = resp.json().get("dag_runs", [])
        return {
            "dag_id":     dag_id,
            "total_runs": len(runs),
            "runs": [
                {
                    "run_id":         r.get("dag_run_id"),
                    "state":          r.get("state"),
                    "execution_date": r.get("execution_date"),
                    "start_date":     r.get("start_date"),
                    "end_date":       r.get("end_date"),
                    "run_type":       r.get("run_type"),
                    "external_trigger": r.get("external_trigger"),
                }
                for r in runs
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/airflow/dags/{dag_id}/tasks",
    tags=["Airflow — Flow Details"],
    summary="Get task details for a DAG",
)
def get_dag_tasks(
    dag_id: str = Path(default="titanic_data_pipeline", description="DAG identifier")
):
    """Retrieve all task definitions and their configurations."""
    try:
        resp = requests.get(
            f"{AIRFLOW_BASE}/dags/{dag_id}/tasks",
            auth=AIRFLOW_CREDS,
            timeout=10,
        )
        resp.raise_for_status()
        tasks = resp.json().get("tasks", [])
        return {
            "dag_id":      dag_id,
            "total_tasks": len(tasks),
            "tasks": [
                {
                    "task_id":        t.get("task_id"),
                    "operator":       t.get("class_ref", {}).get("class_name"),
                    "trigger_rule":   t.get("trigger_rule"),
                    "retries":        t.get("retries"),
                    "retry_delay":    t.get("retry_delay"),
                    "depends_on_past": t.get("depends_on_past"),
                    "downstream_task_ids": t.get("downstream_task_ids", []),
                }
                for t in tasks
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3.1 + 3.2 — MLFLOW: Model Deployment Details
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/mlflow/experiment",
    tags=["MLflow — Deployment Details"],
    summary="Get experiment and all model run details",
)
def get_experiment_details():
    """
    **Application Detail 3**: Retrieve ML experiment details including
    all model runs, metrics, and deployment status.
    """
    try:
        exp = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
        if not exp:
            raise HTTPException(status_code=404, detail=f"Experiment '{EXPERIMENT_NAME}' not found. Run the ML pipeline first.")

        runs = mlflow_client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
        return {
            "experiment": {
                "id":               exp.experiment_id,
                "name":             exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage":  exp.lifecycle_stage,
            },
            "total_runs": len(runs),
            "model_runs": [
                {
                    "run_id":       r.info.run_id,
                    "model_name":   r.data.params.get("model_name", "N/A"),
                    "status":       r.info.status,
                    "start_time_ms": r.info.start_time,
                    "metrics": {
                        "accuracy":   round(r.data.metrics.get("accuracy",  0), 4),
                        "precision":  round(r.data.metrics.get("precision", 0), 4),
                        "recall":     round(r.data.metrics.get("recall",    0), 4),
                        "f1_score":   round(r.data.metrics.get("f1_score",  0), 4),
                        "roc_auc":    round(r.data.metrics.get("roc_auc",   0), 4),
                    },
                    "params": {
                        "train_size": r.data.params.get("train_size"),
                        "test_size":  r.data.params.get("test_size"),
                        "features":   r.data.params.get("features"),
                    },
                }
                for r in runs
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/mlflow/models",
    tags=["MLflow — Deployment Details"],
    summary="List all registered models",
)
def list_registered_models():
    """
    **Application Detail 4**: List all registered models and their versions
    (deployment-ready artifacts stored on S3).
    """
    try:
        models = mlflow_client.search_registered_models()
        return {
            "total_models": len(models),
            "models": [
                {
                    "name":             m.name,
                    "creation_timestamp": m.creation_timestamp,
                    "last_updated":     m.last_updated_timestamp,
                    "versions": [
                        {
                            "version":           v.version,
                            "stage":             v.current_stage,
                            "status":            v.status,
                            "run_id":            v.run_id,
                            "source":            v.source,
                        }
                        for v in m.latest_versions
                    ],
                }
                for m in models
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/s3/artifacts",
    tags=["S3 — Storage Details"],
    summary="List all artifacts and plots stored in S3",
)
def list_s3_artifacts():
    """
    **Application Detail 5**: Retrieve list of all stored artifacts
    (EDA plots, processed data, model artifacts) from S3.
    """
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        response = s3.list_objects_v2(Bucket=S3_BUCKET)
        contents = response.get("Contents", [])
        return {
            "bucket":        S3_BUCKET,
            "total_objects": len(contents),
            "objects": [
                {
                    "key":           obj["Key"],
                    "size_bytes":    obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                }
                for obj in sorted(contents, key=lambda x: x["Key"])
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Running Everything

### Step 8 — Copy files to correct locations

```bash
cd ~/mlpipeline

# ── Copy DAG to Airflow dags folder ──────────────────────────────────────────
cp dags/data_pipeline_dag.py ~/airflow/dags/
echo "DAG copied"
```

### Step 9 — Initialize and start Airflow

```bash
source ~/mlpipeline/venv/bin/activate

# Initialize Airflow database (first time only)
export AIRFLOW_HOME=~/airflow
airflow db init

# Create Airflow admin user (first time only)
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin

echo "Airflow initialized"
```

Now start Airflow services using `screen` so they keep running after you close the terminal:

```bash
# Terminal / Screen 1 — Airflow webserver
screen -S airflow-web
source ~/mlpipeline/venv/bin/activate
export AIRFLOW_HOME=~/airflow
airflow webserver --port 8080
# Press Ctrl+A then D to detach

# Terminal / Screen 2 — Airflow scheduler
screen -S airflow-scheduler
source ~/mlpipeline/venv/bin/activate
export AIRFLOW_HOME=~/airflow
airflow scheduler
# Press Ctrl+A then D to detach
```

### Step 10 — Start MLflow tracking server

```bash
screen -S mlflow
source ~/mlpipeline/venv/bin/activate
mlflow server \
  --backend-store-uri sqlite:///~/mlpipeline/mlflow.db \
  --default-artifact-root s3://$S3_BUCKET/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
# Press Ctrl+A then D to detach
```

### Step 11 — Run the ML pipeline

```bash
# In a regular terminal (not screen):
source ~/mlpipeline/venv/bin/activate
cd ~/mlpipeline/ml
python ml_pipeline.py
```

You will see output like:
```
Training: RandomForest
  Accuracy   : 0.8433  (84.33%)
  Precision  : 0.8316
  Recall     : 0.7826
  F1 Score   : 0.8063
  ROC-AUC    : 0.9021
  MLflow Run ID: abc123...

Training: LogisticRegression
  Accuracy   : 0.8022  (80.22%)
  ...
Best Model (by F1): RandomForest
```

### Step 12 — Start FastAPI

```bash
screen -S fastapi
source ~/mlpipeline/venv/bin/activate
cd ~/mlpipeline/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Press Ctrl+A then D to detach
```

### Step 13 — Verify everything is running

```bash
# Check all screens are running:
screen -ls
# Should show: airflow-web, airflow-scheduler, mlflow, fastapi

# Test API:
curl http://localhost:8000/health
# Expected: {"api":"ok","mlflow":"ok","airflow":"ok"}

curl http://localhost:8000/airflow/dags
curl http://localhost:8000/mlflow/experiment
```

### Access the UIs in your browser

| Service   | URL                                      |
|-----------|------------------------------------------|
| Airflow   | `http://<EC2_PUBLIC_IP>:8080`            |
| MLflow    | `http://<EC2_PUBLIC_IP>:5000`            |
| FastAPI   | `http://<EC2_PUBLIC_IP>:8000/docs`       |
| CloudWatch| AWS Console → CloudWatch → Log Groups → `/mlpipeline/airflow` |

> **Default Airflow login**: username=`admin`, password=`admin`

### Step 14 — Enable the DAG in Airflow

1. Open `http://<EC2_IP>:8080`
2. Find `titanic_data_pipeline` in the DAG list
3. Toggle the **ON/OFF** switch to **ON**
4. The DAG will now run automatically every 3 minutes
5. Go to **Graph View** to see the task dependencies

---

## Screenshots Guide

Take these screenshots for your submission document. They are required for full marks.

| # | Screenshot | Where to find it |
|---|---|---|
| 1 | **Airflow DAG list** showing `titanic_data_pipeline` | `http://<IP>:8080/home` |
| 2 | **DAG Graph view** — showing 3 tasks connected in sequence | Airflow → titanic_data_pipeline → Graph |
| 3 | **DAG Schedule** showing `*/3 * * * *` | Airflow → DAG details page |
| 4 | **Successful DAG run** (all tasks green) | Airflow → titanic_data_pipeline → Runs |
| 5 | **Task logs — Ingestion** (row count, columns) | Airflow → Task Instance → Log |
| 6 | **Task logs — Preprocessing** (missing values, imputation) | Airflow → Preprocessing task → Log |
| 7 | **Task logs — EDA** (correlations, feature importance) | Airflow → EDA task → Log |
| 8 | **CloudWatch Log Group** `/mlpipeline/airflow` | AWS Console → CloudWatch → Logs |
| 9 | **CloudWatch log stream** showing pipeline messages | CW → Log streams → DataIngestion |
| 10 | **S3 Bucket** showing `plots/`, `data/`, `mlflow-artifacts/` | AWS Console → S3 → your bucket |
| 11 | **5 EDA plots** (download from S3 and insert into doc) | S3 → plots/ folder |
| 12 | **MLflow Experiments page** showing `Titanic_Survival_Prediction` | `http://<IP>:5000` |
| 13 | **MLflow Runs table** — both models side by side | MLflow → Experiments → Compare |
| 14 | **MLflow Run detail** — all 8 metrics for Random Forest | MLflow → click on RF run |
| 15 | **MLflow Run detail** — all 8 metrics for Logistic Regression | MLflow → click on LR run |
| 16 | **MLflow Artifact** — model saved to S3 | MLflow → Run → Artifacts |
| 17 | **FastAPI Swagger UI** at `/docs` | `http://<IP>:8000/docs` |
| 18 | **`GET /airflow/dags`** — executed in Swagger, showing response | Swagger → try it out |
| 19 | **`GET /airflow/dags/{dag_id}/runs`** — JSON response | Swagger → try it out |
| 20 | **`GET /mlflow/experiment`** — showing both model metrics | Swagger → try it out |

---

## Mark Breakdown

| Component | Marks | Covered By |
|---|---|---|
| **1.1 Business Understanding** | ✅ | Titanic: predict passenger survival (binary classification) |
| **1.2 Data Ingestion** | ✅ | `ingest_data()` — reads from S3, logs shape/columns/sample |
| **1.3 Data Preprocessing** | ✅ | `preprocess_data()` — summary stats, missing check, median imputation, normalization |
| **1.4 EDA** | ✅ | `perform_eda()` — correlation matrix, encoding, binning, feature importance, 5 plots |
| **1.5 DataOps** | ✅ | Airflow DAG `*/3 * * * *`, CloudWatch logging, Airflow dashboard |
| **2.1 Model Preparation** | ✅ | Random Forest + Logistic Regression with feature engineering |
| **2.2 Model Training** | ✅ | 70/30 stratified split, 5-fold cross-validation |
| **2.3 Model Evaluation** | ✅ | Accuracy, confusion matrix, classification report |
| **2.4 MLOps** | ✅ | MLflow: 8 metrics logged (accuracy, precision, recall, F1, ROC-AUC, CV, specificity, TP/FP/FN/TN) |
| **3.1 Built-in APIs** | ✅ | Airflow REST API + MLflow API accessed via FastAPI |
| **3.2 Display Details** | ✅ | 5 endpoints: DAG list, runs, tasks, experiment, registered models |
| **Submission doc + video** | ✅ | 20 screenshots above + screen-record all 4 UIs |
| **Virtual Lab (AWS EC2)** | ✅ | Deployed on EC2 with S3 + CloudWatch = +1 bonus mark |

---

## Useful Commands Reference

```bash
# ── Manage screen sessions ────────────────────────────────────────────────────
screen -ls                          # List all sessions
screen -r airflow-web               # Reattach to Airflow webserver
screen -r mlflow                    # Reattach to MLflow
screen -r fastapi                   # Reattach to FastAPI

# ── Trigger DAG manually (without waiting 3 min) ──────────────────────────────
source ~/mlpipeline/venv/bin/activate
airflow dags trigger titanic_data_pipeline

# ── Check DAG status ──────────────────────────────────────────────────────────
airflow dags list
airflow dags state titanic_data_pipeline $(date -u +%Y-%m-%dT%H:%M:%S)

# ── View CloudWatch logs from CLI ─────────────────────────────────────────────
aws logs describe-log-groups --log-group-name-prefix /mlpipeline
aws logs get-log-events \
  --log-group-name /mlpipeline/airflow \
  --log-stream-name DataIngestion \
  --limit 20

# ── List S3 plots ─────────────────────────────────────────────────────────────
aws s3 ls s3://$S3_BUCKET/plots/
aws s3 cp s3://$S3_BUCKET/plots/ ./plots/ --recursive   # Download all plots

# ── Restart everything after EC2 reboot ──────────────────────────────────────
source ~/.bashrc
source ~/mlpipeline/venv/bin/activate
screen -S airflow-web       -dm bash -c "export AIRFLOW_HOME=~/airflow; airflow webserver --port 8080"
screen -S airflow-scheduler -dm bash -c "export AIRFLOW_HOME=~/airflow; airflow scheduler"
screen -S mlflow            -dm bash -c "mlflow server --backend-store-uri sqlite:///~/mlpipeline/mlflow.db --default-artifact-root s3://$S3_BUCKET/mlflow-artifacts --host 0.0.0.0 --port 5000"
screen -S fastapi           -dm bash -c "cd ~/mlpipeline/api && uvicorn main:app --host 0.0.0.0 --port 8000"
```

---

*CCZG506 — API-driven Cloud Native Solutions | Assignment I | BITS Pilani WILP*
