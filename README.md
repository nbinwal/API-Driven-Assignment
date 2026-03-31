# CCZG506: API-Driven Cloud Native ML Pipeline

## 1. Project Overview

This project implements a cloud-based data science / machine learning application using the Titanic dataset. It satisfies the assignment requirements by providing: a data pipeline for ingestion, preprocessing, EDA, and monitoring; a machine learning pipeline for model training and evaluation; and an API layer to retrieve application details. The assignment requires the data workflow to run every 3 minutes, the ML pipeline to use two algorithms with a 70/30 split, and the model monitoring section to log at least four metrics. 

## 2. Business Problem

The business objective is to predict whether a Titanic passenger survived based on passenger features such as class, sex, age, family structure, fare, and embarkation port. This is a classification problem and is suitable for comparing multiple supervised learning models.

## 3. Dataset Details

The dataset used is the Titanic dataset with 891 rows and 12 columns. The most important columns for modeling are `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`. The dataset contains missing values mainly in `Cabin`, `Age`, and `Embarked`, so the preprocessing step must handle these carefully.

## 4. Tech Stack

* **Orchestration / DataOps**: Prefect Cloud
* **Experiment Tracking / MLOps**: MLflow
* **API Layer**: FastAPI
* **Storage**: Local EC2 file system
* **Libraries**: pandas, scikit-learn, matplotlib, seaborn, psutil

## 5. Repository Structure

```text
mlpipeline/
├── data/
│   ├── titanic.csv
│   └── titanic_processed.csv
├── plots/
│   ├── correlation_heatmap.png
│   ├── survival_by_sex.png
│   ├── survival_by_pclass.png
│   ├── survival_dist.png
│   └── confusion_matrix.png
├── flows/
│   └── data_pipeline.py
├── ml/
│   └── ml_pipeline.py
├── api/
│   └── main.py
├── requirements.txt
└── README.md
```

## 6. Setup Instructions

### 6.1 Create folders

```bash
mkdir -p data plots
```

### 6.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 6.3 Place the dataset

Put the Titanic CSV file here:

```text
data/titanic.csv
```

## 7. Code Adjustments for This Dataset

This dataset needs a few small updates compared to the original COVID-based reference code:

1. `Age` has missing values, so fill them with the median.
2. `Embarked` has missing values, so fill them with the mode.
3. `Cabin` has many missing values, so drop it for modeling.
4. `Name`, `Ticket`, and `PassengerId` are not useful for baseline modeling, so drop them.
5. Create a simple engineered feature such as `FamilySize = SibSp + Parch + 1`.
6. Use `Sex` and `Embarked` encoding before model training.
7. Add at least one bivariate chart in EDA, such as survival by sex or class.
8. Log confusion matrix and system metrics to make the MLOps section stronger.

## 8. `requirements.txt`

```text
prefect==2.19.9
mlflow==2.13.0
fastapi==0.111.0
uvicorn==0.29.0
pandas==2.2.2
scikit-learn==1.4.2
matplotlib==3.9.0
seaborn==0.13.2
psutil
```

## 9. Project Execution Order

Run the project in this order so each component has the files it needs.

### Step 1: Start the MLflow UI

Open a terminal and run:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Then open:

```text
http://localhost:5000
```

### Step 2: Log in to Prefect Cloud

Before starting the flow, authenticate Prefect on the EC2 instance:

```bash
prefect cloud login --key <YOUR_PREFECT_KEY>
```

Make sure the login is successful and the workspace is connected.

### Step 3: Run the Prefect data pipeline

From the project root, run:

```bash
python flows/data_pipeline.py
```

This will:

* read `data/titanic.csv`
* preprocess the data
* save `data/titanic_processed.csv`
* generate EDA plots in `plots/`
* schedule the flow to run every 3 minutes
* send execution logs to Prefect Cloud

### Step 4: Run the ML training script

After the processed CSV is available, run:

```bash
python ml/ml_pipeline.py
```

This will:

* load `data/titanic_processed.csv`
* split the data into 70% training and 30% testing sets
* train two models
* log metrics and parameters to MLflow
* save the trained models in MLflow

### Step 5: Start the API server

Run:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/docs
```

## 10. Data Pipeline

The assignment requires data ingestion, preprocessing, EDA, and automation every 3 minutes with activity logs visible on a cloud dashboard. 

### `flows/data_pipeline.py`

```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from prefect import flow, task, get_run_logger

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

@task(name="Data Ingestion")
def ingest_data():
    logger = get_run_logger()
    df = pd.read_csv("data/titanic.csv")
    logger.info(f"Ingested {len(df)} records from local storage.")
    return df

@task(name="Data Pre-processing")
def preprocess(df):
    logger = get_run_logger()

    logger.info(f"Summary Statistics:\n{df.describe(include='all').to_string()}")
    logger.info(f"Missing Values:\n{df.isnull().sum().to_string()}")
    logger.info(f"Data Types:\n{df.dtypes.to_string()}")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

    scaler = MinMaxScaler()
    df[["Age", "Fare", "SibSp", "Parch", "FamilySize"]] = scaler.fit_transform(
        df[["Age", "Fare", "SibSp", "Parch", "FamilySize"]]
    )

    df.to_csv("data/titanic_processed.csv", index=False)
    return df

@task(name="Exploratory Data Analysis")
def run_eda(df):
    numeric_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    df["Survived"].value_counts().plot(kind="bar")
    plt.title("Survival Distribution")
    plt.tight_layout()
    plt.savefig("plots/survival_dist.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Sex", hue="Survived")
    plt.title("Survival by Sex")
    plt.tight_layout()
    plt.savefig("plots/survival_by_sex.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Pclass", hue="Survived")
    plt.title("Survival by Passenger Class")
    plt.tight_layout()
    plt.savefig("plots/survival_by_pclass.png")
    plt.close()

    return "EDA charts saved to plots/"

@flow(name="Titanic-DataOps-Local")
def data_ops_flow():
    raw_data = ingest_data()
    clean_data = preprocess(raw_data)
    run_eda(clean_data)

if __name__ == "__main__":
    data_ops_flow.serve(name="titanic-local-deployment", cron="*/3 * * * *")
```

## 11. Machine Learning Pipeline

The assignment requires two algorithms, a 70/30 split, and at least four logged metrics. 

### `ml/ml_pipeline.py`

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import psutil
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def prepare_data():
    df = pd.read_csv("data/titanic_processed.csv")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"
    ]
    feature_cols += [c for c in df.columns if c.startswith("Embarked_")]

    X = df[feature_cols]
    y = df["Survived"]
    return X, y

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        start_time = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        end_time = time.time()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        confusion = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time_seconds", end_time - start_time)

        mlflow.log_metric("true_positive", int(confusion[1][1]))
        mlflow.log_metric("false_positive", int(confusion[0][1]))
        mlflow.log_metric("true_negative", int(confusion[0][0]))
        mlflow.log_metric("false_negative", int(confusion[1][0]))

        mlflow.log_metric("system_cpu_usage", psutil.cpu_percent(interval=1))
        mlflow.log_metric("system_memory_usage", psutil.virtual_memory().percent)

        if model_name == "RandomForest":
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("n_estimators", model.n_estimators)
        else:
            mlflow.log_param("max_iter", model.max_iter)

        mlflow.sklearn.log_model(model, "model")

        print(f"Successfully logged {model_name} to MLflow.")
        print(f"Accuracy: {accuracy:.4f}")

def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    mlflow.set_experiment("Titanic_Survival_Local")

    models = {
        "RandomForest": RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    for model_name, model in models.items():
        train_and_log_model(model_name, model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
```

## 12. API Layer

The assignment requires using built-in APIs to retrieve and display at least two application details. 

### `api/main.py`

```python
from fastapi import FastAPI
import mlflow
import mlflow.tracking

app = FastAPI(title="Titanic Local ML Application API")
client = mlflow.tracking.MlflowClient()

@app.get("/application-details")
def get_details():
    experiment = client.get_experiment_by_name("Titanic_Survival_Local")

    return {
        "Application_Objective": "Titanic Survival Prediction",
        "Storage_Type": "EC2 Local File System",
        "MLflow_Experiment_ID": experiment.experiment_id if experiment else "Not Initialized",
        "MLflow_Tracking_URI": mlflow.get_tracking_uri(),
        "Artifact_Location": experiment.artifact_location if experiment else "N/A"
    }

@app.get("/pipeline-status")
def get_status():
    return {
        "Status": "Operational",
        "DataOps_Frequency": "Every 3 Minutes"
    }
```

## 13. Useful Commands

### Open MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### Run the data pipeline

```bash
python flows/data_pipeline.py
```

### Run the ML pipeline

```bash
python ml/ml_pipeline.py
```

### Run the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Test API endpoints

```bash
curl http://127.0.0.1:8000/application-details
curl http://127.0.0.1:8000/pipeline-status
```

## 14. What to Capture for Submission

The submission must include a Word/PDF document with screenshots and explanation, plus a video demonstration. The assignment also encourages using a virtual lab, which can earn bonus credit. 

### Recommended screenshots

* Prefect Cloud dashboard showing successful runs every 3 minutes
* Data pipeline terminal output
* MLflow UI showing both trained models and metrics
* FastAPI `/docs` page
* API response from `/application-details`
* Project folder structure showing `data/` and `plots/`

## 15. Submission Checklist

* Word/PDF document with explanation and screenshots
* Video demo of the entire workflow
* Prefect Cloud proof of scheduled execution
* MLflow experiment proof
* API demo proof
* Virtual lab proof if available

## 16. Final Notes

* Keep `titanic.csv` in the `data/` folder before running anything.
* Run the Prefect flow first so the processed dataset is created.
* Run the ML pipeline after preprocessing completes.
* Use Prefect Cloud for flow visibility and scheduling.
* Use MLflow UI for model comparison and metric tracking.
