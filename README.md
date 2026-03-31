# CCZG506: API-Driven Cloud Native ML Pipeline

This project implements a cloud-based data science / machine learning application using the Titanic dataset. It satisfies the assignment requirements by providing a data pipeline for ingestion, preprocessing, EDA, and monitoring; a machine learning pipeline for model training and evaluation; and an API layer to retrieve application details. The assignment requires the data workflow to run every 3 minutes, the ML pipeline to train two algorithms using a 70/30 split, and the model monitoring section to log at least four metrics. 

## 1. Business Problem

Predict whether a Titanic passenger survived based on passenger attributes such as class, sex, age, fare, family size, and embarkation port.

## 2. Tech Stack

* **Orchestration / DataOps**: Prefect Cloud
* **Experiment Tracking / MLOps**: MLflow
* **API Layer**: FastAPI
* **Storage**: Local EC2 file system
* **Libraries**: pandas, scikit-learn, matplotlib, seaborn, psutil

## 3. Project Structure

```text
mlpipeline/
├── data/
│   ├── titanic.csv
│   └── titanic_processed.csv
├── plots/
│   ├── correlation_heatmap.png
│   ├── survival_dist.png
│   ├── survival_by_sex.png
│   └── survival_by_pclass.png
├── flows/
│   └── data_pipeline.py
├── ml/
│   └── ml_pipeline.py
├── api/
│   └── main.py
├── requirements.txt
└── README.md
```

## 4. Create the Project From Scratch

### Step 1: Create all folders

```bash
mkdir -p mlpipeline/{data,plots,flows,ml,api}
cd mlpipeline
```

### Step 2: Create all files

```bash
touch requirements.txt README.md
touch data/titanic.csv
touch flows/data_pipeline.py
touch ml/ml_pipeline.py
touch api/main.py
```

### Step 3: Put the dataset in place

Copy your Titanic file into:

```text
data/titanic.csv
```

If your file is named `Titanic-Dataset.csv`, copy it like this:

```bash
cp /path/to/Titanic-Dataset.csv data/titanic.csv
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

If needed, install directly too:

```bash
pip install prefect mlflow fastapi uvicorn pandas scikit-learn matplotlib seaborn psutil
```

### Step 5: Log in to Prefect Cloud

```bash
prefect cloud login --key <YOUR_PREFECT_KEY>
```

### Step 6: Start MLflow UI

Open a new terminal and run:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open in browser:

```text
http://localhost:5000
```

### Step 7: Run the Prefect data pipeline

From the project root:

```bash
python flows/data_pipeline.py
```

### Step 8: Run the ML pipeline

After `data/titanic_processed.csv` is created:

```bash
python ml/ml_pipeline.py
```

### Step 9: Start the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Step 10: Open the API docs

```text
http://localhost:8000/docs
```

### Step 11: Test the API endpoints

```bash
curl http://127.0.0.1:8000/application-details
curl http://127.0.0.1:8000/pipeline-status
```

## 5. `requirements.txt`

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

## 6. Data Pipeline

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
    df = df.copy()

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
    logger.info("Saved processed dataset to data/titanic_processed.csv")
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

### What this pipeline does

1. reads `data/titanic.csv`
2. prints summary statistics, missing values, and data types
3. fills missing `Age` values with the median
4. fills missing `Embarked` values with the mode
5. creates `FamilySize` and `IsAlone`
6. drops `Cabin`, `Name`, `Ticket`, and `PassengerId`
7. normalizes numeric columns
8. saves `data/titanic_processed.csv`
9. creates EDA plots in `plots/`
10. schedules the flow every 3 minutes in Prefect Cloud

## 7. Machine Learning Pipeline

The assignment requires two algorithms, a 70/30 split, model evaluation, and at least four logged metrics. 

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

mlflow.set_tracking_uri("file:./mlruns")

def prepare_data():
    df = pd.read_csv("data/titanic_processed.csv")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

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

### What this pipeline does

1. loads `data/titanic_processed.csv`
2. encodes `Sex`
3. encodes `Embarked`
4. selects the final feature set
5. splits the dataset into 70% training and 30% testing
6. trains `RandomForestClassifier`
7. trains `LogisticRegression`
8. logs accuracy, precision, recall, F1, confusion matrix values, system metrics, and training time to MLflow
9. saves each trained model as an MLflow artifact

## 8. API Layer

The assignment requires built-in APIs to retrieve and display at least two application details. 

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

## 9. Exact Execution Order

### Step 1: Create folders and files

Use the commands in Section 4.

### Step 2: Put the dataset in `data/titanic.csv`

Make sure the Titanic CSV is present before running anything.

### Step 3: Install dependencies

Use the commands in Section 4.

### Step 4: Log in to Prefect Cloud

```bash
prefect cloud login --key <YOUR_PREFECT_KEY>
```

### Step 5: Start MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### Step 6: Run the Prefect data pipeline

```bash
python flows/data_pipeline.py
```

### Step 7: Verify Prefect Cloud

Open the Prefect Cloud dashboard and confirm:

* deployment name: `titanic-local-deployment`
* schedule: every 3 minutes
* flow runs are appearing
* task logs are visible for ingestion, preprocessing, and EDA

### Step 8: Run the ML pipeline

```bash
python ml/ml_pipeline.py
```

### Step 9: Verify MLflow

Open:

```text
http://localhost:5000
```

Confirm:

* `RandomForest` run
* `LogisticRegression` run
* metrics
* model artifacts

### Step 10: Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Step 11: Open API docs

```text
http://localhost:8000/docs
```

### Step 12: Test the API

```bash
curl http://127.0.0.1:8000/application-details
curl http://127.0.0.1:8000/pipeline-status
```

## 10. What Will Be Visible in Prefect Cloud

After running the data pipeline, Prefect Cloud should show:

* the deployment `titanic-local-deployment`
* a schedule that triggers every 3 minutes
* flow runs for each scheduled execution
* the task graph containing:

  * `ingest_data`
  * `preprocess`
  * `run_eda`
* logs for each task
* run status, start time, end time, and duration

This is the expected DataOps proof for the assignment. 

## 11. What to Capture for Submission

The submission should include a Word/PDF document with screenshots and explanation, plus a video demonstration. The assignment also encourages using a virtual lab, which can earn bonus credit. 

### Screenshots to include

* Prefect Cloud dashboard showing successful runs every 3 minutes
* Prefect task logs
* MLflow UI showing both models and metrics
* FastAPI `/docs` page
* API response from `/application-details`
* Folder structure showing `data/` and `plots/`

### Video should show

* the Prefect Cloud dashboard
* the data pipeline run
* the MLflow UI
* the ML pipeline run
* the FastAPI docs
* the API endpoint response

## 12. Final Checklist

* `data/titanic.csv` is present
* `plots/` exists
* Prefect Cloud login is done
* Data pipeline runs successfully
* MLflow UI opens and shows the runs
* ML pipeline completes successfully
* FastAPI server starts successfully
* API endpoints return JSON
* screenshots and video are ready for submission

## 13. Final Notes

* Run the Prefect flow first so `titanic_processed.csv` is created.
* Run the ML pipeline only after preprocessing is complete.
* Keep the MLflow UI terminal open while testing.
* Keep the Prefect Cloud flow terminal open so scheduled runs continue.
* The code above is aligned to the Titanic dataset columns and is structured to run cleanly from a fresh setup.
