# CCZG506: API-Driven Cloud Native ML Pipeline

## 1. Project Overview

This project builds a cloud-based data science / machine learning application using the Titanic dataset. It fulfills the three assignment objectives: a data pipeline, a machine learning pipeline, and API access. The assignment also requires the data pipeline to be scheduled every 3 minutes, the ML pipeline to use two algorithms with a 70/30 train-test split, and the model monitoring section to log at least four metrics. 

## 2. What This Project Does

* Ingests the Titanic dataset from a local `data/` folder
* Cleans and preprocesses the dataset
* Performs exploratory data analysis and saves charts in `plots/`
* Runs the data pipeline on a 3-minute schedule through Prefect Cloud
* Trains and evaluates two machine learning models
* Logs metrics and artifacts to MLflow
* Exposes application details through a FastAPI endpoint

## 3. Tech Stack

* **Orchestration / DataOps**: Prefect Cloud
* **Experiment Tracking / MLOps**: MLflow
* **API Layer**: FastAPI
* **Storage**: Local EC2 file system
* **Libraries**: pandas, scikit-learn, matplotlib, seaborn, psutil

## 4. Repository Structure

```text
mlpipeline/
├── data/
│   ├── titanic.csv
│   └── titanic_processed.csv
├── plots/
│   ├── correlation_heatmap.png
│   └── survival_dist.png
├── flows/
│   └── data_pipeline.py
├── ml/
│   └── ml_pipeline.py
├── api/
│   └── main.py
├── requirements.txt
└── README.md
```

## 5. Dataset Setup

1. Download the Titanic dataset from a public source such as Kaggle.
2. Rename it to `titanic.csv`.
3. Place it inside the `data/` folder.

Your project should read the raw data from:

```text
data/titanic.csv
```

## 6. Installation Steps

### Step 1: Create the folders

Run this in the project root:

```bash
mkdir -p data plots
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Prefect and MLflow if needed separately

```bash
pip install prefect mlflow
```

## 7. `requirements.txt`

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

## 8. Running the Project

## 8.1 Prefect Cloud Setup

This project uses Prefect Cloud for orchestration. The data pipeline should be registered and scheduled from the EC2 environment, and the run history should appear in the Prefect Cloud dashboard. The assignment explicitly asks for automation every 3 minutes and cloud dashboard visibility. 

### What to do in Prefect Cloud

1. Create or open your Prefect Cloud workspace.
2. Copy your Prefect API key / workspace credentials.
3. In the EC2 terminal, log in to Prefect Cloud:

   ```bash
   prefect cloud login --key <YOUR_PREFECT_KEY>
   ```
4. Confirm the login is successful.
5. Keep the deployment terminal open after starting the flow so the schedule can keep running.
6. Open the Prefect Cloud dashboard and verify that the flow runs appear every 3 minutes.

### What to do on EC2

In the project root, run:

```bash
python flows/data_pipeline.py
```

This should register the flow and start the scheduled execution.

## 8.2 MLflow Setup

The ML pipeline logs model metrics, parameters, and artifacts to MLflow.

### What to do in Terminal 1

Start MLflow on port 5000:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### What to check in the browser

Open:

```text
http://localhost:5000
```

You should see the experiment runs, logged metrics, and saved models.

## 8.3 Run the ML Pipeline

### What to do in Terminal 2

Execute the ML training script:

```bash
python ml/ml_pipeline.py
```

This will:

* load `data/titanic_processed.csv`
* split the data into train and test sets
* train two models
* log accuracy, precision, recall, and F1 score
* save the trained model to MLflow

## 8.4 Run the API

### What to do in Terminal 3

Start the FastAPI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Open the API documentation

Visit:

```text
http://localhost:8000/docs
```

## 9. Project Files

## 9.1 `flows/data_pipeline.py`

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

    logger.info(f"Summary Statistics:\n{df.describe().to_string()}")
    logger.info(f"Missing Values:\n{df.isnull().sum().to_string()}")
    logger.info(f"Data Types:\n{df.dtypes.to_string()}")

    df["Age"] = df["Age"].fillna(df["Age"].median())

    scaler = MinMaxScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

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

    return "EDA charts saved to plots/"

@flow(name="Titanic-DataOps-Local")
def data_ops_flow():
    raw_data = ingest_data()
    clean_data = preprocess(raw_data)
    run_eda(clean_data)

if __name__ == "__main__":
    data_ops_flow.serve(name="titanic-local-deployment", cron="*/3 * * * *")
```

## 9.2 `ml/ml_pipeline.py`

```python
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_models():
    df = pd.read_csv("data/titanic_processed.csv")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(max_depth=3, n_estimators=100),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    mlflow.set_experiment("Titanic_Survival_Local")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

            if name == "RandomForest":
                mlflow.log_param("max_depth", 3)

            mlflow.sklearn.log_model(model, "model")
            print(f"Successfully logged {name} to MLflow.")

if __name__ == "__main__":
    train_models()
```

## 9.3 `api/main.py`

```python
from fastapi import FastAPI
import mlflow.tracking
import mlflow

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

## 10. What to Demonstrate in Screenshots / Video

The submission should include a Word/PDF document with screenshots and explanation, plus a video showing the full workflow. The assignment also encourages using a virtual lab, which can earn bonus credit. 

### Required evidence

* Prefect Cloud dashboard showing the scheduled flow
* MLflow UI showing the two model runs and logged metrics
* Terminal showing the data pipeline running
* Terminal showing the ML pipeline execution
* FastAPI `/docs` page
* API output from `/application-details`
* Folder structure showing `data/` and `plots/`

## 11. API Checks

### Check application details

```bash
curl http://127.0.0.1:8000/application-details
```

### Check pipeline status

```bash
curl http://127.0.0.1:8000/pipeline-status
```

## 12. Submission Checklist

* Word/PDF document with project explanation and screenshots
* Video demonstration of the complete project
* Prefect Cloud dashboard proof
* MLflow experiment screenshots
* Clear mention of group member contribution
* Evidence of virtual lab usage, if used

## 13. Notes

* Keep the `data/` and `plots/` folders in the project root.
* Make sure `titanic.csv` is present before running the pipeline.
* Run the Prefect flow first so that `titanic_processed.csv` is created before ML training.
* Run the ML pipeline only after the data pipeline has completed successfully.

---
