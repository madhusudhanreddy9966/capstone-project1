import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging
from mlflow.tracking import MlflowClient


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
os.environ["MLFLOW_TRACKING_TOKEN"] = dagshub_token

repo_owner = "madhusudhanreddy8074"
repo_name = "capstone-project1"

# Set up MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/madhusudhanreddy8074/capstone-project1.mlflow")
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# https://dagshub.com/madhusudhanreddy8074/capstone-project1.mlflow
# dagshub.init(repo_owner="ramasujith85", repo_name="capstone-project", mlflow=True)
# -------------------------------------------------------------------------------------


def load_model(file_path: str):
    """Load the trained model from a file."""
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    logging.info("Model loaded from %s", file_path)
    return model


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    logging.info("Data loaded from %s", file_path)
    return df


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
    }


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=4)
    logging.info("Metrics saved to %s", file_path)


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and artifact path to a JSON file."""
    model_info = {"run_id": run_id, "model_path": model_path}
    with open(file_path, "w") as file:
        json.dump(model_info, file, indent=4)
    logging.info("Model info saved to %s", file_path)


def main():
    # -------------------------------------------------------------------------
    # SAFE EXPERIMENT HANDLING (DAGSHUB COMPATIBLE)
    # -------------------------------------------------------------------------
    experiment_name = "my-dvc-pipeline-v2"
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
    # -------------------------------------------------------------------------

    with mlflow.start_run() as run:
        clf = load_model("./models/model.pkl")
        test_data = load_data("./data/processed/test_bow.csv")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model parameters to MLflow
        if hasattr(clf, "get_params"):
            for param_name, param_value in clf.get_params().items():
                mlflow.log_param(param_name, param_value)

        # 🔥 CORRECT MODEL LOGGING (IMPORTANT FIX)
        model_info = mlflow.sklearn.log_model(clf,artifact_path="model"
)

        save_model_info(
             run.info.run_id,
             model_info.model_uri,
            "reports/experiment_info.json"
)


        # Log metrics file
        mlflow.log_artifact("reports/metrics.json")


if __name__ == "__main__":
    main()