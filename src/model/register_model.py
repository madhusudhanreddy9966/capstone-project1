# register model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
from mlflow.tracking import MlflowClient

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# -------------------- AUTH SETUP (CORRECT) --------------------

dagshub_user = os.getenv("DAGSHUB_USER")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_user or not dagshub_token:
    raise EnvironmentError("DAGSHUB_USER or DAGSHUB_TOKEN is not set")

# Safety gate (KEEP this)
if not os.getenv("CAPSTONE_TEST"):
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    "https://dagshub.com/madhusudhanreddy8074/capstone-project1.mlflow"
)

# --------------------------------------------------------------


# Set up MLflow tracking URI
#mlflow.set_tracking_uri("https://dagshub.com/madhusudhanreddy8074/capstone-project1.mlflow")
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri("https://dagshub.com/ramasujith85/capstone-project.mlflow")
# dagshub.init(repo_owner="ramasujith85", repo_name="capstone-project", mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    with open(file_path, "r") as file:
        model_info = json.load(file)
    logging.info("Model info loaded from %s", file_path)
    return model_info


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = model_info["model_path"]  # this is now FULL URI

        print(f"Registering model from: {model_uri}")

        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging",
            archive_existing_versions=False
        )

        logging.info(
            f"Model {model_name} version {result.version} registered and moved to Staging"
        )

    except Exception as e:
        logging.error("Error during model registration: %s", e)
        raise


def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)

    except Exception as e:
        logging.error("Failed to complete the model registration process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()