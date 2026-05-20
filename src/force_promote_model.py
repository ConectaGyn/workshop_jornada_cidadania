from mlflow.tracking import MlflowClient

MODEL_NAME = "telco_churn_model"
VERSION = 2

client = MlflowClient()

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=VERSION,
    stage="Staging",
    archive_existing_versions=True
)

print(f"Modelo versão {VERSION} promovido para Staging")
