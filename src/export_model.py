import mlflow
import mlflow.sklearn
import os
import shutil

MODEL_NAME = "telco_churn_model"
MODEL_STAGE = "Staging"
EXPORT_PATH = "model_artifact"

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# Remove diretório anterior, se existir
if os.path.exists(EXPORT_PATH):
    shutil.rmtree(EXPORT_PATH)

model = mlflow.sklearn.load_model(model_uri)

mlflow.sklearn.save_model(
    sk_model=model,
    path=EXPORT_PATH
)

print("Modelo exportado com sucesso.")
