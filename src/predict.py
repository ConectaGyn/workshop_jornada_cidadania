import pandas as pd
import mlflow
import mlflow.sklearn


# ----------------------------------
# Settings
# ----------------------------------

MODEL_NAME = "telco_churn_model"
MODEL_STAGE = "Staging"


# ----------------------------------
# Load model from MLflow Registry
# ----------------------------------

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.sklearn.load_model(model_uri)


# ----------------------------------
# Example of input data
# ----------------------------------

sample = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

X = pd.DataFrame([sample])


# ----------------------------------
# Prediction
# ----------------------------------

proba = model.predict_proba(X)[0, 1]

print(f"Probabilidade de churn: {proba:.4f}")
