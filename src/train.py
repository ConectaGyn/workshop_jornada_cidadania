import json
import os

import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

import category_encoders as ce


# ----------------------------------
# 1) Carrega parâmetros do DVC
# ----------------------------------

with open("params.yaml") as f:
    p = yaml.safe_load(f)

EXPERIMENT_NAME = p["train"]["experiment_name"]
MODEL_NAME      = p["train"]["model_name"]
SEED            = p["train"]["seed"]

mlflow.set_experiment(EXPERIMENT_NAME)


# ----------------------------------
# 2) Carrega dataset versionado
# ----------------------------------

DATA_PATH = "data/raw/telco_churn.csv"

df = pd.read_csv(DATA_PATH)
df = df.drop("customerID", axis=1)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["Churn"])
df = df.dropna(thresh=df.shape[1] - 2)


# ----------------------------------
# 3) Feature Engineering
# ----------------------------------

target_col = "Churn"
X = df.drop(columns=[target_col])
y = df[target_col]

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=p["train"]["test_size"],
    random_state=SEED,
    stratify=y,
)


# ----------------------------------
# 4) Pré-processamento
# ----------------------------------

cat_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=0.3)

preprocess = ColumnTransformer(
    transformers=[("cat_encoder", cat_encoder, cat_cols)]
)


# ----------------------------------
# 5) Modelo + busca de hiperparâmetros (via params.yaml)
# ----------------------------------

gb_clf = GradientBoostingClassifier(random_state=SEED)

param_dist = {
    "model__learning_rate":    [p["model"]["learning_rate"]],
    "model__n_estimators":     [p["model"]["n_estimators"]],
    "model__max_depth":        [p["model"]["max_depth"]],
    "model__min_samples_split":[p["model"]["min_samples_split"]],
    "model__min_samples_leaf": [p["model"]["min_samples_leaf"]],
}

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", gb_clf),
])

kf = StratifiedKFold(
    n_splits=p["train"]["n_splits"],
    shuffle=True,
    random_state=SEED,
)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=p["train"]["n_iter"],
    cv=kf,
    scoring="roc_auc",
    random_state=SEED,
    n_jobs=-1,
)


# ----------------------------------
# 6) Treinamento + MLflow Tracking
# ----------------------------------

with mlflow.start_run(run_name="gb_random_search"):

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    mlflow.log_params(random_search.best_params_)

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_pred_proba)
    ll  = log_loss(y_test, y_pred_proba)

    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("log_loss", ll)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )

    print("Best params:", random_search.best_params_)
    print(f"Test ROC-AUC: {roc:.4f}")
    print(f"Test Log Loss: {ll:.4f}")


# ----------------------------------
# 7) Salva métricas para o DVC
# ----------------------------------

os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump({"roc_auc": round(roc, 6), "log_loss": round(ll, 6)}, f, indent=2)

print("Métricas salvas em metrics/metrics.json")
