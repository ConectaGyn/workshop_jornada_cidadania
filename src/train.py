import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

import category_encoders as ce


# ----------------------------------
# 1) MLflow Config
# ----------------------------------

EXPERIMENT_NAME = "telco_churn_gb"
MODEL_NAME = "telco_churn_model"
SEED = 0

mlflow.set_experiment(EXPERIMENT_NAME)


# ----------------------------------
# 2) Load dataset
# ----------------------------------

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(url)
df = df.drop('customerID', axis=1)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df = df.dropna(subset=['Churn'])
df = df.dropna(thresh=df.shape[1] - 2)


# ----------------------------------
# 3) Feature Engineering
# ----------------------------------

target_col = 'Churn'

X = df.drop(columns=[target_col])
y = df[target_col]

cat_cols = [c for c in X.columns if X[c].dtype == 'object']
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)


# ----------------------------------
# 4) Preprocessing
# ----------------------------------

cat_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=0.3)

preprocess = ColumnTransformer(
    transformers=[
        ('cat_encoder', cat_encoder, cat_cols),
    ]
)


# ----------------------------------
# 5) Modelo + Hyperparameter Search
# ----------------------------------

gb_clf = GradientBoostingClassifier(random_state=SEED)

# param_dist = {
#     'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'model__n_estimators': [50, 100, 200],
#     'model__max_depth': [1, 3, 5],
#     'model__min_samples_split': [2, 5, 10],
#     'model__min_samples_leaf': [1, 2, 4],
# }

# TESTE CI/CD: modelo propositalmente subótimo
param_dist = {
    'model__learning_rate': [0.5],     # alto demais
    'model__n_estimators': [10],       # poucos estimadores
    'model__max_depth': [1],           # árvore rasa
    'model__min_samples_split': [50],  # muito restritivo
    'model__min_samples_leaf': [20],
}


pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', gb_clf)
])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=kf,
    scoring='roc_auc',
    random_state=SEED,
    n_jobs=-1
)


# ----------------------------------
# 6) Treinamento + MLflow Tracking
# ----------------------------------

with mlflow.start_run(run_name="gb_random_search"):

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    # Log parâmetros do melhor modelo
    mlflow.log_params(random_search.best_params_)

    # Avaliação final
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_pred_proba)
    ll = log_loss(y_test, y_pred_proba)

    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("log_loss", ll)

    # Log modelo (pipeline completo)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

    print("Best params:", random_search.best_params_)
    print(f"Test ROC-AUC: {roc:.4f}")
    print(f"Test Log Loss: {ll:.4f}")
