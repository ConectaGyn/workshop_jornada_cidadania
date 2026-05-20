import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "telco_churn_model"
METRIC_NAME = "roc_auc"
MIN_DELTA = 0.0  # exige ser >= modelo atual

client = MlflowClient()

# Última run (novo modelo)
experiment = client.get_experiment_by_name("telco_churn_gb")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

new_run = runs[0]
new_metric = new_run.data.metrics.get(METRIC_NAME)

if new_metric is None:
    raise RuntimeError("Nova run não contém a métrica necessária.")

print(f"Nova métrica {METRIC_NAME}: {new_metric:.4f}")

# Buscar modelo atual em Staging (se existir)
staging_versions = client.get_latest_versions(
    MODEL_NAME, stages=["Staging"]
)

if not staging_versions:
    print("Nenhum modelo em Staging. Gate liberado.")
    exit(0)

current_version = staging_versions[0]
current_run = client.get_run(current_version.run_id)
current_metric = current_run.data.metrics.get(METRIC_NAME)

if current_metric is None:
    raise RuntimeError("Modelo atual em Staging não possui a métrica.")

print(f"Métrica atual em Staging: {current_metric:.4f}")

# Comparação
if new_metric + MIN_DELTA >= current_metric:
    print("Gate aprovado. Novo modelo é melhor ou equivalente.")
    exit(0)
else:
    raise RuntimeError(
        f"Gate reprovado: {new_metric:.4f} < {current_metric:.4f}"
    )
