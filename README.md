# Workshop — Jornada Cidadã PUC

Workshop prático de MLOps com foco em reprodutibilidade, rastreabilidade de dados/modelos e deploy em produção usando ferramentas gratuitas.

> **Nota:** Este workshop será expandido com novos módulos em breve.

---

## Módulo — MLOps com Predição de Churn + DVC

**Problema:** Prever se um cliente de telecom irá cancelar o serviço (churn), usando o dataset IBM Telco Customer Churn.

Este módulo combina dois pilares:

| Pilar | Ferramenta | Responsabilidade |
|---|---|---|
| Rastreamento de experimentos | MLflow | Métricas, parâmetros, Model Registry |
| Versionamento de dados/pipeline | DVC | Dataset, params, reprodutibilidade |

---

## Stack completa

| Etapa | Ferramenta |
|---|---|
| Treinamento | scikit-learn |
| Rastreamento de experimentos | MLflow (local) |
| Versionamento de dados | DVC |
| API de inferência | FastAPI |
| Frontend | Streamlit |
| CI/CD | GitHub Actions |
| Deploy | Render (free tier) |

---

## Pipeline DVC

O pipeline é definido em `dvc.yaml` e reproduzível com um único comando:

```bash
dvc repro
```

```
download → train → gate → export
```

| Estágio | Script | O que faz |
|---|---|---|
| `download` | `src/download_data.py` | Baixa o CSV do IBM Telco e salva em `data/raw/` |
| `train` | `src/train.py` | Lê `params.yaml`, treina o modelo, registra no MLflow |
| `gate` | `src/metric_gate.py` | Valida ROC-AUC vs. modelo em Staging |
| `export` | `src/export_model.py` | Exporta artefato do MLflow Staging para `model_artifact/` |

### Hiperparâmetros (`params.yaml`)

Todos os hiperparâmetros ficam em `params.yaml` — edite e rode `dvc repro` para ver o impacto:

```yaml
model:
  learning_rate: 0.5
  n_estimators: 10
  max_depth: 1
  ...
```

Compare experimentos com:

```bash
dvc params diff
dvc metrics diff
```

---

## Estrutura do projeto

```
├── dvc.yaml                  # Definição do pipeline DVC
├── params.yaml               # Hiperparâmetros do modelo
├── requirements.txt          # Dependências principais
├── requirements-api.txt      # Dependências da API
├── requirements-frontend.txt # Dependências do frontend
├── data/
│   └── raw/
│       └── telco_churn.csv   # Dataset (versionado pelo DVC)
├── metrics/
│   └── metrics.json          # Métricas geradas (roc_auc, log_loss)
├── model_artifact/           # Artefato exportado do MLflow
└── src/
    ├── download_data.py      # Estágio 1: download do dataset
    ├── train.py              # Estágio 2: treinamento + MLflow
    ├── metric_gate.py        # Estágio 3: gate de qualidade
    ├── export_model.py       # Estágio 4: export do artefato
    ├── app.py                # API FastAPI
    └── streamlit_app.py      # Interface web
```

---

## Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Inicializar o DVC (apenas na primeira vez)
dvc init

# 3. Rodar o pipeline completo
dvc repro

# 4. Ver o DAG do pipeline
dvc dag

# 5. Subir a API
uvicorn src.app:app --reload

# 6. Subir o frontend
streamlit run src/streamlit_app.py
```

---

## Fluxo CI/CD

```
Treino local → MLflow (staging) → PR aprovado → Deploy automático → Endpoint público
```

---

## Créditos

### MLOps Pipeline

Baseado e adaptado do repositório:

**[mlops-churn-probability](https://github.com/pedrohrafael/mlops-churn-probability)** — por [@pedrohrafael](https://github.com/pedrohrafael)

> Pipeline MLOps completo usando ferramentas gratuitas, do treinamento local ao deploy em produção — com foco em clareza arquitetural e reprodutibilidade para fins educacionais.

### Versionamento de Dados

Integração de versionamento de dados e pipeline utilizando:

**[DVC — Data Version Control](https://github.com/treeverse/dvc)** — por [Iterative / treeverse](https://github.com/treeverse)

> Ferramenta open-source (Apache-2.0) para versionamento de dados e modelos de ML, rastreamento de experimentos e pipelines reproduzíveis — sem depender de servidores externos.

---

## Licença

Distribuído para fins educacionais no contexto da Jornada Cidadã PUC.
