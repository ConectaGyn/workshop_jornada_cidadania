# Workshop — Jornada Cidadã PUC

Workshop prático de MLOps com foco em reprodutibilidade, rastreabilidade de modelos e deploy em produção usando ferramentas gratuitas.

---

## Sobre este workshop

Este material foi desenvolvido para a **Jornada Cidadã da PUC** com o objetivo de apresentar, de forma acessível, um pipeline completo de Machine Learning em produção — desde o treinamento local até a inferência em nuvem.

> **Nota:** Este workshop será expandido com novos módulos em breve.

---

## Módulo atual — MLOps com Predição de Churn

**Problema:** Prever se um cliente de telecom irá cancelar o serviço (churn), usando o dataset IBM Telco Customer Churn.

**Tecnologias utilizadas:**

| Etapa | Ferramenta |
|---|---|
| Treinamento | scikit-learn |
| Rastreamento de experimentos | MLflow (local) |
| API de inferência | FastAPI |
| Frontend | Streamlit |
| CI/CD | GitHub Actions |
| Deploy | Render (free tier) |

**Fluxo do pipeline:**

```
Treino local → MLflow (staging) → PR aprovado → Deploy automático → Endpoint público
```

### Estrutura do projeto

```
src/
├── train.py              # Treinamento e registro no MLflow
├── predict.py            # Lógica de predição
├── app.py                # API FastAPI
├── streamlit_app.py      # Interface web
├── export_model.py       # Exporta artefato para deploy
├── metric_gate.py        # Valida métricas antes de promover modelo
└── force_promote_model.py
model_artifact/           # Artefato do modelo treinado
```

### Como executar localmente

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar o modelo
python src/train.py

# Subir a API
uvicorn src.app:app --reload

# Subir o frontend
streamlit run src/streamlit_app.py
```

---

## Créditos

Este módulo de MLOps é baseado e adaptado do repositório:

**[mlops-churn-probability](https://github.com/pedrohrafael/mlops-churn-probability)** — por [@pedrohrafael](https://github.com/pedrohrafael)

> Pipeline MLOps completo usando ferramentas gratuitas, do treinamento local ao deploy em produção — com foco em clareza arquitetural e reprodutibilidade para fins educacionais.

---

## Licença

Distribuído para fins educacionais no contexto da Jornada Cidadã PUC.
