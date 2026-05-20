import time
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import mlflow.sklearn
import logging
from typing import Optional


# -------------------------------------------------
# Configuração básica de logging
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Inicialização da aplicação
# -------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description="API para previsão de churn usando modelo treinado com MLflow",
    version="1.0.0"
)


# -------------------------------------------------
# Carregamento do modelo (fail fast)
# -------------------------------------------------

try:
    model = mlflow.sklearn.load_model("model_artifact")
    logger.info("Modelo carregado com sucesso.")
except Exception as e:
    logger.exception("Falha ao carregar o modelo.")
    raise RuntimeError("Erro crítico: modelo não pôde ser carregado.") from e


# -------------------------------------------------
# Schema de entrada (validação automática)
# -------------------------------------------------

class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)


# -------------------------------------------------
# Schema de saída
# -------------------------------------------------

class ChurnOutput(BaseModel):
    churn_probability: float
    model_version: Optional[str] = None


# -------------------------------------------------
# Endpoint de saúde (obrigatório em produção)
# -------------------------------------------------




@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    start_time = time.time()

    try:
        # Payload mínimo válido
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

        _ = model.predict_proba(X)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "healthy",
            "model_loaded": True,
            "inference_ok": True,
            "latency_ms": latency_ms
        }

    except Exception as e:
        logger.exception("Health check failed")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "model_loaded": model is not None,
                "inference_ok": False
            }
        )


# -------------------------------------------------
# Endpoint de predição
# -------------------------------------------------

@app.post("/predict", response_model=ChurnOutput)
def predict(payload: ChurnInput):
    try:
        # Converte payload validado em DataFrame
        X = pd.DataFrame([payload.model_dump()])

        # Predição
        proba = model.predict_proba(X)[0, 1]

        return {
            "churn_probability": round(float(proba), 6),
            "model_version": "staging-export"
        }

    except Exception as e:
        logger.exception("Erro durante a predição.")
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao gerar predição."
        )


# -------------------------------------------------
# Tratamento global de erros de validação
# -------------------------------------------------

@app.exception_handler(ValidationError)
def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning("Erro de validação no payload.")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Payload inválido",
            "details": exc.errors()
        }
    )


# -------------------------------------------------
# Tratamento global de erros inesperados
# -------------------------------------------------

@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Erro inesperado.")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erro inesperado no servidor"
        }
    )
