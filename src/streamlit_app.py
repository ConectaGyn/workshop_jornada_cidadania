import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO, BytesIO
import time

st.set_page_config(
    page_title="Churn Prediction — MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Constantes de domínio
# --------------------------------------------------

GENDER_OPTIONS = ["Female", "Male"]
YES_NO = ["Yes", "No"]
YES_NO_NOPHONE = ["Yes", "No", "No phone service"]
YES_NO_NOINET = ["Yes", "No", "No internet service"]
INTERNET_OPTIONS = ["DSL", "Fiber optic", "No"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60}

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("📊 Churn Prediction")
    st.caption("MLOps Demo — Telco Customer Churn")
    st.divider()

    api_url = st.text_input(
        "URL da API",
        value="http://localhost:8000",
        help="Endereço base da API FastAPI (local ou Render)",
    )
    api_url = api_url.rstrip("/")

    st.divider()
    page = st.radio(
        "Navegação",
        ["🎯 Predição Individual", "📦 Predição em Lote", "🔍 Status da API", "ℹ️ Sobre o Projeto"],
    )


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _risk_label(prob: float) -> tuple[str, str]:
    if prob < RISK_THRESHOLDS["low"]:
        return "Baixo Risco", "#28a745"
    if prob < RISK_THRESHOLDS["medium"]:
        return "Risco Moderado", "#fd7e14"
    return "Alto Risco", "#dc3545"


def _gauge(prob: float) -> go.Figure:
    label, color = _risk_label(prob)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 48}},
            title={"text": f"<b>{label}</b>", "font": {"size": 20, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin={"t": 60, "b": 0, "l": 20, "r": 20})
    return fig


def _predict(payload: dict, api_url: str) -> dict | None:
    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar à API. Verifique se ela está rodando e a URL está correta.")
    except requests.exceptions.Timeout:
        st.error("A API demorou demais para responder (timeout 15s).")
    except requests.exceptions.HTTPError as e:
        st.error(f"Erro HTTP {e.response.status_code}: {e.response.text}")
    return None


def _health(api_url: str) -> dict | None:
    try:
        resp = requests.get(f"{api_url}/health", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _customer_form(prefix: str = "") -> dict:
    """Renderiza o formulário e retorna o payload como dict."""
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("👤 Dados Pessoais")
        gender = st.selectbox("Gênero", GENDER_OPTIONS, key=f"{prefix}gender")
        senior = st.selectbox("Idoso", ["Não (0)", "Sim (1)"], key=f"{prefix}senior")
        partner = st.selectbox("Tem parceiro(a)?", YES_NO, key=f"{prefix}partner")
        dependents = st.selectbox("Tem dependentes?", YES_NO, key=f"{prefix}dependents")
        tenure = st.number_input(
            "Tempo de contrato (meses)", min_value=0, max_value=120, value=12, key=f"{prefix}tenure"
        )

    with c2:
        st.subheader("📡 Serviços")
        phone_service = st.selectbox("Serviço telefônico", YES_NO, key=f"{prefix}phone")
        multiple_lines = st.selectbox("Múltiplas linhas", YES_NO_NOPHONE, key=f"{prefix}multlines")
        internet_service = st.selectbox("Serviço de internet", INTERNET_OPTIONS, key=f"{prefix}internet")
        online_security = st.selectbox("Segurança online", YES_NO_NOINET, key=f"{prefix}security")
        online_backup = st.selectbox("Backup online", YES_NO_NOINET, key=f"{prefix}backup")
        device_protection = st.selectbox("Proteção do dispositivo", YES_NO_NOINET, key=f"{prefix}device")
        tech_support = st.selectbox("Suporte técnico", YES_NO_NOINET, key=f"{prefix}tech")
        streaming_tv = st.selectbox("Streaming TV", YES_NO_NOINET, key=f"{prefix}tv")
        streaming_movies = st.selectbox("Streaming Filmes", YES_NO_NOINET, key=f"{prefix}movies")

    with c3:
        st.subheader("💳 Contrato e Pagamento")
        contract = st.selectbox("Tipo de contrato", CONTRACT_OPTIONS, key=f"{prefix}contract")
        paperless = st.selectbox("Fatura sem papel", YES_NO, key=f"{prefix}paperless")
        payment = st.selectbox("Método de pagamento", PAYMENT_OPTIONS, key=f"{prefix}payment")
        monthly = st.number_input(
            "Mensalidade (R$)", min_value=0.01, max_value=200.0, value=65.0,
            step=0.01, format="%.2f", key=f"{prefix}monthly"
        )
        total = st.number_input(
            "Total cobrado (R$)", min_value=0.0, max_value=10000.0, value=780.0,
            step=0.01, format="%.2f", key=f"{prefix}total"
        )

    return {
        "gender": gender,
        "SeniorCitizen": int(senior.startswith("Sim")),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }


# ==================================================
# PÁGINA 1 — Predição Individual
# ==================================================

if page == "🎯 Predição Individual":
    st.title("🎯 Predição Individual de Churn")
    st.caption("Preencha os dados do cliente e clique em **Prever** para obter a probabilidade de cancelamento.")
    st.divider()

    with st.form("predict_form"):
        payload = _customer_form(prefix="ind_")
        submitted = st.form_submit_button("🔮 Prever Probabilidade de Churn", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("Consultando modelo..."):
            result = _predict(payload, api_url)

        if result:
            prob = result["churn_probability"]
            label, color = _risk_label(prob)
            version = result.get("model_version", "—")

            st.divider()
            col_gauge, col_info = st.columns([1, 1])

            with col_gauge:
                st.plotly_chart(_gauge(prob), use_container_width=True)

            with col_info:
                st.markdown(f"### Resultado")
                st.metric("Probabilidade de Churn", f"{prob * 100:.2f}%")
                st.metric("Classificação de Risco", label)
                st.metric("Versão do Modelo", version)

                st.divider()
                if prob < RISK_THRESHOLDS["low"]:
                    st.success("✅ Cliente com baixo risco de cancelamento. Manter estratégia atual de retenção.")
                elif prob < RISK_THRESHOLDS["medium"]:
                    st.warning("⚠️ Cliente com risco moderado. Considere ações proativas de retenção.")
                else:
                    st.error("🚨 Cliente com alto risco de cancelamento. Intervenção imediata recomendada.")

            with st.expander("📋 Payload enviado à API"):
                st.json(payload)


# ==================================================
# PÁGINA 2 — Predição em Lote
# ==================================================

elif page == "📦 Predição em Lote":
    st.title("📦 Predição em Lote")
    st.caption("Faça upload de um CSV com múltiplos clientes e obtenha as probabilidades de churn para todos.")
    st.divider()

    REQUIRED_COLS = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]

    with st.expander("📄 Formato esperado do CSV"):
        sample_data = {
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 34],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [29.85, 1889.5],
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

        csv_bytes = sample_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Baixar CSV de exemplo",
            data=csv_bytes,
            file_name="clientes_exemplo.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Erro ao ler o CSV: {e}")
            st.stop()

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Colunas faltando no CSV: {missing}")
            st.stop()

        st.success(f"{len(df)} clientes carregados.")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("🔮 Prever para todos os clientes", type="primary", use_container_width=True):
            results = []
            errors = []
            progress = st.progress(0, text="Enviando predições...")

            for i, row in df.iterrows():
                payload = row[REQUIRED_COLS].to_dict()
                payload["SeniorCitizen"] = int(payload["SeniorCitizen"])
                payload["tenure"] = int(payload["tenure"])
                payload["MonthlyCharges"] = float(payload["MonthlyCharges"])
                payload["TotalCharges"] = float(payload["TotalCharges"])

                res = _predict(payload, api_url)
                if res:
                    results.append(res["churn_probability"])
                else:
                    results.append(None)
                    errors.append(i)

                progress.progress((i + 1) / len(df), text=f"Processando cliente {i + 1}/{len(df)}...")

            progress.empty()

            df_result = df.copy()
            df_result["churn_probability"] = results
            df_result["churn_pct"] = df_result["churn_probability"].apply(
                lambda x: f"{x * 100:.2f}%" if x is not None else "Erro"
            )
            df_result["risco"] = df_result["churn_probability"].apply(
                lambda x: _risk_label(x)[0] if x is not None else "Erro"
            )

            if errors:
                st.warning(f"⚠️ {len(errors)} clientes com erro de predição (linhas: {errors})")

            st.divider()
            st.subheader("📊 Resultados")

            valid = df_result["churn_probability"].dropna()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total de clientes", len(df_result))
            m2.metric("Alto risco (≥60%)", int((valid >= 0.60).sum()))
            m3.metric("Risco moderado (30–60%)", int(((valid >= 0.30) & (valid < 0.60)).sum()))
            m4.metric("Baixo risco (<30%)", int((valid < 0.30).sum()))

            col_dist, col_table = st.columns([1, 1])

            with col_dist:
                fig = px.histogram(
                    valid * 100,
                    nbins=20,
                    labels={"value": "Probabilidade de Churn (%)"},
                    title="Distribuição das Probabilidades",
                    color_discrete_sequence=["#4C72B0"],
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                risk_counts = df_result["risco"].value_counts().reset_index()
                risk_counts.columns = ["Risco", "Clientes"]
                colors = {
                    "Alto Risco": "#dc3545",
                    "Risco Moderado": "#fd7e14",
                    "Baixo Risco": "#28a745",
                    "Erro": "#6c757d",
                }
                fig2 = px.pie(
                    risk_counts,
                    values="Clientes",
                    names="Risco",
                    title="Distribuição por Nível de Risco",
                    color="Risco",
                    color_discrete_map=colors,
                )
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)

            display_cols = ["churn_pct", "risco"] + [c for c in REQUIRED_COLS if c in df_result.columns]
            st.dataframe(df_result[display_cols], use_container_width=True)

            csv_out = df_result.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Baixar resultados completos (CSV)",
                data=csv_out,
                file_name="predicoes_churn.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ==================================================
# PÁGINA 3 — Status da API
# ==================================================

elif page == "🔍 Status da API":
    st.title("🔍 Status da API")
    st.caption(f"Monitoramento do endpoint `{api_url}`")
    st.divider()

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        check = st.button("🔄 Verificar agora", type="primary", use_container_width=True)

    auto_refresh = st.checkbox("Atualizar automaticamente a cada 30s")

    if auto_refresh:
        time.sleep(0)
        st.rerun() if "last_check" in st.session_state and (time.time() - st.session_state.last_check) > 30 else None

    if check or auto_refresh:
        st.session_state.last_check = time.time()

        with st.spinner("Verificando API..."):
            health = _health(api_url)

        st.divider()

        if health is None:
            st.error(f"❌ API indisponível em `{api_url}`")
            st.info("Verifique se a API está rodando: `uvicorn src.app:app --reload`")
        else:
            status = health.get("status", "unknown")
            model_loaded = health.get("model_loaded", False)
            inference_ok = health.get("inference_ok", False)
            latency = health.get("latency_ms", None)

            if status == "healthy":
                st.success("✅ API Saudável")
            else:
                st.error("❌ API com problemas")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", status.upper())
            c2.metric("Modelo carregado", "✅ Sim" if model_loaded else "❌ Não")
            c3.metric("Inferência OK", "✅ Sim" if inference_ok else "❌ Não")
            c4.metric("Latência", f"{latency} ms" if latency else "—")

            st.divider()
            st.subheader("📋 Resposta completa")
            st.json(health)

    st.divider()
    st.subheader("🔗 Endpoints disponíveis")
    st.markdown(f"""
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | [`{api_url}/health`]({api_url}/health) | Health check com latência real |
| `POST` | `{api_url}/predict` | Predição de churn (JSON) |
| `GET` | [`{api_url}/docs`]({api_url}/docs) | Swagger UI (documentação interativa) |
| `GET` | [`{api_url}/redoc`]({api_url}/redoc) | ReDoc (documentação alternativa) |
""")


# ==================================================
# PÁGINA 4 — Sobre o Projeto
# ==================================================

elif page == "ℹ️ Sobre o Projeto":
    st.title("ℹ️ Sobre o Projeto")
    st.divider()

    st.markdown("""
## MLOps Churn Prediction — Free Tier

Este projeto demonstra um **fluxo completo de MLOps** utilizando apenas ferramentas gratuitas.
O objetivo é mostrar boas práticas reais sem custo de infraestrutura.
""")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Problema de Negócio")
        st.markdown("""
- **Dataset**: IBM Telco Customer Churn
- **Tarefa**: Classificação binária (churn / não churn)
- **Objetivo**: Prever a probabilidade de um cliente cancelar o serviço

**Por que isso importa?**
Adquirir um novo cliente custa até **5x mais** do que reter um existente.
Identificar clientes em risco antecipadamente permite ações proativas de retenção.
""")

        st.subheader("🤖 Modelo")
        st.markdown("""
| Componente | Detalhe |
|-----------|---------|
| Algoritmo | Gradient Boosting Classifier |
| Busca de hiperparâmetros | RandomizedSearchCV |
| Validação cruzada | StratifiedKFold (5 folds) |
| Encoding | TargetEncoder (variáveis categóricas) |
| Métrica principal | ROC-AUC |
| Gate automático | Sem regressão vs. Staging |
""")

    with col2:
        st.subheader("🏗️ Arquitetura")
        st.code("""
┌─────────────────────┐
│   Ambiente Local    │
│─────────────────────│
│  Treino (sklearn)   │
│  MLflow Tracking    │
│  Model Registry     │
└──────────┬──────────┘
           │ export
           ▼
┌─────────────────────┐
│  GitHub (develop)   │
│─────────────────────│
│  Modelo aprovado    │
│  Gate automático    │
│  CI/CD Actions      │
└──────────┬──────────┘
           │ PR → main
           ▼
┌─────────────────────┐
│  Cloud (Render)     │
│─────────────────────│
│  FastAPI            │
│  /health /predict   │
└─────────────────────┘
        """, language="text")

        st.subheader("🛠️ Stack Tecnológica")
        st.markdown("""
| Etapa | Ferramenta |
|-------|-----------|
| ML Core | scikit-learn |
| Experimentação | MLflow |
| Feature Encoding | category-encoders |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| CI/CD | GitHub Actions |
| Deploy | Render (Free Tier) |
""")

    st.divider()
    st.subheader("🔄 Pipeline CI/CD")

    tab1, tab2 = st.tabs(["Branch develop (Treino)", "Branch main (Deploy)"])
    with tab1:
        st.markdown("""
**Trigger**: Push em `src/train.py` ou toda segunda às 03h UTC

```
1. pip install -r requirements.txt
2. python src/train.py          → Treina e registra no MLflow
3. python src/metric_gate.py   → Valida ROC-AUC (sem regressão)
4. Promoção automática → Staging
5. python src/export_model.py  → Exporta model_artifact/
6. git commit + push            → Versiona o artefato
```
""")
    with tab2:
        st.markdown("""
**Trigger**: Merge de `develop` → `main`

```
1. Trigger Render deploy via API
   └─ Header: Authorization: Bearer $RENDER_API_KEY
   └─ Service ID: $RENDER_SERVICE_ID
2. Render faz pull do main e reinicia a API
```
""")

    st.divider()
    st.subheader("📊 Interpretação dos Resultados")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("🟢 **Baixo Risco** (< 30%)\nCliente estável. Manter estratégia de engajamento atual.")
    with c2:
        st.warning("🟡 **Risco Moderado** (30–60%)\nSinal de alerta. Considere ofertas personalizadas ou contato proativo.")
    with c3:
        st.error("🔴 **Alto Risco** (≥ 60%)\nIntervenção imediata. Priorize ações de retenção urgentes.")

    st.divider()
    st.caption("Projeto desenvolvido como demonstração de MLOps com free tier. GitHub: mlops-churn-probability")
