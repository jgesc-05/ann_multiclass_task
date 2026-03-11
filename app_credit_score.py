import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Prediccion Credit Score", page_icon="💳", layout="wide")


# Canonical base features expected before one-hot Occupation columns.
BASE_FEATURES = [
    "Age",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance",
    "Num_of_Loans_Total",
]

# Typical occupation values for the dataset; with drop_first one category is omitted.
OCCUPATION_RAW_VALUES = [
    "Architect",
    "Developer",
    "Doctor",
    "Engineer",
    "Entrepreneur",
    "Journalist",
    "Lawyer",
    "Manager",
    "Mechanic",
    "Media_Manager",
    "Musician",
    "Scientist",
    "Teacher",
    "Writer",
]

PAYMENT_BEHAVIOUR_MAP = {
    "Low_spent_Small_value_payments": 0,
    "High_spent_Medium_value_payments": 1,
    "High_spent_Large_value_payments": 2,
    "Low_spent_Medium_value_payments": 3,
    "High_spent_Small_value_payments": 4,
    "Low_spent_Large_value_payments": 5,
}

PAYMENT_BEHAVIOUR_OPTIONS = list(PAYMENT_BEHAVIOUR_MAP.keys())

CREDIT_MIX_LABELS = {
    "Bad": "Mala",
    "Standard": "Estándar",
    "Good": "Buena",
}

MIN_PAYMENT_LABELS = {
    "No": "No",
    "Yes": "Sí",
}

PAYMENT_BEHAVIOUR_LABELS = {
    "Low_spent_Small_value_payments": "Gasto bajo y pagos pequeños",
    "High_spent_Medium_value_payments": "Gasto alto y pagos medianos",
    "High_spent_Large_value_payments": "Gasto alto y pagos grandes",
    "Low_spent_Medium_value_payments": "Gasto bajo y pagos medianos",
    "High_spent_Small_value_payments": "Gasto alto y pagos pequeños",
    "Low_spent_Large_value_payments": "Gasto bajo y pagos grandes",
}

PROFESION_LABEL_TO_RAW = {
    "Arquitecto": "Architect",
    "Desarrollador": "Developer",
    "Doctor": "Doctor",
    "Ingeniero": "Engineer",
    "Emprendedor": "Entrepreneur",
    "Periodista": "Journalist",
    "Abogado": "Lawyer",
    "Director": "Manager",
    "Mecánico": "Mechanic",
    "Director de medios": "Media_Manager",
    "Músico": "Musician",
    "Científico": "Scientist",
    "Profesor": "Teacher",
    "Escritor": "Writer",
}

MODEL_CANDIDATES = [
    "modelo_credit_score.keras",
    "modelo_credit_score (1).keras",
]

SCALER_CANDIDATES = [
    "scaler_credit_score.joblib",
    "scaler_credit_score (1).joblib",
]

PCA_CANDIDATES = [
    "pca_credit_score.joblib",
    "pca_credit_score (1).joblib",
]


FEATURE_COLUMNS_CANDIDATES = [
    "feature_columns_credit_score.joblib"
]


def find_first_existing(candidates):
    for name in candidates:
        path = Path(name)
        if path.exists():
            return path
    return None


def default_feature_columns():
    default_occupation_dummies = [f"Occupation_{v}" for v in OCCUPATION_RAW_VALUES]
    return BASE_FEATURES + default_occupation_dummies


def load_feature_columns():
    path = find_first_existing(FEATURE_COLUMNS_CANDIDATES)
    if path is None:
        return default_feature_columns()

    if path.suffix == ".joblib":
        loaded = joblib.load(path)
        return list(loaded)

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        return list(loaded)

    return default_feature_columns()


@st.cache_resource
def load_artifacts():
    model_path = find_first_existing(MODEL_CANDIDATES)
    if model_path is None:
        raise FileNotFoundError(
            "No se encontro el modelo. Se esperaba: modelo_credit_score.keras"
        )

    scaler_path = find_first_existing(SCALER_CANDIDATES)
    if scaler_path is None:
        raise FileNotFoundError("No se encontro scaler_credit_score.joblib")

    pca_path = find_first_existing(PCA_CANDIDATES)

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path) if pca_path is not None else None

    # Use the scaler's own feature names as the authoritative source so the
    # column order always matches what the scaler was trained with.
    if hasattr(scaler, "feature_names_in_"):
        feature_columns = list(scaler.feature_names_in_)
    else:
        feature_columns = load_feature_columns()

    return (
        model,
        scaler,
        pca,
        feature_columns,
        model_path.name,
        scaler_path.name,
        pca_path.name if pca_path is not None else None,
    )


def build_input_row(data, feature_columns):
    row = {col: 0 for col in feature_columns}

    base_values = {
        "Age": float(data["edad"]),
        "Num_Bank_Accounts": int(data["num_cuentas"]),
        "Num_Credit_Card": int(data["num_tarjetas"]),
        "Interest_Rate": float(data["tasa_interes"]),
        "Num_of_Loan": int(data["num_prestamos"]),
        "Delay_from_due_date": int(data["dias_demora"]),
        "Num_of_Delayed_Payment": int(data["pagos_demorados"]),
        "Changed_Credit_Limit": float(data["changed_credit_limit"]),
        "Num_Credit_Inquiries": int(data["consultas_crediticias"]),
        "Credit_Mix": {"Bad": 0, "Standard": 1, "Good": 2}[data["credit_mix"]],
        "Outstanding_Debt": float(data["deuda_pendiente"]),
        "Credit_Utilization_Ratio": float(data["credit_utilization"]),
        "Credit_History_Age": float(data["historial_crediticio"]),
        "Payment_of_Min_Amount": {"No": 0, "Yes": 1}[data["pago_minimo"]],
        "Total_EMI_per_month": float(data["emi"]),
        "Amount_invested_monthly": float(data["gasto_mensual"]),
        "Payment_Behaviour": PAYMENT_BEHAVIOUR_MAP[data["payment_behavior"]],
        "Monthly_Balance": float(data["balance_mensual"]),
        "Num_of_Loans_Total": int(data["total_loans"]),
    }

    for key, value in base_values.items():
        if key in row:
            row[key] = value

    selected_raw_occupation = PROFESION_LABEL_TO_RAW[data["profesion"]]
    occupation_col = f"Occupation_{selected_raw_occupation}"
    if occupation_col in row:
        row[occupation_col] = 1

    return pd.DataFrame([row], columns=feature_columns)


st.title("Modelo de predicción de Credit Score con ANN")
st.subheader("Por: Juan Escobar")

st.image(
    "https://images.unsplash.com/photo-1563013544-824ae1b704d3",
    caption="Credit Card - Imagen libre de derechos",
    width=520,
)

try:
    model, scaler, pca, feature_columns, model_file, scaler_file, pca_file = load_artifacts()
except Exception as e:
    st.error(f"No fue posible cargar artefactos: {e}")
    st.stop()


st.header("Ingrese los datos del cliente")

col1, col2, col3 = st.columns(3)

with col1:
    edad = st.slider("Edad", 0.0, 100.0, 30.0)
    num_cuentas = st.slider("Número de cuentas bancarias", 0, 20, 2)
    num_tarjetas = st.slider("Número de tarjetas de crédito", 0, 20, 2)
    tasa_interes = st.slider("Tasa de interés", 0.0, 50.0, 10.0)
    num_prestamos = st.slider("Número de préstamos", 0, 20, 1)
    dias_demora = st.slider("Demora desde fecha de pago", -50, 100, 0)
    pagos_demorados = st.slider("Número de pagos demorados", 0, 50, 0)

with col2:
    changed_credit_limit = st.slider("Límite de crédito cambiado", -10000.0, 10000.0, 0.0)
    consultas_crediticias = st.slider("Número de consultas crediticias", 0, 50, 5)
    credit_mix = st.selectbox(
        "Combinación de crédito",
        ["Bad", "Standard", "Good"],
        format_func=lambda option: CREDIT_MIX_LABELS[option],
    )
    deuda_pendiente = st.slider("Deuda pendiente", 0.0, 200000.0, 10000.0)
    credit_utilization = st.slider("Ratio de utilización de crédito", 0.0, 100.0, 30.0)
    historial_crediticio = st.slider("Antigüedad de historial crediticio (años)", 0.0, 50.0, 5.0)
    pago_minimo = st.selectbox(
        "Pago de valor mínimo",
        ["No", "Yes"],
        format_func=lambda option: MIN_PAYMENT_LABELS[option],
    )

with col3:
    emi = st.slider("Total EMI mensual", 0.0, 20000.0, 1000.0)
    gasto_mensual = st.slider("Cantidad invertida por mes", 0.0, 50000.0, 2000.0)
    payment_behavior = st.selectbox(
        "Comportamiento de pago",
        PAYMENT_BEHAVIOUR_OPTIONS,
        format_func=lambda option: PAYMENT_BEHAVIOUR_LABELS[option],
    )
    balance_mensual = st.slider("Balance mensual", -50000.0, 50000.0, 1000.0)
    total_loans = st.slider("Número total de préstamos", 0, 20, 1)
    profesion = st.selectbox("Profesión", list(PROFESION_LABEL_TO_RAW.keys()))

if st.button("Predecir", type="primary"):
    payload = {
        "edad": edad,
        "num_cuentas": num_cuentas,
        "num_tarjetas": num_tarjetas,
        "tasa_interes": tasa_interes,
        "num_prestamos": num_prestamos,
        "dias_demora": dias_demora,
        "pagos_demorados": pagos_demorados,
        "changed_credit_limit": changed_credit_limit,
        "consultas_crediticias": consultas_crediticias,
        "credit_mix": credit_mix,
        "deuda_pendiente": deuda_pendiente,
        "credit_utilization": credit_utilization,
        "historial_crediticio": historial_crediticio,
        "pago_minimo": pago_minimo,
        "emi": emi,
        "gasto_mensual": gasto_mensual,
        "payment_behavior": payment_behavior,
        "balance_mensual": balance_mensual,
        "total_loans": total_loans,
        "profesion": profesion,
    }

    x_input = build_input_row(payload, feature_columns)

    expected_by_scaler = getattr(scaler, "n_features_in_", x_input.shape[1])
    if x_input.shape[1] != expected_by_scaler:
        st.error(
            "El numero de variables no coincide con el scaler. "
            f"Esperadas por scaler: {expected_by_scaler}, construidas por app: {x_input.shape[1]}."
        )
        st.info(
            "Guarda en entrenamiento un archivo con columnas exactas, por ejemplo: "
            "feature_columns_credit_score.joblib"
        )
        st.stop()

    x_scaled = scaler.transform(x_input)

    if pca is not None:
        expected_by_pca = getattr(pca, "n_features_in_", x_scaled.shape[1])
        if x_scaled.shape[1] != expected_by_pca:
            st.error(
                "El numero de variables no coincide con el PCA. "
                f"Esperadas por PCA: {expected_by_pca}, entregadas por scaler: {x_scaled.shape[1]}."
            )
            st.stop()
        x_model_input = pca.transform(x_scaled)
    else:
        x_model_input = x_scaled

    expected_by_model = model.input_shape[-1]
    if x_model_input.shape[1] != expected_by_model:
        st.error(
            "La dimension de entrada no coincide con el modelo. "
            f"Modelo espera: {expected_by_model}, app entrega: {x_model_input.shape[1]}."
        )
        st.info(
            "Si el modelo fue entrenado con PCA, debes exportar y colocar "
            "pca_credit_score.joblib junto con el scaler y el modelo."
        )
        st.stop()

    pred = model.predict(x_model_input, verbose=0)
    clase = int(np.argmax(pred, axis=1)[0])
    probs = pred[0]

    if clase == 0:
        st.error(f"Credit Score predicho: Malo ({clase})")
    elif clase == 1:
        st.warning(f"Credit Score predicho: Medio ({clase})")
    else:
        st.success(f"Credit Score predicho: Bueno ({clase})")

st.markdown(
    """
    <div style="
        margin-top: 2rem;
        padding: 0.9rem 1rem;
        text-align: center;
        background-color: #e5e7eb;
        color: #4b5563;
        border-radius: 10px;
        font-size: 0.95rem;
    ">
        Desarrollado en la UNAB, 2026 - Taller de Ciencia de Datos
    </div>
    """,
    unsafe_allow_html=True,
)




