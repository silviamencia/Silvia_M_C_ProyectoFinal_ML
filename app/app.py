import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# 1. Título
# -------------------------------
st.title("Predicción de Respuesta Oportuna")
st.write("Esta aplicación predice si habrá una respuesta oportuna basado en las características del cliente.")

# -------------------------------
# 2. Cargar objetos con rutas relativas
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)
encoders = joblib.load(ENCODERS_PATH)


st.sidebar.header("Ingrese los datos del cliente")

# -------------------------------
# 3. Crear inputs dinámicos
# -------------------------------
input_data = {}

for col in feature_columns:
    if col in encoders:  # columna categórica
        # Mostrar selectbox con las categorías originales
        categories = encoders[col].classes_.tolist()
        input_data[col] = st.sidebar.selectbox(col, categories)
    else:  # columna numérica
        input_data[col] = st.sidebar.number_input(col, value=0.0)

# Crear DataFrame
input_df = pd.DataFrame([input_data], columns=feature_columns)

# -------------------------------
# 4. Transformar categóricas con LabelEncoder
# -------------------------------
for col, le in encoders.items():
    input_df[col] = le.transform(input_df[col])

# -------------------------------
# 5. Escalar datos
# -------------------------------
input_scaled = scaler.transform(input_df)

# -------------------------------
# 6. Predicción
# -------------------------------
if st.button("Predecir"):
    prediction = model.predict(input_scaled)
    prediction_prob = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.success("✅ La respuesta será oportuna")
    else:
        st.error("❌ La respuesta no será oportuna")

    st.info(f"Probabilidad de Respuesta Oportuna: {prediction_prob[0][1]:.2f}")
    st.info(f"Probabilidad de Respuesta No Oportuna: {prediction_prob[0][0]:.2f}")



column_info = joblib.load(os.path.join(BASE_DIR, "column_info.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

input_data = {}
for col in feature_columns:
    if column_info[col]["type"] == "numeric":
        input_data[col] = st.sidebar.number_input(col, value=0.0)
    else:
        # Para categóricas usamos selectbox con las categorías originales
        input_data[col] = st.sidebar.selectbox(col, column_info[col]["categories"])

input_df = pd.DataFrame([input_data], columns=feature_columns)

# Transformar con LabelEncoder
for col, le in encoders.items():
    input_df[col] = le.transform(input_df[col])

