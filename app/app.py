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
# 3. Inputs dinámicos
# -------------------------------
input_data = {}       # valores codificados para el modelo
original_values = {}  # valores originales para mostrar al usuario

for col in feature_columns:
    if col in encoders:  # columna categórica
        categories = encoders[col].classes_.tolist()
        selected_value = st.sidebar.selectbox(col, categories)
        input_data[col] = encoders[col].transform([selected_value])[0]  # numérico
        original_values[col] = selected_value  # legible
    else:  # columna numérica
        val = st.sidebar.number_input(col, value=0.0)
        input_data[col] = val
        original_values[col] = val  # lo guardamos igual para mostrarlo luego

# -------------------------------
# 4. Mostrar SIEMPRE los valores originales
# -------------------------------
st.subheader("Valores seleccionados (originales):")
st.table(pd.DataFrame([original_values]))  # muestra en tabla legible

# -------------------------------
# 5. DataFrame para el modelo
# -------------------------------
input_df = pd.DataFrame([input_data], columns=feature_columns)

# -------------------------------
# 6. Escalar datos
# -------------------------------
input_scaled = scaler.transform(input_df)

# -------------------------------
# 7. Predicción
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
