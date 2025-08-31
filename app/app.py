# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# 1. Título de la app
# -------------------------------
st.title("Predicción de Respuesta Oportuna")
st.write("Esta aplicación predice si habrá una respuesta oportuna basado en las características del cliente.")

# -------------------------------
# 2. Cargar modelo y scaler con rutas relativas
# -------------------------------
BASE_DIR = os.path.dirname(__file__)  # carpeta donde está app.py
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------
# 3. Definir columnas de entrada
# -------------------------------
# Cambia esta lista por todas tus features de X
feature_columns = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]

st.sidebar.header("Ingrese los datos del cliente")

# -------------------------------
# 4. Crear inputs dinámicos
# -------------------------------
input_data = {}
for col in feature_columns:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

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

