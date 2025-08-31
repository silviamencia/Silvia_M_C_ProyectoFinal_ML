# app.py

import streamlit as st
import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(__file__)  # carpeta donde está app.py
model_path = os.path.join(BASE_DIR, "rf_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

#model = joblib.load(model_path)
#scaler = joblib.load(scaler_path)

st.title("Predicción de Respuesta Oportuna")
st.write("Esta aplicación predice si habrá una respuesta oportuna basado en las características del cliente.")

# -------------------------------
# 1. Cargar modelo y scaler
# -------------------------------
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# 2. Definir columnas de entrada
# -------------------------------
# Cambia esta lista por todas tus features de X
feature_columns = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]

st.sidebar.header("Ingrese los datos del cliente")

# -------------------------------
# 3. Crear inputs dinámicos
# -------------------------------
input_data = {}
for col in feature_columns:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

# -------------------------------
# 4. Escalar datos
# -------------------------------
input_scaled = scaler.transform(input_df)

# -------------------------------
# 5. Predicción
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
