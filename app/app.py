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
# 3. Crear inputs dinámicos (mostrando categorías originales)
# -------------------------------
input_data = {}
original_values = {}  # Para almacenar los valores originales

for col in feature_columns:
    if col in encoders:  # columna categórica
        # Mostrar selectbox con las categorías originales
        categories = encoders[col].classes_.tolist()
        selected_value = st.sidebar.selectbox(col, categories)
        input_data[col] = encoders[col].transform([selected_value])[0]  # Guardar el valor codificado
        original_values[col] = selected_value  # Guardar el valor original para mostrar
    else:  # columna numérica
        input_data[col] = st.sidebar.number_input(col, value=0.0)

# -------------------------------
# 4. Mostrar los valores seleccionados (originales)
# -------------------------------
st.subheader("Valores seleccionados:")
for col, value in original_values.items():
    st.write(f"{col}: {value}")

for col in feature_columns:
    if col not in original_values:  # Para columnas numéricas
        st.write(f"{col}: {input_data[col]}")

# Crear DataFrame
input_df = pd.DataFrame([input_data], columns=feature_columns)

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
