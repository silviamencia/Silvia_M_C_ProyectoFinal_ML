import streamlit as st
import pandas as pd
import joblib
import os
import traceback

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

st.title("Predicción de Respuesta Oportuna")
st.write("Debes seleccionar obligatoriamente **State** y **Product**.")

# Cargar objetos
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)
encoders = joblib.load(ENCODERS_PATH)

input_data = {}
original_values = {}

# -------------------------------
# Inputs obligatorios
# -------------------------------
for col in ["State", "Product"]:
    if col in encoders:
        categories = encoders[col].classes_.tolist()  # <-- VALORES ORIGINALES
        selected_value = st.sidebar.selectbox(f"{col}", categories)
        input_data[col] = encoders[col].transform([selected_value])[0]
        original_values[col] = selected_value
    else:
        st.error(f"No encontré encoder para la columna {col}")

# -------------------------------
# Completar el resto de columnas con valores por defecto
# -------------------------------
for col in feature_columns:
    if col not in ["State", "Product"]:
        if col in encoders:
            default_val = encoders[col].classes_[0]
            input_data[col] = encoders[col].transform([default_val])[0]
            original_values[col] = default_val
        else:
            input_data[col] = 0.0
            original_values[col] = 0.0

# -------------------------------
# Mostrar lo que el usuario eligió
# -------------------------------
st.subheader("Valores seleccionados (originales):")
st.write(original_values)

# -------------------------------
# Predicción
# -------------------------------
input_df = pd.DataFrame([input_data], columns=feature_columns)

if st.button("Predecir"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_prob = model.predict_proba(input_scaled)[0]

        if prediction[0] == 1:
            st.success("✅ La respuesta será oportuna")
        else:
            st.error("❌ La respuesta no será oportuna")

        st.info(f"Prob. Oportuna: {prediction_prob[1]:.2f}")
        st.info(f"Prob. No Oportuna: {prediction_prob[0]:.2f}")

    except Exception as e:
        st.error("Error en la predicción")
        st.text(traceback.format_exc())

        st.text(traceback.format_exc())

