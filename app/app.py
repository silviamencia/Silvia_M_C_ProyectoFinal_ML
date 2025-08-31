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
original_values = {}  # valores originales en texto/numérico

for col in feature_columns:
    if col in encoders:  # columna categórica
        categories = encoders[col].classes_.tolist()
        selected_value = st.sidebar.selectbox(col, categories)   # el usuario ve los valores originales
        input_data[col] = encoders[col].transform([selected_value])[0]  # numérico para el modelo
        original_values[col] = selected_value  # guardamos el valor categórico en texto
    else:  # columna numérica
        val = st.sidebar.number_input(col, value=0.0)
        input_data[col] = val
        original_values[col] = val  # guardamos el valor tal cual

# -------------------------------
# 4. Mostrar siempre valores categóricos originales
# -------------------------------
st.subheader("Valores seleccionados (originales):")
st.dataframe(pd.DataFrame([original_values]))

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

    st.subheader("Resultado de la predicción:")
    if prediction[0] == 1:
        st.success("✅ La respuesta será oportuna")
    else:
        st.error("❌ La respuesta no será oportuna")

    # Probabilidades
    st.info(f"Probabilidad de Respuesta Oportuna: {prediction_prob[0][1]:.2f}")
    st.info(f"Probabilidad de Respuesta No Oportuna: {prediction_prob[0][0]:.2f}")

    # Tabla resumen final con valores originales + predicción
    resumen = original_values.copy()
    resumen["Predicción"] = "Oportuna" if prediction[0] == 1 else "No Oportuna"
    resumen["Prob. Oportuna"] = round(prediction_prob[0][1], 2)
    resumen["Prob. No Oportuna"] = round(prediction_prob[0][0], 2)

    st.subheader("Resumen completo:")
    st.table(pd.DataFrame([resumen]))

