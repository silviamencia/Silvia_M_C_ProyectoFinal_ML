import streamlit as st
import pandas as pd
import joblib
import os
import traceback

# -------------------------------
# Config / carga de artefactos
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

st.title("Predicción de Respuesta Oportuna")
st.write("Debes seleccionar obligatoriamente **State** y **Product**. Las demás variables se completan automáticamente.")

# Cargar objetos
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)   # lista con el orden de las columnas
encoders = joblib.load(ENCODERS_PATH)          # dict: {columna: encoder}

# -------------------------------
# Utilidades
# -------------------------------
def safe_transform_category(enc, value):
    """Convierte categoría en número según el encoder."""
    try:
        if hasattr(enc, "classes_"):
            return int(enc.transform([value])[0])
        if hasattr(enc, "categories_"):
            return int(enc.transform([[value]])[0][0])
        if hasattr(enc, "transform"):
            return enc.transform([value])[0]
    except Exception:
        return 0

def get_first_category(enc):
    """Devuelve la primera categoría del encoder."""
    try:
        if hasattr(enc, "classes_"):
            return enc.classes_[0]
        if hasattr(enc, "categories_"):
            return enc.categories_[0][0]
    except Exception:
        return None

# -------------------------------
# Inputs obligatorios: State y Product
# -------------------------------
st.sidebar.header("Ingrese datos del cliente")

input_data = {}
original_values = {}

for col in feature_columns:
    if col in ["State", "Product"]:  # obligatorios
        if col in encoders:
            cats = None
            try:
                if hasattr(encoders[col], "classes_"):
                    cats = list(encoders[col].classes_)
                elif hasattr(encoders[col], "categories_"):
                    cats = list(encoders[col].categories_[0])
            except Exception:
                cats = None

            if cats:
                selected = st.sidebar.selectbox(f"{col}", options=cats, key=f"sel_{col}")
                original_values[col] = selected
                input_data[col] = safe_transform_category(encoders[col], selected)
            else:
                text_val = st.sidebar.text_input(f"{col} (texto libre)", value="", key=f"text_{col}")
                original_values[col] = text_val
                input_data[col] = safe_transform_category(encoders[col], text_val)
        else:
            val = st.sidebar.number_input(f"{col} (numérico)", value=0.0, key=f"num_{col}")
            original_values[col] = val
            input_data[col] = float(val)
    else:
        # Rellenar automáticamente el resto
        if col in encoders:
            first_cat = get_first_category(encoders[col])
            if first_cat is not None:
                original_values[col] = first_cat
                input_data[col] = safe_transform_category(encoders[col], first_cat)
            else:
                original_values[col] = "Valor por defecto"
                input_data[col] = 0
        else:
            original_values[col] = 0.0
            input_data[col] = 0.0

# -------------------------------
# Mostrar valores originales
# -------------------------------
st.subheader("Valores usados (originales, antes del encoder):")
st.dataframe(pd.DataFrame([original_values]))

# -------------------------------
# Preparar datos y predicción
# -------------------------------
input_df = pd.DataFrame([input_data], columns=feature_columns)

if st.button("Predecir"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_prob = model.predict_proba(input_scaled)[0]

        st.subheader("Resultado de la predicción:")
        if int(prediction[0]) == 1:
            st.success("✅ La respuesta será oportuna")
        else:
            st.error("❌ La respuesta no será oportuna")

        st.info(f"Probabilidad de Respuesta Oportuna: {prediction_prob[1]:.2f}")
        st.info(f"Probabilidad de Respuesta No Oportuna: {prediction_prob[0]:.2f}")

        resumen = original_values.copy()
        resumen["Predicción"] = "Oportuna" if int(prediction[0]) == 1 else "No Oportuna"
        resumen["Prob. Oportuna"] = round(prediction_prob[1], 2)
        resumen["Prob. No Oportuna"] = round(prediction_prob[0], 2)

        st.subheader("Resumen completo:")
        st.table(pd.DataFrame([resumen]))

    except Exception as e:
        st.error("Error al hacer la predicción.")
        st.text(traceback.format_exc())

