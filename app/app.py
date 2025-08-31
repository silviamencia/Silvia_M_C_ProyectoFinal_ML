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
st.write("Selecciona qué columnas quieres rellenar; las demás se usan con valores por defecto.")

# Cargar objetos (asume que existen y están guardados con joblib)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)   # lista con el orden de las columnas
encoders = joblib.load(ENCODERS_PATH)          # dict: {columna: encoder}

# -------------------------------
# Utilidades
# -------------------------------
def safe_transform_category(enc, value):
    """
    Intenta transformar una categoría con distintos tipos de encoders comunes.
    Si falla, devuelve 0 (fallback seguro).
    """
    try:
        # LabelEncoder-like (classes_)
        if hasattr(enc, "classes_"):
            return int(enc.transform([value])[0])
        # OrdinalEncoder-like (categories_)
        if hasattr(enc, "categories_"):
            # OrdinalEncoder espera 2D
            out = enc.transform([[value]])[0][0]
            # si es entero flotante, devolver int
            try:
                return int(out) if float(out).is_integer() else float(out)
            except Exception:
                return out
        # Otros con transform genérico
        if hasattr(enc, "transform"):
            out = enc.transform([value])[0]
            return out
    except Exception:
        return 0

def get_first_category(enc):
    """Devuelve la primera categoría conocida por el encoder, o None."""
    try:
        if hasattr(enc, "classes_"):
            return enc.classes_[0]
        if hasattr(enc, "categories_"):
            return enc.categories_[0][0]
    except Exception:
        return None
    return None

# -------------------------------
# Sidebar: selección de columnas a rellenar (1 o más)
# -------------------------------
st.sidebar.header("Configuración de entrada")
cols_to_fill = st.sidebar.multiselect(
    "Elige las columnas que quieres rellenar manualmente (por defecto 1):",
    options=feature_columns,
    default=[feature_columns[0]] if feature_columns else []
)

if not cols_to_fill:
    st.sidebar.warning("No has seleccionado ninguna columna: se usará la primera por defecto.")
    cols_to_fill = [feature_columns[0]]

# -------------------------------
# Recoger inputs: las elegidas por el usuario; el resto por defecto
# -------------------------------
input_data = {}
original_values = {}

for col in feature_columns:
    if col in cols_to_fill:
        # Creamos un control para que el usuario rellene esa columna
        if col in encoders:
            # Intentar obtener categorías para ofrecer un selectbox
            cats = None
            try:
                if hasattr(encoders[col], "classes_"):
                    cats = list(encoders[col].classes_)
                elif hasattr(encoders[col], "categories_"):
                    cats = list(encoders[col].categories_[0])
            except Exception:
                cats = None

            if cats:
                selected = st.sidebar.selectbox(f"{col} (categoría)", options=cats, key=f"sel_{col}")
                original_values[col] = selected
                input_data[col] = safe_transform_category(encoders[col], selected)
            else:
                # Si no podemos listar categorías, dejamos un text_input y tratamos como "no codificable"
                text_val = st.sidebar.text_input(f"{col} (texto libre)", value="", key=f"text_{col}")
                original_values[col] = text_val
                # Intentamos transformar, si falla, fallback a 0
                try:
                    input_data[col] = safe_transform_category(encoders[col], text_val)
                except Exception:
                    input_data[col] = 0
        else:
            # Numérica: pedir number_input
            val = st.sidebar.number_input(f"{col} (numérico)", value=0.0, key=f"num_{col}")
            original_values[col] = val
            input_data[col] = float(val)
    else:
        # Columna NO seleccionada por el usuario -> usar valor por defecto
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
# Mostrar valores originales antes del encoder
# -------------------------------
st.subheader("Valores usados (originales, antes del encoder):")
st.dataframe(pd.DataFrame([original_values]))

# Mostrar qué columnas el usuario ha rellenado
st.caption(f"Columnas rellenadas por usuario: {', '.join(cols_to_fill)}")

# -------------------------------
# Preparar DataFrame para el modelo
# -------------------------------
input_df = pd.DataFrame([input_data], columns=feature_columns)

# Botón de predicción
if st.button("Predecir"):
    # Mostrar un expander con la información enviada al modelo
    with st.expander("DEBUG: Datos enviados al modelo (input_df)"):
        st.write(input_df)

    # Escalado y predicción con manejo de errores
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error("Error al escalar los datos antes de predecir. Revisa que el scaler y las columnas coincidan.")
        st.text(traceback.format_exc())
        st.stop()

    try:
        prediction = model.predict(input_scaled)
    except Exception as e:
        st.error("Error en model.predict(). Revisa que el modelo sea compatible con las entradas.")
        st.text(traceback.format_exc())
        st.stop()

    # Probabilidades (si el modelo tiene predict_proba)
    prediction_prob = None
    if hasattr(model, "predict_proba"):
        try:
            prediction_prob = model.predict_proba(input_scaled)[0]
        except Exception:
            prediction_prob = None

    # Mostrar resultado
    st.subheader("Resultado de la predicción:")
    if int(prediction[0]) == 1:
        st.success("✅ La respuesta será oportuna")
    else:
        st.error("❌ La respuesta no será oportuna")

    if prediction_prob is not None:
        st.info(f"Probabilidad de Respuesta Oportuna: {prediction_prob[1]:.2f}")
        st.info(f"Probabilidad de Respuesta No Oportuna: {prediction_prob[0]:.2f}")

    # Resumen final: valores originales + predicción
    resumen = original_values.copy()
    resumen["Predicción"] = "Oportuna" if int(prediction[0]) == 1 else "No Oportuna"
    if prediction_prob is not None:
        resumen["Prob. Oportuna"] = round(prediction_prob[1], 2)
        resumen["Prob. No Oportuna"] = round(prediction_prob[0], 2)

    st.subheader("Resumen completo:")
    st.table(pd.DataFrame([resumen]))

    # Mostrar scaled en DEBUG
    with st.expander("DEBUG: Datos escalados (input_scaled)"):
        st.write(input_scaled)

