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
# 2. Cargar objetos
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))

# -------------------------------
# 3. Cargar DataFrame de entrenamiento para categorías
# -------------------------------
df_train = pd.read_csv(os.path.join(BASE_DIR, "datos_entrenamiento.csv"))

# -------------------------------
# 4. Detectar tipo de columna y categorías
# -------------------------------
column_info = {}
for col in feature_columns:
    if pd.api.types.is_numeric_dtype(df_train[col]):
        column_info[col] = {"type": "numeric"}
    else:
        column_info[col] = {"type": "categorical", "categories": sorted(df_train[col].dropna().unique())}

# -------------------------------
# 5. Inputs del usuario
# -------------------------------
st.sidebar.header("Ingrese los datos del cliente")
input_data = {}

for col in feature_columns:
    col_type = column_info[col]["type"]
    if col_type == "numeric":
        input_data[col] = st.sidebar.number_input(col, value=0.0)
    else:  # categórica
        input_data[col] = st.sidebar.selectbox(col, column_info[col]["categories"])

# Crear DataFrame con tipos originales
input_df = pd.DataFrame([input_data], columns=feature_columns)

# -------------------------------
# 6. Transformar categóricas con LabelEncoder
# -------------------------------
for col, le in encoders.items():
    try:
        input_df[col] = le.transform(input_df[col])
    except ValueError:
        st.error(f"La categoría ingresada para {col} no está en el conjunto de entrenamiento.")
        st.stop()

# -------------------------------
# 7. Escalar datos
# -------------------------------
input_scaled = scaler.transform(input_df)

# -------------------------------
# 8. Predicción
# -------------------------------
if st.button("Predecir"):
    prediction = model.predict(input_scaled)
    prediction_prob = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.success("✅ La respuesta será oportuna")
    else:
        st.error("❌ La respuesta no será oportuna")

    st.write("Probabilidades:")
    st.write(f"Oportuna: {prediction_prob[0][1]:.2f}")
    st.write(f"No Oportuna: {prediction_prob[0][0]:.2f}")
