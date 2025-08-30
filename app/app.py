import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Predicción de Respuesta de Compañía", layout="centered")
st.title("Predicción: ¿La compañía respondió a tiempo?")

# Lista de opciones
estados = ['AE', 'AK', 'AL', 'AP', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC',
           'DE', 'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY',
           'LA', 'MA', 'MD', 'ME', 'MH', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC',
           'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA',
           'PR', 'PW', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI', 'VT',
           'WA', 'WI', 'WV', 'WY']

productos = ['Debt collection', 'Mortgage', 'Credit card', 'Consumer loan',
             'Bank account or service', 'Payday loan', 'Credit reporting',
             'Money transfers', 'Student loan', 'Prepaid card',
             'Other financial service']

# Cargar mappings
try:
    with open("estado_mapping.pkl", "rb") as f:
        estado_mapping = pickle.load(f)
    with open("producto_mapping.pkl", "rb") as f:
        producto_mapping = pickle.load(f)
except FileNotFoundError:
    st.error("No se pudieron cargar los archivos de mapeo. Asegúrate de que existan.")
    st.stop()

# Selección del usuario
estado = st.selectbox("¿En qué estado vives?", options=estados)
producto = st.selectbox("Producto relacionado con la queja", options=productos)

# Codificar inputs
estado_encoded = estado_mapping.get(estado, -1)
producto_encoded = producto_mapping.get(producto, -1)

if estado_encoded == -1 or producto_encoded == -1:
    st.error("Error: el estado o producto seleccionado no está mapeado correctamente.")
    st.stop()

# Ruta del modelo
MODEL_PATH = "models/final_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"No se encontró el modelo en {MODEL_PATH}")
    st.stop()
else:
    try:
        with open(MODEL_PATH, "rb") as f:
            modelo = pickle.load(f)
        st.success("Modelo cargado correctamente 🎉")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.stop()

# Predicción
if st.button("Predecir"):
    entrada = np.array([[estado_encoded, producto_encoded]], dtype=float)
    try:
        prediccion = modelo.predict(entrada)[0]
        etiquetas = {0: "No respondió a tiempo", 1: "Respondió a tiempo"}
        st.success(f"Predicción del modelo: {etiquetas.get(prediccion, prediccion)}")

        if hasattr(modelo, "predict_proba"):
            probabilidad_clase_1 = modelo.predict_proba(entrada)[0][1]
            st.info(f"Probabilidad de que la compañía responda a tiempo: {probabilidad_clase_1:.2f}")
    except Exception as e:
        st.error(f"Ocurrió un error al predecir: {e}")
