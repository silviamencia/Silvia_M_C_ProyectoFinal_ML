import streamlit as st
import pickle
import numpy as np
import os

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
with open("estado_mapping.pkl", "rb") as f:
    estado_mapping = pickle.load(f)

with open("producto_mapping.pkl", "rb") as f:
    producto_mapping = pickle.load(f)

# Selección del usuario
estado = st.selectbox("¿En qué estado vives?", options=estados)
producto = st.selectbox("Producto relacionado con la queja", options=productos)

# Codificar inputs
estado_encoded = estado_mapping.get(estado, -1)
producto_encoded = producto_mapping.get(producto, -1)

# Ruta del modelo
MODEL_PATH = "models/final_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"No se encontró el modelo en {MODEL_PATH}")
else:
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)
    st.success("Modelo cargado correctamente 🎉")

    if st.button("Predecir"):
        entrada = np.array([[estado_encoded, producto_encoded]])
        prediccion = modelo.predict(entrada)
        st.success(f"Predicción del modelo: {prediccion[0]}")

        if hasattr(modelo, "predict_proba"):
            probabilidad_clase_1 = modelo.predict_proba(entrada)[0][1]
            st.write(f"Probabilidad de la clase 1: {probabilidad_clase_1:.2f}")
