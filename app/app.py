import streamlit as st
import pickle
import numpy as np
import os

st.title("Predicción: ¿La compañía respondió a tiempo?")

# Carpeta base (donde está app.py)
BASE_DIR = os.path.dirname(__file__)

# Rutas de los archivos
estado_mapping_path = os.path.join(BASE_DIR, "estado_mapping.pkl")
producto_mapping_path = os.path.join(BASE_DIR, "producto_mapping.pkl")
model_path = os.path.join(BASE_DIR, "final_model.pkl")

# Verificar que existan los archivos
for path in [estado_mapping_path, producto_mapping_path, model_path]:
    if not os.path.exists(path):
        st.error(f"No se encontró el archivo: {os.path.basename(path)}")
        st.stop()

# Cargar mappings
with open(estado_mapping_path, "rb") as f:
    estado_mapping = pickle.load(f)

with open(producto_mapping_path, "rb") as f:
    producto_mapping = pickle.load(f)

# Listas de opciones
estados = list(estado_mapping.keys())
productos = list(producto_mapping.keys())

# Inputs del usuario
estado = st.selectbox("¿En qué estado vives?", options=estados)
producto = st.selectbox("Producto relacionado con la queja", options=productos)

# Codificar inputs
estado_encoded = estado_mapping.get(estado, -1)
producto_encoded = producto_mapping.get(producto, -1)

# Cargar modelo
with open(model_path, "rb") as f:
    modelo = pickle.load(f)
st.success("Modelo cargado correctamente 🎉")

# Botón de predicción
if st.button("Predecir"):
    entrada = np.array([[estado_encoded, producto_encoded]])
    prediccion = modelo.predict(entrada)
    st.success(f"Predicción del modelo: {prediccion[0]}")

    if hasattr(modelo, "predict_proba"):
        probabilidad_clase_1 = modelo.predict_proba(entrada)[0][1]
        st.write(f"Probabilidad de la clase 1: {probabilidad_clase_1:.2f}")
