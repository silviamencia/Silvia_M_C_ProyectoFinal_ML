# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Ruta a la carpeta "models" que está en Github

MODEL_PATH = os.path.join("models", "final_model.pkl")

st.title("Predicción: ¿La compañía respondió a tiempo?")

# Verificar que existe antes de cargarlo
if not os.path.exists(MODEL_PATH):
    st.error(f"No se encontró el modelo en {MODEL_PATH}")
else:
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)

    st.success("Modelo cargado correctamente 🎉")

with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

# Título de la app
#st.title("Aplicación de Predicción con Modelo Entrenado")

st.write("Introduce los valores de las características para obtener la predicción.")

# Ajusta los inputs según las variables de tu dataset
# Inputs del usuario (categóricos)
estado = st.text_input("¿En qué estado vives?", value="CDMX")
producto = st.text_input("Producto relacionado con la queja", value="Tarjeta")
company = st.text_input("Nombre de la compañía", value="Banco ABC")

# Botón para predecir
if st.button("Predecir"):
    # Crear arreglo con la entrada
    entrada = np.array([[estado, producto, company]])
    
    # Hacer predicción
    prediccion = modelo.predict(entrada)
    
    st.success(f"Predicción del modelo: {prediccion[0]}")

    # Mostrar la probabilidad de la clase 1
    probabilidad_clase_1 = modelo.predict_proba(entrada)[0][1]
    st.write(f"Probabilidad de la clase 1: {probabilidad_clase_1}")
