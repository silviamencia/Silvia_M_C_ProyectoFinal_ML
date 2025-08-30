# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Ruta a la carpeta "models" que está en Github

MODEL_PATH = os.path.join("models", "final_model.pkl")

st.title("Predicción: ¿La compañía respondió a tiempo?")

# Lista de opciones (puedes ajustarlas según tu dataset)
estados = ['TX', 'MA', 'CA', nan, 'OH', 'NJ', 'ND', 'RI', 'CO', 'UT', 'AL',
       'PA', 'NY', 'NC', 'GA', 'IL', 'WI', 'MI', 'FL', 'CT', 'OR', 'VA',
       'WA', 'TN', 'MD', 'IA', 'KY', 'LA', 'OK', 'NE', 'KS', 'MO', 'NH',
       'IN', 'DC', 'NV', 'ME', 'NM', 'SC', 'AZ', 'AP', 'MS', 'MN', 'ID',
       'HI', 'PR', 'WV', 'WY', 'AK', 'VI', 'MT', 'DE', 'AR', 'AE', 'SD',
       'GU', 'VT', 'MH', 'PW', 'AS']
productos = ['Debt collection', 'Mortgage', 'Credit card', 'Consumer loan',
       'Bank account or service', 'Payday loan', 'Credit reporting',
       'Money transfers', 'Student loan', 'Prepaid card',
       'Other financial service']

 

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
estado = st.selectbox("¿En qué estado vives?", options=estados)
producto = st.selectbox("Producto relacionado con la queja", options=productos)


# Botón para predecir
if st.button("Predecir"):
    # Crear arreglo con la entrada
    entrada = np.array([[estado, producto]])
    
    # Hacer predicción
    prediccion = modelo.predict(entrada)
    
    st.success(f"Predicción del modelo: {prediccion[0]}")

    # Mostrar la probabilidad de la clase 1
    probabilidad_clase_1 = modelo.predict_proba(entrada)[0][1]
    st.write(f"Probabilidad de la clase 1: {probabilidad_clase_1}")
