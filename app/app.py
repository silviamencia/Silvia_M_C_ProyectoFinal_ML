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
with open("models/estado_mapping.pkl", "rb") as f:
    estado_mapping = pickle.load(f)

with open("models/estado_mapping.pkl", "rb") as f:
    producto_mapping = pickle.load(f)

 
# Verificar que existe antes de cargarlo
if not os.path.exists(MODEL_PATH):
    st.error(f"No se encontró el modelo en {MODEL_PATH}")
else:
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)

    st.success("Modelo cargado correctamente 🎉")

#with open(MODEL_PATH, "rb") as f:
    #modelo = pickle.load(f)

# Título de la app
#st.title("Aplicación de Predicción con Modelo Entrenado")

st.write("Introduce los valores de las características para obtener la predicción.")

# Ajusta los inputs según las variables de tu dataset
# Inputs del usuario (categóricos)

estado = st.selectbox("¿En qué estado vives?", options=estados)
producto = st.selectbox("Producto relacionado con la queja", options=productos)

# Transformar inputs  usando los mappings
estado_encoded = estado_mapping.get(estado, -1)
producto_encoded = producto_mapping.get(producto, -1)


# Botón para predecir
if st.button("Predecir"):
    # Crear arreglo con la entrada
    entrada = np.array([[estado_encoded, producto_encoded]], dtype=np.float32)
    
    # Hacer predicción
    prediccion = modelo.predict(entrada)
    
    st.success(f"Predicción del modelo: {prediccion[0]}")

    # Mostrar la probabilidad de la clase 1
    probabilidad_clase_1 = modelo.predict_proba(entrada)[0][1]
    st.write(f"Probabilidad de la clase 1: {probabilidad_clase_1}")
