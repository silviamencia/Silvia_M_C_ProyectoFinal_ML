# app.py
import streamlit as st
import pickle
import pandas as pd
import os

MODEL_PATH = os.path.join("models", "final_model.pkl")

st.title("Predicción: ¿La compañía respondió a tiempo?")

# =========================
# 1. Cargar modelo
# =========================
if not os.path.exists(MODEL_PATH):
    st.error(f"No se encontró el modelo en {MODEL_PATH}. Entrena primero con train_model.py")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

st.success("Modelo cargado correctamente 🎉")

st.write("Introduce los valores de las características para obtener la predicción:")

# =========================
# 2. Inputs del usuario
# =========================
# 👇 estos deben coincidir con X_train
estado = st.selectbox("¿En qué estado vives?", 
    options=['AE','AK','AL','AP','AR','AS','AZ','CA','CO','CT','DC','DE','FL','GA','GU',
             'HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MH','MI','MN','MO',
             'MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','PR',
             'PW','RI','SC','SD','TN','TX','UT','VA','VI','VT','WA','WI','WV','WY'])

producto = st.selectbox("Producto relacionado con la queja", 
    options=['Debt collection','Mortgage','Credit card','Consumer loan',
             'Bank account or service','Payday loan','Credit reporting',
             'Money transfers','Student loan','Prepaid card',
             'Other financial service'])

# 👇 ejemplo si tenías una variable numérica
# monto = st.number_input("Monto de la transacción", min_value=0, value=100)

# =========================
# 3. Predicción
# =========================
if st.button("Predecir"):
    entrada = pd.DataFrame([{
        "estado": estado,
        "producto": producto,
        # "monto": monto,  # agrega todas las columnas necesarias
    }])

    prediccion = modelo.predict(entrada)
    st.success(f"Predicción del modelo: {prediccion[0]}")

    if hasattr(modelo, "predict_proba"):
        prob = modelo.predict_proba(entrada)[0][1]
        st.write(f"Probabilidad de la clase 1 (respuesta a tiempo): {prob:.2f}")
