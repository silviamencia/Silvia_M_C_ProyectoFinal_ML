from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# Cargar objetos (asume que existen y están guardados con joblib)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)  # lista con el orden de las columnas
encoders = joblib.load(ENCODERS_PATH)  # dict: {columna: encoder}

# -------------------------------
# FastAPI Setup
# -------------------------------
app = FastAPI()

# -------------------------------
# Model Input (utilizando Pydantic)
# -------------------------------

class ModelInput(BaseModel):
    # Define las columnas que esperas como parámetros de entrada
    # Asegúrate de que coincidan con las columnas de tu modelo
    # Asumimos que todas las columnas son de tipo float, ajusta según tu modelo
    input_data: dict

# -------------------------------
# Utilidades
# -------------------------------
def safe_transform_category(enc, value):
    try:
        # LabelEncoder-like (classes_)
        if hasattr(enc, "classes_"):
            return int(enc.transform([value])[0])
        # OrdinalEncoder-like (categories_)
        if hasattr(enc, "categories_"):
            out = enc.transform([[value]])[0][0]
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
    try:
        if hasattr(enc, "classes_"):
            return enc.classes_[0]
        if hasattr(enc, "categories_"):
            return enc.categories_[0][0]
    except Exception:
        return None
    return None

# -------------------------------
# Endpoint: predicción
# -------------------------------
@app.post("/predict")
async def predict(input: ModelInput):
    try:
        input_data = input.input_data
        original_values = {}

        # Preparar los datos, similar a Streamlit pero en formato de JSON
        for col in feature_columns:
            value = input_data.get(col, None)
            if value is not None:
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
                        # Si hay categorías, transformamos con el encoder
                        original_values[col] = value
                        input_data[col] = safe_transform_category(encoders[col], value)
                    else:
                        # Si no es una categoría, tratamos como numérico
                        original_values[col] = value
                        input_data[col] = float(value)
                else:
                    # Si no hay encoder, tratamos como numérico
                    original_values[col] = value
                    input_data[col] = float(value)
            else:
                # Si no hay valor, usamos por defecto
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

        # Crear DataFrame para pasar al modelo
        input_df = pd.DataFrame([input_data], columns=feature_columns)

        # Escalar y predecir
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        # Probabilidades si el modelo tiene predict_proba
        prediction_prob = None
        if hasattr(model, "predict_proba"):
            prediction_prob = model.predict_proba(input_scaled)[0]

        # Preparar la respuesta
        result = {
            "prediction": "Oportuna" if int(prediction[0]) == 1 else "No Oportuna",
            "probabilidad_oportuna": prediction_prob[1] if prediction_prob else None,
            "probabilidad_no_oportuna": prediction_prob[0] if prediction_prob else None,
            "original_values": original_values
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

