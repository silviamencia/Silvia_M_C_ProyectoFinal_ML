#Código para evaluar el modelo utilizando los datos de prueba.
#IMPORTAR LIBRERÍAS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import pickle
import yaml
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


print(os.getcwd())
#df = pd.read_csv('C:/Users/silvi/Documents/DATA_SCIENCE/TheBridge - copia/DSPT2025-ML/Proyecto final/data/quejas-clientes.csv')
#df.head()
#os.chdir(r"/content/sample_data/")
df_scaled= pd.read_csv("df_scaled.csv")

#ENTRENAMIENTO CON RANDOM FOREST
#SEPARACIÓN DE DATOS
#Separación entre variables explicativas y variable objetivo "Timely response?"
X =df_scaled.drop(["Timely response?"], axis = 1)
y=df_scaled["Timely response?"]

# Partición dejando un 75% para entrenar y un 25% de test.

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.25, random_state=42)

# RANDOM FOREST
# Crear el objeto SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
# Crear pipeline con SMOTE y entrenar el modelo

RF=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=None, max_features='sqrt', max_leaf_nodes=None,min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1, oob_score=False, random_state=42, verbose=0,warm_start=False)
pipeline = Pipeline([('smote', smote), ('RF', RF)])
# Crear y entrenar el modelo

pipeline.fit(X_train, y_train)

# Predecir con el conjunto de prueba y evaluar
y_pred = pipeline.predict(X_test)

print("RF:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

