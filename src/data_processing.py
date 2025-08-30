#IMPORTACIÓN DE LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

print(os.getcwd())
df = pd.read_csv("quejas-clientes.csv")

#Elimino la columna Unnamed ya que coincide con la del índice.
#Elimino la variable de Complaint ID porque no me aporta mucha información
#Elimino la variable Consumer disputed ya que tiene demasiados valores faltantes
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('Complaint ID', axis=1, inplace=True)
df.drop('Consumer disputed?', axis=1, inplace=True)
#Convierto las columnas a datetime para poder trabajar con ellas

df["Date received"] = pd.to_datetime(df["Date received"])
df["Date sent to company"] = pd.to_datetime(df["Date sent to company"])

#Creo una nueva columna con la resta de las columnas, Date received (Fecha de recepción de la queja, cuando se recopilan las quejas) y Date sent to company (fecha en la que fue remitida a la
#empresa para que se pueda responder o gestinar el caso). Calculo el tiempo en días de ese retraso desde que se recibe la queja en algún sistema que recoge las quejas hasta que ésta es remitida
#a la empresa
df["Retraso envi�o di�as"] = (df["Date sent to company"] - df["Date received"]).dt.days

#Relleno Sub-product con NI (no informado)
#Relleno Sub-issue con NI (no informado)
df["Sub-product"] = df["Sub-product"].fillna("NI")
df["Sub-issue"] = df["Sub-issue"].fillna("NI")

#He descargado un csv con zips para rellenar los NANs de la columna ZIP
#df_zips = pd.read_csv(r'C:\Users\silvi\Documents\DATA_SCIENCE\TheBridge - copia\DSPT2025-ML\Proyecto final\Silvia_Proyecto_Final_ML\notebooks\uszips.csv')
df_zips = pd.read_csv("uszips.csv")

#Función para rellenar los valores faltantes de la columna ZIP code en el df
def rellenar_zips(zip_code):
    if pd.isna(zip_code):
        return random.choice(df_zips["zip"])
    else:
        return zip_code

df["ZIP code"] = df["ZIP code"].apply(rellenar_zips)

#Paso la columna ZIP code de float64 a entero
df["ZIP code"] = df["ZIP code"].astype(int)

#Quiero rellenar los NAN de la columna State con los valores obtenidos del df_zips
# Crear un diccionario ZIP → State
zip_to_state = df_zips.set_index('zip')['state_id'].to_dict()

# Rellenar los NaN en df['State'] usando ZIP code
df["State"] = df["State"].fillna(df['ZIP code'].map(zip_to_state))

#Hay 30 NANs de la columna State que no se han rellenado usando ZIP code, como son sólo 30 los elimino
df.dropna(subset=['State'], inplace=True)

#También elimino los 2 valores faltantes de la columna Issue
df.dropna(subset=['Issue'], inplace=True)

#Exporto a csv este df sin haber aún dividido en variables numéricas y categóricas.
df.to_csv('df.csv', index=False)

#Hay que pasar todas las columnas categóricas a números para poder entrenar, para ello separo en dos dfs,uno con variables numéricas y otro con las categóricas
#Transformo las categorías en números con LabelEncoder
df_num = df.select_dtypes(include=['int64'])
df_cat = df.select_dtypes(include=['object'])

#CORRELACIÓN DE LAS VARIABLES NUMÉRICAS

corr_matriz = df_num.corr()
print(corr_matriz)

#OUTLIERS

#Reviso la existencia de outliers en el DataFrame, saco un boxplot únicamente de la variable Retraso envío días, ya que no tiene sentido
#hacerlo de la variable ZIP

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_num['Retraso envi�o di�as'])
plt.title('Boxplot de outliers de Retraso envi�o di�as')
plt.show()

#########################TRANSFORAMCIÓN DE LOS DATOS############################################
#Es necesario convertir todas las columnas categóricas en valores numéricos para poder utilizarlas en el entrenamiento del modelo.

#Transformo las categorías en números con LabelEncoder

le = LabelEncoder()
df_cat = df_cat.apply(le.fit_transform)

#Uno de nuevo el df_cat ya convertido a números y df_num
data_completo= pd.concat([df_cat, df_num], axis=1)

#Mi variable objetivo es Timely reponse? y quiero ver si hay imbalanceo y podría haber problemas para predecir la clase minoritaria
distribucion_clases = data_completo['Timely response?'].value_counts()
print(distribucion_clases)

#Normalizo el Dataframe ya que hay valores muy dispares
#Quito la variable objetivo para normalizar pero primer la guardo
columna_objetivo = data_completo['Timely response?']  # Serie
data_completo_2 = data_completo.drop('Timely response?', axis=1)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(data_completo_2)
df_scaled = pd.DataFrame(df_scaled, columns=data_completo_2.columns)
#Vuelvo a añadir la columna de la variable objetivo,si no le pongo .values no me une bien esta columna de vuelta y me salen NaN
df_scaled['Timely response?'] = columna_objetivo.values

#Paso el DataFrame a csv
df_scaled .to_csv('df_scaled.csv', index=False)

df.info()
