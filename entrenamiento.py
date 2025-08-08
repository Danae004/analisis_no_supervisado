import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

# 1. Importar el Dataset con los datos de entrenamiento

df_datos_clientes = pd.read_csv("dataset/clientes_entrenamiento.csv")

print(df_datos_clientes.info())
print(df_datos_clientes.head())

#2. Convertir el Dataframe a un Array de Numpy
X = df_datos_clientes.values
print(X)

# 3. Entrenar el modelo
modelo = KMeans(n_clusters=2, random_state=1234, n_init=10)
modelo.fit(X)

# Guardar el modelo
joblib.dump(modelo, "modelo_segmentacion_clientes.pkl")

# 4. Análisis del modelo
df_datos_clientes['clusters'] = modelo.labels_
analisis = df_datos_clientes.groupby('clusters').mean()
print(analisis)
# 5. Graficar los clusters
centroides = modelo.cluster_centers_
etiquetas = modelo.labels_

cluster0 = X[etiquetas == 0]
cluster1 = X[etiquetas == 1]
cluster2 = X[etiquetas == 2]

# Colocar los puntos de cada cluster
plt.scatter(cluster0[:,0],cluster0[:,1], c='red', label='clientes de temporada')
plt.scatter(cluster1[:,0],cluster1[:,1], c='blue', label='Clientes VIP')
plt.scatter(cluster2[:,0],cluster2[:,1], c='green', label='clientes de ofertas')

# Colocar los centroides de cada cluster
plt.scatter(centroides[:,0],centroides[:,1],marker='x', c='black', label='centroides')

# Colocar titulos y etiquetas
plt.title('Segmentacion de clientes')
plt.xlabel('Gasto total')
plt.ylabel('Vistas')
plt.legend()
plt.grid(True)

os.makedirs('graficas', exist_ok=True)
plt.savefig('graficas/clusters.png')