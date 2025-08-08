import numpy as np
import joblib

modelo = joblib.load("modelo_segmentacion_clientes.pkl")

datos_prueba = np.array([
    [50,3],
    [600,4]
])

clusters = modelo.predict(datos_prueba)
print(clusters)