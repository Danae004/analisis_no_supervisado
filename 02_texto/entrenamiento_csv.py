import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import joblib

# Leer archivo Excel
df = pd.read_excel("noticias.xlsx")

# Lista de textos
titulos_noticias = df["Noticia"].dropna().tolist()

# Configuración NLTK
nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

spanish_stop_words = stopwords.words("spanish")

# Vectorización TF-IDF
vectorizador = TfidfVectorizer(stop_words=spanish_stop_words)
X = vectorizador.fit_transform(titulos_noticias)

# Entrenamiento del modelo KMeans
modelo = KMeans(n_clusters=3, random_state=1234, n_init=10)
modelo.fit(X)

# Guardar modelo entrenado
joblib.dump(modelo, "modelo_texto.pkl")

# Crear DataFrame con resultados
resultados = pd.DataFrame({
    "Noticia": titulos_noticias,
    "Cluster": modelo.labels_
})

# Ordenar por número de cluster
resultados = resultados.sort_values(by="Cluster").reset_index(drop=True)

# Imprimir
for _, row in resultados.iterrows():
    print(f"Cluster {row['Cluster']}: {row['Noticia']}")
