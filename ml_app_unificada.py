
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="🧠 Machine Learning Didáctico", layout="wide")
st.title("🧠 Machine Learning desde Cero - App Educativa")

# Introducción teórica
st.header("📘 ¿Qué es Machine Learning?")
st.markdown("""
Machine Learning (ML) es una rama de la inteligencia artificial que permite a las máquinas aprender automáticamente a partir de los datos sin ser explícitamente programadas.

## 📚 Tipos de ML
- **Supervisado:** Se entrena con datos etiquetados (ej: precio de casas)
- **No Supervisado:** Se descubren patrones sin etiquetas (ej: agrupamiento de clientes)
- **Por Refuerzo:** El modelo aprende a partir de recompensas y errores (ej: videojuegos o robots)

## 📐 Fundamentos Matemáticos
- **Regresión lineal:** \( y = a \cdot x + b \)
- **Distancia Euclidiana en KNN:** \( d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} \)
- **Función de activación en redes neuronales:** \( \sigma(x) = \frac{1}{1 + e^{-x}} \)

## 📖 Documentación recomendada
- [Scikit-learn](https://scikit-learn.org/stable/)
- [DeepLearning.ai](https://www.deeplearning.ai/)
- Libro: “Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow”
""")

st.divider()

# 1. KNN
st.subheader("📊 Clasificación con KNN")
st.markdown("Clasifica frutas según peso y textura (1 = suave, 0 = rugosa).")

data = {
    "peso": [150, 180, 200, 120, 170, 130, 160, 110, 190, 210],
    "textura": [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    "fruta": ["manzana", "manzana", "manzana", "naranja", "manzana", "naranja", "manzana", "naranja", "manzana", "manzana"]
}
df = pd.DataFrame(data)
X = df[["peso", "textura"]]
y = df["fruta"]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
peso = st.slider("📦 Peso de fruta (g)", 100, 220, 150, key="peso_knn")
textura = st.radio("🧸 Textura", ["Suave", "Rugosa"], key="textura_knn")
textura_val = 1 if textura == "Suave" else 0
pred = knn.predict([[peso, textura_val]])[0]
st.success(f"🔍 Fruta clasificada: **{pred}**")

fig1, ax1 = plt.subplots()
for f in df["fruta"].unique():
    subset = df[df["fruta"] == f]
    ax1.scatter(subset["peso"], subset["textura"], label=f)
ax1.scatter(peso, textura_val, color="red", label="Tu fruta", marker="x", s=100)
ax1.legend(); ax1.set_xlabel("Peso"); ax1.set_ylabel("Textura")
st.pyplot(fig1)

st.divider()

# 2. Regresión Lineal
st.subheader("📈 Regresión Lineal - Precio de casas")
st.markdown("Relación entre tamaño de casa (m2) y su precio estimado.")

np.random.seed(0)
m2 = np.random.randint(40, 200, 50)
precio = m2 * 3000 + np.random.normal(0, 20000, 50).astype(int)
df2 = pd.DataFrame({"m2": m2, "precio": precio})
X2 = df2[["m2"]]; y2 = df2["precio"]
modelo = LinearRegression().fit(X2, y2)
m2_input = st.slider("📐 Tamaño de casa (m2)", 40, 200, 100)
pred2 = modelo.predict([[m2_input]])[0]
st.success(f"💰 Precio estimado: ${int(pred2):,}")

fig2, ax2 = plt.subplots()
sns.regplot(data=df2, x="m2", y="precio", ax=ax2)
ax2.scatter(m2_input, pred2, color="red", label="Tu casa"); ax2.legend()
st.pyplot(fig2)

st.divider()

# 3. Clustering
st.subheader("🔍 Agrupamiento con KMeans")
st.markdown("Se agrupan puntos según cercanía sin usar etiquetas.")

X3, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.5, random_state=42)
k = st.slider("📌 Número de clusters", 2, 6, 4)
modelo3 = KMeans(n_clusters=k, n_init=10)
y_pred = modelo3.fit_predict(X3)

fig3, ax3 = plt.subplots()
ax3.scatter(X3[:, 0], X3[:, 1], c=y_pred, cmap='viridis')
centros = modelo3.cluster_centers_
ax3.scatter(centros[:, 0], centros[:, 1], c='red', s=200, label='Centros')
ax3.legend()
st.pyplot(fig3)

st.divider()

# 4. Red Neuronal
st.subheader("🧠 Clasificación de Dígitos con Red Neuronal")
st.markdown("Red neuronal multicapa (MLP) que clasifica imágenes de dígitos (0–9).")

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=42)
modelo4 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
modelo4.fit(X_train, y_train)
idx = st.slider("🔢 Imagen de dígito", 0, len(digits.images)-1, 10)
fig4, ax4 = plt.subplots()
ax4.imshow(digits.images[idx], cmap='gray')
st.pyplot(fig4)
pred4 = modelo4.predict([digits.data[idx]])[0]
st.success(f"🧾 Predicción del modelo: **{pred4}** (real: {digits.target[idx]})")
