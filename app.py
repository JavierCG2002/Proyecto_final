import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")
st.title("✈️ Análisis de Satisfacción de Pasajeros de Aerolínea")

# 1. Cargar los datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("train.csv")  # Asegúrate de tener el CSV localmente
    return df

df = cargar_datos()

# 2. Mostrar una vista previa
st.subheader("Vista previa del conjunto de datos")
st.dataframe(df.head())

# 3. Filtros interactivos
st.sidebar.title("🎛️ Filtros")
tipo_cliente = st.sidebar.selectbox("Tipo de Cliente", df['Customer Type'].unique())
df_filtrado = df[df['Customer Type'] == tipo_cliente]

# 4. Visualización de satisfacción por tipo de cliente
st.subheader(f"Satisfacción de clientes '{tipo_cliente}'")
fig, ax = plt.subplots()
sns.countplot(data=df_filtrado, x="satisfaction", palette="pastel", ax=ax)
st.pyplot(fig)

# 5. Opinión sobre el wifi de los NO satisfechos
st.subheader("Opinión sobre el wifi (clientes no satisfechos)")
wifi_opinion = df[df["satisfaction"] == "neutral or dissatisfied"]["Inflight wifi service"]
fig2, ax2 = plt.subplots()
wifi_opinion.value_counts().sort_index().plot.pie(
    autopct="%1.1f%%", startangle=90, colors=sns.color_palette("Blues"), ax=ax2
)
ax2.set_ylabel("")
ax2.set_title("Opinión sobre el wifi")
st.pyplot(fig2)
