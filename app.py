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

import streamlit as st
from PIL import Image

# Cargar la imagen
image = Image.open("imagenes/Captura de pantalla 2025-05-28 192158.png")
new_image = image.resize((600, 400))
# Mostrarla
st.image(new_image)

# 3. Filtros interactivos
st.sidebar.title("Filtros")
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

st.header("✈️ Predicción de satisfacción")

# Codificación para variables categóricas
df_model = df.copy()
df_model.dropna(inplace=True)
df_model['satisfaction'] = df_model['satisfaction'].apply(lambda x: 1 if x == "satisfied" else 0)

X = df_model.drop(['id', 'satisfaction'], axis=1)
y = df_model['satisfaction']

# Convertir categóricas a dummy variables
X = pd.get_dummies(X)

# Train/Test split y modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Mostrar accuracy
accuracy = accuracy_score(y_test, modelo_rf.predict(X_test))
st.success(f"📊 Accuracy del modelo: {accuracy:.2f}")

# Formulario de predicción
st.subheader("🧪 Simula un pasajero para ver su satisfacción")
with st.form("form_prediccion"):
    gender = st.selectbox("Género", df_model['Gender'].unique())
    customer_type = st.selectbox("Tipo de cliente", df_model['Customer Type'].unique())
    age = st.slider("Edad", 7, 85, 30)
    travel_type = st.selectbox("Tipo de viaje", df_model['Type of Travel'].unique())
    travel_class = st.selectbox("Clase", df_model['Class'].unique())
    flight_distance = st.slider("Distancia del vuelo", 30, 5000, 1000)
    wifi = st.slider("Servicio wifi (0-5)", 0, 5, 3)
    food = st.slider("Comida y bebida (0-5)", 0, 5, 3)
    online_boarding = st.slider("Embarque online (0-5)", 0, 5, 3)

    submit = st.form_submit_button("Predecir satisfacción")

if submit:
    input_df = pd.DataFrame({
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Age': [age],
        'Type of Travel': [travel_type],
        'Class': [travel_class],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [wifi],
        'Food and drink': [food],
        'Online boarding': [online_boarding]
    })

    input_encoded = pd.get_dummies(input_df)
    # Igualar columnas a X
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    pred = modelo_rf.predict(input_encoded)[0]
    resultado = "✅ Satisfecho" if pred == 1 else "❌ No satisfecho"
    st.subheader("Resultado de la predicción:")
    st.info(resultado)

