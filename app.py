import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix 

st.set_page_config(layout="wide")
st.title("✈️ Análisis de Satisfacción de Pasajeros de una Aerolínea")

# 1. Carga del dataset
@st.cache_data
def cargar_datos():
    df = pd.read_csv("train.csv")
    return df

df = cargar_datos()

st.subheader("Introducción")

st.markdown("""
En este proyecto se tratará la creación y el estudio de un programa de **predicción del nivel de satisfacción** 
de los pasajeros de una aerolínea, utilizando para ello algoritmos de aprendizaje automático sobre datos reales 
recogidos de las experiencias de usuarios.

El objetivo es comprender qué factores influyen más en la percepción del cliente, y construir un modelo predictivo 
que permita anticipar si un pasajero estará satisfecho o no.
""")

# 2. Mostrar una vista previa
st.subheader("Vista previa del conjunto de datos")
st.dataframe(df.head())

import streamlit as st
from PIL import Image

# Cargar la imagen
image = Image.open("imagenes/Captura de pantalla 2025-05-28 192158.png").resize((600, 400))
# Mostrarla
st.image(image)


datos_tabla_modelos = {
    "Datos de entrenamiento": [
        "Random Forest Classifier",
        "Decision Tree Classifier",
        "KNeighbors Classifier",
        "SVM Linear"
    ],
    "Datos normales": [
        "95,58",
        "94,52",
        "64,2 (k = 20)",
        "-"
    ],
    "Datos escalados": [
        "95,54",
        "94,56",
        "93,0 (k = 9)",
        "87,59"
    ],
    "Análisis de Componentes Principales\n(95 % varianza)": [
        "91,63",
        "88,77",
        "82,0 (k = 20)",
        "95,36"
    ]
}

tabla_modelos = pd.DataFrame(datos_tabla_modelos)

# Mostrar tabla en Streamlit
st.subheader("📊 Comparativa de modelos y su precision")
st.table(tabla_modelos)


st.header("✈️ Predicción de satisfacción")

# Codificación para variables categóricas
df_model = df.copy()
df_model.dropna(inplace=True)
df_model['satisfaction'] = df_model['satisfaction'].apply(lambda x: 1 if x == "satisfied" else 0)

X = df_model.drop(['id', 'satisfaction'], axis=1)
y = df_model['satisfaction']

# Convertir categóricas a dummy variables
X = pd.get_dummies(X)

# Train/test, split y modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, modelo_rf.predict(X_test))
st.success(f"📊 Accuracy del modelo: {accuracy:.2f}")

# Importancia de las variables
feature_importances = pd.Series(modelo_rf.feature_importances_, index=X_train.columns)
feature_importances_percent = (feature_importances * 100).round(2)
feature_importances_percent_sorted = feature_importances_percent.sort_values(ascending=False)

# Convertir a DataFrame
importance_df = feature_importances_percent_sorted.reset_index()
importance_df.columns = ['Feature', 'Importance (%)']

# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance (%)', y='Feature', palette='Blues_d', ax=ax)
ax.set_title('Importancia de características (%) - Random Forest')
ax.set_xlabel('Importancia (%)')
ax.set_ylabel('Características')
plt.tight_layout()

# Mostrar en Streamlit
st.pyplot(fig)

# Formulario de predicción
st.subheader("🧪 Simula un pasajero para ver su satisfacción")
with st.form("form_prediccion"):
    gender = st.selectbox("Género", df_model['Gender'].unique())
    customer_type = st.selectbox("Tipo de cliente", df_model['Customer Type'].unique())
    age = st.slider("Edad", 7, 85, 30)
    travel_type = st.selectbox("Tipo de viaje", df_model['Type of Travel'].unique())
    travel_class = st.selectbox("Clase", df_model['Class'].unique())
    departure_convenient = st.slider("Hora de salida/llegada conveniente", 0, 5, 3)
    flight_distance = st.slider("Distancia del vuelo", 30, 5000, 1000)
    online_booking = st.slider("Facilidad de reserva online", 0, 5, 3)
    wifi = st.slider("Servicio wifi (0 es 'no aplicable')", 0, 5, 3)
    food = st.slider("Comida y bebida (0 es 'no aplicable')", 0, 5, 3)
    online_boarding = st.slider("Embarque online (0 es 'no aplicable')", 0, 5, 3)
    gate_location = st.slider("Ubicación de la puerta de embarque (0 es 'no aplicable')", 0, 5, 3)
    seat_comfort = st.slider("Comodidad del asiento (0 es 'no aplicable')", 0, 5, 3)
    entertainment = st.slider("Entretenimiento a bordo (0 es 'no aplicable')", 0, 5, 3)
    onboard_service = st.slider("Servicio a bordo (0 es 'no aplicable')", 0, 5, 3)
    leg_room = st.slider("Satisfacción con el espacio para las piernas (0 es 'no aplicable')", 0, 5, 3)
    baggage = st.slider("Satisfacción con el manejo del equipaje (0 es 'no aplicable')", 0, 5, 3)
    checkin = st.slider("Satisfacción con el Check-in (0 es 'no aplicable')", 0, 5, 3)
    inflight_service = st.slider("Atención en vuelo (0 es 'no aplicable')", 0, 5, 3)
    cleanliness = st.slider("Satisfacción con la limpieza en el avión", 0, 5, 3)
    dep_delay = st.number_input("Minutos de retraso en la salida", min_value=0)
    arr_delay = st.number_input("Minutos de retraso en la llegada", min_value=0)

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
        'Departure/Arrival time convenient': [departure_convenient],
        'Ease of Online booking': [online_booking],
        'Gate location': [gate_location],
        'Food and drink': [food],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [entertainment],
        'On-board service': [onboard_service],
        'Leg room service': [leg_room],
        'Baggage handling': [baggage],
        'Check-in service': [checkin],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
        'Departure Delay in Minutes': [dep_delay],
        'Arrival Delay in Minutes': [arr_delay]
    })

    input_encoded = pd.get_dummies(input_df)
    # Igualar columnas a X
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    pred = modelo_rf.predict(input_encoded)[0]
    resultado = "✅ Satisfecho" if pred == 1 else "❌ No satisfecho"
    st.subheader("Resultado de la predicción:")
    st.info(resultado)

