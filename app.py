import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix 
from PIL import Image

st.set_page_config(layout="wide")
st.title("Análisis de Satisfacción de Pasajeros de una Aerolínea")

# Carga del dataset
@st.cache_data
def cargar_datos():
    df = pd.read_csv("satisfaccion_aerolinea.csv")
    return df

st.subheader("Introducción")

st.markdown("""
En este proyecto se tratará la creación y el estudio de un programa de **predicción del nivel de satisfacción** 
de los pasajeros de una aerolínea, utilizando para ello algoritmos de aprendizaje automático sobre datos reales 
recogidos de las experiencias de usuarios.

El objetivo es comprender qué factores influyen más en la percepción del cliente, y construir un modelo predictivo 
que permita anticipar si un pasajero estará satisfecho o no.
""")

st.subheader("El porqué de esta elección")
st.markdown("""
De las pocas veces que he tenido la oportunidad de viajar en avión, he disfrutado no solo del vuelo en sí, 
sino también de todo el proceso: desde el embarque hasta la facturación.  
Esto me llevó a, en el momento de tener que elegir un tema para este trabajo, inclinarme por analizar la 
valoración y satisfacción (o no) de las personas que han volado.  

Más allá de la motivación personal, es sabido que cualquier empresa busca, al menos en teoría, obtener 
feedback de sus clientes sobre los servicios ofrecidos y en qué puntos se podrían mejorar.  
En el sector aéreo, mejorar esta experiencia puede suponer una ventaja competitiva significativa.  
El uso de IA y modelos predictivos ofrece a las aerolíneas una herramienta poderosa para actuar de 
forma proactiva: mejorar la calidad del servicio, reducir costes y fidelizar clientes.
""")

st.subheader("Obtención y tratamiento de datos")
st.markdown("""
La búsqueda de un conjunto de datos acorde con la idea del proyecto se inició primero en páginas de datasets 
públicos como [datos.gob.es](https://datos.gob.es) o el Instituto Nacional de Estadística.  
Tras no encontrar un conjunto de datos satisfactorio, se pasó a buscar en la plataforma Kaggle, donde 
finalmente se eligió el dataset **"Passenger Satisfaction"**, creado por el usuario **John D**.
Puedes consultarse en el siguiente enlace:  
[https://www.kaggle.com/datasets/johndddddd/customer-satisfaction/data](https://www.kaggle.com/datasets/johndddddd/customer-satisfaction/data)
""")

# Cargar la imagen
image = Image.open("imagenes/imagen_passenger_satisaction.png")
# Mostrarla
st.image(image)

st.subheader("Propuesta de proyecto")
st.markdown("""
Lo que se propone en este proyecto es utilizar las valoraciones que una aerolínea ha recogido de sus usuarios —desde la compra del billete online hasta el uso del wifi en vuelo—, incluyendo también una valoración final del cliente sobre si ha estado satisfecho o no.  

Con estos datos, se construirá un modelo capaz de predecir si un cliente estará satisfecho, sin necesidad de que lo indique explícitamente.  
Una vez entrenado el modelo, se analizará qué factores tienen mayor influencia en la percepción de los usuarios, para detectar posibles áreas de mejora por parte de la aerolínea.
""")

df_model = cargar_datos()
df_model.drop(columns=['Unnamed: 0'], inplace=True)

#  Mostrar una vista previa
st.subheader("Vista del conjunto de datos")
st.dataframe(df_model.head(7))


# Codificación para variables categóricas
df_model['Arrival Delay in Minutes'].fillna(0, inplace = True)
df_model.drop(columns=['id'], inplace=True)
columnas_cat = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in columnas_cat:
    df_model[col] = df_model[col].astype('category')
    df_model[col] = df_model[col].cat.codes


X = df_model.drop(['satisfaction'], axis=1)
y = df_model['satisfaction']

# Convertir categóricas a dummy variables
X = pd.get_dummies(X)

# Train/test, split y modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred_rfm = modelo_rf.predict(X_test)

st.subheader("Elección y justificación de modelos o técnicas de IA/ML.")

st.markdown("""
Se eligió el modelo de aprendizaje supervisado, ya que el objetivo del proyecto es predecir una categoría específica 
– la satisfacción del cliente–, dada un conjunto de características observadas. Dado que la variable objetivo en los 
datos está etiquetada, ‘satisfaction’, el aprendizaje supervisado es la opción obvia. 
Además, elegimos el algoritmo Random Forest porque tiene una alta precisión, ya que maneja bien tanto las variables 
categóricas como las numéricas, y ofrece una medida de la importancia de cada variable. Esto es importante para 
nuestro objetivo, ya que podemos identificar qué factor es el más influyente en la satisfacción del pasajero.
""")

# Tabla de los modelos
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
st.table(tabla_modelos)

st.header("Resultados de Random Forest")

st.markdown("""
El modelo obtuvo una **precisión aproximada del 96%**, lo cual indica una alta capacidad de acierto en la 
predicción del nivel de satisfacción de los pasajeros.

Otras métricas, como el Error Cuadrático Medio, el Error Absoluto Medio o el Coeficiente de Determinación, 
sugieren que el modelo también **captura con eficacia** la relación entre las variables de entrada y el 
objetivo. Aunque estas métricas son más comunes en modelos de regresión, pueden proporcionar una idea 
general de la calidad del modelo:

""")
col1, col2, col3 = st.columns(3)
col1.metric("MSE", "0.04")
col2.metric("MAE", "0.04")
col3.metric("R²", "0.82")
st.markdown("""
---
            
A continuación se muestra la matriz de confusión, que permite visualizar el rendimiento del modelo respecto a 
las clases predichas frente a las reales:
""")

# Calcular la matriz
cm = confusion_matrix(y_test, y_pred_rfm)

# Mostrarla como gráfico
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Neutral o no satisfechos", "Satisfechos"],
    yticklabels=["Neutral o no satisfechos", "Satisfechos"],
    ax=ax
)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusión Random Forest")
st.pyplot(fig)

# Extraer valores y calcular tasas de error
TP = cm[0, 0]
FN = cm[0, 1]
FP = cm[1, 0]
TN = cm[1, 1]

fn_rate = FN / (TP + FN) * 100
fp_rate = FP / (FP + TN) * 100

st.markdown(f"""
- **Errores tipo FN** (se predijo "no satisfecho" pero era "satisfecho"): `{fn_rate:.2f}%`
- **Errores tipo FP** (se predijo "satisfecho" pero era "no satisfecho"): `{fp_rate:.2f}%`
""")

st.markdown(f"""
Lo que más nos interesa del modelo resultante es como este realizo la predicción y que 
grado de importancia tuvo cada uno de los valores. Mediante el atributo ‘feature_importances_’ 
del modelo Random Forest podemos obtener la importancia relativa de cada variable (feature) 
en las predicciones del modelo, como se puede ver:
""")

# Tabla de importancia de características
datos_tabla_persos = {
    'Característica': [
        'Class', 'Online boarding', 'Inflight wifi service', 'Type of Travel',
        'Inflight entertainment', 'Seat comfort', 'Ease of Online booking',
        'Age', 'Customer Type', 'Flight Distance', 'Inflight service',
        'Checkin service', 'Baggage handling', 'Cleanliness',
        'Leg room service', 'On-board service', 'Gate location',
        'Departure/Arrival time convenient', 'Arrival Delay in Minutes',
        'Food and drink', 'Departure Delay in Minutes', 'Gender'
    ],
    'Importancia (%)': [
        16.25, 14.55, 13.32, 7.81, 5.99, 4.12, 4.01, 3.49, 3.44, 3.37,
        3.15, 3.14, 2.78, 2.70, 2.25, 2.23, 1.80, 1.47, 1.30, 1.26, 1.13, 0.46
    ]
}

# Crear DataFrame
tabla_pesos = pd.DataFrame(datos_tabla_persos)

# Mostrar tabla en Streamlit
st.title("Importancia de Características - Random Forest")
st.dataframe(tabla_pesos.style.format({'Importancia (%)': '{:.2f}'}))


st.markdown("""
-   Las tres primeras variables más importantes — **Class**, **Online boarding** y 
    **Type of Travel** — abarcan aproximadamente el **44% del peso total** del modelo Random Forest.
    Si a estas se les suman otras tres variables relevantes, como **Inflight wifi service** 
    e **Inflight entertainment**, el peso acumulado alcanza alrededor del **58%**.

-   A partir de la sexta variable, **Seat comfort**, el **peso individual de cada variable 
    empieza a descender notablemente**, situándose generalmente entre el **2% y el 4%** del total.

-   Finalmente, las variables menos influyentes — desde **Departure/Arrival time convenient** 
    hacia abajo — no llegan al **1,5%** de importancia individual en el modelo.
""")

st.subheader("Interpretación de Resultados")

st.markdown("""
Los resultados del modelo Random Forest destacan la importancia de varias variables clave en la 
predicción de la satisfacción de los pasajeros:

-   **Class (16,25%)**: La clase en la que viaja el pasajero (Business, Eco, etc.) es el factor 
    más influyente. Tiene sentido, ya que el nivel de comodidad y atención varía considerablemente 
    según la clase elegida.

-   **Online boarding (14,55%)**: Representa la experiencia del pasajero al hacer el check-in online. 
    Un proceso fácil y rápido mejora notablemente la percepción del servicio.

- **Type of Travel (7,81%)** y **Customer Type (3,44%)**: Viajar por negocios o por placer y ser un 
    cliente habitual o nuevo también influyen en la satisfacción. Por ejemplo, los pasajeros frecuentes 
    pueden tener expectativas más claras o elevadas.

- **Inflight wifi service (13,32%)**: La disponibilidad y calidad del wifi a bordo se valora mucho 
    actualmente, sobre todo por quienes necesitan mantenerse conectados durante el vuelo.
            
En general, los resultados muestran que lo más importante para los pasajeros no es solo lo que pasa 
dentro del avión, sino también cómo se sienten desde antes de volar. Cosas como la clase en la que viajan, 
si el viaje es por trabajo o por ocio, y si pueden hacer todo fácil desde el móvil (reserva y embarque) 
pesan mucho en cómo ven todo el servicio. 
""")


# Accuracy
accuracy = accuracy_score(y_test, modelo_rf.predict(X_test))
st.success(f"Accuracy del modelo: {accuracy:.2f}")

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
st.subheader("Formulario de predicción")
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
    resultado = "Satisfecho" if pred == 1 else "No satisfecho"
    st.subheader("Resultado de la predicción:")
    st.info(resultado)

