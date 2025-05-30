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
Lo que se propone en este proyecto es utilizar las valoraciones que una aerolínea ha recogido de sus usuarios 
—desde la compra del billete online hasta el uso del wifi en vuelo—, incluyendo también una valoración final 
del cliente sobre si ha estado satisfecho o no.  

Con estos datos, se construirá un modelo capaz de predecir si un cliente estará satisfecho, sin necesidad de 
que lo indique explícitamente. Una vez entrenado el modelo, se analizará qué factores tienen mayor influencia 
en la percepción de los usuarios, para detectar posibles áreas de mejora por parte de la aerolínea.
""")

df_original = cargar_datos()

#  Mostrar una vista previa
st.subheader("Vista del conjunto de datos del dataset:")
st.dataframe(df_original.head(7))

st.header("Evaluación de la calidad y adecuación del conjunto de datos")

st.markdown("""
Este dataset contiene un total de **129.880 filas** y **24 columnas**. Las columnas incluidas en el conjunto de datos son:

- **Gender**: Género del pasajero (Female, Male)
- **Customer Type**: Tipo de cliente (Loyal customer, Disloyal customer)
- **Age**: Edad
- **Type of Travel**: Tipo de viaje (Personal, Business)
- **Class**: Clase del vuelo (Business, Eco, Eco Plus)
- **Flight Distance**: Distancia del vuelo

**Valoraciones del servicio del 0 al 5**, donde 0 indica "No aplicable" (es decir, el cliente no usó o no valoró ese servicio):
- Inflight wifi service
- Departure/Arrival time convenient
- Ease of Online booking
- Gate location
- Food and drink
- Online boarding
- Seat comfort
- Inflight entertainment
- On-board service
- Leg room service
- Baggage handling
- Check-in service
- Inflight service
- Cleanliness

- **Departure Delay in Minutes**
- **Arrival Delay in Minutes**
- **Satisfaction**: Nivel de satisfacción (Satisfied, Neutral or Dissatisfied)
""")

st.markdown("""
Con esto se puede ver que se trata de un conjunto de datos bastante extenso. Del análisis realizado 
se destacan los siguientes puntos:

-   Las columnas de valoraciones del servicio requieren interpretación específica: los valores 0 
    **no significan necesariamente una mala experiencia**, sino que el servicio no fue utilizado. 
    Por tanto, **se conservaron** porque aportan información sobre la disponibilidad o el uso de los servicios.
-   La columna **Satisfaction**, que será la variable objetivo, tiene dos valores:
    - Neutral or Dissatisfied: 73.452 registros
    - Satisfied: 56.428 registros  
    Esto representa una distribución **relativamente equilibrada**, lo que **no requiere técnicas de balanceo** como sobremuestreo o submuestreo.
-   Solo se encontraron **393 valores nulos** en la columna Arrival Delay in Minutes.
""")

st.header("Procesado, limpieza y transformación de los datos")

st.markdown("""
-   Los valores nulos en`Arrival Delay in Minutes fueron reemplazados por 0, asumiendo que son vuelos sin retraso registrado.
-   La única columna eliminada fue **id**, al no aportar valor al modelo.
-   Las variables categóricas (como Type of Travel, Customer Type, Gender, etc.) fueron **codificadas numéricamente** 
    para que los algoritmos de machine learning pudieran procesarlas correctamente.
Una vez el dataset ha sido creado se dispone se parar los datos de entrada y salida, y los datos de entrenamiento 
y de prueba del modelo:
            
X = df.drop(columns=['satisfaction'])\n
y = df['satisfaction']\n

Entrenamiento: 20 % de test\n
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n           

Escalar features con StandardScaler\n
scaler = StandardScaler()\n
X_train_scaled = scaler.fit_transform(X_train)\n
X_test_scaled = scaler.transform(X_test)\n

Reducir la dimensionalidad del dataset, manteniendo un 95% de la varianza.\n
pca = PCA(n_components=0.95)\n
X_train_pca = pca.fit_transform(X_train_scaled)\n
X_test_pca = pca.transform(X_test_scaled)\n         
           
""")


# Codificación para variables categóricas
df_model = df_original.copy()
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

#  Mostrar el dataset ya procesado
st.subheader("Vista del dataset ya procesado:")
st.dataframe(df_model.head(7))

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
        "**95,58**",
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

# Mostrar matriz
fig, ax = plt.subplots(figsize=(4.5, 4))
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
#plt.tight_layout()

# Mostrar en Streamlit
st.pyplot(fig, use_container_width=False)

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

# Importancia de las variables
feature_importances = pd.Series(modelo_rf.feature_importances_, index=X_train.columns)
feature_importances_percent = (feature_importances * 100).round(2)
feature_importances_percent_sorted = feature_importances_percent.sort_values(ascending=False)

# Convertir a un dataFrame
importance_df = feature_importances_percent_sorted.reset_index()
importance_df.columns = ['Feature', 'Importance (%)']

# Grafico de las variables
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance (%)', y='Feature', palette='Blues_d', ax=ax)
ax.set_title('Importancia de características (%) - Random Forest')
ax.set_xlabel('Importancia (%)')
ax.set_ylabel('Características')
plt.tight_layout()

# Mostrar en Streamlit
st.pyplot(fig, use_container_width = False)

# Crear DataFrame
tabla_pesos = pd.DataFrame(datos_tabla_persos)

# Mostrar tabla en Streamlit
st.title("Importancia de los Valores - Random Forest")
st.dataframe(tabla_pesos.style.format({'Importancia (%)': '{:.2f}'}), width=400)


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

# Gráficos del dataset original
# Clase
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.countplot(data=df_original, x='Class', hue='satisfaction', palette='Set2', ax=ax)

ax.set_title('Satisfacción según las clases clase')
ax.set_xlabel('Clase del vuelo')
ax.set_ylabel('Cantidad')
ax.legend(title='Satisfacción')
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# Tipo de viaje
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.countplot(data=df_original, x='Type of Travel', hue='satisfaction', palette='Set2', ax=ax)

ax.set_title('Satisfacción según el tipo de viaje')
ax.set_xlabel('Tipo de viaje')
ax.set_ylabel('Cantidad')
ax.legend(title='Satisfacción')
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# Wifi
valoracion_wifi = df_original['Inflight wifi service'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(4.5, 4))
sns.barplot(x=valoracion_wifi.index, y=valoracion_wifi.values, palette="Blues", ax=ax)

ax.set_title("Valoraciones del servicio de WiFi (0 = no aplicable)")
ax.set_xlabel("Valoración")
ax.set_ylabel("Cantidad de pasajeros")
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# Wifi
valoracion_boarding = df_original['Online boarding'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(4.5, 4))
sns.barplot(x=valoracion_boarding.index, y=valoracion_boarding.values, palette="Blues", ax=ax)

ax.set_title("Valoraciones del Online boarding (0 = no aplicable)")
ax.set_xlabel("Valoración")
ax.set_ylabel("Cantidad de pasajeros")
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# Satisfaccion por edad
df_prov = df_original.copy()
df_prov['satisfaction_binaria'] = df_prov['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# Agrupar por edad y calcular satisfacción media (porcentaje de satisfechos)
satisfaccion_por_edad = df_prov.groupby('Age')['satisfaction_binaria'].mean().reset_index()

# Crear gráfico
fig, ax = plt.subplots()
ax.plot(satisfaccion_por_edad['Age'], satisfaccion_por_edad['satisfaction_binaria'], color='green')
ax.set_title("Proporción de pasajeros satisfechos según la edad")
ax.set_xlabel("Edad")
ax.set_ylabel("Proporción de satisfacción (0-1)")
plt.tight_layout()
st.pyplot(fig, use_container_width=False)


# Accuracy
accuracy = accuracy_score(y_test, modelo_rf.predict(X_test))
st.success(f"Accuracy del modelo: {accuracy:.2f}")


# Diccionarios de mapeo para variables categóricas
gender_map = {'Male': 1, 'Female': 0}
customer_type_map = {'Loyal Customer': 1, 'disloyal Customer': 0}
travel_type_map = {'Business travel': 1, 'Personal Travel': 0}
travel_class_map = {'Business': 2, 'Eco Plus': 1, 'Eco': 0}

# Formulario de predicción
st.subheader("Formulario de predicción")
with st.form("form_prediccion"):

    gender_label = st.selectbox("Género", list(gender_map.keys()))
    customer_type_label = st.selectbox("Tipo de cliente", list(customer_type_map.keys()))
    age = st.slider("Edad", 7, 85, 30)
    travel_type_label = st.selectbox("Tipo de viaje", list(travel_type_map.keys()))
    travel_class_label = st.selectbox("Clase", list(travel_class_map.keys()))

    gender = gender_map[gender_label]
    customer_type = customer_type_map[customer_type_label]
    travel_type = travel_type_map[travel_type_label]
    travel_class = travel_class_map[travel_class_label]

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

