# Imports
import streamlit as st
import pickle
import numpy as np

# Título y descripción
st.title('🌸 Predicción del Tipo de Flor Iris')
st.subheader('Ingrese las características de la flor y vea el resultado')
st.text('Este modelo utiliza las características de Iris para predecir su especie.')

# Cargar el modelo serializado
import os

# Define la ruta relativa (retrocedemos un nivel con '..' y luego accedemos a 'Data')
file_path = os.path.join('..', '..', 'Data', 'modelo_prueba_iris.pkl')

# Abre el archivo
with open(file_path, 'rb') as file:
    model = pickle.load(file)

# Crear sliders para ingresar las características de Iris
st.sidebar.header('📊 Características de la Flor Iris')
sepal_length = st.sidebar.slider('Sepal Length (cm)', 0.0, 10.0, 5.0)
sepal_width = st.sidebar.slider('Sepal Width (cm)', 0.0, 10.0, 3.0)
petal_length = st.sidebar.slider('Petal Length (cm)', 0.0, 10.0, 4.0)
petal_width = st.sidebar.slider('Petal Width (cm)', 0.0, 10.0, 1.0)

# Crear un botón para predecir
if st.sidebar.button('Predecir Tipo de Flor'):
    # Preparar los datos como un array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]  # Realizar la predicción
    species_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.write(f'🌼 **El modelo predice que la flor es: `{species_dict[prediction]}`**')

# Mostrar una imagen de las flores Iris
st.image('https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg', caption='Flores Iris', use_column_width=True)

# Agregar un footer
st.markdown(
    """
    ---
    **Creado por [Tu Nombre]**  
    Este modelo fue entrenado para fines educativos usando datos de Iris.
    """
)
