# Imports
import streamlit as st
import pickle
import numpy as np
import os

# Título y descripción
st.title('🌸 Predicción del Tipo de Flor Iris')
st.subheader('Ingrese las características de la flor y vea el resultado')
st.text('Este modelo utiliza las características de Iris para predecir su especie.')

# Definir la ruta del modelo
# Nos aseguramos de acceder al archivo modelo_prueba_iris.pkl dentro de la carpeta Data
file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'modelo_prueba_iris.pkl')

# Verificar si el archivo existe antes de intentar cargarlo
if os.path.exists(file_path):
    # Cargar el modelo serializado
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error('¡No se encontró el archivo del modelo! Asegúrate de que esté en la carpeta Data.')

# Crear sliders para ingresar las características de Iris
st.sidebar.header('📊 Características de la Flor Iris')
sepal_length = st.sidebar.slider('Sepal Length (cm)', 0.0, 10.0, 5.0)
sepal_width = st.sidebar.slider('Sepal Width (cm)', 0.0, 10.0, 3.0)
petal_length = st.sidebar.slider('Petal Length (cm)', 0.0, 10.0, 4.0)
petal_width = st.sidebar.slider('Petal Width (cm)', 0.0, 10.0, 1.0)

# Crear un botón para predecir
if st.sidebar.button('Predecir Tipo de Flor'):
    if 'model' in locals():
        # Preparar los datos como un array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]  # Realizar la predicción
        species_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        st.write(f'🌼 **El modelo predice que la flor es: `{species_dict[prediction]}`**')
    else:
        st.error("¡El modelo no está cargado correctamente!")

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
