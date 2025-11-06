# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np

# =============================================================================
# CONFIGURACIÓN INICIAL Y ESTILOS
# =============================================================================
st.set_page_config(
    page_title="Tasador Predictivo VALTEL",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilos 
st.markdown("""
    <style>
    /* Estilo del título y acentos */
    .big-title {
        color: #00FFFF; /* Cian */
        font-weight: 700;
        font-size: 2.2em;
        padding-bottom: 0.5em;
        text-align: center;
    }
    /* Estilo del botón */
    .stButton>button {
        background-color: #00FFFF;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #FF00FF; /* Magenta */
        color: #FFFFFF;
        box-shadow: 0 0 10px #FF00FF;
    }
    /* Estilo del resultado (Métrica) */
    .stMetric > div {
        background-color: #161B22;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF00FF; /* Magenta */
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# CARGA DEL MODELO (Asegúrate de que el archivo .pkl esté en GitHub)
# =============================================================================
MODELO_FILE = 'modelo_tasacion_valtel.pkl'

@st.cache_resource
def load_model(file_path):
    """Carga el pipeline de Machine Learning usando joblib."""
    try:
        if not os.path.exists(file_path):
            return None
        pipeline = joblib.load(file_path)
        return pipeline
    except Exception as e:
        st.error(f"Error al cargar el modelo. ¿Está 'modelo_tasacion_valtel.pkl' subido? Detalle: {e}")
        return None

final_pipeline = load_model(MODELO_FILE)

# =============================================================================
# DEFINICIÓN DE VARIABLES Y VALORES (CLAVE PARA EL ERROR DE COLUMNAS)
# =============================================================================

# Lista EXACTA de columnas que el pipeline espera recibir (ORDEN Y NOMBRE EXACTOS)
COLUMNAS_ESPERADAS = [
    'Marca', 'Potencia (CV)', 'Año', 'Popularidad', 'Consumo Ciudad', 
    'Consumo Carretera', 'Cilindros', 'Tamaño', 'Transmisión', 
    'Puertas', 'Tracción', 'Mercado', 'Estilo'
]

# Valores por defecto para las 13 columnas. Usaremos valores promedio o frecuentes.
VALORES_DEFECTO = {
    # Numéricos por defecto 
    'Potencia (CV)': 150, 
    'Año': 2018, 
    'Popularidad': 1000, 
    'Consumo Ciudad': 20, 
    'Consumo Carretera': 25,
    'Cilindros': 4,
    'Puertas': 4,
    
    # Categóricos por defecto 
    'Marca': 'Otro', 
    'Tamaño': 'Midsize', 
    'Transmisión': 'Automática', 
    'Tracción': 'Delantera', 
    'Mercado': 'Lujo',
    'Estilo': 'Sedan', 
}

# Opciones de selección (Ajusta estas listas si tus categorías son diferentes)
MARCAS = ['Audi', 'BMW', 'Chevrolet', 'Nissan', 'Toyota', 'Ford', 'Honda', 'Otro'] 
TAMAÑOS = ['Compact', 'Midsize', 'Large']
TRANSMISIONES = ['Automática', 'Manual']
ESTILOS = ['Sedan', 'SUV', 'Coupe', 'Wagon', 'Hatchback']
TRACCIONES = ['Delantera', 'Trasera', 'AWD']
CILINDROS = [4, 6, 8]
PUERTAS = [2, 4]


# =============================================================================
# INTERFAZ Y LÓGICA DE PREDICCIÓN
# =============================================================================

st.markdown('<div class="big-title">🚗 Tasador Predictivo VALTEL</div>', unsafe_allow_html=True)
st.write("Estime el precio de un vehículo de segunda mano para elaborar ofertas competitivas.")


if final_pipeline:
    
    # ------------------ SIDEBAR: INPUTS DEL USUARIO ------------------
    st.sidebar.header("🔧 Parámetros del Vehículo")

    # Inputs directos para el usuario
    user_marca = st.sidebar.selectbox("Marca", MARCAS, index=0)
    user_potencia = st.sidebar.slider("Potencia (CV)", min_value=50, max_value=600, value=150, step=10)
    user_antiguedad = st.sidebar.slider("Antigüedad (Años)", min_value=0, max_value=25, value=5, step=1)
    
    # Inputs secundarios
    st.sidebar.subheader("Otras Especificaciones")
    user_transmision = st.sidebar.selectbox("Transmisión", TRANSMISIONES)
    user_estilo = st.sidebar.selectbox("Estilo de Carrocería", ESTILOS)
    user_traccion = st.sidebar.selectbox("Tracción", TRACCIONES)
    
    # Inputs de números discretos
    user_cilindros = st.sidebar.selectbox("Cilindros", CILINDROS, index=CILINDROS.index(VALORES_DEFECTO['Cilindros']) if VALORES_DEFECTO['Cilindros'] in CILINDROS else 0)
    user_puertas = st.sidebar.selectbox("Puertas", PUERTAS, index=PUERTAS.index(VALORES_DEFECTO['Puertas']) if VALORES_DEFECTO['Puertas'] in PUERTAS else 0)


    # ------------------ CONSTRUCCIÓN DEL DATAFRAME ------------------

    # 1. Calcular el Año (Modelo espera 'Año', no 'Antigüedad')
    current_year = datetime.now().year
    user_year = current_year - user_antiguedad
    
    
    # 2. Recopilar datos del usuario y valores por defecto
    datos_usuario_input = {
        # Variables de Usuario
        'Marca': user_marca,
        'Potencia (CV)': user_potencia,
        'Año': user_year, 
        'Transmisión': user_transmision,
        'Estilo': user_estilo,
        'Tracción': user_traccion,
        'Cilindros': user_cilindros,
        'Puertas': user_puertas,
        
        # Variables de Relleno (Defaults) para que el DataFrame esté completo
        'Popularidad': VALORES_DEFECTO['Popularidad'],
        'Consumo Ciudad': VALORES_DEFECTO['Consumo Ciudad'],
        'Consumo Carretera': VALORES_DEFECTO['Consumo Carretera'],
        'Tamaño': VALORES_DEFECTO['Tamaño'],
        'Mercado': VALORES_DEFECTO['Mercado'],
    }
    
    # 3. Construir el DataFrame con TODAS las columnas y el ORDEN correcto
    df_prediccion = pd.DataFrame([datos_usuario_input], columns=COLUMNAS_ESPERADAS)


    # ------------------ CÁLCULO Y PREDICCIÓN ------------------

    st.markdown("---")
    
    if st.button('Calcular Precio de Tasación', use_container_width=True):
        with st.spinner('Realizando tasación...'):
            try:
                # El pipeline recibe el DataFrame con TODAS las columnas necesarias
                precio_predicho = final_pipeline.predict(df_prediccion)[0]
                
                # Presentar el resultado
                st.subheader("💰 Resultado de la Tasación")
                st.metric("Precio de Venta Estimado", 
                          f"€ {precio_predicho:,.0f}", 
                          help="Precio competitivo basado en las características del vehículo.")
                
                # Mensaje de información
                st.info(f"Modelo tasado: {df_prediccion['Marca'].iloc[0]} | {df_prediccion['Potencia (CV)'].iloc[0]} CV | Antigüedad: {user_antiguedad} años")


            except Exception as e:
                # Manejo de errores de predicción
                st.error("Error al predecir. Esto puede deberse a un valor que no existía en los datos de entrenamiento.")
                st.exception(e)
                st.write("DataFrame enviado (Debugging):")
                st.dataframe(df_prediccion)

else:
    st.warning("La aplicación no puede funcionar. Por favor, asegúrese de que el archivo 'modelo_tasacion_valtel.pkl' exista y sea accesible en el repositorio de GitHub.")
