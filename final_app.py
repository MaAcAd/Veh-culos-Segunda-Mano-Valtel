
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np

# =============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO (Paleta VALTEL Optimizada)
# =============================================================================
st.set_page_config(
    page_title="Tasador VALTEL",
    page_icon="🚗",
    layout="centered", # Centrado para un look de app simple y compacto
    initial_sidebar_state="expanded"
)

# Paleta de Colores Extraída
COLOR_BACKGROUND = "#0a090f" # Negro/Gris muy oscuro (Fondo Principal)
COLOR_BACKGROUND_SIDEBAR = "#101015" # Fondo de Sidebar
COLOR_TEXT = "#FAFAFA" # Texto principal
COLOR_PRIMARY_ACCENT = "#F3B71D" # AMARILLO/DORADO (Acento principal)
COLOR_SECONDARY_ACCENT = "#40BCD8" # Cián (Acento secundario/Etiquetas)
COLOR_HOVER_ACCENT = "#FF00FF" # Magenta para el hover del botón

# Inyección de CSS para un diseño limpio, enfocado y elegante
st.markdown(f"""
    <style>
    /* Fondo principal y de la app */
    .stApp {{
        background-color: {COLOR_BACKGROUND};
        color: {COLOR_TEXT};
    }}
    /* Barra lateral */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_BACKGROUND_SIDEBAR};
        border-right: 3px solid {COLOR_PRIMARY_ACCENT};
    }}
    [data-testid="stSidebar"] h2, .css-j7q09m, [data-testid="stSidebar"] h3 {{
        color: {COLOR_PRIMARY_ACCENT}; /* Título y subtítulos de la sidebar en Amarillo */
        font-weight: 600;
    }}
    
    /* Título Principal (Centrado y en color de acento) */
    .big-title {{
        color: {COLOR_PRIMARY_ACCENT};
        font-size: 2.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5em;
    }}
    
    /* Botón de Predicción */
    .stButton>button {{
        background-color: {COLOR_PRIMARY_ACCENT};
        color: {COLOR_BACKGROUND}; /* Texto oscuro para alto contraste */
        font-weight: bold;
        font-size: 1.1em;
        border-radius: 8px;
        padding: 10px 20px;
        width: 100%;
        border: none;
        transition: all 0.2s ease-in-out;
    }}
    .stButton>button:hover {{
        background-color: {COLOR_HOVER_ACCENT}; /* Magenta para el hover */
        color: {COLOR_TEXT};
        box-shadow: 0 0 15px {COLOR_HOVER_ACCENT};
    }}
    
    /* Resultado de la Métrica */
    [data-testid="stMetric"] {{
        margin-top: 50px;
        background-color: #1A1A1A; /* Un gris oscuro para el contenedor */
        padding: 30px;
        border-radius: 12px;
        border: 1px solid {COLOR_PRIMARY_ACCENT};
        box-shadow: 0 4px 15px rgba(243, 183, 29, 0.4);
        text-align: center;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLOR_PRIMARY_ACCENT}; /* Valor en Amarillo/Dorado */
        font-size: 3.5em;
        font-weight: 700;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 1.5em;
        color: {COLOR_SECONDARY_ACCENT}; /* Etiqueta en Cián */
    }}
    
    /* Titulo de la sección de Insights */
    .insights-header {{
        color: {COLOR_SECONDARY_ACCENT};
        font-size: 1.8em;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# CARGA DEL MODELO (Sin cambios)
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
# DEFINICIÓN DE VARIABLES Y VALORES (Ajustada para quitar 'Tracción')
# =============================================================================
COLUMNAS_ESPERADAS = [
    'Marca', 'CV', 'Año', 'Popularidad', 'Consumo Ciudad', 
    'Consumo Carretera', 'Cilindros', 'Tamaño', 'Transmisión', 
    'Puertas', 'Mercado', 'Estilo', 'Combustible'
    # NOTA: 'Tracción' se elimina aquí y se asume un valor por defecto en el DF.
]

# Las variables de Tracción y Tamaño deben tener un default para el pipeline, aunque no se pidan.
VALORES_DEFECTO = {
    'CV': 150, 'Año': 2018, 'Popularidad': 1000, 'Consumo Ciudad': 20, 
    'Consumo Carretera': 25, 'Cilindros': 4, 'Puertas': 4, 'Marca': 'Otro', 
    'Tamaño': 'Midsize', 'Transmisión': 'Automática', 'Tracción': 'Delantera', # Se mantiene el default
    'Mercado': 'Lujo', 'Estilo': 'Sedan', 'Combustible': 'Gasolina',
}
MARCAS = ['Audi', 'BMW', 'Chevrolet', 'Nissan', 'Toyota', 'Ford', 'Honda', 'Otro'] 
TRANSMISIONES = ['Automática', 'Manual']
ESTILOS = ['Sedan', 'SUV', 'Coupe', 'Wagon', 'Hatchback']
CILINDROS = [4, 6, 8]
PUERTAS = [2, 4]
COMBUSTIBLES = ['Gasolina', 'Diesel', 'Híbrido']
TAMAÑOS = ['Compact', 'Midsize', 'Large']


# =============================================================================
# INTERFAZ Y LÓGICA DE PREDICCIÓN (Optimizada para la compactación)
# =============================================================================

# ------------------ PANTALLA PRINCIPAL: TÍTULO Y RESULTADO ------------------
st.markdown('<div class="big-title">TASADOR PREDICTIVO VALTEL</div>', unsafe_allow_html=True)
st.write(
    '<p style="text-align: center; color: #BBB;">Modelo de Regresión para la Tarificación de Vehículos de Segunda Mano.</p>', 
    unsafe_allow_html=True
)


# ------------------ BARRA LATERAL: INPUTS DEL USUARIO COMPACTOS ------------------
st.sidebar.image("https://i.imgur.com/gQY8XjO.png", width=80) # Logo simulado
st.sidebar.header("🔧 Parámetros de Tasación")

# 1. Inputs Clave (CV, Año)
user_marca = st.sidebar.selectbox("Marca", MARCAS, index=0)
user_cv = st.sidebar.slider("Potencia (CV)", min_value=50, max_value=600, value=150, step=10)
user_antiguedad = st.sidebar.slider("Antigüedad (Años)", min_value=0, max_value=25, value=5, step=1)

st.sidebar.markdown("---")

# 2. Inputs Discretos y Secundarios (en columnas para compactar)
st.sidebar.subheader("Especificaciones")
col_combustible, col_transmision = st.sidebar.columns(2)
with col_combustible:
    user_combustible = st.selectbox("Combustible", COMBUSTIBLES, index=0)
with col_transmision:
    user_transmision = st.selectbox("Transmisión", TRANSMISIONES, index=0)

col_estilo, col_cyl, col_doors = st.sidebar.columns(3)
with col_estilo:
    user_estilo = st.selectbox("Estilo", ESTILOS, index=0)
with col_cyl:
    user_cilindros = st.selectbox("Cilindros", CILINDROS, index=0)
with col_doors:
    user_puertas = st.selectbox("Puertas", PUERTAS, index=1)
    
# Botón de predicción en la pantalla principal
btn_predict = st.button('Estimar Precio de Tasación', use_container_width=True)
st.markdown("---")

# Contenedor para el resultado
result_container = st.empty()


# ------------------ LÓGICA DE PREDICCIÓN Y RESULTADO ------------------

if final_pipeline and btn_predict:
    with st.spinner('Realizando cálculo predictivo...'):
        # 1. Calcular el Año
        current_year = datetime.now().year
        user_year = current_year - user_antiguedad
        
        # 2. Recopilar datos y defaults (Asegurar que 'Tracción' está presente para el pipeline)
        datos_usuario_input = {
            'Marca': user_marca, 'CV': user_cv, 'Año': user_year, 
            'Transmisión': user_transmision, 'Estilo': user_estilo, 
            'Cilindros': user_cilindros, 'Puertas': user_puertas, 
            'Combustible': user_combustible,
            
            # Relleno de defaults (necesarios para el pipeline)
            'Popularidad': VALORES_DEFECTO['Popularidad'],
            'Consumo Ciudad': VALORES_DEFECTO['Consumo Ciudad'],
            'Consumo Carretera': VALORES_DEFECTO['Consumo Carretera'],
            'Tamaño': VALORES_DEFECTO['Tamaño'],
            'Mercado': VALORES_DEFECTO['Mercado'],
            # TRACCIÓN: Se añade con un valor por defecto que no es preguntado al usuario
            'Tracción': VALORES_DEFECTO['Tracción'], 
        }
        
        # 3. Construir el DataFrame, ajustando las columnas esperadas al modelo
        df_prediccion = pd.DataFrame([datos
