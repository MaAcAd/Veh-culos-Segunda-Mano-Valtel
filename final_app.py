# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np

# =============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO (Basado en la Presentación de Canva)
# =============================================================================
st.set_page_config(
    page_title="Tasador VALTEL",
    page_icon="🚗",
    layout="wide",  # Usamos layout ancho para un look más profesional
    initial_sidebar_state="expanded"
)

# Paleta de Colores Extraída de la Presentación
COLOR_BACKGROUND = "#0a090f" # Negro/Gris muy oscuro
COLOR_BACKGROUND_SIDEBAR = "#101015" # Un poco más claro
COLOR_TEXT = "#FAFAFA" # Blanco
COLOR_PRIMARY_ACCENT = "#F3B71D" # Amarillo/Dorado (del gráfico de Tarta)
COLOR_SECONDARY_ACCENT = "#40BCD8" # Cián/Turquesa (del gráfico de Tarta)
COLOR_TERTIARY_ACCENT = "#E020F5" # Magenta (del gráfico de Tarta)

# Inyección de CSS para un diseño personalizado
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
        border-right: 1px solid {COLOR_PRIMARY_ACCENT};
    }}
    [data-testid="stSidebar"] h2 {{
        color: {COLOR_SECONDARY_ACCENT}; /* Título de la sidebar en Cián */
    }}
    
    /* Título Principal */
    .title-text {{
        color: {COLOR_PRIMARY_ACCENT}; /* Título en Amarillo/Dorado */
        font-size: 2.5em;
        font-weight: 700;
        padding-bottom: 0.2em;
    }}
    /* Subtítulo (Autor) */
    .author-text {{
        color: {COLOR_TEXT};
        font-size: 1.2em;
        font-style: italic;
        margin-bottom: 20px;
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
        background-color: {COLOR_TERTIARY_ACCENT}; /* Hover en Magenta */
        color: {COLOR_TEXT}; /* Texto blanco en hover */
        box-shadow: 0 0 15px {COLOR_TERTIARY_ACCENT};
    }}
    
    /* Resultado de la Métrica */
    [data-testid="stMetric"] {{
        background-color: #1A1A1A;
        padding: 20px;
        border-radius: 10px;
        border-left: 7px solid {COLOR_PRIMARY_ACCENT}; /* Borde Amarillo */
    }}
    [data-testid="stMetricValue"] {{
        color: {COLOR_PRIMARY_ACCENT}; /* Valor en Amarillo */
        font-size: 2.8em;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 1.2em;
        color: {COLOR_TEXT};
    }}
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# CARGA DEL MODELO (Sin cambios)
# =============================================================================
MODELO_FILE = 'modelo_tasacion_valtel.pkl'

@st.cache_resource
def load_model(file_path):
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
# DEFINICIÓN DE VARIABLES Y VALORES (Sin cambios, mantiene la corrección)
# =============================================================================
COLUMNAS_ESPERADAS = [
    'Marca', 'CV', 'Año', 'Popularidad', 'Consumo Ciudad', 
    'Consumo Carretera', 'Cilindros', 'Tamaño', 'Transmisión', 
    'Puertas', 'Tracción', 'Mercado', 'Estilo', 'Combustible'
]
VALORES_DEFECTO = {
    'CV': 150, 'Año': 2018, 'Popularidad': 1000, 'Consumo Ciudad': 20, 
    'Consumo Carretera': 25, 'Cilindros': 4, 'Puertas': 4, 'Marca': 'Otro', 
    'Tamaño': 'Midsize', 'Transmisión': 'Automática', 'Tracción': 'Delantera', 
    'Mercado': 'Lujo', 'Estilo': 'Sedan', 'Combustible': 'Gasolina',
}
MARCAS = ['Audi', 'BMW', 'Chevrolet', 'Nissan', 'Toyota', 'Ford', 'Honda', 'Otro'] 
TAMAÑOS = ['Compact', 'Midsize', 'Large']
TRANSMISIONES = ['Automática', 'Manual']
ESTILOS = ['Sedan', 'SUV', 'Coupe', 'Wagon', 'Hatchback']
TRACCIONES = ['Delantera', 'Trasera', 'AWD']
CILINDROS = [4, 6, 8]
PUERTAS = [2, 4]
COMBUSTIBLES = ['Gasolina', 'Diesel', 'Híbrido']

# =============================================================================
# INTERFAZ Y LÓGICA DE PREDICCIÓN (Rediseño de Layout)
# =============================================================================

# ------------------ TÍTULO Y AUTOR (en la página principal) ------------------
st.markdown('<div class="title-text">OPTIMIZACIÓN DE PRECIOS VEHICULARES PARA VALTEL</div>', unsafe_allow_html=True)
st.markdown('<div class="author-text">Un proyecto de: María Aceituno Adrados</div>', unsafe_allow_html=True)
st.markdown("---")

# Layout de la página principal: 2 columnas
col1, col2 = st.columns([1.5, 1]) # Columna izquierda más ancha

# ------------------ COLUMNA 1: CONTEXTO DEL PROYECTO ------------------
with col1:
    st.subheader("Contexto del Proyecto")
    st.markdown("""
    Valtel requiere diversificar su oferta incursionando en el mercado de vehículos de segunda mano para sus clientes (no nacionales).
    
    Se requiere un Modelo de Regresión basado en Machine Learning para capturar los patrones complejos del mercado y generar la tarifa competitiva de forma automatizada.
    """)
    st.markdown("---")
    st.subheader("Instrucciones")
    st.markdown("""
    1.  Utilice los controles en la **barra lateral (izquierda)** para introducir las características del vehículo.
    2.  Haga clic en el botón **'Estimar Precio'**.
    3.  El precio de tasación sugerido aparecerá a la derecha.
    """)


# ------------------ BARRA LATERAL: INPUTS DEL USUARIO ------------------
st.sidebar.image("https://i.imgur.com/gQY8XjO.png", width=100) # Un logo genérico de coche
st.sidebar.header("🔧 Tasador Vehicular")

user_marca = st.sidebar.selectbox("Marca", MARCAS, index=0)
user_cv = st.sidebar.slider("Potencia (CV)", min_value=50, max_value=600, value=150, step=10)
user_antiguedad = st.sidebar.slider("Antigüedad (Años)", min_value=0, max_value=25, value=5, step=1)
user_combustible = st.sidebar.selectbox("Tipo de Combustible", COMBUSTIBLES)
user_transmision = st.sidebar.selectbox("Transmisión", TRANSMISIONES)
user_estilo = st.sidebar.selectbox("Estilo de Carrocería", ESTILOS)
user_traccion = st.sidebar.selectbox("Tracción", TRACCIONES)
user_cilindros = st.sidebar.selectbox("Cilindros", CILINDROS, index=0)
user_puertas = st.sidebar.selectbox("Puertas", PUERTAS, index=1)

# Botón de predicción al final de la sidebar
btn_predict = st.sidebar.button('Estimar Precio')

# ------------------ COLUMNA 2: RESULTADO DE LA PREDICCIÓN ------------------
with col2:
    st.subheader("Resultado de la Tasación")
    
    if final_pipeline and btn_predict:
        with st.spinner('Calculando...'):
            # Lógica de construcción del DataFrame (sin cambios)
            current_year = datetime.now().year
            user_year = current_year - user_antiguedad
            
            datos_usuario_input = {
                'Marca': user_marca, 'CV': user_cv, 'Año': user_year, 
                'Transmisión': user_transmision, 'Estilo': user_estilo, 
                'Tracción': user_traccion, 'Cilindros': user_cilindros, 
                'Puertas': user_puertas, 'Combustible': user_combustible,
                
                # Relleno de defaults
                'Popularidad': VALORES_DEFECTO['Popularidad'],
                'Consumo Ciudad': VALORES_DEFECTO['Consumo Ciudad'],
                'Consumo Carretera': VALORES_DEFECTO['Consumo Carretera'],
                'Tamaño': VALORES_DEFECTO['Tamaño'],
                'Mercado': VALORES_DEFECTO['Mercado'],
            }
            
            df_prediccion = pd.DataFrame([datos_usuario_input], columns=COLUMNAS_ESPERADAS)
            
            try:
                # Predicción
                precio_predicho = final_pipeline.predict(df_prediccion)[0]
                
                # Mostrar resultado
                st.metric("Precio de Venta Sugerido", 
                          f"€ {precio_predicho:,.0f}", 
                          help="Estimación basada en el modelo de regresión (R² > 0.90)")
                
                st.caption(f"Detalles: {user_marca} | {user_cv} CV | {user_antiguedad} años")

            except Exception as e:
                st.error("Error al predecir.")
                st.exception(e)

    elif not final_pipeline:
        st.error("Modelo no cargado. Revise el archivo 'modelo_tasacion_valtel.pkl'.")
    else:
        st.info("Introduzca los datos en la barra lateral y haga clic en 'Estimar Precio' para ver el resultado.")
