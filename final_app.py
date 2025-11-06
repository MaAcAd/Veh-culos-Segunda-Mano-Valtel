# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np

# =============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO (Basado en la Paleta VALTEL)
# =============================================================================
st.set_page_config(
    page_title="Tasador VALTEL",
    page_icon="🚗",
    layout="centered", # De vuelta al layout centrado para un look de app simple
    initial_sidebar_state="expanded"
)

# Paleta de Colores Extraída
COLOR_BACKGROUND = "#0a090f" # Negro/Gris muy oscuro
COLOR_BACKGROUND_SIDEBAR = "#101015"
COLOR_TEXT = "#FAFAFA"
COLOR_PRIMARY_ACCENT = "#F3B71D" # AMARILLO/DORADO (Acento principal)
COLOR_SECONDARY_ACCENT = "#40BCD8" # Cián

# Inyección de CSS para un diseño limpio y enfocado
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
        border-right: 3px solid {COLOR_PRIMARY_ACCENT}; /* Borde más grueso y en acento */
    }}
    [data-testid="stSidebar"] h2, .css-j7q09m {{
        color: {COLOR_PRIMARY_ACCENT}; /* Título de la sidebar en Amarillo */
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
        background-color: #FF00FF; /* Usamos Magenta para el hover */
        color: {COLOR_TEXT};
        box-shadow: 0 0 15px #FF00FF;
    }}
    
    /* Resultado de la Métrica */
    [data-testid="stMetric"] {{
        margin-top: 50px;
        background-color: #1A1A1A; /* Un gris oscuro para el contenedor */
        padding: 30px;
        border-radius: 12px;
        border: 1px solid {COLOR_PRIMARY_ACCENT};
        box-shadow: 0 4px 15px rgba(243, 183, 29, 0.4); /* Sombra suave de acento */
        text-align: center;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLOR_PRIMARY_ACCENT}; /* Valor en Amarillo/Dorado */
        font-size: 3.5em; /* Valor grande */
        font-weight: 700;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 1.5em;
        color: {COLOR_SECONDARY_ACCENT}; /* Etiqueta en Cián */
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
# DEFINICIÓN DE VARIABLES Y VALORES (Sin cambios, manteniendo la funcionalidad)
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
TRANSMISIONES = ['Automática', 'Manual']
ESTILOS = ['Sedan', 'SUV', 'Coupe', 'Wagon', 'Hatchback']
TRACCIONES = ['Delantera', 'Trasera', 'AWD']
CILINDROS = [4, 6, 8]
PUERTAS = [2, 4]
COMBUSTIBLES = ['Gasolina', 'Diesel', 'Híbrido']
TAMAÑOS = ['Compact', 'Midsize', 'Large'] # Aunque no se pide al usuario, es necesario en el DF

# =============================================================================
# INTERFAZ Y LÓGICA DE PREDICCIÓN (Demo Enfocada)
# =============================================================================

# ------------------ PANTALLA PRINCIPAL: TÍTULO Y RESULTADO ------------------
st.markdown('<div class="big-title">TASADOR PREDICTIVO VALTEL</div>', unsafe_allow_html=True)
st.write(
    '<p style="text-align: center; color: #BBB;">Modelo de Regresión para la Tarificación de Vehículos de Segunda Mano.</p>', 
    unsafe_allow_html=True
)


# ------------------ BARRA LATERAL: INPUTS DEL USUARIO ------------------
st.sidebar.image("https://i.imgur.com/gQY8XjO.png", width=80) # Logo simulado
st.sidebar.header("🔧 Parámetros del Vehículo")

# Agrupación de inputs en la sidebar
user_marca = st.sidebar.selectbox("Marca", MARCAS, index=0)
user_cv = st.sidebar.slider("Potencia (CV)", min_value=50, max_value=600, value=150, step=10)
user_antiguedad = st.sidebar.slider("Antigüedad (Años)", min_value=0, max_value=25, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Especificaciones Adicionales")

# Inputs secundarios
user_combustible = st.sidebar.selectbox("Tipo de Combustible", COMBUSTIBLES)
user_transmision = st.sidebar.selectbox("Transmisión", TRANSMISIONES)
user_estilo = st.sidebar.selectbox("Estilo de Carrocería", ESTILOS)
user_traccion = st.sidebar.selectbox("Tracción", TRACCIONES)

# Inputs de números discretos
col_cyl, col_doors = st.sidebar.columns(2)
with col_cyl:
    user_cilindros = st.selectbox("Cilindros", CILINDROS, index=0, label_visibility="collapsed")
with col_doors:
    user_puertas = st.selectbox("Puertas", PUERTAS, index=1, label_visibility="collapsed")
    
# Reinsertamos los labels para los selectboxes
st.sidebar.write("Cilindros:")
st.sidebar.selectbox("Cilindros", CILINDROS, index=0, key='cyl_key', label_visibility="collapsed")
st.sidebar.write("Puertas:")
st.sidebar.selectbox("Puertas", PUERTAS, index=1, key='door_key', label_visibility="collapsed")


# ------------------ LÓGICA DE PREDICCIÓN Y RESULTADO ------------------

# Botón de predicción al final de la pantalla (más visible)
btn_predict = st.button('Estimar Precio de Tasación', use_container_width=True)
st.markdown("---")

# Contenedor para el resultado
result_container = st.empty()

if final_pipeline and btn_predict:
    with st.spinner('Realizando cálculo predictivo...'):
        # 1. Calcular el Año
        current_year = datetime.now().year
        user_year = current_year - user_antiguedad
        
        # 2. Recopilar datos y defaults
        datos_usuario_input = {
            'Marca': user_marca, 'CV': user_cv, 'Año': user_year, 
            'Transmisión': user_transmision, 'Estilo': user_estilo, 
            'Tracción': user_traccion, 'Cilindros': user_cilindros, 
            'Puertas': user_puertas, 'Combustible': user_combustible,
            
            # Relleno de defaults (necesarios para el pipeline)
            'Popularidad': VALORES_DEFECTO['Popularidad'],
            'Consumo Ciudad': VALORES_DEFECTO['Consumo Ciudad'],
            'Consumo Carretera': VALORES_DEFECTO['Consumo Carretera'],
            'Tamaño': VALORES_DEFECTO['Tamaño'],
            'Mercado': VALORES_DEFECTO['Mercado'],
        }
        
        # 3. Construir el DataFrame
        df_prediccion = pd.DataFrame([datos_usuario_input], columns=COLUMNAS_ESPERADAS)
        
        try:
            # 4. Predicción
            precio_predicho = final_pipeline.predict(df_prediccion)[0]
            
            # 5. Mostrar resultado en el contenedor principal
            result_container.metric(
                "PRECIO DE VENTA SUGERIDO (EUR)", 
                f"€ {precio_predicho:,.0f}", 
                help="Precio estimado por el modelo de Machine Learning."
            )
            
            # Información de resumen debajo del resultado
            st.markdown(f"""
                <div style="text-align: center; color: {COLOR_TEXT}; margin-top: 15px;">
                    <small>Tasando: **{user_marca}** | **{user_cv} CV** | **{user_antiguedad}** años</small>
                </div>
            """, unsafe_allow_html=True)


        except Exception as e:
            st.error("Error al predecir. Verifique los parámetros o la consistencia de los datos de entrada.")
            # st.exception(e) # Comentamos esto para no mostrar el traceback en la demo

elif not final_pipeline:
    st.warning("El modelo no ha podido ser cargado. Asegúrese de que el archivo .pkl esté correctamente subido.")
else:
    # Mensaje inicial antes de la primera predicción
    result_container.info("Introduzca las especificaciones del vehículo en la barra lateral para generar una estimación de precio instantánea.")
