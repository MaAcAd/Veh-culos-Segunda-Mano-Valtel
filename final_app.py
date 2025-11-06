# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np

# =============================================================================
# DEFINICIÓN DE PALETA NEÓN VALTEL
# =============================================================================
NEON_CIAN = '#00FFFF'     # Color de acento principal (Botón, Títulos)
NEON_MAGENTA = '#FF33FF'  # Color de acento secundario (Hover, Bordes)
NEON_LIMA = '#CCFF00'     # Color de acento terciario

COLOR_BACKGROUND = "#0a090f" # Negro/Gris muy oscuro (Fondo Principal)
COLOR_BACKGROUND_SIDEBAR = "#101015" # Fondo de Sidebar
COLOR_TEXT = '#FFFFFF' # Texto principal

# =============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO (Paleta NEON Aplicada)
# =============================================================================
st.set_page_config(
    page_title="Tasador VALTEL",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Inyección de CSS para un diseño limpio, enfocado y elegante con NEON
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
        border-right: 3px solid {NEON_CIAN}; 
    }}
    [data-testid="stSidebar"] h2, .css-j7q09m, [data-testid="stSidebar"] h3 {{
        color: {NEON_CIAN}; /* Título y subtítulos de la sidebar en NEON CIAN */
        font-weight: 600;
    }}
    
    /* Título Principal (Centrado y en color de acento) */
    .big-title {{
        color: {NEON_CIAN};
        font-size: 2.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5em;
    }}
    
    /* Botón de Predicción */
    .stButton>button {{
        background-color: {NEON_CIAN};
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
        background-color: {NEON_MAGENTA}; /* NEON MAGENTA para el hover */
        color: {COLOR_TEXT};
        box-shadow: 0 0 15px {NEON_MAGENTA};
    }}
    
    /* Resultado de la Métrica */
    [data-testid="stMetric"] {{
        margin-top: 50px;
        background-color: #1A1A1A; /* Un gris oscuro para el contenedor */
        padding: 30px;
        border-radius: 12px;
        border: 1px solid {NEON_CIAN};
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.4);
        text-align: center;
    }}
    [data-testid="stMetricValue"] {{
        color: {NEON_CIAN}; /* Valor en NEON CIAN */
        font-size: 3.5em;
        font-weight: 700;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 1.5em;
        color: {NEON_MAGENTA}; /* Etiqueta en NEON MAGENTA */
    }}
    
    /* Titulo de la sección de Insights */
    .insights-header {{
        color: {NEON_MAGENTA};
        font-size: 1.8em;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# CARGA DEL MODELO
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
# DEFINICIÓN DE VARIABLES Y VALORES (SIN TRACCIÓN EN EL INPUT)
# =============================================================================
# NOTA: 'Tracción' se elimina de la lista de inputs al usuario, pero se mantiene 
# en el DataFrame final con un valor por defecto si el modelo lo requiere.
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
CILINDROS = [4, 6, 8]
PUERTAS = [2, 4]
COMBUSTIBLES = ['Gasolina', 'Diesel', 'Híbrido']
# 'Tamaño' y 'Mercado' se mantienen en el DF de predicción con valores por defecto

# =============================================================================
# INTERFAZ Y LÓGICA DE PREDICCIÓN (Diseño Compacto)
# =============================================================================

# ------------------ PANTALLA PRINCIPAL: TÍTULO Y RESULTADO ------------------
st.markdown('<div class="big-title">TASADOR PREDICTIVO VALTEL</div>', unsafe_allow_html=True)
st.write(
    '<p style="text-align: center; color: #BBB;">Modelo de Regresión para la Tarificación de Vehículos de Segunda Mano.</p>', 
    unsafe_allow_html=True
)


# ------------------ BARRA LATERAL: INPUTS DEL USUARIO COMPACTOS ------------------
st.sidebar.image("https://i.imgur.com/gQY8XjO.png", width=80) 
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
        
        # 2. Recopilar datos y defaults (Asegurar que 'Tracción' y otros defaults están presentes)
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
            'Tracción': VALORES_DEFECTO['Tracción'], # Asignamos el valor por defecto
        }
        
        # 3. Construir el DataFrame, asegurando que todas las COLUMNAS_ESPERADAS estén presentes
        df_prediccion = pd.DataFrame([datos_usuario_input], columns=COLUMNAS_ESPERADAS)
        
        try:
            # 4. Predicción
            precio_predicho = final_pipeline.predict(df_prediccion)[0]
            
            # 5. Mostrar resultado
            result_container.metric(
                "PRECIO DE VENTA SUGERIDO (EUR)", 
                f"€ {precio_predicho:,.0f}", 
                help="Precio estimado por el modelo de Machine Learning."
            )
            
            st.markdown(f"""
                <div style="text-align: center; color: {COLOR_TEXT}; margin-top: 15px;">
                    <small>Tasando: **{user_marca}** | **{user_cv} CV** | **{user_antiguedad}** años</small>
                </div>
            """, unsafe_allow_html=True)


        except Exception as e:
            st.error("Error al predecir. Verifique los parámetros o la consistencia de los datos de entrada.")
            # st.exception(e) # Para debug
            
elif not final_pipeline:
    st.warning("El modelo no ha podido ser cargado. Asegúrese de que el archivo .pkl esté correctamente subido.")
else:
    # Mensaje inicial antes de la primera predicción
    result_container.info("Introduzca las especificaciones del vehículo en la barra lateral para generar una estimación de precio instantánea.")


# =============================================================================
# SECCIÓN DE INSIGHTS DEL MODELO (El Resumen Compacto y Elegante)
# =============================================================================
st.markdown("---")
st.markdown('<div class="insights-header">💡 Insights Clave del Modelo VALTEL</div>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 1.1em; color: #FFF;">
El <b>Precio de Venta</b> está impulsado principalmente por el <b style="color: #FF33FF;">Rendimiento</b> y el <b style="color: #FF33FF;">Lujo</b>, no por la popularidad o la eficiencia.
</p>
""", unsafe_allow_html=True)


st.markdown("##### 🚀 Predictores de Precio (Matriz de Correlación):")
data = {
    "Variable": ["Potencia (CV)", "Cilindros", "Popularidad"],
    "Relación": ["Fuerte Positiva", "Media Positiva", "Casi Nula"]
}
st.table(data)


st.markdown("##### ⛽ Compensaciones y Segmentos:")
st.markdown(f"""
- Los vehículos más potentes y caros tienden a tener un **peor rendimiento de combustible** (mayor consumo).
- La popularidad es inversamente proporcional al precio: los más populares son los más **asequibles**.
- **Segmentos de Lujo:** Marcas como <b style="color: #FF33FF;">Cadillac</b> y vehículos <b style="color: #FF33FF;">Eléctricos</b> o de gasolina premium tienen los precios más altos.
""", unsafe_allow_html=True)
