# -*- coding: utf-8 -*-
"""
FINAL_APP.PY: Aplicación Web Mínima Viable (MVP) para la Tasación de Vehículos.
Utiliza Streamlit para la interfaz de usuario y carga el modelo entrenado
para realizar predicciones en tiempo real.
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. CARGA DEL MODELO Y RECURSOS ---
# El modelo completo (pipeline + estimador) fue guardado en la fase de entrenamiento.
# ¡IMPORTANTE!: Este archivo 'modelo_tasacion_valtel.pkl' debe estar en el mismo repositorio de GitHub.
try:
    pipeline = joblib.load('modelo_tasacion_valtel.pkl')
    st.session_state['model_loaded'] = True
except FileNotFoundError:
    st.session_state['model_loaded'] = False
    st.error("Error: El archivo 'modelo_tasacion_valtel.pkl' no fue encontrado. Asegúrese de que el modelo entrenado esté en el repositorio.")

# Definir las categorías que el modelo espera (ejemplo basado en variables comunes)
TRANSMISION_OPCIONES = ['Automática', 'Manual']
COMBUSTIBLE_OPCIONES = ['Gasolina', 'Diesel', 'Híbrido', 'Eléctrico']
MARCA_OPCIONES = ['Mercedes', 'BMW', 'Audi', 'Volkswagen', 'Ford', 'Otro'] # Ejemplo de marcas

# --- 2. CONFIGURACIÓN DE LA PÁGINA (Estilo y Título) ---
st.set_page_config(
    page_title="VALTEL: Tasador Predictivo de Vehículos",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilo NEON Suave (usando Markdown para inyección de CSS)
st.markdown("""
    <style>
    .reportview-container {
        background: #0e1117; /* Fondo oscuro */
    }
    .stButton>button {
        background-color: #00CCCC; /* Cian Neón */
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: 2px solid #00CCCC;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #CC00CC; /* Magenta Neón en hover */
        border-color: #CC00CC;
        color: white;
    }
    h1 { color: #00CCCC; } /* Título Cian */
    h2 { color: #CC00CC; } /* Subtítulos Magenta */
    .stTextInput>div>div>input, .stSelectbox>div>div {
        border-color: #39CC14; /* Borde verde suave */
    }
    .stSuccess {
        background-color: #0c430c; /* Fondo de éxito oscuro */
        border-left: 5px solid #39CC14; /* Borde verde brillante */
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("VALTEL: Tasador Predictivo - Demo")
st.markdown("Herramienta desarrollada con **XGBoost Regressor** ($R^2=0.9706$)")

# --- 3. FORMULARIO DE ENTRADA DE DATOS ---
if st.session_state['model_loaded']:
    with st.form(key='tasacion_form'):
        st.header("Características del Vehículo")
        
        # Columna 1: Variables Continuas (Las más Importantes)
        col1, col2 = st.columns(2)

        with col1:
            # 1. Potencia (CV) - Factor más importante según la Diapositiva 5/6
            cv = st.number_input(
                "Potencia (CV):",
                min_value=50, max_value=800, value=150, step=10,
                help="El factor más influyente en el precio (65% de importancia)."
            )
            
            # 2. Antigüedad (Años) - Segundo factor más importante
            antiguedad = st.slider(
                "Antigüedad (Años):",
                min_value=0, max_value=20, value=3, step=1,
                help="Depreciación: Años desde la primera matriculación."
            )

        with col2:
            # 3. Kilometraje (Km) - Factor menos importante
            kilometraje = st.number_input(
                "Kilometraje (Km):",
                min_value=100, max_value=500000, value=50000, step=1000,
                help="Kilómetros recorridos."
            )

            # 4. Marca (Variable categórica) - Asumimos que incluimos la marca en el modelo
            marca = st.selectbox(
                "Marca del Vehículo:",
                options=MARCA_OPCIONES,
                index=3,
                help="La marca influye debido al valor residual."
            )

        st.subheader("Otras Características")
        col3, col4 = st.columns(2)

        with col3:
            # 5. Transmisión - Factor clave en la gama del vehículo
            transmision = st.radio(
                "Tipo de Transmisión:",
                options=TRANSMISION_OPCIONES,
                index=0,
                help="Automática vs Manual."
            )
        
        with col4:
            # 6. Combustible
            combustible = st.selectbox(
                "Tipo de Combustible:",
                options=COMBUSTIBLE_OPCIONES,
                index=0
            )


        # Botón para enviar la solicitud
        submit_button = st.form_submit_button(label='TASAR VEHÍCULO 🚀')

        # --- 4. LÓGICA DE PREDICCIÓN ---
        if submit_button:
            # 1. Crear el DataFrame de entrada (Debe coincidir EXACTAMENTE con el formato de entrenamiento)
            # Adaptar esto según el nombre exacto de las columnas en su modelo.
            datos_entrada = pd.DataFrame({
                'CV': [cv],
                'Antiguedad': [antiguedad],
                'Kilometraje': [kilometraje],
                'Transmision': [transmision],
                'Combustible': [combustible],
                'Marca': [marca], # Asumimos que 'Marca' fue codificada en el Pipeline
            })
            
            # 2. Realizar la Predicción
            try:
                prediccion = pipeline.predict(datos_entrada)[0]
                
                # 3. Mostrar Resultado
                precio_formateado = f"€{prediccion:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
                
                st.success("✅ Tasación Realizada con Éxito")
                st.balloons() # Pequeña celebración

                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 3px solid #39CC14; border-radius: 10px; background-color: rgba(0, 50, 0, 0.4);">
                    <h2 style="color: #FFFFFF; margin-bottom: 0px;">PRECIO ESTIMADO DE VENTA</h2>
                    <h1 style="font-size: 4em; color: #CC00CC; margin-top: 5px;">{precio_formateado}</h1>
                    <p style="color: #FFFFFF; font-size: 0.9em;">Basado en nuestro modelo XGBoost (Error medio de ±9.639 €)</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error al predecir. El modelo puede no estar recibiendo los datos esperados: {e}")

# --- 5. MENSAJE SI EL MODELO NO CARGA ---
else:
    st.warning("⚠️ No se puede cargar la aplicación porque el modelo no se encontró.")
