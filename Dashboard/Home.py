# 📊_Dashboard_Principal.py
import streamlit as st

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Dashboard Central Analisis Vehiculos Electricos",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Navegación")
st.sidebar.success("Selecciona un análisis del menú para comenzar.")
st.sidebar.divider()
st.sidebar.info("Este dashboard es un proyecto en desarrollo.")


# --- PÁGINA PRINCIPAL ---

st.title("📊 Dashboard Interactivo: Analisis Vehiculos Electricos")

st.markdown("""
¡Bienvenido! Este espacio centraliza análisis visuales sobre diferentes aspectos relevantes para la
 adopción de vehiculos electricos en colombia.
 Utiliza el menú de navegación en la **barra lateral izquierda** para sumergirte en cada tema específico.
""")

st.divider()

# --- Sección de Presentación de Dashboards (Usando Columnas) ---
st.header("Explora los Dashboards Disponibles:")

# Cambiado a 4 columnas, ajusta el 'gap' si es necesario ('medium' o 'small')
col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.subheader("🚗 Vehículos Eléctricos")
    st.markdown("""
    Analiza la adopción de vehículos eléctricos en el país. Visualiza registros por:
    *   Año y mes
    *   Marca y clase
    *   Ubicación geográfica (Departamento)
    """)
    if st.button("Ir a Vehículos Eléctricos", key="btn_ev"):
        st.info("Selecciona '🚗 Vehiculos Electricos' en la barra lateral.")


with col2:
    st.subheader("⚡ Consumo Energético")
    st.markdown("""
    Explora los patrones de consumo de energía eléctrica a nivel nacional. Descubre:
    *   Tendencias diarias, mensuales y anuales
    *   Curvas de carga típicas (patrón horario)
    *   Comparativas entre días laborables y fines de semana
    """)
    if st.button("Ir a Consumo Energético", key="btn_energy"):
        st.info("Selecciona '⚡ Consumo Energetico' en la barra lateral.")


with col3:
    st.subheader("💨 Emisiones GEI")
    st.markdown("""
    Visualiza las emisiones de Gases de Efecto Invernadero (GEI), con énfasis en:
    *   El sector de transporte por carretera
    *   Comparativa con las emisiones totales nacionales
    *   Evolución temporal por categoría
    """)
    if st.button("Ir a Emisiones GEI", key="btn_co2"):
        st.info("Selecciona '💨 Emisiones GEI' en la barra lateral.")


with col4:
    st.subheader("🔗 Correlaciones y Factores") 
    st.markdown("""
    Examina las relaciones temporales entre la adopción de EVs y factores externos:
    *   Indicadores económicos (IPC, IPP)
    *   Precios (Gasolina, EVs)
    *   Matriz y gráficos de correlación.
    """)
    if st.button("Ir a Correlaciones", key="btn_corr"):
        # Ajusta el texto para que coincida con el nombre exacto en la barra lateral
        st.info("Selecciona '📈 Relaciones EV' o 'Correlaciones' en la barra lateral.")


st.divider()

# --- Sección de Ayuda o Contexto Adicional ---
st.markdown("""
### ¿Cómo usar este dashboard?
1.  **Navega:** Usa el menú en la barra lateral izquierda para elegir el dashboard que deseas explorar.
2.  **Interactúa:** Dentro de cada dashboard, encontrarás filtros (en la barra lateral) para ajustar los datos visualizados (ej: por fecha, región, tipo).
3.  **Explora:** Los gráficos son interactivos. Puedes pasar el cursor sobre ellos para ver detalles, hacer zoom o seleccionar/deseleccionar categorías en las leyendas.
""")

st.info("💡 **Nota:** La carga inicial de algunos dashboards puede tomar unos segundos mientras se obtienen y procesan los datos.")

# --- Pie de página (Opcional) ---
st.divider()
st.markdown("""
*Dashboard creado con Python, Streamlit y Plotly.*
*Fuentes de datos: Datos.gov.co, API XM, runt.gov.co*
*Desarrollado por: Shadow Team*
""")