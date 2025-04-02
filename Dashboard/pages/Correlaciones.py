# pages/4_📈_Relaciones_EV.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# --- CONFIGURACIÓN (Opcional) ---
st.set_page_config(page_title="Relaciones Mercado EV", page_icon="🔗", layout="wide")

st.title("🔗 Relaciones Temporales: Mercado EV y Factores Externos")
st.markdown("Análisis de la correlación entre adopción de EVs (BEV/PHEV), interés de búsqueda, precios e indicadores macroeconómicos (datos mensuales).")

# --- FUNCIÓN DE CARGA DE DATOS CSV ---
@st.cache_data
def load_final_data(filepath):
    try:
        df = pd.read_csv(filepath)
        st.success(f"Datos cargados desde '{filepath}'")
        return df
    except FileNotFoundError:
        st.error(f"Error: El archivo '{filepath}' no se encontró.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV '{filepath}': {e}")
        return None

# --- CARGA Y PREPROCESAMIENTO ---
FILE_FINAL = 'Dashboard\df_final.csv' # Asegúrate que este es el nombre correcto
df_orig_final = load_final_data(FILE_FINAL)

if df_orig_final is None:
    st.stop()

st.sidebar.header("Preprocesamiento (Relaciones):")
log_relaciones = []

df_final = df_orig_final.copy()

# 1. Limpiar nombres de columnas
original_cols = df_final.columns.tolist()
df_final.columns = df_final.columns.str.lower().str.replace(' ', '_')
# Asegurar nombres clave si vienen diferentes
df_final = df_final.rename(columns={
    'fecha_registro': 'fecha_registro', # Ya debería estar bien
    'google_trends': 'google_trends',
    'precio_gasolina': 'precio_gasolina',
    'precio_promedio_bev': 'precio_promedio_bev',
    'precio_promedio_phev': 'precio_promedio_phev',
    # Asegurar que bev, phev, ipc, ipp están en minúsculas
})
new_cols = df_final.columns.tolist()
renamed_count = sum(1 for o, n in zip(original_cols, new_cols) if o != n)
log_relaciones.append(f"✓ {renamed_count} columnas renombradas.")

# 2. Convertir 'fecha_registro' a datetime (asumiendo formato YYYY-MM o similar)
if 'fecha_registro' in df_final.columns:
    try:
        # Intentar convertir, puede necesitar especificar formato si falla
        df_final['fecha_registro'] = pd.to_datetime(df_final['fecha_registro'], errors='coerce')
        na_dates = df_final['fecha_registro'].isna().sum()
        if na_dates > 0:
            log_relaciones.append(f"✓ 'fecha_registro' a datetime ({na_dates} NaT eliminados).")
            df_final.dropna(subset=['fecha_registro'], inplace=True)
        else:
            log_relaciones.append("✓ 'fecha_registro' convertida a datetime.")
        # Ordenar por fecha, importante para series temporales
        df_final = df_final.sort_values('fecha_registro')
    except Exception as e:
        log_relaciones.append(f"✗ Error convirtiendo 'fecha_registro': {e}")
        st.stop()
else:
    log_relaciones.append("✗ Columna 'fecha_registro' no encontrada.")
    st.stop()

# 3. Convertir todas las demás columnas a numérico
numeric_cols = ['bev', 'phev', 'google_trends', 'ipc', 'ipp', 'precio_gasolina', 'precio_promedio_bev', 'precio_promedio_phev']
cols_converted = 0
cols_with_errors = []
for col in numeric_cols:
    if col in df_final.columns:
        original_type = df_final[col].dtype
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        if df_final[col].isna().sum() > 0 and not pd.api.types.is_numeric_dtype(original_type):
            cols_with_errors.append(col)
        cols_converted += 1
    else:
        log_relaciones.append(f"✗ Columna numérica esperada '{col}' no encontrada.")

log_relaciones.append(f"✓ {cols_converted} columnas procesadas a numérico.")
if cols_with_errors:
    log_relaciones.append(f"   (Columnas con valores no numéricos -> NaN: {cols_with_errors})")

# Eliminar filas con NaN en columnas clave si es necesario (opcional, depende de análisis)
# df_final.dropna(subset=['bev', 'phev'], inplace=True)

# Mostrar log
for log in log_relaciones:
    st.sidebar.write(log)

# --- FILTROS (Opcional para este dataset pequeño, pero puede ser útil) ---
# Podríamos filtrar por un rango de fechas si hubiera más datos,
# pero con <100 puntos, mostraremos todo por defecto.
# Podríamos añadir un filtro para seleccionar qué variables correlacionar.
st.sidebar.header("Opciones de Visualización")
# Ejemplo: Seleccionar variables para scatter plot
numeric_cols_available = [col for col in numeric_cols if col in df_final.columns]
var_x = st.sidebar.selectbox("Variable Eje X (Scatter Plot):", options=numeric_cols_available, index=numeric_cols_available.index('google_trends') if 'google_trends' in numeric_cols_available else 0)
var_y = st.sidebar.selectbox("Variable Eje Y (Scatter Plot):", options=numeric_cols_available, index=numeric_cols_available.index('bev') if 'bev' in numeric_cols_available else 1)


# --- KPIs (Último mes registrado) ---
st.header("Últimos Datos Registrados")
if not df_final.empty:
    last_data = df_final.iloc[-1] # Última fila (datos más recientes)
    last_date = last_data['fecha_registro'].strftime('%Y-%m')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"BEV ({last_date})", f"{last_data.get('bev', 'N/A'):,.0f}")
    col2.metric(f"PHEV ({last_date})", f"{last_data.get('phev', 'N/A'):,.0f}")
    col3.metric(f"Google Trends ({last_date})", f"{last_data.get('google_trends', 'N/A'):.1f}")
    col4.metric(f"Precio Gasolina ({last_date})", f"${last_data.get('precio_gasolina', 'N/A'):,.0f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric(f"IPC ({last_date})", f"{last_data.get('ipc', 'N/A'):.2f}")
    col6.metric(f"IPP ({last_date})", f"{last_data.get('ipp', 'N/A'):.2f}")
    col7.metric(f"Precio Prom. BEV ({last_date})", f"${last_data.get('precio_promedio_bev', 'N/A'):,.0f}")
    col8.metric(f"Precio Prom. PHEV ({last_date})", f"${last_data.get('precio_promedio_phev', 'N/A'):,.0f}")

else:
    st.warning("No hay datos procesados para mostrar KPIs.")


# --- VISUALIZACIONES ---
st.header("Evolución Temporal de Variables")

if df_final.empty:
    st.warning("No hay datos procesados para mostrar visualizaciones.")
else:
    # --- Gráficos de Líneas Temporales ---
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.subheader("Adopción Vehículos Eléctricos (Mensual)")
        fig_ev = px.line(df_final, x='fecha_registro', y=['bev', 'phev'],
                         title="Registros Mensuales BEV vs PHEV",
                         labels={'value': 'Número de Vehículos', 'fecha_registro': 'Fecha', 'variable': 'Tipo Vehículo'},
                         markers=True)
        fig_ev.update_layout(yaxis_title="Número de Vehículos")
        st.plotly_chart(fig_ev, use_container_width=True)

        st.subheader("Interés de Búsqueda y Precios Promedio")
        fig_trends_price = px.line(df_final, x='fecha_registro', y=['google_trends', 'precio_promedio_bev', 'precio_promedio_phev'],
                                   title="Google Trends 'Carro Eléctrico' vs Precios Promedio",
                                   labels={'value': 'Valor / Índice', 'fecha_registro': 'Fecha', 'variable': 'Variable'},
                                   markers=True)
        # Podríamos necesitar ejes Y separados aquí, pero Plotly Express no lo facilita.
        # Mostrarlos juntos puede ser engañoso si las escalas son muy diferentes.
        # Considerar gráficos separados o normalización si es necesario.
        fig_trends_price.update_layout(yaxis_title="Valor / Precio ($)")
        st.plotly_chart(fig_trends_price, use_container_width=True)


    with col_t2:
        st.subheader("Indicadores Económicos")
        fig_econ = px.line(df_final, x='fecha_registro', y=['ipc', 'ipp', 'precio_gasolina'],
                         title="Evolución IPC, IPP y Precio Gasolina",
                         labels={'value': 'Índice / Precio', 'fecha_registro': 'Fecha', 'variable': 'Indicador'},
                         markers=True)
        fig_econ.update_layout(yaxis_title="Valor Índice / Precio ($)")
        st.plotly_chart(fig_econ, use_container_width=True)

    st.divider()
    st.header("Análisis de Correlaciones")

    # --- Matriz de Correlación ---
    st.subheader("Matriz de Correlación Lineal")
    # Seleccionar solo columnas numéricas disponibles para la correlación
    corr_cols = [col for col in numeric_cols_available if col in df_final.columns]
    if len(corr_cols) > 1:
        correlation_matrix = df_final[corr_cols].corr()
        fig_corr = px.imshow(correlation_matrix,
                             text_auto=True, # Muestra los valores de correlación
                             aspect="auto", # Ajusta el tamaño
                             color_continuous_scale='RdBu_r', # Escala de color Rojo-Azul (r=reversed)
                             title="Correlación entre Variables Numéricas")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No hay suficientes columnas numéricas para calcular una matriz de correlación.")

    # --- Scatter Plot Interactivo ---
    st.subheader(f"Relación entre: {var_x.replace('_',' ').title()} y {var_y.replace('_',' ').title()}")
    if var_x in df_final.columns and var_y in df_final.columns:
        fig_scatter = px.scatter(df_final, x=var_x, y=var_y,
                                 title=f"Scatter Plot: {var_y.replace('_',' ').title()} vs {var_x.replace('_',' ').title()}",
                                 labels={var_x: var_x.replace('_',' ').title(), var_y: var_y.replace('_',' ').title()},
                                 trendline="ols", # Añade línea de regresión lineal ordinaria
                                 hover_data=['fecha_registro']) # Muestra la fecha al pasar el cursor
        st.plotly_chart(fig_scatter, use_container_width=True)
        # Mostrar resultados de la regresión OLS
        try:
            results = px.get_trendline_results(fig_scatter)
            st.write("Resultados Regresión Lineal (OLS):")
            st.dataframe(results.px_fit_results.iloc[0].summary().tables[0]) # Mostrar tabla de resumen
            st.dataframe(results.px_fit_results.iloc[0].summary().tables[1]) # Mostrar tabla de coeficientes
        except Exception as e:
             st.warning(f"No se pudieron obtener los resultados de la regresión: {e}")

    else:
        st.warning("Selecciona variables válidas en los filtros laterales para el Scatter Plot.")


# --- DATOS TABULARES (Opcional) ---
st.divider()
st.header("Exploración de Datos")
if st.checkbox("Mostrar tabla de datos procesados", key='tabla_relaciones'):
    if not df_final.empty:
        st.dataframe(df_final)
    else:
        st.write("No hay datos procesados para mostrar.")

if st.checkbox("Mostrar información del DataFrame", key='info_relaciones'):
     if not df_final.empty:
         buffer = StringIO()
         df_final.info(buf=buffer)
         s = buffer.getvalue()
         st.text(s)
     else:
        st.write("No hay datos procesados para mostrar información.")