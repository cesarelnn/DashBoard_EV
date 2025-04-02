import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# ---------------- CONFIGURACIÓN DE LA PÁGINA -----------------
st.set_page_config(
    page_title="Dashboard Vehículos Eléctricos CO",
    page_icon="⚡",
    layout="wide"
)

# ----------------- FUNCIÓN DE CARGA Y CACHÉ --------------------------
# Usar caché para no recargar el archivo cada vez que interactuamos con un widget
@st.cache_data # Nueva sintaxis de caché en Streamlit > 1.18.0
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: El archivo '{filepath}' no se encontró. Asegúrate de que esté en el mismo directorio que el script.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {e}")
        return None

# Cargar los datos
df_original = load_data('Dashboard\VehiculosElectricos.csv')

# Si la carga falló, detener la ejecución del script
if df_original is None:
    st.stop()

# -------- COPIA Y PREPROCESAMIENTO --------
df = df_original.copy()
st.sidebar.header("Preprocesamiento Aplicado:")
preprocessing_log = []

# 1. Limpiar nombres de columnas (minúsculas, sin espacios, sin tildes)
original_cols = df.columns.tolist()
df.columns = df.columns.str.lower().str.replace(' ', '_').str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
new_cols = df.columns.tolist()
col_rename_map = {orig: new for orig, new in zip(original_cols, new_cols) if orig != new}
if col_rename_map:
    preprocessing_log.append(f"✓ Renombrado columnas: {len(col_rename_map)} cambiadas (ej: '{list(col_rename_map.keys())[0]}' -> '{list(col_rename_map.values())[0]}')")
else:
    preprocessing_log.append("✓ Nombres de columna ya estaban en formato adecuado.")


# 2. Convertir 'fecha_registro' a datetime
if 'fecha_registro' in df.columns:
    original_dtype = str(df['fecha_registro'].dtype)
    # Intentar convertir, manejando posibles formatos diversos
    df['fecha_registro'] = pd.to_datetime(df['fecha_registro'], errors='coerce', dayfirst=False) # Asume MM/DD/YYYY o YYYY-MM-DD etc. Ajustar dayfirst=True si es DD/MM/YYYY
    if pd.api.types.is_datetime64_any_dtype(df['fecha_registro']):
        na_count = df['fecha_registro'].isna().sum()
        if na_count > 0:
            preprocessing_log.append(f"✓ 'fecha_registro' convertida a datetime ({na_count} fechas no válidas -> NaT).")
        else:
            preprocessing_log.append("✓ 'fecha_registro' convertida a datetime.")
    else:
        preprocessing_log.append(f"✗ Error convirtiendo 'fecha_registro' (tipo original: {original_dtype}). Se mantuvo como objeto.")
else:
    preprocessing_log.append("✗ Columna 'fecha_registro' no encontrada.")

# 3. Convertir 'ano_registro' a numérico (entero)
if 'ano_registro' in df.columns:
    original_dtype = str(df['ano_registro'].dtype)
    df['ano_registro'] = pd.to_numeric(df['ano_registro'], errors='coerce')
    # Tratar los NaN antes de convertir a Int64 (que soporta NaN) o int (que no)
    na_count_before = df['ano_registro'].isna().sum()
    
    
    if not df['ano_registro'].isna().all(): # Si hay algún valor no NaN
        try:
            # Int64 permite mantener NaNs si los hay
            df['ano_registro'] = df['ano_registro'].astype('Int64')
            na_count_after = df['ano_registro'].isna().sum()
            preprocessing_log.append(f"✓ 'ano_registro' convertido a numérico (Int64). {na_count_after} valores no válidos.")
        except Exception as e:
             preprocessing_log.append(f"✗ Error convirtiendo 'ano_registro' a Int64: {e}")
    else:
        preprocessing_log.append("✓ 'ano_registro' contiene solo valores no válidos/vacíos.")

else:
    preprocessing_log.append("✗ Columna 'ano_registro' no encontrada.")


# 4. Limpiar strings (quitar espacios extra) en columnas categóricas comunes
categorical_cols = ['combustible', 'estado', 'clase', 'servicio', 'marca', 'municipio', 'departamento']
cleaned_cols_count = 0
for col in categorical_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.strip()
        cleaned_cols_count += 1
if cleaned_cols_count > 0:
    preprocessing_log.append(f"✓ Espacios extra eliminados de {cleaned_cols_count} columnas de texto.")

# Mostrar log de preprocesamiento en la barra lateral
for log_entry in preprocessing_log:
    st.sidebar.write(log_entry)

# ----------------- TÍTULO DEL DASHBOARD --------------------
st.title("⚡ Dashboard de Vehículos Eléctricos en Colombia")
st.markdown("Análisis interactivo de registros de vehículos eléctricos basado en `VehiculosElectricos.csv`")

# ----------------- FILTROS EN SIDEBAR --------------------
st.sidebar.header("Filtros Interactivos")

# Crear copia para filtrar sin afectar el df preprocesado original
df_filtered = df.copy()

# --- Filtro por Estado ---
if 'estado' in df_filtered.columns:
    estados_disponibles = ['Todos'] + sorted(df_filtered['estado'].unique().tolist())
    selected_estado = st.sidebar.selectbox("Estado del Vehículo:", options=estados_disponibles, index=0) # Default 'Todos'
    if selected_estado != 'Todos':
        df_filtered = df_filtered[df_filtered['estado'] == selected_estado]
else:
    st.sidebar.warning("Columna 'estado' no encontrada para filtrar.")

# --- Filtro por Departamento ---
if 'departamento' in df_filtered.columns:
    departamentos_disponibles = sorted(df_filtered['departamento'].unique().tolist())
    selected_departamentos = st.sidebar.multiselect("Departamento(s):", options=departamentos_disponibles, default=departamentos_disponibles)
    if selected_departamentos:
        df_filtered = df_filtered[df_filtered['departamento'].isin(selected_departamentos)]
    else: # Si el usuario deselecciona todo
        df_filtered = pd.DataFrame(columns=df.columns) # Dataframe vacío
else:
    st.sidebar.warning("Columna 'departamento' no encontrada para filtrar.")


# --- Filtro por Año de Registro ---
if 'ano_registro' in df_filtered.columns and df_filtered['ano_registro'].notna().any():
    min_year = int(df_filtered['ano_registro'].min())
    max_year = int(df_filtered['ano_registro'].max())
    selected_years = st.sidebar.slider(
        "Año de Registro:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year) # Rango por defecto: todos los años
    )
    # Asegurarse de que el filtro se aplique correctamente incluso con NaNs
    df_filtered = df_filtered[
        (df_filtered['ano_registro'] >= selected_years[0]) &
        (df_filtered['ano_registro'] <= selected_years[1]) |
        (df_filtered['ano_registro'].isna()) # Incluir NaNs si no se han filtrado antes
    ]
else:
    st.sidebar.warning("Columna 'ano_registro' no válida o no encontrada para filtrar.")


# --- Filtro por Marca ---
if 'marca' in df_filtered.columns:
    marcas_disponibles = sorted(df_filtered['marca'].unique().tolist())
    selected_marcas = st.sidebar.multiselect("Marca(s):", options=marcas_disponibles, default=marcas_disponibles)
    if selected_marcas:
         df_filtered = df_filtered[df_filtered['marca'].isin(selected_marcas)]
    else:
        df_filtered = pd.DataFrame(columns=df.columns) # Dataframe vacío
else:
    st.sidebar.warning("Columna 'marca' no encontrada para filtrar.")


# --- Filtro por Clase ---
if 'clase' in df_filtered.columns:
    clases_disponibles = sorted(df_filtered['clase'].unique().tolist())
    selected_clases = st.sidebar.multiselect("Clase(s) de Vehículo:", options=clases_disponibles, default=clases_disponibles)
    if selected_clases:
        df_filtered = df_filtered[df_filtered['clase'].isin(selected_clases)]
    else:
        df_filtered = pd.DataFrame(columns=df.columns) # Dataframe vacío
else:
    st.sidebar.warning("Columna 'clase' no encontrada para filtrar.")

# ----------------- MOSTRAR MÉTRICAS PRINCIPALES (KPIs) ---------------
st.header("Resumen General")
total_registros_filtrados = len(df_filtered)
total_marcas_unicas = df_filtered['marca'].nunique() if 'marca' in df_filtered.columns else 0
total_departamentos_unicos = df_filtered['departamento'].nunique() if 'departamento' in df_filtered.columns else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Vehículos (Filtrados)", f"{total_registros_filtrados:,}")
col2.metric("Marcas Únicas (Filtradas)", f"{total_marcas_unicas:,}")
col3.metric("Departamentos Únicos (Filtrados)", f"{total_departamentos_unicos:,}")


# ----------------- VISUALIZACIONES (usando df_filtered) ---------------
st.header("Análisis Visual")

# Verificar si hay datos después de filtrar antes de intentar graficar
if df_filtered.empty:
    st.warning("No hay datos disponibles para mostrar con los filtros seleccionados.")
else:
    # --- Layout de Gráficos ---
    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        # 1. Registros por Año (si la columna existe y es válida)
        st.subheader("Vehículos Registrados por Año")
        if 'ano_registro' in df_filtered.columns and df_filtered['ano_registro'].notna().any():
            registros_por_ano = df_filtered['ano_registro'].value_counts().sort_index().reset_index()
            registros_por_ano.columns = ['Año', 'Cantidad']
            fig_anos = px.bar(registros_por_ano, x='Año', y='Cantidad',
                              title="Número de Vehículos Eléctricos Registrados por Año",
                              labels={'Año':'Año de Registro', 'Cantidad':'Número de Vehículos'},
                              text='Cantidad')
            fig_anos.update_layout(xaxis_type='category', xaxis_title="Año", yaxis_title="Cantidad")
            st.plotly_chart(fig_anos, use_container_width=True)
        else:
            st.info("Gráfico de registros por año no disponible (columna 'ano_registro' inválida o faltante).")


        # 3. Distribución por Clase
        st.subheader("Distribución por Clase de Vehículo")
        if 'clase' in df_filtered.columns:
            vehiculos_por_clase = df_filtered['clase'].value_counts().reset_index()
            vehiculos_por_clase.columns = ['Clase', 'Cantidad']
            fig_clase = px.bar(vehiculos_por_clase.sort_values('Cantidad', ascending=True),
                               x='Cantidad', y='Clase', orientation='h',
                               title="Cantidad de Vehículos por Clase",
                               labels={'Clase':'Clase de Vehículo', 'Cantidad':'Número de Vehículos'},
                               text='Cantidad')
            fig_clase.update_layout(yaxis_title="Clase", xaxis_title="Cantidad")
            st.plotly_chart(fig_clase, use_container_width=True)
        else:
            st.info("Gráfico de distribución por clase no disponible (columna 'clase' faltante).")


    with col_vis2:
        # 2. Distribución por Marca (Top N + Otros)
        st.subheader("Distribución por Marca")
        if 'marca' in df_filtered.columns:
            max_marcas_pie = 10 # Mostrar las N marcas principales + "Otros"
            vehiculos_por_marca = df_filtered['marca'].value_counts().reset_index()
            vehiculos_por_marca.columns = ['Marca', 'Cantidad']

            if len(vehiculos_por_marca) > max_marcas_pie:
                top_marcas = vehiculos_por_marca.nlargest(max_marcas_pie - 1, 'Cantidad')
                otros_sum = vehiculos_por_marca.nsmallest(len(vehiculos_por_marca) - (max_marcas_pie - 1), 'Cantidad')['Cantidad'].sum()
                if otros_sum > 0:
                    otros_df = pd.DataFrame([{'Marca': 'Otros', 'Cantidad': otros_sum}])
                    vehiculos_por_marca_pie = pd.concat([top_marcas, otros_df], ignore_index=True)
                else: # Si la suma de otros es 0, no añadir fila 'Otros'
                    vehiculos_por_marca_pie = top_marcas
            else: # Si hay menos de N marcas, mostrarlas todas
                vehiculos_por_marca_pie = vehiculos_por_marca

            fig_marcas = px.pie(vehiculos_por_marca_pie,
                               names='Marca', values='Cantidad',
                               title=f"Distribución por Marca (Top {max_marcas_pie-1} y Otros)",
                               hole=0.4) # Gráfico de dona
            fig_marcas.update_traces(textposition='outside', textinfo='percent+label', pull=[0.05] * len(vehiculos_por_marca_pie)) # Separar un poco los slices
            st.plotly_chart(fig_marcas, use_container_width=True)

            # Opcional: Mostrar tabla completa si se agruparon marcas
            if len(vehiculos_por_marca) > max_marcas_pie:
                 with st.expander("Ver todas las marcas y sus cantidades"):
                     st.dataframe(vehiculos_por_marca.style.format({"Cantidad": "{:,}"}))

        else:
            st.info("Gráfico de distribución por marca no disponible (columna 'marca' faltante).")


        # 4. Distribución Geográfica (Top N Departamentos)
        st.subheader("Distribución por Departamento")
        if 'departamento' in df_filtered.columns:
            top_n_dptos = 15
            vehiculos_por_dpto = df_filtered['departamento'].value_counts().reset_index()
            vehiculos_por_dpto.columns = ['Departamento', 'Cantidad']
            fig_dpto = px.bar(vehiculos_por_dpto.head(top_n_dptos), # Tomar los N primeros (ya están ordenados por value_counts)
                                x='Departamento', y='Cantidad',
                                title=f"Top {top_n_dptos} Departamentos por Cantidad de Vehículos",
                                labels={'Departamento':'Departamento', 'Cantidad':'Número de Vehículos'},
                                text='Cantidad')
            fig_dpto.update_layout(xaxis_tickangle=-45, xaxis_title="Departamento", yaxis_title="Cantidad")
            st.plotly_chart(fig_dpto, use_container_width=True)
        else:
            st.info("Gráfico de distribución por departamento no disponible (columna 'departamento' faltante).")


    # 5. Serie de Tiempo (si fecha_registro es válida)
    st.subheader("Evolución Temporal de Registros (Mensual)")
    if 'fecha_registro' in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered['fecha_registro']):
        # Asegurarse de que no haya NaT antes de agrupar
        df_temporal = df_filtered.dropna(subset=['fecha_registro']).copy()
        if not df_temporal.empty:
            # Agrupar por mes (Periodo para mejor manejo de fechas)
            df_temporal['mes_registro'] = df_temporal['fecha_registro'].dt.to_period('M')
            registros_mensuales = df_temporal.groupby('mes_registro').size().reset_index(name='Cantidad')
            # Convertir Periodo a timestamp para Plotly (inicio del periodo)
            registros_mensuales['mes_registro'] = registros_mensuales['mes_registro'].dt.to_timestamp()
            registros_mensuales = registros_mensuales.sort_values('mes_registro')

            fig_tiempo = px.line(registros_mensuales, x='mes_registro', y='Cantidad',
                                 title="Número de Registros por Mes",
                                 labels={'mes_registro': 'Mes', 'Cantidad': 'Número de Vehículos'},
                                 markers=True) # Puntos en la línea
            fig_tiempo.update_layout(xaxis_title="Mes de Registro", yaxis_title="Cantidad")
            st.plotly_chart(fig_tiempo, use_container_width=True)
        else:
             st.info("No hay datos de fecha válidos para mostrar la serie temporal con los filtros actuales.")
    else:
        st.info("Gráfico de serie temporal no disponible (columna 'fecha_registro' inválida o faltante).")



# ----------------- MOSTRAR DATOS TABULARES (Opcional) ---------
st.header("Exploración de Datos")
if st.checkbox("Mostrar tabla de datos filtrados"):
    st.dataframe(df_filtered)

if st.checkbox("Mostrar descripción estadística de columnas numéricas"):
    if not df_filtered.empty:
        st.write(df_filtered.describe())
    else:
        st.write("No hay datos filtrados para describir.")

if st.checkbox("Mostrar información del DataFrame (tipos de datos, no nulos)"):
     # Usar buffer para capturar la salida de df.info()
     from io import StringIO
     buffer = StringIO()
     df_filtered.info(buf=buffer)
     s = buffer.getvalue()
     st.text(s)