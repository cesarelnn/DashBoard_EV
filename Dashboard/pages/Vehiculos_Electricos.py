import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # No usado directamente, se puede quitar
# import seaborn as sns # No usado directamente, se puede quitar
import plotly.express as px
import streamlit as st
from io import StringIO # Importado para df.info()

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
    """Carga datos desde un archivo CSV, con manejo de errores."""
    try:
        # Especificar encoding puede ayudar con caracteres especiales
        df = pd.read_csv(filepath, encoding='utf-8')
        # Alternativamente, prueba 'latin1' si utf-8 falla:
        # df = pd.read_csv(filepath, encoding='latin1')
        return df
    except FileNotFoundError:
        st.error(f"Error: El archivo '{filepath}' no se encontró. Asegúrate de que la ruta sea correcta (ej. 'Dashboard/VehiculosElectricos.csv').")
        return None
    except UnicodeDecodeError:
        st.error(f"Error de codificación al leer '{filepath}'. Intenta especificar 'utf-8' o 'latin1' en pd.read_csv.")
        # Intenta con latin1 como fallback
        try:
            df = pd.read_csv(filepath, encoding='latin1')
            st.warning("Se cargó el archivo usando codificación 'latin1'. Revisa si los caracteres especiales se ven bien.")
            return df
        except Exception as e:
            st.error(f"Error al cargar con 'latin1' también: {e}")
            return None
    except Exception as e:
        st.error(f"Error general al cargar el archivo CSV '{filepath}': {e}")
        return None

# Cargar los datos (Ajusta la ruta si es necesario)
# Usar / en lugar de \ para mejor compatibilidad entre sistemas operativos
DATA_FILEPATH = 'Dashboard/VehiculosElectricos.csv'
df_original = load_data(DATA_FILEPATH)

# Si la carga falló, detener la ejecución del script
if df_original is None:
    st.warning("La carga de datos inicial falló. El dashboard no puede continuar.")
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
    preprocessing_log.append(f"✓ Renombrado {len(col_rename_map)} columnas (ej: '{list(col_rename_map.keys())[0]}' -> '{list(col_rename_map.values())[0]}')")
else:
    preprocessing_log.append("✓ Nombres de columna ya en formato adecuado.")


# 2. Convertir 'fecha_registro' a datetime
if 'fecha_registro' in df.columns:
    original_dtype = str(df['fecha_registro'].dtype)
    # Intentar convertir, manejando posibles formatos diversos
    df['fecha_registro'] = pd.to_datetime(df['fecha_registro'], errors='coerce', dayfirst=False) # Asume MM/DD/YYYY o YYYY-MM-DD etc. Ajustar dayfirst=True si es DD/MM/YYYY
    if pd.api.types.is_datetime64_any_dtype(df['fecha_registro']):
        na_count = df['fecha_registro'].isna().sum()
        if na_count > 0:
            preprocessing_log.append(f"✓ 'fecha_registro' a datetime ({na_count} fechas no válidas -> NaT).")
        else:
            preprocessing_log.append("✓ 'fecha_registro' convertida a datetime.")
    else:
        preprocessing_log.append(f"✗ Error convirtiendo 'fecha_registro' (tipo original: {original_dtype}). Se mantuvo como objeto.")
else:
    preprocessing_log.append("✗ Columna 'fecha_registro' no encontrada.")

# 3. Convertir 'ano_registro' a numérico (entero)
if 'ano_registro' in df.columns:
    original_dtype = str(df['ano_registro'].dtype)
    # Primero convertir a numérico (float, para permitir NaN)
    df['ano_registro'] = pd.to_numeric(df['ano_registro'], errors='coerce')
    na_count_before = df['ano_registro'].isna().sum()

    if not df['ano_registro'].isna().all(): # Si hay algún valor no NaN
        try:
            # Int64 permite mantener NaNs si los hay
            df['ano_registro'] = df['ano_registro'].astype('Int64')
            na_count_after = df['ano_registro'].isna().sum()
            msg = f"✓ 'ano_registro' a Int64."
            if na_count_after > 0:
                msg += f" ({na_count_after} valores no válidos/NaN)"
            preprocessing_log.append(msg)
        except Exception as e:
             preprocessing_log.append(f"✗ Error convirtiendo 'ano_registro' a Int64: {e}")
    elif na_count_before == len(df):
        preprocessing_log.append("✓ 'ano_registro' contiene solo valores no válidos/vacíos.")
    else: # La columna existe pero no se pudo convertir o estaba vacía
         preprocessing_log.append(f"✓ 'ano_registro' (tipo {original_dtype}) procesado. {na_count_before} valores no válidos.")

else:
    preprocessing_log.append("✗ Columna 'ano_registro' no encontrada.")


# 4. Limpiar strings (quitar espacios extra) en columnas categóricas comunes
#    También manejar NaNs antes de aplicar .str
categorical_cols = ['combustible', 'estado', 'clase', 'servicio', 'marca', 'municipio', 'departamento']
cleaned_cols_count = 0
for col in categorical_cols:
    if col in df.columns and df[col].dtype == 'object':
        # Guardar NaNs, aplicar strip, restaurar NaNs si es necesario (aunque strip en NaN no da error)
        df[col] = df[col].str.strip()
        cleaned_cols_count += 1
if cleaned_cols_count > 0:
    preprocessing_log.append(f"✓ Espacios extra eliminados de {cleaned_cols_count} columnas de texto.")

# Mostrar log de preprocesamiento en la barra lateral
for log_entry in preprocessing_log:
    st.sidebar.write(log_entry)

# ----------------- TÍTULO DEL DASHBOARD --------------------
st.title("⚡ Dashboard de Vehículos Eléctricos en Colombia")
st.markdown("Análisis interactivo de registros de vehículos eléctricos basado en el archivo cargado.")

# ----------------- FILTROS EN SIDEBAR --------------------
st.sidebar.header("Filtros Interactivos")

# Crear copia para filtrar sin afectar el df preprocesado original
# Importante: Crear la copia ANTES de empezar a filtrar
df_filtered = df.copy()

# --- Filtro por Estado ---
if 'estado' in df.columns:
    # Obtener opciones desde el DataFrame *original* preprocesado para no perder opciones
    estados_opciones = ['Todos'] + sorted(df['estado'].dropna().unique().tolist())
    selected_estado = st.sidebar.selectbox("Estado del Vehículo:", options=estados_opciones, index=0) # Default 'Todos'
    if selected_estado != 'Todos':
        df_filtered = df_filtered[df_filtered['estado'] == selected_estado]
else:
    st.sidebar.warning("Columna 'estado' no encontrada para filtrar.")

# --- Filtro por Combustible ---
if 'combustible' in df.columns:
    # Opciones basadas en los datos filtrados hasta este punto
    combustibles_disponibles = sorted(df_filtered['combustible'].dropna().unique().tolist())
    if combustibles_disponibles:
        selected_combustibles = st.sidebar.multiselect(
            "Combustible(s):",
            options=combustibles_disponibles,
            default=combustibles_disponibles # Default: todos los disponibles seleccionados
        )
        if selected_combustibles:
            # Aplicar el filtro al DataFrame ya filtrado por los pasos anteriores
            df_filtered = df_filtered[df_filtered['combustible'].isin(selected_combustibles)]
        else:
            # Si el usuario deselecciona todos, mostrar dataframe vacío
            df_filtered = pd.DataFrame(columns=df.columns)
    else:
        st.sidebar.info("No hay tipos de combustible disponibles con los filtros actuales.")
else:
    st.sidebar.warning("Columna 'combustible' no encontrada para filtrar.")
# --- Filtro por Departamento ---
if 'departamento' in df.columns:
    # Opciones desde el original preprocesado
    departamentos_opciones = sorted(df['departamento'].dropna().unique().tolist())
    # Default = todas las opciones disponibles inicialmente
    selected_departamentos = st.sidebar.multiselect("Departamento(s):", options=departamentos_opciones, default=departamentos_opciones)
    if selected_departamentos:
        # Filtrar el df_filtered actual
        df_filtered = df_filtered[df_filtered['departamento'].isin(selected_departamentos)]
    else: # Si el usuario deselecciona todo
        df_filtered = pd.DataFrame(columns=df.columns) # Dataframe vacío
else:
    st.sidebar.warning("Columna 'departamento' no encontrada para filtrar.")


# --- Filtro por Año de Registro ---
# Comprobar si la columna existe Y si tiene al menos un valor numérico válido
if 'ano_registro' in df.columns and df_filtered['ano_registro'].notna().any():
    # Calcular min/max sobre los datos *filtrados hasta ahora* que no sean NaN
    min_year = int(df_filtered['ano_registro'].dropna().min())
    max_year = int(df_filtered['ano_registro'].dropna().max())

    # Asegurar que min_year no sea mayor que max_year (puede pasar si solo queda 1 año)
    if min_year <= max_year:
        selected_years = st.sidebar.slider(
            "Año de Registro:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year) # Rango por defecto: todos los años disponibles en df_filtered
        )
        # Aplicar filtro sobre df_filtered
        # Incluir NaNs si el usuario no los ha filtrado explícitamente de otra forma
        df_filtered = df_filtered[
            (df_filtered['ano_registro'] >= selected_years[0]) &
            (df_filtered['ano_registro'] <= selected_years[1]) |
            (df_filtered['ano_registro'].isna())
        ]
    else:
        st.sidebar.info("Solo un año disponible con los filtros actuales.")
        # No es necesario filtrar más por año si solo hay uno
elif 'ano_registro' in df.columns:
     st.sidebar.info("No hay años de registro válidos con los filtros actuales.")
else:
    st.sidebar.warning("Columna 'ano_registro' no válida o no encontrada para filtrar.")


# --- Filtro por Marca ---
if 'marca' in df.columns:
    # Opciones basadas en los datos filtrados *hasta este punto*
    marcas_disponibles = sorted(df_filtered['marca'].dropna().unique().tolist())
    if marcas_disponibles: # Solo mostrar si hay opciones
        # Default: todas las marcas disponibles DESPUÉS de aplicar filtros anteriores
        selected_marcas = st.sidebar.multiselect("Marca(s):", options=marcas_disponibles, default=marcas_disponibles)
        if selected_marcas:
             df_filtered = df_filtered[df_filtered['marca'].isin(selected_marcas)]
        else:
            df_filtered = pd.DataFrame(columns=df.columns) # Dataframe vacío
    else:
        st.sidebar.info("No hay marcas disponibles con los filtros actuales.")
else:
    st.sidebar.warning("Columna 'marca' no encontrada para filtrar.")

# --- Filtro por Clase ---
if 'clase' in df.columns:
    # Opciones basadas en los datos filtrados hasta este punto
    clases_disponibles = sorted(df_filtered['clase'].dropna().unique().tolist())
    if clases_disponibles:
        selected_clases = st.sidebar.multiselect("Clase(s) de Vehículo:", options=clases_disponibles, default=clases_disponibles)
        if selected_clases:
            df_filtered = df_filtered[df_filtered['clase'].isin(selected_clases)]
        else:
            df_filtered = pd.DataFrame(columns=df.columns) # Dataframe vacío
    else:
        st.sidebar.info("No hay clases disponibles con los filtros actuales.")
else:
    st.sidebar.warning("Columna 'clase' no encontrada para filtrar.")



# ----------------- MOSTRAR MÉTRICAS PRINCIPALES (KPIs) ---------------
st.header("Resumen General (Datos Filtrados)")

# Recalcular métricas con el df_filtered final
total_registros_filtrados = len(df_filtered)
total_marcas_unicas = df_filtered['marca'].nunique() if 'marca' in df_filtered.columns else 0
total_departamentos_unicos = df_filtered['departamento'].nunique() if 'departamento' in df_filtered.columns else 0
# Podríamos añadir un KPI de combustibles si quisiéramos:
total_combustibles_unicos = df_filtered['combustible'].nunique() if 'combustible' in df_filtered.columns else 0


col1, col2, col3 = st.columns(3)
col1.metric("Total Vehículos", f"{total_registros_filtrados:,}")
col2.metric("Marcas Únicas", f"{total_marcas_unicas:,}")
col3.metric("Departamentos Únicos", f"{total_departamentos_unicos:,}")
# Ejemplo de cómo añadir el nuevo KPI si se quiere:
# col4.metric("Combustibles Únicos", f"{total_combustibles_unicos:,}")


# ----------------- VISUALIZACIONES (usando df_filtered final) ---------------
st.header("Análisis Visual")

# Verificar si hay datos DESPUÉS de aplicar TODOS los filtros
if df_filtered.empty:
    st.warning("No hay datos disponibles para mostrar con la combinación de filtros seleccionada.")
else:
    # --- Layout de Gráficos ---
    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        # 1. Registros por Año (si la columna existe y es válida en el df_filtered)
        st.subheader("Vehículos Registrados por Año")
        # Comprobar si hay datos válidos en la columna AÑO del df FILTRADO
        if 'ano_registro' in df_filtered.columns and df_filtered['ano_registro'].notna().any():
            # Agrupar por año, contar y ordenar
            registros_por_ano = df_filtered.dropna(subset=['ano_registro'])['ano_registro'].astype(int).value_counts().sort_index().reset_index()
            registros_por_ano.columns = ['Año', 'Cantidad']
            fig_anos = px.bar(registros_por_ano, x='Año', y='Cantidad',
                              # Título dinámico opcional
                              # title=f"Registros por Año ({selected_years[0]}-{selected_years[1]})",
                              title="Número de Vehículos Registrados por Año",
                              labels={'Año':'Año de Registro', 'Cantidad':'Número de Vehículos'},
                              text='Cantidad')
            # Ajustar ejes y formato de texto
            fig_anos.update_traces(textposition='outside', texttemplate='%{text:,}')
            fig_anos.update_layout(xaxis_type='category', xaxis_title="Año", yaxis_title="Cantidad")
            st.plotly_chart(fig_anos, use_container_width=True)
        else:
            st.info("Gráfico de registros por año no disponible para la selección actual.")


        # 3. Distribución por Clase
        st.subheader("Distribución por Clase de Vehículo")
        if 'clase' in df_filtered.columns and df_filtered['clase'].notna().any():
            vehiculos_por_clase = df_filtered['clase'].value_counts().reset_index()
            vehiculos_por_clase.columns = ['Clase', 'Cantidad']
            fig_clase = px.bar(vehiculos_por_clase.sort_values('Cantidad', ascending=True),
                               x='Cantidad', y='Clase', orientation='h',
                               title="Cantidad de Vehículos por Clase",
                               labels={'Clase':'Clase de Vehículo', 'Cantidad':'Número de Vehículos'},
                               text='Cantidad')
            fig_clase.update_traces(textposition='outside', texttemplate='%{text:,}')
            fig_clase.update_layout(yaxis_title="Clase", xaxis_title="Cantidad")
            st.plotly_chart(fig_clase, use_container_width=True)
        else:
            st.info("Gráfico de distribución por clase no disponible para la selección actual.")


    with col_vis2:
        # 2. Distribución por Marca (Top N + Otros)
        st.subheader("Distribución por Marca")
        if 'marca' in df_filtered.columns and df_filtered['marca'].notna().any():
            max_marcas_pie = 10 # Mostrar las N marcas principales + "Otros"
            vehiculos_por_marca = df_filtered['marca'].value_counts().reset_index()
            vehiculos_por_marca.columns = ['Marca', 'Cantidad']

            if len(vehiculos_por_marca) > max_marcas_pie:
                # Asegurar que max_marcas_pie-1 no sea negativo si hay pocas marcas
                top_n = max(1, max_marcas_pie - 1)
                top_marcas = vehiculos_por_marca.nlargest(top_n, 'Cantidad')
                # Calcular 'Otros' solo si hay más marcas que las top_n
                if len(vehiculos_por_marca) > top_n:
                     otros_sum = vehiculos_por_marca.nsmallest(len(vehiculos_por_marca) - top_n, 'Cantidad')['Cantidad'].sum()
                     if otros_sum > 0:
                         otros_df = pd.DataFrame([{'Marca': 'Otros', 'Cantidad': otros_sum}])
                         vehiculos_por_marca_pie = pd.concat([top_marcas, otros_df], ignore_index=True)
                     else: # Suma de otros es 0
                         vehiculos_por_marca_pie = top_marcas
                else: # No hay 'Otros' que calcular
                    vehiculos_por_marca_pie = top_marcas

            else: # Si hay menos de N marcas, mostrarlas todas
                vehiculos_por_marca_pie = vehiculos_por_marca

            # Solo graficar si tenemos datos en vehiculos_por_marca_pie
            if not vehiculos_por_marca_pie.empty:
                fig_marcas = px.pie(vehiculos_por_marca_pie,
                                   names='Marca', values='Cantidad',
                                   title=f"Distribución por Marca (Top {len(top_marcas) if 'top_marcas' in locals() else len(vehiculos_por_marca_pie)}{' y Otros' if 'otros_df' in locals() else ''})",
                                   hole=0.4) # Gráfico de dona
                fig_marcas.update_traces(textposition='outside', textinfo='percent+label', pull=[0.05] * len(vehiculos_por_marca_pie))
                st.plotly_chart(fig_marcas, use_container_width=True)

                # Opcional: Mostrar tabla completa si se agruparon marcas
                if 'otros_df' in locals():
                     with st.expander("Ver todas las marcas y sus cantidades"):
                         st.dataframe(vehiculos_por_marca.style.format({"Cantidad": "{:,}"}))
            else:
                 st.info("No hay datos de marca para mostrar en el gráfico circular con la selección actual.")

        else:
            st.info("Gráfico de distribución por marca no disponible para la selección actual.")


        # 4. Distribución Geográfica (Top N Departamentos)
        st.subheader("Distribución por Departamento")
        if 'departamento' in df_filtered.columns and df_filtered['departamento'].notna().any():
            top_n_dptos = 15
            vehiculos_por_dpto = df_filtered['departamento'].value_counts().reset_index()
            vehiculos_por_dpto.columns = ['Departamento', 'Cantidad']
            # Asegurar que no intentamos mostrar más de los que hay
            n_to_show = min(top_n_dptos, len(vehiculos_por_dpto))

            if n_to_show > 0:
                fig_dpto = px.bar(vehiculos_por_dpto.head(n_to_show),
                                    x='Departamento', y='Cantidad',
                                    title=f"Top {n_to_show} Departamentos por Cantidad de Vehículos",
                                    labels={'Departamento':'Departamento', 'Cantidad':'Número de Vehículos'},
                                    text='Cantidad')
                fig_dpto.update_traces(textposition='outside', texttemplate='%{text:,}')
                fig_dpto.update_layout(xaxis_tickangle=-45, xaxis_title="Departamento", yaxis_title="Cantidad")
                st.plotly_chart(fig_dpto, use_container_width=True)
            else:
                 st.info("No hay datos de departamento para mostrar en el gráfico de barras con la selección actual.")
        else:
            st.info("Gráfico de distribución por departamento no disponible para la selección actual.")


    # 5. Serie de Tiempo (si fecha_registro es válida en el df_filtered)
    st.subheader("Evolución Temporal de Registros (Mensual)")
    if 'fecha_registro' in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered['fecha_registro']) and df_filtered['fecha_registro'].notna().any():
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
             st.info("No hay datos de fecha válidos para mostrar la serie temporal con la selección actual.")
    else:
        st.info("Gráfico de serie temporal no disponible (columna 'fecha_registro' inválida o faltante en la selección actual).")



# ----------------- MOSTRAR DATOS TABULARES (Opcional) ---------
st.header("Exploración de Datos")
if st.checkbox("Mostrar tabla de datos filtrados"):
    if not df_filtered.empty:
        # Mostrar un número limitado de filas por defecto para no sobrecargar
        st.dataframe(df_filtered) # Streamlit maneja la paginación básica
        st.caption(f"Mostrando {len(df_filtered):,} registros filtrados.")
    else:
        st.write("No hay datos en la tabla para mostrar con los filtros seleccionados.")

if st.checkbox("Mostrar descripción estadística de columnas numéricas"):
    if not df_filtered.empty:
        # Seleccionar solo columnas numéricas para describe()
        df_numeric = df_filtered.select_dtypes(include=np.number)
        if not df_numeric.empty:
            st.write(df_numeric.describe())
        else:
            st.write("No hay columnas numéricas en los datos filtrados para describir.")
    else:
        st.write("No hay datos filtrados para describir.")

if st.checkbox("Mostrar información del DataFrame Filtrado (tipos, no nulos)"):
     if not df_filtered.empty:
         buffer = StringIO()
         df_filtered.info(buf=buffer)
         s = buffer.getvalue()
         st.text(s)
     else:
         st.write("No hay datos filtrados para mostrar información.")
