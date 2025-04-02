# pages/3_💨_Emisiones_CO2.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Para añadir anotaciones o más control
from io import StringIO

# --- CONFIGURACIÓN (Opcional) ---
st.set_page_config(page_title="Análisis Emisiones CO2", page_icon="💨", layout="wide")

st.title("💨 Análisis de Emisiones de Gases de Efecto Invernadero (GEI)")
st.markdown("Enfoque en emisiones totales y del sector transporte por carretera.")

# --- FUNCIÓN DE CARGA DE DATOS EXCEL ---
@st.cache_data
def load_excel_data(filepath):
    try:
        df = pd.read_excel(filepath)
        st.success(f"Datos cargados desde '{filepath}'")
        return df
    except FileNotFoundError:
        st.error(f"Error: El archivo '{filepath}' no se encontró. Asegúrate de que la ruta sea correcta.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el archivo Excel '{filepath}': {e}")
        return None

# --- CARGA Y PREPROCESAMIENTO INICIAL ---
# Ajusta las rutas si tus archivos no están en 'Dashboard/' relativo al script principal
FILE_CO2 = 'Dashboard\EmsionesCo2.xlsx'
FILE_GEI_TOTAL = 'Dashboard\GasesEfectoInvernadero.xlsx'

df_co2_orig = load_excel_data(FILE_CO2)
df_gei_total_orig = load_excel_data(FILE_GEI_TOTAL)

# Detener si alguna carga falló
if df_co2_orig is None or df_gei_total_orig is None:
    st.warning("No se pudieron cargar todos los archivos de datos. El análisis puede estar incompleto.")
    st.stop()

st.sidebar.header("Preprocesamiento (Emisiones):")
log_emisiones = []

def clean_emission_df(df, name):
    """Función auxiliar para limpiar dataframes de emisiones."""
    if df is None: return None, []
    df_clean = df.copy()
    log = []
    original_cols = df_clean.columns.tolist()
    # Limpieza básica: minúsculas, guion bajo
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    # Específico: asegurar nombres clave
    df_clean = df_clean.rename(columns={
        'año': 'ano', # Asegurar consistencia si viene como 'año'
        'clasificacion': 'clasificacion',
        'total_emisiones': 'total_emisiones' # Asegurar este nombre clave
    })
    new_cols = df_clean.columns.tolist()
    renamed_count = sum(1 for o, n in zip(original_cols, new_cols) if o != n)
    log.append(f"✓ ({name}) {renamed_count} cols renombradas.")

    # Convertir tipos
    if 'ano' in df_clean.columns:
        df_clean['ano'] = pd.to_numeric(df_clean['ano'], errors='coerce').astype('Int64')
        log.append(f"✓ ({name}) 'ano' a numérico Int64.")
    else: log.append(f"✗ ({name}) Columna 'ano' no encontrada.")

    if 'total_emisiones' in df_clean.columns:
        df_clean['total_emisiones'] = pd.to_numeric(df_clean['total_emisiones'], errors='coerce')
        log.append(f"✓ ({name}) 'total_emisiones' a numérico float.")
    else: log.append(f"✗ ({name}) Columna 'total_emisiones' no encontrada.")

    # Eliminar filas donde año o emisiones sean NaN después de la conversión
    rows_before = len(df_clean)
    df_clean.dropna(subset=['ano', 'total_emisiones'], inplace=True)
    rows_after = len(df_clean)
    if rows_before > rows_after:
         log.append(f"✓ ({name}) {rows_before - rows_after} filas con NaN en 'ano'/'total_emisiones' eliminadas.")

    return df_clean, log

# Limpiar ambos dataframes
df_co2, log1 = clean_emission_df(df_co2_orig, "CO2")
df_gei_total, log2 = clean_emission_df(df_gei_total_orig, "GEI Total")
log_emisiones.extend(log1)
log_emisiones.extend(log2)


# Preparar DataFrame para Gráfico de Comparación (Gráfico 3)
comparison_df = pd.DataFrame()
if df_gei_total is not None:
    road_transport_data = df_gei_total[df_gei_total['clasificacion'] == "1.A.3.b. Transporte por carretera"].copy()
    national_data = df_gei_total[df_gei_total['clasificacion'] == "Nacional"].copy()

    if not road_transport_data.empty and not national_data.empty:
        comparison_df = pd.merge(
            road_transport_data[['ano', 'total_emisiones']],
            national_data[['ano', 'total_emisiones']],
            on='ano',
            suffixes=('_carretera', '_nacional')
        )
        # Calcular porcentaje
        comparison_df['porcentaje_carretera'] = (
            (comparison_df['total_emisiones_carretera'] / comparison_df['total_emisiones_nacional']) * 100
        ).fillna(0) # Evitar división por cero si nacional es 0
        log_emisiones.append("✓ DataFrame de comparación ('carretera' vs 'nacional') creado.")
    else:
        log_emisiones.append("✗ No se encontraron datos 'Nacional' o 'Transporte por carretera' en GEI Total para comparación.")
else:
    log_emisiones.append("✗ No se pudo procesar GEI Total para comparación.")

# Mostrar log
for log in log_emisiones:
    st.sidebar.write(log)

# --- FILTROS EN SIDEBAR ---
st.sidebar.header("Filtros (Emisiones)")

# Verificar que df_co2 existe y tiene datos antes de crear filtros
if df_co2 is not None and not df_co2.empty:
    # Filtro por Año
    min_year_co2 = int(df_co2['ano'].min())
    max_year_co2 = int(df_co2['ano'].max())
    selected_years = st.sidebar.slider(
        "Selecciona Rango de Años:",
        min_value=min_year_co2,
        max_value=max_year_co2,
        value=(min_year_co2, max_year_co2), # Default todo el rango
        key='year_slider_co2'
    )

    # Filtro por Clasificación
    all_classifications = sorted(df_co2['clasificacion'].unique().tolist())
    # Seleccionar 'Transporte por carretera' por defecto si existe
    default_selection = ["1.A.3.b. Transporte por carretera"] if "1.A.3.b. Transporte por carretera" in all_classifications else []
    if not default_selection and all_classifications: # Si no está, seleccionar la primera
        default_selection = [all_classifications[0]]

    selected_classifications = st.sidebar.multiselect(
        "Selecciona Clasificación(es):",
        options=all_classifications,
        default=default_selection,
        key='classification_multi_co2'
    )

    # Aplicar filtros
    df_co2_filtered = df_co2[
        (df_co2['ano'] >= selected_years[0]) &
        (df_co2['ano'] <= selected_years[1]) &
        (df_co2['clasificacion'].isin(selected_classifications))
    ].copy()

    # Filtrar comparison_df también por año
    if not comparison_df.empty:
        comparison_df_filtered = comparison_df[
            (comparison_df['ano'] >= selected_years[0]) &
            (comparison_df['ano'] <= selected_years[1])
        ].copy()
    else:
        comparison_df_filtered = pd.DataFrame() # Vacío si no se pudo crear

else:
    st.sidebar.warning("No hay datos de CO2 procesados para aplicar filtros.")
    df_co2_filtered = pd.DataFrame() # Asegurar que sean DataFrames vacíos
    comparison_df_filtered = pd.DataFrame()


# --- KPIs ---
st.header("Indicadores Clave (Periodo Seleccionado)")
if not df_co2_filtered.empty:
    total_emissions_filtered = df_co2_filtered['total_emisiones'].sum()
    avg_yearly_emissions = df_co2_filtered.groupby('ano')['total_emisiones'].sum().mean()

    col1, col2 = st.columns(2)
    col1.metric("Emisiones Totales (Selección)", f"{total_emissions_filtered:,.1f} ktCO2eq")
    col2.metric("Promedio Anual (Selección)", f"{avg_yearly_emissions:,.1f} ktCO2eq/año")

    # KPI específico para transporte carretera si está seleccionado
    if "1.A.3.b. Transporte por carretera" in selected_classifications:
        emissions_road_latest = df_co2_filtered[
            (df_co2_filtered['clasificacion'] == "1.A.3.b. Transporte por carretera") &
            (df_co2_filtered['ano'] == selected_years[1]) # Último año del rango
        ]['total_emisiones'].sum()
        st.metric(f"Emisiones Carretera ({selected_years[1]})", f"{emissions_road_latest:,.1f} ktCO2eq")

else:
    st.warning("No hay datos filtrados para mostrar KPIs.")


# --- VISUALIZACIONES ---
st.header("Análisis Visual de Emisiones")

if df_co2_filtered.empty:
    st.warning("Selecciona filtros con datos para ver las visualizaciones.")
else:
    # --- Gráfico 1: Emisiones Transporte por Carretera (Recreado) ---
    st.subheader("Evolución Emisiones: Transporte por Carretera")
    df_road_filtered = df_co2_filtered[df_co2_filtered['clasificacion'] == "1.A.3.b. Transporte por carretera"]
    if not df_road_filtered.empty:
        yearly_emissions_road = df_road_filtered.groupby('ano')['total_emisiones'].sum().reset_index()
        fig1 = px.line(yearly_emissions_road, x='ano', y='total_emisiones',
                       title="Emisiones Anuales (1.A.3.b. Transporte por carretera)",
                       labels={'ano': 'Año', 'total_emisiones': 'Total Emisiones (ktCO2eq)'},
                       markers=True, text='total_emisiones') # Añadir texto
        fig1.update_traces(textposition='top center', texttemplate='%{text:.0f}') # Formato texto
        fig1.update_layout(xaxis_title="Año", yaxis_title="Emisiones (ktCO2eq)")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Selecciona '1.A.3.b. Transporte por carretera' en los filtros para ver este gráfico.")


    # --- Gráfico 2: Comparativa Clasificaciones Seleccionadas (Recreado) ---
    st.subheader("Comparativa Emisiones por Clasificación Seleccionada")
    # Usar df_co2_filtered directamente, ya contiene las clasificaciones seleccionadas
    if len(selected_classifications) > 0:
         # Agrupar por si hay múltiples entradas por año/clasificación (aunque no debería según formato usual)
        df_plot2 = df_co2_filtered.groupby(['ano', 'clasificacion'])['total_emisiones'].sum().reset_index()

        fig2 = px.line(df_plot2, x='ano', y='total_emisiones', color='clasificacion',
                       title="Emisiones Anuales por Clasificación",
                       labels={'ano': 'Año', 'total_emisiones': 'Total Emisiones (ktCO2eq)', 'clasificacion': 'Clasificación'},
                       markers=True) # Poner marcadores
        # Añadir texto para años específicos (más complejo en Plotly Express)
        # Alternativa: Tooltips interactivos (vienen por defecto)
        # Si necesitas texto sí o sí como en matplotlib, se requiere plotly.graph_objects
        fig2.update_layout(xaxis_title="Año", yaxis_title="Emisiones (ktCO2eq)", legend_title_text='Clasificación')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Selecciona al menos una clasificación en los filtros para ver este gráfico.")


    # --- Gráfico 3: Comparación Carretera vs Nacional (Recreado) ---
    st.subheader("Comparación: Emisiones Carretera vs Total Nacional")
    if not comparison_df_filtered.empty:
        # Melt para facilitar graficar con color
        df_melted_comp = pd.melt(comparison_df_filtered,
                                 id_vars=['ano', 'porcentaje_carretera'],
                                 value_vars=['total_emisiones_carretera', 'total_emisiones_nacional'],
                                 var_name='tipo_emision', value_name='emisiones')
        df_melted_comp['tipo_emision'] = df_melted_comp['tipo_emision'].replace({
            'total_emisiones_carretera': 'Transporte Carretera',
            'total_emisiones_nacional': 'Total Nacional'
        })

        fig3 = px.line(df_melted_comp, x='ano', y='emisiones', color='tipo_emision',
                       title="Comparación Emisiones Transporte Carretera vs Total Nacional",
                       labels={'ano': 'Año', 'emisiones': 'Total Emisiones (ktCO2eq)', 'tipo_emision': 'Tipo'},
                       markers=True)

        # Añadir el porcentaje como anotación (similar a tu código matplotlib)
        # Necesitamos iterar sobre el dataframe *antes* de derretir para tener los valores en la misma fila
        annotations = []
        for i, row in comparison_df_filtered.iterrows():
            annotations.append(dict(
                x=row['ano'],
                y=row['total_emisiones_carretera'], # Posicionar sobre la línea de carretera
                text=f"{row['porcentaje_carretera']:.1f}%",
                showarrow=False,
                yshift=10 # Desplazar un poco hacia arriba
            ))
        fig3.update_layout(annotations=annotations)
        fig3.update_layout(xaxis_title="Año", yaxis_title="Emisiones (ktCO2eq)", legend_title_text='Tipo')
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No se pudieron cargar/procesar los datos necesarios para la comparación Nacional vs Carretera.")


    # --- Gráfico Adicional 1: Top N Emisiones por Clasificación (Bar Chart) ---
    st.subheader("Top Clasificaciones por Emisiones Totales (Periodo Seleccionado)")
    if len(selected_classifications) > 1: # Tiene sentido si hay varias para comparar
        top_n = 10 # Mostrar las 10 principales
        total_by_class = df_co2_filtered.groupby('clasificacion')['total_emisiones'].sum().reset_index()
        top_classes = total_by_class.nlargest(top_n, 'total_emisiones').sort_values('total_emisiones', ascending=True) # Ascendente para bar chart horizontal

        fig4 = px.bar(top_classes, y='clasificacion', x='total_emisiones',
                      orientation='h', # Barras horizontales
                      title=f"Top {top_n} Clasificaciones por Emisiones Totales ({selected_years[0]}-{selected_years[1]})",
                      labels={'clasificacion': 'Clasificación', 'total_emisiones': 'Emisiones Totales (ktCO2eq)'},
                      text='total_emisiones')
        fig4.update_traces(texttemplate='%{text:,.0f}') # Formato texto barra
        fig4.update_layout(yaxis_title="Clasificación", xaxis_title="Emisiones Totales (ktCO2eq)")
        st.plotly_chart(fig4, use_container_width=True)
    elif len(selected_classifications) == 1:
        st.info("Solo una clasificación seleccionada. El gráfico de barras comparativas no es aplicable.")


    # --- Gráfico Adicional 2: Composición en el Último Año (Pie Chart) ---
    st.subheader(f"Composición de Emisiones por Clasificación ({selected_years[1]})")
    df_last_year = df_co2_filtered[df_co2_filtered['ano'] == selected_years[1]]
    if not df_last_year.empty:
        # Agrupar por si acaso hay duplicados
        df_pie = df_last_year.groupby('clasificacion')['total_emisiones'].sum().reset_index()
        fig5 = px.pie(df_pie, names='clasificacion', values='total_emisiones',
                      title=f"Distribución de Emisiones por Clasificación en {selected_years[1]}",
                      hole=0.3) # Gráfico de dona
        fig5.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info(f"No hay datos disponibles para el año {selected_years[1]} con los filtros actuales.")


# --- DATOS TABULARES (Opcional) ---
st.header("Exploración de Datos (Emisiones)")
show_tables = st.expander("Mostrar Tablas de Datos Filtrados")
with show_tables:
    if st.checkbox("Mostrar tabla CO2 Filtrada", key='tabla_co2'):
        if not df_co2_filtered.empty:
            st.dataframe(df_co2_filtered)
        else: st.write("No hay datos CO2 filtrados.")

    if st.checkbox("Mostrar tabla Comparación Filtrada", key='tabla_comp'):
        if not comparison_df_filtered.empty:
            st.dataframe(comparison_df_filtered.style.format({
                'total_emisiones_carretera': '{:,.2f}',
                'total_emisiones_nacional': '{:,.2f}',
                'porcentaje_carretera': '{:.1f}%'
            }))
        else: st.write("No hay datos de comparación filtrados.")

    if st.checkbox("Mostrar información DataFrame CO2 procesado", key='info_co2'):
        if df_co2 is not None:
            buffer = StringIO()
            df_co2.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        else: st.write("DataFrame CO2 no disponible.")
