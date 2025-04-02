# pages/2_⚡_Consumo_Energetico.py
from pydataxm import *
import datetime as dt
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import StringIO
st.set_page_config(page_title="Análisis Consumo Energético", page_icon="⚡", layout="wide") # 
st.title("⚡ Análisis del Consumo Energético Nacional por Horas")
st.markdown("Visualización interactiva de los datos de consumo horario (2010-2023) obtenidos vía API XM.")

#objetoAPI = pydataxm.ReadDB()

# --- FUNCIÓN PARA OBTENER Y CACHEAR DATOS DE LA API ---
@st.cache_data(ttl=3600) # Cachear por 1 hora (3600 segundos) para no sobrecargar la API
def fetch_consumption_data(start_date, end_date):
    """Obtiene los datos de demanda real desde la API de pydataxm."""
    try:
        st.info(f"Consultando API XM para DemaReal desde {start_date} hasta {end_date}...")
        api = pydataxm.ReadDB()
        df = api.request_data(
            "DemaReal",
            "Sistema",
            start_date,
            end_date
        )
        st.success("Datos obtenidos correctamente de la API.")
        if df is None or df.empty:
            st.warning("La consulta a la API no devolvió datos para el periodo seleccionado.")
            return pd.DataFrame() # Devuelve DataFrame vacío para evitar errores posteriores
        return df
    except Exception as e:
        st.error(f"Error al conectar o solicitar datos a la API de XM: {e}")
        return pd.DataFrame() # Devuelve DataFrame vacío en caso de error


# --- OBTENER DATOS ---
start_date_api = dt.date(2010, 1, 1)
end_date_api = dt.date(2023, 12, 31)
df_original_energy = fetch_consumption_data(start_date_api, end_date_api)

# Verificar si la carga de datos fue exitosa antes de continuar
if df_original_energy.empty:
    st.error("No se pudieron obtener los datos de consumo. El análisis no puede continuar.")
    st.stop() # Detiene la ejecución del script si no hay datos




st.sidebar.header("Preprocesamiento (Consumo):")
log_energy = []

with st.spinner("Procesando datos de consumo... Esto puede tardar un momento."):
    df_energy = df_original_energy.copy()

    # 1. Limpiar nombres de columnas
    original_cols = df_energy.columns.tolist()
    df_energy.columns = df_energy.columns.str.lower().str.replace(' ', '_')
    new_cols = df_energy.columns.tolist()
    renamed_count = sum(1 for o, n in zip(original_cols, new_cols) if o != n)
    log_energy.append(f"✓ {renamed_count} columnas renombradas (minúsculas, guion bajo).")

    # 2. Convertir 'date' a datetime
    if 'date' in df_energy.columns:
        df_energy['date'] = pd.to_datetime(df_energy['date'], errors='coerce')
        na_dates = df_energy['date'].isna().sum()
        if na_dates > 0:
            log_energy.append(f"✓ 'date' convertida a datetime ({na_dates} fechas inválidas -> NaT).")
            df_energy.dropna(subset=['date'], inplace=True) # Eliminar filas sin fecha válida
        else:
            log_energy.append("✓ 'date' convertida a datetime.")
    else:
        log_energy.append("✗ Columna 'date' no encontrada.")
        st.stop() # Detener si no hay columna de fecha

    # 3. Identificar columnas horarias y convertir a numérico
    hour_cols = [col for col in df_energy.columns if col.startswith('values_hour')]
    if not hour_cols:
         log_energy.append("✗ No se encontraron columnas 'values_hourXX'.")
         st.stop()

    converted_numeric = 0
    for col in hour_cols:
        if df_energy[col].dtype != 'float64' and df_energy[col].dtype != 'int64':
             original_non_numeric = df_energy[col].apply(type).eq(str).sum() # Contar strings antes
             df_energy[col] = pd.to_numeric(df_energy[col], errors='coerce')
             if original_non_numeric > 0 or df_energy[col].isna().sum() > 0:
                 converted_numeric += 1
    if converted_numeric > 0:
        log_energy.append(f"✓ {len(hour_cols)} columnas horarias convertidas a numérico (NaN si hubo errores).")
    else:
         log_energy.append(f"✓ {len(hour_cols)} columnas horarias ya eran numéricas.")


    # 4. Transformar de formato ancho a largo (Unpivot/Melt)
    id_vars = ['id', 'values_code', 'date'] # Columnas a mantener fijas
    # Asegurarse que las id_vars existen antes de usarlas
    id_vars = [col for col in id_vars if col in df_energy.columns]

    df_long = pd.melt(df_energy,
                      id_vars=id_vars,
                      value_vars=hour_cols,
                      var_name='hora_col', # Nombre temporal para la columna que contiene 'values_hourXX'
                      value_name='consumo') # Nombre de la columna con los valores de consumo
    log_energy.append("✓ Datos transformados a formato largo (una fila por hora).")

    # 5. Extraer la hora numérica y crear timestamp completo
    # Extraer el número de la hora de 'values_hourXX' (ej: 'values_hour01' -> 1)
    df_long['hora'] = df_long['hora_col'].str.extract(r'(\d+)$').astype(int)
    # Ajustar hora 24 a 0 para cálculos correctos de timedelta y asignar al día correcto
    # (La hora 24 de un día es la hora 0 del día siguiente, pero aquí representa el *final* del intervalo 23-24 del día original)
    # Para simplificar el análisis *diario*, mantenemos la hora 24 como 24 o la mapeamos a 0 según la necesidad del gráfico.
    # Para el timestamp, trataremos la hora 24 como la última hora del día dado.
    # Creamos el timestamp sumando la hora (menos 1, ya que las horas van 1-24) a la fecha.
    # Hora 1 -> +0 horas, Hora 24 -> +23 horas
    df_long['timestamp'] = df_long['date'] + pd.to_timedelta(df_long['hora'] - 1, unit='h')
    df_long.drop(columns=['hora_col'], inplace=True) # Eliminar columna temporal
    log_energy.append("✓ Columna 'timestamp' (fecha+hora) y 'hora' (numérica) creadas.")

    # 6. Extraer características temporales adicionales
    df_long['año'] = df_long['timestamp'].dt.year
    df_long['mes'] = df_long['timestamp'].dt.month
    df_long['dia_semana'] = df_long['timestamp'].dt.day_name()
    df_long['dia_mes'] = df_long['timestamp'].dt.day
    df_long['numero_mes'] = df_long['timestamp'].dt.month # Numérico para ordenar
    df_long['nombre_mes'] = df_long['timestamp'].dt.month_name()

    # Ordenar días de la semana
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_long['dia_semana'] = pd.Categorical(df_long['dia_semana'], categories=days_order, ordered=True)
    log_energy.append("✓ Características temporales (año, mes, día semana, etc.) extraídas.")

    # 7. Calcular consumo diario para algunos análisis
    df_daily_sum = df_energy.set_index('date')[hour_cols].sum(axis=1).reset_index(name='consumo_diario_total')
    log_energy.append("✓ Calculado consumo diario total.")

# Mostrar log en sidebar
for log in log_energy:
    st.sidebar.write(log)

# --- FILTROS EN SIDEBAR ---
st.sidebar.header("Filtros (Consumo)")

# Filtrar por rango de fechas (usando el 'date' original para la selección)
min_date = df_energy['date'].min().date()
max_date = df_energy['date'].max().date()

selected_date_range = st.sidebar.date_input(
    "Selecciona Rango de Fechas:",
    value=(min_date, max_date), # Default: todo el rango
    min_value=min_date,
    max_value=max_date,
    key='date_range_consumo'
)

# Aplicar filtro de fecha (manejar el caso de una sola fecha seleccionada)
if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    # Filtrar el dataframe *largo* y el *diario*
    mask_long = (df_long['date'] >= pd.to_datetime(start_date)) & (df_long['date'] <= pd.to_datetime(end_date))
    mask_daily = (df_daily_sum['date'] >= pd.to_datetime(start_date)) & (df_daily_sum['date'] <= pd.to_datetime(end_date))
    df_long_filtered = df_long[mask_long].copy()
    df_daily_filtered = df_daily_sum[mask_daily].copy()
else:
    # Si solo se selecciona una fecha o ninguna, mostrar todo por defecto (o manejar como error)
    st.sidebar.warning("Selecciona un rango de fechas válido (inicio y fin). Mostrando todos los datos.")
    df_long_filtered = df_long.copy()
    df_daily_filtered = df_daily_sum.copy()


# --- KPIs ---
st.header("Resumen del Periodo Seleccionado")

if not df_long_filtered.empty:
    total_consumo = df_long_filtered['consumo'].sum()
    avg_daily_consumo = df_daily_filtered['consumo_diario_total'].mean()
    # Hora pico promedio (hora del día con mayor consumo promedio)
    avg_hourly = df_long_filtered.groupby('hora')['consumo'].mean()
    peak_hour = avg_hourly.idxmax()
    peak_consumption = avg_hourly.max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Consumo Total (GWh aprox.)", f"{total_consumo / 1e9:.2f} GWh") # Asumiendo Wh -> GWh
    col2.metric("Promedio Diario (MWh aprox.)", f"{avg_daily_consumo / 1e6:.2f} MWh") # Asumiendo Wh -> MWh
    col3.metric(f"Hora Pico Promedio", f"Hora {peak_hour} ({peak_consumption / 1e6:.2f} MWh)")
else:
    st.warning("No hay datos disponibles para el rango de fechas seleccionado.")


# --- VISUALIZACIONES ---
st.header("Análisis Visual del Consumo")

if df_long_filtered.empty:
    st.warning("Selecciona un rango de fechas con datos para ver las visualizaciones.")
else:
    # 1. Serie Temporal General (Consumo Diario Total)
    st.subheader("Evolución del Consumo Diario Total")
    fig_daily_trend = px.line(df_daily_filtered, x='date', y='consumo_diario_total',
                              title="Consumo Energético Diario Total",
                              labels={'date': 'Fecha', 'consumo_diario_total': 'Consumo Total Diario (Wh)'}) # Ajustar unidad si es diferente
    fig_daily_trend.update_layout(xaxis_title="Fecha", yaxis_title="Consumo Diario (Wh)")
    st.plotly_chart(fig_daily_trend, use_container_width=True)

    # Opción: Agregación Mensual/Anual si el rango es grande
    if len(df_daily_filtered) > 90: # Mostrar agregados si hay más de ~3 meses de datos
        st.subheader("Tendencias Agregadas")
        df_long_filtered['año_mes'] = df_long_filtered['timestamp'].dt.to_period('M').astype(str)
        df_monthly = df_long_filtered.groupby('año_mes')['consumo'].sum().reset_index()
        df_monthly['año_mes'] = pd.to_datetime(df_monthly['año_mes']) # Convertir a timestamp para plot

        fig_monthly_trend = px.line(df_monthly, x='año_mes', y='consumo',
                                     title="Consumo Energético Mensual Total",
                                     labels={'año_mes': 'Mes', 'consumo': 'Consumo Total Mensual (Wh)'})
        fig_monthly_trend.update_layout(xaxis_title="Mes", yaxis_title="Consumo Mensual (Wh)")
        st.plotly_chart(fig_monthly_trend, use_container_width=True)


    # 2. Patrón Diario Promedio (Curva de Carga)
    st.subheader("Patrón de Consumo Diario Promedio (Curva de Carga)")
    avg_hourly_consumption = df_long_filtered.groupby('hora')['consumo'].mean().reset_index()
    fig_daily_pattern = px.line(avg_hourly_consumption, x='hora', y='consumo',
                                title="Consumo Promedio por Hora del Día",
                                labels={'hora': 'Hora del Día (1-24)', 'consumo': 'Consumo Promedio (Wh)'},
                                markers=True)
    fig_daily_pattern.update_layout(xaxis=dict(tickmode='linear', dtick=1), xaxis_title="Hora del Día", yaxis_title="Consumo Promedio (Wh)")
    st.plotly_chart(fig_daily_pattern, use_container_width=True)

    # 3. Comparativa Diario: Fin de Semana vs. Entre Semana
    st.subheader("Comparativa: Consumo Horario Promedio (Entre Semana vs. Fin de Semana)")
    df_long_filtered['tipo_dia'] = np.where(df_long_filtered['dia_semana'].isin(['Saturday', 'Sunday']), 'Fin de Semana', 'Entre Semana')
    avg_hourly_comp = df_long_filtered.groupby(['hora', 'tipo_dia'])['consumo'].mean().reset_index()
    fig_daily_comp = px.line(avg_hourly_comp, x='hora', y='consumo', color='tipo_dia',
                             title="Consumo Horario Promedio: Entre Semana vs. Fin de Semana",
                             labels={'hora': 'Hora del Día', 'consumo': 'Consumo Promedio (Wh)', 'tipo_dia': 'Tipo de Día'},
                             markers=True)
    fig_daily_comp.update_layout(xaxis=dict(tickmode='linear', dtick=1), xaxis_title="Hora del Día", yaxis_title="Consumo Promedio (Wh)")
    st.plotly_chart(fig_daily_comp, use_container_width=True)


    #4. Distribución Semanal y Mensual (Box Plots)
    st.subheader("Distribución del Consumo Diario")

    # >>> INICIO: Añadir columnas de fecha a df_daily_filtered <<<
    if not df_daily_filtered.empty:
        #print (df_daily_filtered.info())
        # Asegurarse que la columna de fecha es datetime
        df_daily_filtered['date'] = pd.to_datetime(df_daily_filtered['date'])

        # Extraer características de fecha directamente en df_daily_filtered
        df_daily_filtered['dia_semana'] = df_daily_filtered['date'].dt.day_name()
        df_daily_filtered['nombre_mes'] = df_daily_filtered['date'].dt.month_name()
        df_daily_filtered['numero_mes'] = df_daily_filtered['date'].dt.month

        # Ordenar días de la semana (igual que antes)
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        try:
             df_daily_filtered['dia_semana'] = pd.Categorical(df_daily_filtered['dia_semana'], categories=days_order, ordered=True)
        except ValueError as ve:
             st.warning(f"No se pudo ordenar día de la semana categóricamente: {ve}") # Manejar si hay nombres inesperados

        # Ordenar meses para el gráfico (usando numero_mes)
        df_daily_filtered = df_daily_filtered.sort_values('numero_mes')

    # >>> FIN: Añadir columnas de fecha a df_daily_filtered <<<

    # Ahora podemos crear los boxplots usando df_daily_filtered directamente
    col_box1, col_box2 = st.columns(2)
    with col_box1:
        if not df_daily_filtered.empty and 'dia_semana' in df_daily_filtered.columns:
            fig_weekly_box = px.box(df_daily_filtered, x='dia_semana', y='consumo_diario_total',
                                    # No es necesario category_orders si ya es Categórico Ordenado
                                    title="Distribución del Consumo Diario por Día de la Semana",
                                    labels={'dia_semana': 'Día de la Semana', 'consumo_diario_total': 'Consumo Diario Total (Wh)'})
            fig_weekly_box.update_layout(xaxis_title="Día de la Semana", yaxis_title="Consumo Diario (Wh)")
            st.plotly_chart(fig_weekly_box, use_container_width=True)
        elif df_daily_filtered.empty:
             st.info("No hay datos diarios para el gráfico semanal.")
        else:
             st.error("Falta la columna 'dia_semana' para el gráfico semanal.")


    with col_box2:
        if not df_daily_filtered.empty and 'nombre_mes' in df_daily_filtered.columns:
             # No necesitamos el merge, las columnas ya están
             # Ya está ordenado por numero_mes
            fig_monthly_box = px.box(df_daily_filtered, x='nombre_mes', y='consumo_diario_total',
                                     title="Distribución del Consumo Diario por Mes",
                                     labels={'nombre_mes': 'Mes', 'consumo_diario_total': 'Consumo Diario Total (Wh)'})
            fig_monthly_box.update_layout(xaxis_title="Mes", yaxis_title="Consumo Diario (Wh)")
            st.plotly_chart(fig_monthly_box, use_container_width=True)
        elif df_daily_filtered.empty:
             st.info("No hay datos diarios para el gráfico mensual.")
        else:
             st.error("Falta la columna 'nombre_mes' para el gráfico mensual.")

    # 5. Heatmap de Consumo
    st.subheader("Heatmap de Consumo Promedio (Hora vs. Día de la Semana)")
    heatmap_data = df_long_filtered.groupby(['hora', 'dia_semana'])['consumo'].mean().reset_index()
    # Pivotar para formato heatmap
    heatmap_pivot = heatmap_data.pivot(index='dia_semana', columns='hora', values='consumo')
    # Reordenar filas por día de la semana
    heatmap_pivot = heatmap_pivot.reindex(days_order)

    fig_heatmap = px.imshow(heatmap_pivot,
                            title="Consumo Promedio por Hora y Día de la Semana",
                            labels=dict(x="Hora del Día", y="Día de la Semana", color="Consumo Promedio (Wh)"),
                            x=heatmap_pivot.columns, # Horas 1-24
                            y=heatmap_pivot.index,   # Días ordenados
                            aspect="auto", # Ajustar aspecto
                            color_continuous_scale="viridis") # Escala de color
    fig_heatmap.update_xaxes(side="bottom", tickmode='linear', dtick=2)
    st.plotly_chart(fig_heatmap, use_container_width=True)


# --- DATOS TABULARES (Opcional) ---
st.header("Exploración de Datos (Consumo)")
if st.checkbox("Mostrar tabla de datos horarios procesados (filtrados)", key='tabla_consumo_long'):
    st.dataframe(df_long_filtered)

if st.checkbox("Mostrar tabla de consumo diario total (filtrados)", key='tabla_consumo_daily'):
    st.dataframe(df_daily_filtered)

if st.checkbox("Mostrar información del DataFrame horario procesado", key='info_consumo'):
     buffer = StringIO()
     df_long_filtered.info(buf=buffer)
     s = buffer.getvalue()
     st.text(s)