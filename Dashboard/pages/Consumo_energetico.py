# pages/2_⚡_Consumo_Energetico.py
# Se elimina la importación de pydataxm: from pydataxm import *
import datetime as dt # Se mantiene por si se usa en alguna parte, aunque no parece estrictamente necesario ahora
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import StringIO
import os # Necesario para construir la ruta del archivo

st.set_page_config(page_title="Análisis Consumo Energético", page_icon="⚡", layout="wide")
st.title("⚡ Análisis del Consumo Energético Nacional por Horas")
st.markdown("Visualización interactiva de los datos de consumo horario (basado en archivo CSV local).") # Texto actualizado

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
df_original_energy = load_data('Dashboard/df_original_energy.csv')

# Si la carga falló, detener la ejecución del script
if df_original_energy is None:
    st.stop()



# Verificar si la carga de datos fue exitosa antes de continuar
if df_original_energy.empty:
    st.error("No se pudieron cargar los datos desde el archivo CSV. El análisis no puede continuar.")
    st.stop() # Detiene la ejecución del script si no hay datos

# --------------------------------------------------------------------------
# EL RESTO DEL CÓDIGO PERMANECE EXACTAMENTE IGUAL, YA QUE OPERA
# SOBRE EL DATAFRAME 'df_original_energy', INDEPENDIENTEMENTE DE SU ORIGEN
# (SIEMPRE QUE LA ESTRUCTURA DEL CSV SEA IDÉNTICA A LA RESPUESTA DE LA API)
# --------------------------------------------------------------------------

st.sidebar.header("Preprocesamiento (Consumo):")
log_energy = []

# --- INICIO BLOQUE PREPROCESAMIENTO (SIN CAMBIOS) ---
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
        st.error("La columna 'date' es esencial y no se encontró en el CSV. Verifica el archivo.") # Error más específico
        st.stop() # Detener si no hay columna de fecha

    # 3. Identificar columnas horarias y convertir a numérico
    # IMPORTANTE: Asegúrate que tu CSV tenga columnas llamadas 'Values_Hour01', 'Values_Hour02', etc.
    # Si tienen otro nombre (ej. 'hora_01'), necesitarás ajustar este paso.
    # La limpieza de nombres anterior ya las debería convertir a 'values_hour01', etc.
    hour_cols = [col for col in df_energy.columns if col.startswith('values_hour')]
    if not hour_cols:
         log_energy.append("✗ No se encontraron columnas horarias ('values_hourXX'). Verifica los nombres en el CSV.")
         st.error("No se encontraron columnas con datos horarios ('values_hourXX'). Verifica el archivo CSV.")
         st.stop() # Detener si no hay datos horarios

    converted_numeric = 0
    for col in hour_cols:
        # Verificar si ya es numérico antes de intentar convertir
        if not pd.api.types.is_numeric_dtype(df_energy[col]):
             original_non_numeric = df_energy[col].apply(lambda x: isinstance(x, str)).sum() # Contar strings antes
             df_energy[col] = pd.to_numeric(df_energy[col], errors='coerce')
             if original_non_numeric > 0 or df_energy[col].isna().sum() > 0: # Si había strings o si la conversión generó NaNs
                 converted_numeric += 1
    if converted_numeric > 0:
        log_energy.append(f"✓ {len(hour_cols)} columnas horarias revisadas/convertidas a numérico (NaN si hubo errores).")
    else:
         log_energy.append(f"✓ {len(hour_cols)} columnas horarias ya eran numéricas o no necesitaron conversión.")


    # 4. Transformar de formato ancho a largo (Unpivot/Melt)
    id_vars = ['id', 'values_code', 'date'] # Columnas a mantener fijas
    # Asegurarse que las id_vars existen antes de usarlas
    id_vars_existentes = [col for col in id_vars if col in df_energy.columns]
    if len(id_vars_existentes) != len(id_vars):
        missing_ids = set(id_vars) - set(id_vars_existentes)
        log_energy.append(f"⚠ Advertencia: Faltan columnas ID ({', '.join(missing_ids)}) para el 'melt'. Se usarán las existentes.")
        st.warning(f"Faltan columnas ID ({', '.join(missing_ids)}) en el CSV. Pueden ser necesarias para análisis detallados.")


    df_long = pd.melt(df_energy,
                      id_vars=id_vars_existentes, # Usar solo las que existen
                      value_vars=hour_cols,
                      var_name='hora_col', # Nombre temporal
                      value_name='consumo')
    log_energy.append("✓ Datos transformados a formato largo (una fila por hora).")

    # 5. Extraer la hora numérica y crear timestamp completo
    df_long['hora'] = df_long['hora_col'].str.extract(r'(\d+)$').astype(int)
    df_long['timestamp'] = df_long['date'] + pd.to_timedelta(df_long['hora'] - 1, unit='h')
    df_long.drop(columns=['hora_col'], inplace=True)
    log_energy.append("✓ Columna 'timestamp' (fecha+hora) y 'hora' (numérica) creadas.")

    # 6. Extraer características temporales adicionales
    df_long['año'] = df_long['timestamp'].dt.year
    df_long['mes'] = df_long['timestamp'].dt.month # Numérico para ordenar/filtrar
    df_long['dia_semana'] = df_long['timestamp'].dt.day_name()
    df_long['dia_mes'] = df_long['timestamp'].dt.day
    # Mantener el mes numérico también si se usa después
    df_long['numero_mes'] = df_long['timestamp'].dt.month
    df_long['nombre_mes'] = df_long['timestamp'].dt.month_name()


    # Ordenar días de la semana
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    try: # Añadir try-except por si hay valores inesperados
        df_long['dia_semana'] = pd.Categorical(df_long['dia_semana'], categories=days_order, ordered=True)
    except ValueError as e:
        log_energy.append(f"⚠ Advertencia al ordenar días: {e}")
        st.warning(f"Problema al ordenar días de la semana: {e}. Verifica los valores en la columna 'dia_semana'.")
    log_energy.append("✓ Características temporales (año, mes, día semana, etc.) extraídas.")

    # 7. Calcular consumo diario para algunos análisis
    # Asegúrate de que hour_cols contiene las columnas correctas después de la limpieza
    if all(col in df_energy.columns for col in hour_cols): # Verificar que las columnas existen
        df_daily_sum = df_energy.set_index('date')[hour_cols].sum(axis=1).reset_index(name='consumo_diario_total')
        log_energy.append("✓ Calculado consumo diario total.")
    else:
        log_energy.append("✗ Error al calcular suma diaria: Faltan columnas horarias en df_energy.")
        st.error("Error interno: No se pudieron encontrar todas las columnas horarias para calcular el total diario.")
        df_daily_sum = pd.DataFrame(columns=['date', 'consumo_diario_total']) # DataFrame vacío para evitar errores abajo

# --- FIN BLOQUE PREPROCESAMIENTO ---


# Mostrar log en sidebar
for log in log_energy:
    st.sidebar.write(log)

# --- FILTROS EN SIDEBAR (SIN CAMBIOS) ---
st.sidebar.header("Filtros (Consumo)")

# Filtrar por rango de fechas (usando el 'date' original para la selección)
# Asegurarse de que df_energy['date'] existe y no está vacío antes de calcular min/max
if 'date' in df_energy.columns and not df_energy.empty:
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
        # Asegurar que las columnas 'date' existen antes de filtrar
        if 'date' in df_long.columns:
            mask_long = (df_long['date'] >= pd.to_datetime(start_date)) & (df_long['date'] <= pd.to_datetime(end_date))
            df_long_filtered = df_long[mask_long].copy()
        else:
            st.error("Falta la columna 'date' en los datos largos procesados.")
            df_long_filtered = pd.DataFrame() # Vacío para evitar error

        if 'date' in df_daily_sum.columns:
            mask_daily = (df_daily_sum['date'] >= pd.to_datetime(start_date)) & (df_daily_sum['date'] <= pd.to_datetime(end_date))
            df_daily_filtered = df_daily_sum[mask_daily].copy()
        else:
             st.error("Falta la columna 'date' en los datos diarios agregados.")
             df_daily_filtered = pd.DataFrame(columns=['date', 'consumo_diario_total']) # Vacío

    else:
        # Si solo se selecciona una fecha o ninguna, mostrar todo por defecto (o manejar como error)
        st.sidebar.warning("Selecciona un rango de fechas válido (inicio y fin). Mostrando todos los datos.")
        df_long_filtered = df_long.copy()
        df_daily_filtered = df_daily_sum.copy()

else:
    st.sidebar.error("No se pudo determinar el rango de fechas del archivo CSV. Verifica la columna 'date'.")
    # Crear dataframes vacíos para evitar errores posteriores en KPIs y gráficos
    df_long_filtered = pd.DataFrame()
    df_daily_filtered = pd.DataFrame(columns=['date', 'consumo_diario_total'])


# --- KPIs (SIN CAMBIOS, PERO CON CHEQUEOS ADICIONALES) ---
st.header("Resumen del Periodo Seleccionado")

# Añadir chequeos de que los dataframes no están vacíos y tienen las columnas necesarias
if not df_long_filtered.empty and 'consumo' in df_long_filtered.columns and \
   not df_daily_filtered.empty and 'consumo_diario_total' in df_daily_filtered.columns:

    # Asegurarse de que 'consumo' es numérico antes de sumar/promediar
    if pd.api.types.is_numeric_dtype(df_long_filtered['consumo']):
        total_consumo = df_long_filtered['consumo'].sum()
        # Hora pico promedio (hora del día con mayor consumo promedio)
        if 'hora' in df_long_filtered.columns:
            avg_hourly = df_long_filtered.groupby('hora')['consumo'].mean()
            if not avg_hourly.empty:
                peak_hour = avg_hourly.idxmax()
                peak_consumption = avg_hourly.max()
            else:
                peak_hour = "N/A"
                peak_consumption = 0
        else:
            peak_hour = "N/A"
            peak_consumption = 0
    else:
        st.warning("La columna 'consumo' no es numérica. No se pueden calcular KPIs.")
        total_consumo = 0
        peak_hour = "N/A"
        peak_consumption = 0

    # Asegurarse de que 'consumo_diario_total' es numérico
    if pd.api.types.is_numeric_dtype(df_daily_filtered['consumo_diario_total']):
        avg_daily_consumo = df_daily_filtered['consumo_diario_total'].mean()
    else:
         st.warning("La columna 'consumo_diario_total' no es numérica. No se puede calcular el promedio diario.")
         avg_daily_consumo = 0


    col1, col2, col3 = st.columns(3)
    # Asumiendo que la unidad original es Wh. Ajusta si es kWh, MWh, etc.
    col1.metric("Consumo Total (GWh aprox.)", f"{total_consumo / 1e9:.2f} GWh")
    col2.metric("Promedio Diario (MWh aprox.)", f"{avg_daily_consumo / 1e6:.2f} MWh")
    col3.metric(f"Hora Pico Promedio", f"Hora {peak_hour} ({peak_consumption / 1e6:.2f} MWh)")
else:
    st.warning("No hay datos disponibles para calcular KPIs en el rango de fechas seleccionado o faltan columnas clave ('consumo', 'consumo_diario_total', 'hora').")


# --- VISUALIZACIONES (SIN CAMBIOS EN LA LÓGICA, PERO VERIFICAR COLUMNAS) ---
st.header("Análisis Visual del Consumo")

if df_long_filtered.empty or df_daily_filtered.empty:
    st.warning("Selecciona un rango de fechas con datos (o revisa el archivo CSV si está vacío) para ver las visualizaciones.")
else:
    # Verificar la existencia de columnas antes de cada gráfico

    # 1. Serie Temporal General (Consumo Diario Total)
    st.subheader("Evolución del Consumo Diario Total")
    if 'date' in df_daily_filtered.columns and 'consumo_diario_total' in df_daily_filtered.columns:
        fig_daily_trend = px.line(df_daily_filtered, x='date', y='consumo_diario_total',
                                  title="Consumo Energético Diario Total",
                                  labels={'date': 'Fecha', 'consumo_diario_total': 'Consumo Total Diario (Wh)'}) # Ajustar unidad si es diferente
        fig_daily_trend.update_layout(xaxis_title="Fecha", yaxis_title="Consumo Diario (Wh)")
        st.plotly_chart(fig_daily_trend, use_container_width=True)
    else:
        st.warning("Faltan columnas ('date' o 'consumo_diario_total') para el gráfico de tendencia diaria.")

    # Opción: Agregación Mensual/Anual si el rango es grande
    if len(df_daily_filtered) > 90 and 'timestamp' in df_long_filtered.columns and 'consumo' in df_long_filtered.columns:
        st.subheader("Tendencias Agregadas")
        try: # Añadir try-except por si falla la agregación
            df_long_filtered['año_mes'] = df_long_filtered['timestamp'].dt.to_period('M').astype(str)
            df_monthly = df_long_filtered.groupby('año_mes')['consumo'].sum().reset_index()
            df_monthly['año_mes'] = pd.to_datetime(df_monthly['año_mes']) # Convertir a timestamp para plot

            fig_monthly_trend = px.line(df_monthly, x='año_mes', y='consumo',
                                         title="Consumo Energético Mensual Total",
                                         labels={'año_mes': 'Mes', 'consumo': 'Consumo Total Mensual (Wh)'})
            fig_monthly_trend.update_layout(xaxis_title="Mes", yaxis_title="Consumo Mensual (Wh)")
            st.plotly_chart(fig_monthly_trend, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar gráfico de tendencia mensual: {e}")
    elif len(df_daily_filtered) <= 90:
        pass # No mostrar si el rango es corto
    else:
         st.warning("Faltan columnas ('timestamp' o 'consumo') para el gráfico de tendencia mensual.")


    # 2. Patrón Diario Promedio (Curva de Carga)
    st.subheader("Patrón de Consumo Diario Promedio (Curva de Carga)")
    if 'hora' in df_long_filtered.columns and 'consumo' in df_long_filtered.columns:
        avg_hourly_consumption = df_long_filtered.groupby('hora')['consumo'].mean().reset_index()
        fig_daily_pattern = px.line(avg_hourly_consumption, x='hora', y='consumo',
                                    title="Consumo Promedio por Hora del Día",
                                    labels={'hora': 'Hora del Día (1-24)', 'consumo': 'Consumo Promedio (Wh)'},
                                    markers=True)
        fig_daily_pattern.update_layout(xaxis=dict(tickmode='linear', dtick=1), xaxis_title="Hora del Día", yaxis_title="Consumo Promedio (Wh)")
        st.plotly_chart(fig_daily_pattern, use_container_width=True)
    else:
        st.warning("Faltan columnas ('hora' o 'consumo') para el gráfico de patrón diario.")


    # 3. Comparativa Diario: Fin de Semana vs. Entre Semana
    st.subheader("Comparativa: Consumo Horario Promedio (Entre Semana vs. Fin de Semana)")
    if 'dia_semana' in df_long_filtered.columns and 'hora' in df_long_filtered.columns and 'consumo' in df_long_filtered.columns:
        # Asegurarse de que 'dia_semana' es categórica o string para .isin()
        if pd.api.types.is_categorical_dtype(df_long_filtered['dia_semana']) or pd.api.types.is_string_dtype(df_long_filtered['dia_semana']):
             df_long_filtered['tipo_dia'] = np.where(df_long_filtered['dia_semana'].astype(str).isin(['Saturday', 'Sunday']), 'Fin de Semana', 'Entre Semana')
             avg_hourly_comp = df_long_filtered.groupby(['hora', 'tipo_dia'])['consumo'].mean().reset_index()
             fig_daily_comp = px.line(avg_hourly_comp, x='hora', y='consumo', color='tipo_dia',
                                     title="Consumo Horario Promedio: Entre Semana vs. Fin de Semana",
                                     labels={'hora': 'Hora del Día', 'consumo': 'Consumo Promedio (Wh)', 'tipo_dia': 'Tipo de Día'},
                                     markers=True)
             fig_daily_comp.update_layout(xaxis=dict(tickmode='linear', dtick=1), xaxis_title="Hora del Día", yaxis_title="Consumo Promedio (Wh)")
             st.plotly_chart(fig_daily_comp, use_container_width=True)
        else:
            st.warning("La columna 'dia_semana' no tiene el formato esperado para la comparación.")

    else:
        st.warning("Faltan columnas ('dia_semana', 'hora' o 'consumo') para el gráfico comparativo diario.")


    #4. Distribución Semanal y Mensual (Box Plots)
    st.subheader("Distribución del Consumo Diario")

    # --- Modificación en sección 4 ---
    # El código original ya tenía la lógica correcta para añadir las columnas
    # a df_daily_filtered. Solo añadimos chequeos de existencia de columnas.

    # >>> INICIO: Añadir columnas de fecha a df_daily_filtered (CHEQUEOS AÑADIDOS) <<<
    if not df_daily_filtered.empty and 'date' in df_daily_filtered.columns:
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
             st.warning(f"No se pudo ordenar día de la semana categóricamente: {ve}")

        # Ordenar meses para el gráfico (usando numero_mes)
        df_daily_filtered = df_daily_filtered.sort_values('numero_mes')
    elif df_daily_filtered.empty:
         pass # El mensaje de advertencia general ya se mostró
    else:
         st.warning("Falta la columna 'date' en los datos diarios para extraer características semanales/mensuales.")


    # >>> FIN: Añadir columnas de fecha a df_daily_filtered <<<

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        # Re-chequear que las columnas necesarias existen *después* de intentar crearlas
        if not df_daily_filtered.empty and 'dia_semana' in df_daily_filtered.columns and 'consumo_diario_total' in df_daily_filtered.columns:
            fig_weekly_box = px.box(df_daily_filtered, x='dia_semana', y='consumo_diario_total',
                                    title="Distribución del Consumo Diario por Día de la Semana",
                                    labels={'dia_semana': 'Día de la Semana', 'consumo_diario_total': 'Consumo Diario Total (Wh)'})
            fig_weekly_box.update_layout(xaxis_title="Día de la Semana", yaxis_title="Consumo Diario (Wh)")
            st.plotly_chart(fig_weekly_box, use_container_width=True)
        elif df_daily_filtered.empty:
             st.info("No hay datos diarios para el gráfico semanal.")
        else:
             st.error("Faltan columnas ('dia_semana' o 'consumo_diario_total') para el gráfico semanal.")


    with col_box2:
        if not df_daily_filtered.empty and 'nombre_mes' in df_daily_filtered.columns and 'consumo_diario_total' in df_daily_filtered.columns:
            fig_monthly_box = px.box(df_daily_filtered, x='nombre_mes', y='consumo_diario_total',
                                     title="Distribución del Consumo Diario por Mes",
                                     labels={'nombre_mes': 'Mes', 'consumo_diario_total': 'Consumo Diario Total (Wh)'})
            fig_monthly_box.update_layout(xaxis_title="Mes", yaxis_title="Consumo Diario (Wh)")
            st.plotly_chart(fig_monthly_box, use_container_width=True)
        elif df_daily_filtered.empty:
             st.info("No hay datos diarios para el gráfico mensual.")
        else:
             st.error("Faltan columnas ('nombre_mes' o 'consumo_diario_total') para el gráfico mensual.")
    # --- Fin modificación sección 4 ---


    # 5. Heatmap de Consumo
    st.subheader("Heatmap de Consumo Promedio (Hora vs. Día de la Semana)")
    if 'hora' in df_long_filtered.columns and 'dia_semana' in df_long_filtered.columns and 'consumo' in df_long_filtered.columns and pd.api.types.is_categorical_dtype(df_long_filtered['dia_semana']):
        heatmap_data = df_long_filtered.groupby(['hora', 'dia_semana'])['consumo'].mean().reset_index()
        # Pivotar para formato heatmap
        try: # Pivot puede fallar si hay combinaciones duplicadas
            heatmap_pivot = heatmap_data.pivot(index='dia_semana', columns='hora', values='consumo')
            # Reordenar filas por día de la semana (asegurarse que dia_semana es categórico ordenado)
            heatmap_pivot = heatmap_pivot.reindex(days_order)

            fig_heatmap = px.imshow(heatmap_pivot,
                                    title="Consumo Promedio por Hora y Día de la Semana",
                                    labels=dict(x="Hora del Día", y="Día de la Semana", color="Consumo Promedio (Wh)"),
                                    x=heatmap_pivot.columns,
                                    y=heatmap_pivot.index,
                                    aspect="auto",
                                    color_continuous_scale="viridis")
            fig_heatmap.update_xaxes(side="bottom", tickmode='linear', dtick=2)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar el heatmap: {e}. Verifica si hay datos duplicados por hora/día.")

    else:
        missing_cols_heatmap = [col for col in ['hora', 'dia_semana', 'consumo'] if col not in df_long_filtered.columns]
        st.warning(f"Faltan columnas ({', '.join(missing_cols_heatmap)}) o 'dia_semana' no está correctamente formateado para el heatmap.")


# --- DATOS TABULARES (SIN CAMBIOS) ---
st.header("Exploración de Datos (Consumo)")
if st.checkbox("Mostrar tabla de datos horarios procesados (filtrados)", key='tabla_consumo_long'):
    st.dataframe(df_long_filtered)

if st.checkbox("Mostrar tabla de consumo diario total (filtrados)", key='tabla_consumo_daily'):
    st.dataframe(df_daily_filtered)

if st.checkbox("Mostrar información del DataFrame horario procesado", key='info_consumo'):
     if not df_long_filtered.empty:
         buffer = StringIO()
         df_long_filtered.info(buf=buffer)
         s = buffer.getvalue()
         st.text(s)
     else:
         st.info("El DataFrame horario filtrado está vacío.")
