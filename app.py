import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import librosa
import numpy as np
import pandas as pd
import io
import base64


# ------------- Importacion de funciones personalizadas ----------------
from audio_processing.librosa_utils import (
    cargar_audio_desde_bytes,
    calcular_duracion,
    #graficar_espectrograma_librosa,
    calcular_zcr,
)

from audio_processing.praat_utils import (
    cargar_sonido_praat,
    #graficar_espectrograma_praat,
    obtener_frecuencia_fundamental,
    calcular_jitter_shimmer,
)
from audio_processing.yamnet_filter import (
    filtrar_llanto_yamnet,
)

from audio_processing.cry_detection import (
    detectar_llanto,
    detectar_segmentos_llanto,
)

from utils.energia import graficar_energia
from utils.tiempo import detectar_tiempos_llanto

from utils.visualizacion import (
    graficar_espectrograma_praat_interactivo,
    graficar_curva_f0,
    graficar_zcr_plotly,
)
#-----------------------------------------------------------------------------

st.set_page_config(page_title="Análisis de Llanto Infantil", layout="wide")

# -----------------------------Menú lateral ---------------------------------
st.sidebar.title("🔍 Opciones de Análisis")

mostrar_todos = st.sidebar.checkbox("✅ Mostrar todos")
mostrar_info_general = st.sidebar.checkbox("📄 Información General", value=mostrar_todos, disabled=mostrar_todos)
mostrar_espectrograma = st.sidebar.checkbox("🎛️ Espectrograma", value=mostrar_todos, disabled=mostrar_todos)
mostrar_f0 = st.sidebar.checkbox("📈 Frecuencia Fundamental", value=mostrar_todos, disabled=mostrar_todos)
mostrar_jitter_shimmer = st.sidebar.checkbox("📉 Jitter y Shimmer", value=mostrar_todos, disabled=mostrar_todos)
mostrar_zcr = st.sidebar.checkbox("📊 Zero-Crossing Rate", value=mostrar_todos, disabled=mostrar_todos)
mostrar_llanto = st.sidebar.checkbox("🎚️ Filtrado con YAMNet", value=mostrar_todos, disabled=mostrar_todos)

st.title("👶 Análisis de Llanto Infantil")
# Cargar el archivo .wav
archivo_audio = st.file_uploader("", type=["wav"])

if archivo_audio is not None:
    audio_bytes = archivo_audio.read()
    y, sr = cargar_audio_desde_bytes(audio_bytes)
    duracion = calcular_duracion(y, sr)

    # Mostrar reproductor siempre
    st.audio(archivo_audio, format="audio/wav")

    if mostrar_info_general:
        st.markdown(
            "<h4 style='text-align: center;'>📄 Información General</h4>",
            unsafe_allow_html=True
        )
        st.write(f"🕒 **Duración:** {duracion:.2f} segundos")
        minutos = int(duracion // 60)
        segundos = int(duracion % 60)
        st.write(f"⏱️ **Duración (mm:ss):** {minutos:02d}:{segundos:02d}")
        st.write(f"🎧 **Frecuencia de muestreo:** {sr} Hz")
        st.write(f"📊 **Número de muestras:** {len(y)}")

        canales = 1 if len(y.shape) == 1 else y.shape[0]
        tipo_audio = "Mono" if canales == 1 else "Estéreo"
        st.write(f"🔈 **Canales:** {canales} ({tipo_audio})")

        st.write(f"📈 **Amplitud máxima:** {np.max(np.abs(y)):.3f}")
        rms = np.sqrt(np.mean(np.square(y)))
        st.write(f"🔋 **Energía promedio (RMS):** {rms:.4f}")
        st.write(f"⚖️ **Offset DC (valor medio):** {np.mean(y):.5f}")

    if mostrar_espectrograma:
        st.markdown(
            "<h4 style='text-align: center;'>🎛️ Espectrograma</h4>",
            unsafe_allow_html=True
        )
        with st.expander("ℹ️ "):
            st.write(""" Un espectrograma es una representación visual de cómo varían las frecuencias de una 
                señal de audio a lo largo del tiempo. En el eje horizontal se muestra el tiempo, en el vertical
                 la frecuencia, y la intensidad de color representa la energía (amplitud) de cada frecuencia en 
                 un momento dado.
                 \nEl espectrograma permite observar patrones acústicos específicos del llanto, como la presencia de formantes, 
                 ruidos, interrupciones o picos de energía. Estas características pueden estar relacionadas 
                 con estados fisiológicos o emocionales del bebé y son útiles para distinguir entre llantos 
                 normales y aquellos que podrían indicar un problema médico.
                 \nAl interactuar con el espectrograma, verás tres valores en el cursor:
                 \n- **X**: Tiempo (segundos) -  Indica en qué momento del audio estás posicionado.
                 \n- **Y**: Frecuencia (Hz) - Muestra la frecuencia correspondiente a la posición vertical del cursor.
                 \n- **Z**: Intensidad (dB) - Representa la energía o amplitud de la señal en ese punto, expresada en decibeles.
                 \nEstos valores permiten analizar con precisión las características acústicas del llanto en cada instante del tiempo.
                """)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        snd = cargar_sonido_praat(tmp_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp_file:
            ruta_npz = tmp_file.name
        fig3 = graficar_espectrograma_praat_interactivo(snd, max_freq=5000, guardar_como=ruta_npz)
        st.plotly_chart(fig3, use_container_width=True)
        # Leer el contenido del archivo para la descarga
        with open(ruta_npz, "rb") as f:
            bytes_npz = f.read()

        # Botón de descarga
        st.download_button(
            label="⬇️ Descargar datos del espectrograma (.npz)",
            data=bytes_npz,
            file_name="espectrograma.npz",
            mime="application/octet-stream"
        )
    
    if mostrar_f0:
        st.markdown(
            "<h4 style='text-align: center;'>📈 Frecuencia Fundamental</h4>",
            unsafe_allow_html=True
        )
        with st.expander("ℹ️ "):
            st.write("""
                La frecuencia fundamental (F0) es la frecuencia más baja de una señal periódica y representa
                 el tono percibido del llanto. Está relacionada con la vibración de las cuerdas vocales del bebé.

                \nAlteraciones en la F0 pueden reflejar cambios en el estado neurológico o fisiológico del bebé. 
                \nEl rango típico de la frecuencia fundamental (F0) del llanto de un bebé suele encontrarse entre: 250Hz y 600 Hz.
                \nAunque este rango puede variar dependiendo de:
                \n\tEdad gestacional: Los bebés prematuros tienden a tener F0 más altas.
                \n\tEstado emocional o fisiológico: El llanto por dolor, hambre o incomodidad puede elevar la F0.
                \n\tPatologías: Algunas condiciones neurológicas o respiratorias pueden alterar significativamente el patrón y la F0.
                \nRangos más específicos reportados en estudios:
                \nLlantos normales: 
                \n\tF0 promedio: entre 400 y 500 Hz
                \n\tF0 mínima: alrededor de 250 Hz
                \n\tF0 máxima: puede alcanzar hasta 700 Hz o incluso más en episodios agudos.
                \nLlantos patológicos (como en encefalopatías o síndromes genéticos):
                \n\tPueden mostrar F0 muy elevadas (> 800 Hz) o patrones inusuales
                """)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            snd = cargar_sonido_praat(tmp_path)
            f0_mean, f0_min, f0_max, (f0_times, f0_curve) = obtener_frecuencia_fundamental(snd)
            if f0_mean is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"🟢 **Mínima:** {f0_min:.2f} Hz")
                with col2:
                    st.write(f"🟡 **Media:** {f0_mean:.2f} Hz")
                with col3:
                    st.write(f"🔴 **Máxima:** {f0_max:.2f} Hz")

                # Usar la función actualizada que retorna también los valores válidos
                fig_f0, times_validos, f0_validos = graficar_curva_f0(f0_times, f0_curve)
                print(type(fig_f0))

                st.plotly_chart(fig_f0, use_container_width=True)

                # Crear DataFrame solo con valores filtrados
                df_f0 = pd.DataFrame({
                    'Tiempo (s)': times_validos,
                    'F0 (Hz)': f0_validos
                })

                # CSV en memoria para descarga
                csv_buffer = io.StringIO()
                df_f0.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                # Escapar los caracteres especiales antes de usar en f-string
                csv_data_encoded = csv_data.replace('\n', '%0A').replace(',', '%2C')

                # Botón de descarga alineado a la derecha
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-top: 10px;">
                        <a href="data:text/csv;charset=utf-8,{csv_data_encoded}" download="f0_datos.csv">
                            <button style="background-color: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">
                                📥 Descargar F0 (CSV)
                            </button>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("No se pudo detectar la frecuencia fundamental.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    if mostrar_jitter_shimmer:
        st.markdown(
            "<h4 style='text-align: center;'>📈 Jitter y Shimmer</h4>",
            unsafe_allow_html=True
        )
        with st.expander("ℹ️ "):
            st.write("""
                🔸 Jitter mide la variación ciclo a ciclo en la frecuencia fundamental, es decir, la estabilidad 
                temporal de la vibración vocal.
                \nUn jitter elevado puede reflejar inestabilidad vocal, típicamente asociado con alteraciones 
                neuromusculares, fatiga, dolor, o problemas en la coordinación respiratoria o laríngea.
                \nEl umbral de referencia utilizado en este sistema es: 1.0
                \n🔹 Shimmer mide la variación ciclo a ciclo en la amplitud de la señal, es decir, la estabilidad
                en la intensidad vocal.
                \nUn shimmer elevado puede reflejar esfuerzo vocal, dificultades en el control de la intensidad o 
                alteraciones estructurales en las cuerdas vocales.
                \nEl umbral de referencia utilizado en este sistema es: 3.8
                \nValores anormales de jitter y shimmer pueden señalar disfunciones en el control neuromuscular
                o afectaciones en el sistema respiratorio o laríngeo del bebé. 
                """)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            snd = cargar_sonido_praat(tmp_path)
            jitter, shimmer = calcular_jitter_shimmer(snd)
            jitter_percent = jitter * 100
            shimmer_percent = shimmer * 100
            umbral_jitter = 1.0
            umbral_shimmer = 3.8
            col1, col2 = st.columns(2)
            with col1:
                delta_j = jitter_percent - umbral_jitter
                st.metric("🔸 Jitter", f"{jitter_percent:.2f} %", f"{delta_j:+.2f} %", delta_color="inverse" if delta_j > 0 else "normal")
            with col2:
                delta_s = shimmer_percent - umbral_shimmer
                st.metric("🔹 Shimmer", f"{shimmer_percent:.2f} %", f"{delta_s:+.2f} %", delta_color="inverse" if delta_s > 0 else "normal")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    if mostrar_zcr:
        st.markdown(
            "<h4 style='text-align: center;'>📊 Tasa de Cruce por Cero</h4>",
            unsafe_allow_html=True
        )
        with st.expander("ℹ️ "):
            st.write("""
                La tasa de cruce por cero (ZCR) representa cuántas veces la señal de audio cruza el eje cero 
                (cambia de signo) por unidad de tiempo. Es una métrica simple que indica la cantidad de oscilaciones 
                de alta frecuencia.
                \nUn ZCR alto puede estar asociado con señales más ruidosas o entrecortadas, lo que puede indicar
                angustia, esfuerzo respiratorio o llanto agudo. En cambio, un ZCR bajo sugiere llantos más tonales
                y estables, a menudo asociados con estados menos críticos.
                """)
        zcr = calcular_zcr(y)
        zcr_mean = np.mean(zcr)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"🔄 ZCR media: {zcr_mean:.4f}")
        with col2:
            if zcr_mean < 0.02:
                st.markdown("<span style='color:blue'>🟢 Bajo ZCR</span>", unsafe_allow_html=True)
            elif zcr_mean < 0.05:
                st.markdown("<span style='color:orange'>🟡 Moderado ZCR</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:red'>🔴 Alto ZCR</span>", unsafe_allow_html=True)
        fig_zcr = graficar_zcr_plotly(y, sr)
        st.plotly_chart(fig_zcr, use_container_width=True)
        
    if mostrar_llanto:
        st.markdown(
            "<h4 style='text-align: center;'>🍼 Detección y Filtrado de Llanto (YAMNet)</h4>",
            unsafe_allow_html=True
        )

        threshold = st.slider("🎚️ Umbral de detección (confianza mínima)", 0.0, 1.0, 0.3, 0.05)

        with st.spinner("🔎 Analizando llanto con YAMNet..."):
            resultado = filtrar_llanto_yamnet(audio_bytes, threshold=threshold)

        if resultado is not None:
            audio_filtrado_bytes, sr_filtrado, segmentos = resultado

            st.success(f"✅ Se detectaron {len(segmentos)} segmento(s) de llanto.")
            st.audio(audio_filtrado_bytes, format="audio/wav", start_time=0)

            # Botón para descargar señal filtrada
            st.download_button(
                label="⬇️ Descargar audio filtrado (solo llanto)",
                data=audio_filtrado_bytes,
                file_name="llanto_filtrado.wav",
                mime="audio/wav"
            )

            # Mostrar intervalo de tiempo de cada segmento
            st.markdown("### ⏱️ Segmentos detectados:")
            for i, (start, end) in enumerate(segmentos):
                inicio_seg = start / sr_filtrado
                fin_seg = end / sr_filtrado
                st.write(f"🍼 Segmento {i+1}: {inicio_seg:.2f}s - {fin_seg:.2f}s")

        else:
            st.warning("⚠️ No se detectaron segmentos de llanto con el umbral seleccionado.")
else:
    st.warning("Por favor, sube una muestra de llanto en formato .wav para comenzar.")

    