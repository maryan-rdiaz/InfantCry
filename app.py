import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import librosa
import numpy as np

# ------------- Importacion de funciones personalizadas ----------------
from audio_processing.librosa_utils import (
    cargar_audio_desde_bytes,
    calcular_duracion,
    graficar_espectrograma_librosa,
    calcular_zcr,
)

from audio_processing.praat_utils import (
    cargar_sonido_praat,
    graficar_espectrograma_praat,
    obtener_frecuencia_fundamental,
    calcular_jitter_shimmer,
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
mostrar_info_general = st.sidebar.checkbox("📄 Información General")
mostrar_espectrograma = st.sidebar.checkbox("🎛️ Espectrograma")
mostrar_f0 = st.sidebar.checkbox("📈 Frecuencia Fundamental")
mostrar_jitter_shimmer = st.sidebar.checkbox("📉 Jitter y Shimmer")
mostrar_zcr = st.sidebar.checkbox("📊 Zero-Crossing Rate")
mostrar_llanto = st.sidebar.checkbox("🧠 Detección de Llanto")

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
        st.markdown("#### 📄 Información General")
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
        st.markdown("#### 🎛️ Espectrograma")
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
        fig3 = graficar_espectrograma_praat_interactivo(snd)
        st.plotly_chart(fig3, use_container_width=True)

    if mostrar_f0:
        st.markdown("#### 📈 Frecuencia Fundamental")  # Puedes usar ##, ###, #### para ajustar el tamaño
        with st.expander("ℹ️ "):
            st.write("""
                La frecuencia fundamental (F0) es la frecuencia más baja de una señal periódica y representa
                 el tono percibido del llanto. Está relacionada con la vibración de las cuerdas vocales del bebé.

                \nAlteraciones en la F0 pueden reflejar cambios en el estado neurológico o fisiológico del bebé. 
                Por ejemplo, una F0 muy alta o muy baja, o una F0 inestable, pueden estar asociadas con 
                condiciones como dolor, problemas respiratorios o afecciones neurológicas.
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
                fig_f0 = graficar_curva_f0(f0_times, f0_curve)
                st.plotly_chart(fig_f0, use_container_width=True)
            else:
                st.warning("No se pudo detectar la frecuencia fundamental.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    if mostrar_jitter_shimmer:
        st.markdown("#### 📈 Jitter y Shimmer")  # Puedes usar ##, ###, #### para ajustar el tamaño
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
        st.markdown("#### 📊 Tasa de Cruce por Cero")  # Puedes usar ##, ###, #### para ajustar el tamaño
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
        st.subheader("🧠 Detección de Llanto (en desarrollo)")
        # Aquí puedes incorporar lógica futura

else:
    st.warning("Por favor, sube una muestra de llanto en formato .wav para comenzar.")

    