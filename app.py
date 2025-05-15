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
    graficar_zcr,
)
#-----------------------------------------------------------------------------

st.set_page_config(page_title="Análisis de Llanto Infantil", layout="centered")

# ------------------------Menú lateral ---------------------------
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
        st.subheader("📄 Información General")
        st.write(f"🕒 Duración del audio: {duracion:.2f} segundos")

    if mostrar_espectrograma:
        st.subheader("🎛️ Espectrograma con Praat (Interactivo)")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        snd = cargar_sonido_praat(tmp_path)
        fig3 = graficar_espectrograma_praat_interactivo(snd)
        st.plotly_chart(fig3, use_container_width=True)

    if mostrar_f0:
        st.subheader("📈 Frecuencia Fundamental")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            snd = cargar_sonido_praat(tmp_path)
            f0_mean, f0_min, f0_max, (f0_times, f0_curve) = obtener_frecuencia_fundamental(snd)
            if f0_mean is not None:
                st.write(f"🔸 Mínima: {f0_min:.2f} Hz")
                st.write(f"🔹 Media: {f0_mean:.2f} Hz")
                st.write(f"🔺 Máxima: {f0_max:.2f} Hz")
                fig_f0 = graficar_curva_f0(f0_times, f0_curve)
                st.plotly_chart(fig_f0, use_container_width=True)
            else:
                st.warning("No se pudo detectar la frecuencia fundamental.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    if mostrar_jitter_shimmer:
        st.subheader("📉 Jitter y Shimmer")
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
        st.subheader("📊 Tasa de Cruce por Cero")
        zcr = calcular_zcr(y)
        zcr_mean = np.mean(zcr)
        fig_zcr = graficar_zcr(y, sr)
        st.pyplot(fig_zcr)
        st.write(f"🔄 ZCR media: {zcr_mean:.4f}")
        if zcr_mean < 0.02:
            st.markdown("<span style='color:blue'>🔵 Bajo ZCR</span>", unsafe_allow_html=True)
        elif zcr_mean < 0.05:
            st.markdown("<span style='color:orange'>🟠 Moderado ZCR</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:red'>🔴 Alto ZCR</span>", unsafe_allow_html=True)

    if mostrar_llanto:
        st.subheader("🧠 Detección de Llanto (en desarrollo)")
        # Aquí puedes incorporar lógica futura

else:
    st.warning("Por favor, sube una muestra de llanto en formato .wav para comenzar.")

    