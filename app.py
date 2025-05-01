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
st.title("👶 Análisis de Llanto Infantil")
# Cargar el archivo .wav
st.write("Sube un archivo de audio en formato `.wav` para analizar su contenido acústico.")
archivo_audio = st.file_uploader("📤 Sube el archivo de audio", type=["wav"])

if archivo_audio is not None:
    # Leer el archivo en bytes
    audio_bytes = archivo_audio.read()

    # Cargar audio con librosa desde bytes
    y, sr = cargar_audio_desde_bytes(audio_bytes)

    # Calcular duración
    duracion = calcular_duracion(y, sr)
    
    # Espectrograma con Librosa
    #st.subheader("🎛️ Espectrograma con Librosa (Mel)")
    #fig1 = graficar_espectrograma_librosa(y, sr)
    #st.pyplot(fig1)

    # Guardar el archivo temporal para usarlo con Parselmouth
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    # Espectrograma con Praat (Parselmouth)
    # st.subheader("🎛️ Espectrograma con Praat")
    snd = cargar_sonido_praat(tmp_path)
    #fig2 = graficar_espectrograma_praat(snd)
    #st.pyplot(fig2)

    st.subheader("🎛️ Espectrograma con Praat (Interactivo)")
    fig3 = graficar_espectrograma_praat_interactivo(snd)
    st.plotly_chart(fig3, use_container_width=True)

    # Mostrar reproductor
    st.audio(archivo_audio, format="audio/wav")
    st.write(f"🕒 Duración del audio: {duracion:.2f} segundos")
    
    #-----------------------------------------------------------------------------
    # Frecuencia fundamental con Parselmouth
    st.subheader("📈 Frecuencia Fundamental: ")
    try:
        f0_mean, f0_min, f0_max, (f0_times, f0_curve) = obtener_frecuencia_fundamental(snd)
        if f0_mean is not None:
            st.write(f"🔸 Frecuencia mínima: {f0_min:.2f} Hz")
            st.write(f"🔹 Frecuencia fundamental media: {f0_mean:.2f} Hz")
            st.write(f"🔺 Frecuencia máxima: {f0_max:.2f} Hz")

            fig_f0 = graficar_curva_f0(f0_times, f0_curve)
            st.plotly_chart(fig_f0, use_container_width=True)
        else:
            st.warning("No se pudo detectar la frecuencia fundamental en este archivo.")
    except Exception as e:
        st.error(f"⚠️ Error al procesar el audio con Praat: {e}")

    #-----------------------------------------------------------------------------
    # Jitter y Shimmer
    st.subheader("📉 Variabilidad acústica")
    try:
        jitter, shimmer = calcular_jitter_shimmer(snd)
        jitter_percent = jitter *100;
        shimmer_percent = shimmer *100;
        #st.write(f"🔹 Jitter (local): {jitter_percent:.2f}")
        #st.write(f"🔸 Shimmer (local): {shimmer_percent:.2f}")
        
        # Umbrales de referencia (puedes ajustarlos según la literatura)
        umbral_jitter = 1.0  # %
        umbral_shimmer = 3.8  # %

        col1, col2 = st.columns(2)

        with col1:
            delta_j = jitter_percent - umbral_jitter
            st.metric(
                label="🔸 Jitter (local)",
                value=f"{jitter_percent:.2f} %",
                delta=f"{delta_j:+.2f} %",
                delta_color="inverse" if delta_j > 0 else "normal",
                help="Variación ciclo a ciclo del período. Refleja inestabilidad en la frecuencia."
            )

        with col2:
            delta_s = shimmer_percent - umbral_shimmer
            st.metric(
                label="🔹 Shimmer (local)",
                value=f"{shimmer_percent:.2f} %",
                delta=f"{delta_s:+.2f} %",
                delta_color="inverse" if delta_s > 0 else "normal",
                help="Variación ciclo a ciclo de la amplitud. Refleja inestabilidad en la intensidad."
            )

    except Exception as e:
        st.error(f"⚠️ Error al calcular Jitter/Shimmer: {e}")

    #-----------------------------------------------------------------------------
    # Calcular y mostrar ZCR
    st.subheader("📊 Tasa de Cruces por Cero (ZCR)")
    zcr = calcular_zcr(y)
    zcr_mean = np.mean(zcr)      
    fig_zcr = graficar_zcr(y, sr)  # 'y' es la señal de audio y 'sr' es la tasa de muestreo
    st.pyplot(fig_zcr)
    
    # Interpretación del ZCR
    st.subheader("🎯 Interpretación del ZCR")
    st.write(f"🔄 Tasa de Cruce por Cero media: {zcr_mean:.4f}")

    # Definir umbrales de ejemplo (puedes ajustarlos según el contexto de tus datos)
    if zcr_mean < 0.02:
        st.markdown(
            "<span style='color:blue'>🔵 <b>Bajo ZCR</b>: La señal es más suave, probablemente vocalizaciones continuas o sonidos sostenidos.</span>",
            unsafe_allow_html=True,
        )
    elif zcr_mean < 0.05:
        st.markdown(
            "<span style='color:orange'>🟠 <b>ZCR Moderado</b>: La señal tiene cambios frecuentes pero no excesivos, podría haber llanto intermitente.</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='color:red'>🔴 <b>Alto ZCR</b>: Muchos cruces por cero, posible ruido o sonidos agudos entrecortados.</span>",
            unsafe_allow_html=True,
        )

    #-----------------------------------------------------------------------------
    # Detección de llanto
    #llanto, energia = detectar_llanto(y, sr)

    #if llanto:
    #    st.success("👶 Se detectó llanto en el audio.")
    #else:
    #    st.info("🧘 No se detectó llanto en el audio.")

    #segmentos_llanto = detectar_segmentos_llanto(y, sr)

    # Agregar un slider para ajustar el umbral de detección de llanto
    #st.subheader("🛠️ Configuración de Umbral")
    #umbral_db = st.slider("Selecciona el umbral de energía (dB) para distinguir llanto de silencio", min_value=-60, max_value=0, value=-30, step=1)
    # Detectar segmentos de llanto y silencio usando el umbral ajustable
    #tiempo_llanto, tiempo_silencio, mask_llanto = detectar_tiempos_llanto(y, sr, umbral_db)

    # Visualización de energía
    #graficar_energia(y, sr, energia, umbral_db)
    
    #st.subheader("📊 Métricas de Llanto y Silencio")
    #st.write(f"🔸 Tiempo total de llanto: {tiempo_llanto:.2f} s")
    #st.write(f"🔹 Tiempo total de silencio: {tiempo_silencio:.2f} s")