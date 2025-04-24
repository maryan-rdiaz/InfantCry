import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import librosa

# Importar funciones personalizadas
from audio_processing.librosa_utils import (
    cargar_audio_desde_bytes,
    calcular_duracion,
    graficar_espectrograma_librosa,
)

from audio_processing.praat_utils import (
    cargar_sonido_praat,
    graficar_espectrograma_praat,
    obtener_frecuencia_fundamental,
)

from audio_processing.cry_detection import (
    detectar_llanto,
    detectar_segmentos_llanto,
)

from utils.energia import graficar_energia
from utils.tiempo import detectar_tiempos_llanto


st.set_page_config(page_title="Análisis de Llanto Infantil", layout="centered")
st.title("👶 Análisis de Llanto Infantil")
st.write("Sube un archivo de audio en formato `.wav` para analizar su contenido acústico.")
archivo_audio = st.file_uploader("📤 Sube el archivo de audio", type=["wav"])

if archivo_audio is not None:
    # Mostrar reproductor
    #st.audio(archivo_audio, format="audio/wav")

    # Leer el archivo en bytes
    audio_bytes = archivo_audio.read()

    # Cargar audio con librosa desde bytes
    y, sr = cargar_audio_desde_bytes(audio_bytes)

    # Mostrar duración
    duracion = calcular_duracion(y, sr)
    st.write(f"🕒 Duración del audio: {duracion:.2f} segundos")

    # Espectrograma con Librosa
    st.subheader("🎛️ Espectrograma con Librosa (Mel)")
    fig1 = graficar_espectrograma_librosa(y, sr)
    st.pyplot(fig1)

    # Guardar el archivo temporal para usarlo con Parselmouth
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    # Espectrograma con Praat (Parselmouth)
    st.subheader("🎛️ Espectrograma con Praat")
    snd = cargar_sonido_praat(tmp_path)
    fig2 = graficar_espectrograma_praat(snd)
    st.pyplot(fig2)

    # Frecuencia fundamental con Parselmouth
    st.subheader("📈 Frecuencia Fundamental (F0)")
    try:
        f0 = obtener_frecuencia_fundamental(snd)
        if f0 is not None:
            st.write(f"🔹 Frecuencia fundamental media: {f0:.2f} Hz")
        else:
            st.warning("No se pudo detectar la frecuencia fundamental en este archivo.")
    except Exception as e:
        st.error(f"⚠️ Error al procesar el audio con Praat: {e}")

    # Detección de llanto
    st.subheader("🧠 Detección de Llanto")
    llanto, energia = detectar_llanto(y, sr)

    if llanto:
        st.success("👶 Se detectó llanto en el audio.")
    else:
        st.info("🧘 No se detectó llanto en el audio.")

    segmentos_llanto = detectar_segmentos_llanto(y, sr)

    # Agregar un slider para ajustar el umbral de detección de llanto
    st.subheader("🛠️ Configuración de Umbral")
    umbral_db = st.slider("Selecciona el umbral de energía (dB) para distinguir llanto de silencio", min_value=-60, max_value=0, value=-30, step=1)
    # Detectar segmentos de llanto y silencio usando el umbral ajustable
    tiempo_llanto, tiempo_silencio, mask_llanto = detectar_tiempos_llanto(y, sr, umbral_db)

    # Visualización de energía
    graficar_energia(y, sr, energia, umbral_db)
    # Mostrar reproductor
    st.audio(archivo_audio, format="audio/wav")

    st.subheader("📊 Métricas de Llanto y Silencio")
    st.write(f"🔸 Tiempo total de llanto: {tiempo_llanto:.2f} s")
    st.write(f"🔹 Tiempo total de silencio: {tiempo_silencio:.2f} s")