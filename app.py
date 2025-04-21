import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import parselmouth
import tempfile

st.set_page_config(page_title="Análisis de Llanto Infantil", layout="centered")

st.title("👶 Análisis de Llanto Infantil")
st.write("Sube un archivo de audio en formato `.wav` para analizar su contenido acústico.")

archivo_audio = st.file_uploader("📤 Sube el archivo de audio", type=["wav"])

if archivo_audio is not None:
    # Mostrar reproductor
    st.audio(archivo_audio, format="audio/wav")

    # Leer archivo una sola vez
    audio_bytes = archivo_audio.read()

    # Cargar audio con librosa desde bytes
    import io
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Mostrar duración
    duracion = librosa.get_duration(y=y, sr=sr)
    st.write(f"🕒 Duración del audio: {duracion:.2f} segundos")

    # Espectrograma con Librosa
    st.subheader("🎛️ Espectrograma con Librosa (Mel)")
    fig1, ax1 = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img1 = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax1)
    plt.colorbar(img1, ax=ax1, format="%+2.0f dB")
    st.pyplot(fig1)

    # Guardar el archivo temporal para usarlo en Parselmouth
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

     # Espectrograma con Praat (Parselmouth)
    st.subheader("🎛️ Espectrograma con Praat")
    snd = parselmouth.Sound(tmp_path)

    # Convertir el sonido a un espectrograma
    spectrogram = snd.to_spectrogram(window_length=0.025, maximum_frequency=5000)

    # Obtener la matriz de amplitudes del espectrograma
    spectrogram_values = spectrogram.values

    # Convertir los valores del espectrograma de amplitud a decibeles
    spectrogram_db = 10 * np.log10(np.maximum(spectrogram.values, 1e-10))  # Evita log(0)


    # Graficar el espectrograma de Praat
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    im = ax2.imshow(spectrogram_db, aspect='auto', cmap='inferno', origin='lower', 
                extent=[0, snd.get_total_duration(), 0, 5000])
    ax2.set_xlabel('Tiempo (segundos)')
    ax2.set_ylabel('Frecuencia (Hz)')
    plt.colorbar(im, ax=ax2, format="%+2.0f dB")
    plt.title('Espectrograma con Praat (Parselmouth)')
    st.pyplot(fig2)

    # Frecuencia fundamental con Parselmouth
    st.subheader("📈 Frecuencia Fundamental (F0)")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)  # Guardamos los mismos bytes
        tmp_path = tmp_file.name

    try:
        snd = parselmouth.Sound(tmp_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        if len(pitch_values) > 0:
            f0_median = np.median(pitch_values)
            st.write(f"🔹 Frecuencia fundamental media: {f0_median:.2f} Hz")
        else:
            st.warning("No se pudo detectar la frecuencia fundamental en este archivo.")
    except parselmouth.PraatError as e:
        st.error(f"⚠️ Error al procesar el audio con Praat: {e}")

