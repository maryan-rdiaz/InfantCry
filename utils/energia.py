import librosa
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
   
def graficar_energia(y, sr, energia, umbral_db):
   
    # Convertir energía RMS a decibeles
    energia_db = 10 * np.log10(np.maximum(energia, 1e-10))

    st.subheader("📊 Energía del Audio (en dB)")
    fig, ax = plt.subplots()
    frames = range(len(energia_db))
    t = librosa.frames_to_time(frames, sr=sr)
    ax.plot(t, energia_db, label='Energía (dB)')
    ax.axhline(y=umbral_db, color='r', linestyle='--', label='Umbral de silencio (dB)')
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Energía (dB)")
    ax.legend()
    st.pyplot(fig)
