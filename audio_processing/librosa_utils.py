import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def cargar_audio_desde_bytes(audio_bytes, sr=None):
    import io
    return librosa.load(io.BytesIO(audio_bytes), sr=sr)

def calcular_duracion(y, sr):
    return librosa.get_duration(y=y, sr=sr)

def graficar_espectrograma_librosa(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig


