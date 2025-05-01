import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import librosa
from audio_processing.librosa_utils import calcular_zcr

def graficar_espectrograma_praat_interactivo(snd, max_freq=5000):
    """Genera un espectrograma interactivo a partir de un sonido Parselmouth usando Plotly."""
    spectrogram = snd.to_spectrogram(window_length=0.025, maximum_frequency=max_freq)
    spectrogram_db = 10 * np.log10(np.maximum(spectrogram.values, 1e-10))

    tiempo = np.linspace(0, snd.get_total_duration(), spectrogram_db.shape[1])
    frecuencia = np.linspace(0, max_freq, spectrogram_db.shape[0])

    fig = go.Figure(data=go.Heatmap(
        z=spectrogram_db,
        x=tiempo,
        y=frecuencia,
        colorscale='Inferno',
        colorbar=dict(title='dB'),
        zmin=np.min(spectrogram_db),
        zmax=np.max(spectrogram_db)
    ))

    fig.update_layout(
        title='Espectrograma con Praat (Interactivo)',
        xaxis_title='Tiempo (s)',
        yaxis_title='Frecuencia (Hz)',
        autosize=True,
    )

    return fig


def graficar_curva_f0(times, f0_curve):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times,
        y=f0_curve,
        mode='lines',
        name='F0 (Hz)',
        line=dict(color='royalblue')
    ))

    fig.update_layout(
        title="Curva de Frecuencia Fundamental (F0)",
        xaxis_title="Tiempo (s)",
        yaxis_title="Frecuencia (Hz)",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def graficar_zcr(y, sr):
    """Genera la visualización del Zero-Crossing Rate (ZCR)."""
    zcr = calcular_zcr(y)  # Usamos la función de librosa_utils.py para calcular ZCR

    # Convertir a tiempo (en segundos)
    frames = range(len(zcr))
    t = librosa.frames_to_time(frames, sr=sr)

    # Crear la gráfica
    fig, ax = plt.subplots()
    ax.plot(t, zcr, label='Zero-Crossing Rate', color='b')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('ZCR')
    ax.set_title('Tasa de Cruces por Cero (ZCR) a lo largo del tiempo')
    ax.legend()
    plt.grid(True)

    return fig