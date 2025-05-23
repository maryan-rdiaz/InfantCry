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
        title='Praat (Interactivo)',
        xaxis_title='Tiempo (s)',
        yaxis_title='Frecuencia (Hz)',
        autosize=True,
    )

    return fig


def graficar_curva_f1(times, f0_curve):
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


def graficar_curva_f0(f0_times, f0_curve):
    fig = go.Figure()

    # Línea principal de F0
    fig.add_trace(go.Scatter(
        x=f0_times, y=f0_curve,
        mode='lines',
        name='F0',
        line=dict(color='blue')
    ))

    # Obtener índices del valor mínimo y máximo (ignorando ceros o None)
    f0_array = np.array(f0_curve)
    valid_idx = np.where((f0_array > 0) & ~np.isnan(f0_array))[0]

    if valid_idx.size > 0:
        min_idx = valid_idx[np.argmin(f0_array[valid_idx])]
        max_idx = valid_idx[np.argmax(f0_array[valid_idx])]

        # Punto mínimo
        fig.add_trace(go.Scatter(
            x=[f0_times[min_idx]],
            y=[f0_curve[min_idx]],
            mode='markers+text',
            name='Mínimo',
            marker=dict(color='green', size=10),
            text=[f"{f0_curve[min_idx]:.2f} Hz"],
            textposition="top center"
        ))

        # Punto máximo
        fig.add_trace(go.Scatter(
            x=[f0_times[max_idx]],
            y=[f0_curve[max_idx]],
            mode='markers+text',
            name='Máximo',
            marker=dict(color='red', size=10),
            text=[f"{f0_curve[max_idx]:.2f} Hz"],
            textposition="top center"
        ))

    fig.update_layout(
        title="Curva de Frecuencia Fundamental (F0)",
        xaxis_title="Tiempo (s)",
        yaxis_title="F0 (Hz)",
        template="simple_white"
    )
    return fig


def graficar_zcr_plotly(y, sr, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    frames = range(len(zcr))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=zcr, mode='lines', name='ZCR', line=dict(color='mediumblue')))
    fig.update_layout(
        title='Tasa de Cruce por Cero (ZCR)',
        xaxis_title='Tiempo (s)',
        yaxis_title='ZCR',
        template='simple_white',
        height=300
    )
    return fig