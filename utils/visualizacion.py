import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import librosa
from audio_processing.librosa_utils import calcular_zcr
import os

def graficar_espectrograma_praat_interactivo(snd, max_freq=5000, max_points=200_000, guardar_como=None):
    """
    Genera un espectrograma interactivo reduciendo datos si es necesario.
    Si se indica `guardar_como`, guarda los datos completos del espectrograma en .npz.
    """
    spectrogram = snd.to_spectrogram(window_length=0.025, maximum_frequency=max_freq)
    spectrogram_db = 10 * np.log10(np.maximum(spectrogram.values, 1e-10))

    tiempo = np.linspace(0, snd.get_total_duration(), spectrogram_db.shape[1])
    frecuencia = np.linspace(0, max_freq, spectrogram_db.shape[0])

    # Guardar datos completos si se solicita
    if guardar_como:
        np.savez_compressed(guardar_como,
                            espectrograma=spectrogram_db,
                            tiempo=tiempo,
                            frecuencia=frecuencia)
        print(f"Datos guardados en: {guardar_como}.npz")

    # Reducir puntos solo para la visualización
    total_points = spectrogram_db.size
    if total_points > max_points:
        factor_x = int(np.ceil(spectrogram_db.shape[1] / np.sqrt(max_points)))
        factor_y = int(np.ceil(spectrogram_db.shape[0] / np.sqrt(max_points)))
        spectrogram_db = spectrogram_db[::factor_y, ::factor_x]
        tiempo = tiempo[::factor_x]
        frecuencia = frecuencia[::factor_y]

    # Gráfica interactiva
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

def graficar_curva_f0(f0_times, f0_curve, f0_min_valid=200, f0_max_valid=1000):
    """Genera una gráfica interactiva de F0 mostrando solo puntos válidos (250–600 Hz),
    marca los valores mínimo y máximo, y añade líneas guía para el rango típico de llanto."""
    
    # Convertir a arrays
    times_arr = np.array(f0_times)
    f0_arr = np.array(f0_curve)

    # Rango típico del llanto
    f0_min_llanto = 250
    f0_max_llanto = 600

    # Filtrar valores dentro del rango válido general
    mask = (f0_arr >= f0_min_valid) & (f0_arr <= f0_max_valid) & ~np.isnan(f0_arr)
    times_validos = times_arr[mask]
    f0_validos = f0_arr[mask]

    fig = go.Figure()

    # Puntos válidos
    fig.add_trace(go.Scatter(
        x=times_validos,
        y=f0_validos,
        mode='markers',
        name='F0 (Hz)',
        marker=dict(color='royalblue', size=6, symbol='circle')
    ))

    # Líneas de referencia (estáticas, no dependen de f0_validos)
    fig.add_trace(go.Scatter(
        x=[times_arr.min(), times_arr.max()],
        y=[f0_max_llanto, f0_max_llanto],
        mode="lines",
        name="Límite superior (600 Hz)",
        line=dict(color="orange", dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=[times_arr.min(), times_arr.max()],
        y=[f0_min_llanto, f0_min_llanto],
        mode="lines",
        name="Límite inferior (250 Hz)",
        line=dict(color="lightgreen", dash="dash")
    ))

    # Marcar mínimo y máximo si hay datos
    if len(f0_validos) > 0:
        min_idx = np.argmin(f0_validos)
        max_idx = np.argmax(f0_validos)

        fig.add_trace(go.Scatter(
            x=[times_validos[min_idx]],
            y=[f0_validos[min_idx]],
            mode='markers+text',
            name='Mínimo',
            marker=dict(color='green', size=10),
            text=[f"{f0_validos[min_idx]:.2f} Hz"],
            textposition="top center"
        ))

        fig.add_trace(go.Scatter(
            x=[times_validos[max_idx]],
            y=[f0_validos[max_idx]],
            mode='markers+text',
            name='Máximo',
            marker=dict(color='red', size=10),
            text=[f"{f0_validos[max_idx]:.2f} Hz"],
            textposition="bottom center"
        ))

    fig.update_layout(
        title="Curva de Frecuencia Fundamental (F0)",
        xaxis_title="Tiempo (s)",
        yaxis_title="F0 (Hz)",
        template="simple_white",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True
    )

    return fig, times_validos, f0_validos


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