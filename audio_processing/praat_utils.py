import parselmouth
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def cargar_sonido_praat(tmp_path):
    return parselmouth.Sound(tmp_path)

def graficar_espectrograma_praat(snd, max_freq=5000):
    spectrogram = snd.to_spectrogram(window_length=0.025, maximum_frequency=max_freq)
    spectrogram_db = 10 * np.log10(np.maximum(spectrogram.values, 1e-10))
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(spectrogram_db, aspect='auto', cmap='inferno', origin='lower',
                   extent=[0, snd.get_total_duration(), 0, max_freq])
    ax.set_xlabel('Tiempo (segundos)')
    ax.set_ylabel('Frecuencia (Hz)')
    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    return fig

def obtener_frecuencia_fundamental(snd):
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values != 0]  # Excluir silencios (F0 = 0)

    if len(f0_values) == 0:
        return None, None, None, (None, None)

    f0_mean = np.mean(f0_values)
    f0_min = np.min(f0_values)
    f0_max = np.max(f0_values)
    times = pitch.xs()
    curve = pitch.selected_array['frequency']

    return f0_mean, f0_min, f0_max, (times, curve)

def calcular_jitter_shimmer(snd):
    
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

    jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return jitter_local, shimmer_local

