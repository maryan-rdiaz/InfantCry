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
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]
    if len(pitch_values) > 0:
        return np.median(pitch_values)
    return None
