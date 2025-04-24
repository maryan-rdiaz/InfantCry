import librosa
import numpy as np

def detectar_tiempos_llanto(y, sr, umbral_db=-30):
    frame_length = 2048
    hop_length = 512
    energia = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).flatten()
    energia_db = 10 * np.log10(np.maximum(energia, 1e-10))
    
    mask_llanto = energia_db > umbral_db
    tiempo_total = len(y) / sr
    tiempo_llanto = (np.sum(mask_llanto) * hop_length) / sr
    tiempo_silencio = tiempo_total - tiempo_llanto
    
    return tiempo_llanto, tiempo_silencio, mask_llanto
