import numpy as np
import librosa

def detectar_llanto(y, sr, umbral_energia=0.02):
    # Calcular la energía del audio por frames
    energia = librosa.feature.rms(y=y)[0]
    
    # Determinar si hay llanto con un umbral simple
    llanto_detectado = np.any(energia > umbral_energia)

    return llanto_detectado, energia

def detectar_segmentos_llanto(y, sr, umbral_db=-30, frame_length=2048, hop_length=512):
    """
    Detecta segmentos de llanto en base al nivel de energía.
    Devuelve listas de tiempo de inicio y fin de llanto.
    """
    # Calcular energía en dB
    energia = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    energia_db = 10 * np.log10(energia + 1e-10)
    
    # Determinar frames con energía mayor al umbral
    frames_llanto = energia_db > umbral_db
    tiempos = librosa.frames_to_time(np.arange(len(energia_db)), sr=sr, hop_length=hop_length)

    # Extraer segmentos de llanto
    segmentos = []
    en_llanto = False
    inicio = 0

    for i, es_llanto in enumerate(frames_llanto):
        if es_llanto and not en_llanto:
            en_llanto = True
            inicio = tiempos[i]
        elif not es_llanto and en_llanto:
            en_llanto = False
            fin = tiempos[i]
            segmentos.append((inicio, fin))
    if en_llanto:
        segmentos.append((inicio, tiempos[-1]))

    return segmentos
