import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import soundfile as sf
import os
import io

def cargar_yamnet_model():
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model

def obtener_segmentos_llanto(audio, sr, model, threshold=0.3):
    # Resamplear a 16kHz (requisito de YAMNet)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Ejecutar modelo YAMNet
    scores, embeddings, spectrogram = model(audio)
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    class_names = [line.split(',')[2].strip().strip('"') for line in open(class_map_path).readlines()[1:]]

    # Índice de "Infant cry"
    #cry_index = class_names.index('Infant cry')
    cry_index = next(i for i, name in enumerate(class_names) if 'cry' in name.lower())

    # Obtener etiquetas por frames (~0.96 segundos)
    scores_np = scores.numpy()
    cry_scores = scores_np[:, cry_index]
    mask = cry_scores > threshold

    # Calcular los intervalos (en muestras)
    hop_size = int(0.96 * sr)
    segments = []
    for i, val in enumerate(mask):
        if val:
            start = i * hop_size
            end = min((i + 1) * hop_size, len(audio))
            segments.append((start, end))

    return segments, sr

def extraer_segmentos(audio, segments):
    # Extraer y concatenar
    llanto_segmentos = [audio[start:end] for start, end in segments]
    audio_filtrado = np.concatenate(llanto_segmentos) if llanto_segmentos else np.array([])
    return llanto_segmentos, audio_filtrado

def guardar_segmentos(llanto_segmentos, sr, carpeta_salida="segmentos_llanto"):
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = []
    for i, seg in enumerate(llanto_segmentos):
        path = os.path.join(carpeta_salida, f"llanto_segmento_{i+1}.wav")
        sf.write(path, seg, sr)
        archivos.append(path)
    return archivos

def filtrar_llanto_yamnet(audio_bytes, threshold=0.3):
    """
    Carga un audio en bytes, aplica YAMNet para detectar llanto infantil,
    y devuelve la señal filtrada en WAV (bytes), la tasa de muestreo y los segmentos.

    Retorna:
        audio_filtrado_wav_bytes, sr, segmentos_llanto
    """
    # Cargar modelo
    model = cargar_yamnet_model()

    # Leer el audio
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

    # Obtener segmentos donde hay llanto
    segmentos, sr = obtener_segmentos_llanto(y, sr, model, threshold)

    if not segmentos:
        return None

    # Extraer segmentos
    llanto_segmentos, audio_filtrado = extraer_segmentos(y, segmentos)

    # Guardar en buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_filtrado, sr, format='WAV')
    buffer.seek(0)

    return buffer.read(), sr, segmentos
