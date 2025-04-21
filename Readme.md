# 👶 Sistema de Análisis de Llanto Infantil

Este proyecto es una aplicación interactiva desarrollada en Python usando Streamlit. Permite analizar señales de llanto infantil mediante procesamiento de audio y visualización de parámetros como duración, espectrograma y frecuencia fundamental (F0).

## 🚀 Características

- Carga de archivos `.wav`
- Cálculo de duración de la señal
- Estimación básica de segmentos de sonido y silencio
- Extracción de frecuencia fundamental con Praat (vía Parselmouth)
- Visualización del espectrograma
- Reproducción del audio desde la interfaz

## 📁 Estructura del proyecto


## 🛠️ Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/analisis-llanto-infantil.git
   cd analisis-llanto-infantil

2. Crea un entorno virtual
python -m venv venv
.\venv\Scripts\activate  # en Windows

3. Instala dependencias
pip install -r requirements.txt

4. Ejecuta la aplicación
streamlit run app.py

## Ejemplo de uso

## Futuras mejoras

## Licencia
Este proyecto es parte de una tesis doctoral y no está destinado a uso clínico directo sin validación adicional. Consulta al autor para más detalles.