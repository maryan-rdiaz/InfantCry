## InfantCry

🎧 Sistema de análisis de llanto infantil para apoyar el tamizaje clínico.  
Este sistema permite visualizar y analizar características acústicas como duración, segmentos de sonido y silencio, frecuencia fundamental y espectrograma.

## 📌 Objetivo

Desarrollar una interfaz amigable para el personal médico, que permita la exploración de señales de llanto infantil y sus propiedades acústicas, utilizando herramientas como `librosa`, `matplotlib` y `Praat` mediante `parselmouth`.

## 🛠️ Tecnologías utilizadas
- [Python 3.x](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Librosa](https://librosa.org/)
- [Praat-parselmouth](https://github.com/YannickJadoul/Parselmouth)

## ⚙️ Instalación
1. Clona el repositorio:

   ```bash
   git clone https://github.com/maryan-rdiaz/InfantCry.git
   cd InfantCry
2. Crea y activa un entorno virtual (opcional pero recomendado):
   python -m venv venv
   .\venv\Scripts\activate  # En Windows
3. Instala las dependencias
   pip install -r requeriments.txt
4. Ejecución: streamlit run app.py
   La aplicación se abrirá en tu navegador en http://localhost:8501.
   
## 📁 Estructura del proyecto
InfantCry/
│
├── app.py                  # Código principal de la app Streamlit
├── requeriments.txt        # Lista de dependencias
├── README.md               # Este archivo
├── venv/                   # Entorno virtual (no subir a GitHub)
└── data/                   # Carpeta opcional para guardar audios o resultados

## Ejemplo de uso

## Futuras mejoras

## Licencia
Este proyecto es parte de una tesis doctoral y no está destinado a uso clínico directo sin validación adicional. Consulta al autor para más detalles.
