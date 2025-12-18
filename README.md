# Sistema de procesado de datos en Tiempo Real y en Lote centrado en el Reconocimiento Facial y Análisis de Emociones

## Descripción
Este proyecto implementa un sistema de visión por computadora capaz de identificar personas y detectar sus emociones (Enfado, Felicidad, Neutral, Tristeza) en imágenes o video en tiempo real. Utiliza [PyTorch](https://pytorch.org/) para el modelo de clasificación de emociones y la librería [face_recognition](https://github.com/ageitgey/face_recognition) para la identificación facial.

El sistema permite:
- **Identificación de usuarios**: Compara caras detectadas con una base de datos de identidades conocidas.
- **Detección de emociones**: Clasifica la expresión facial en una de las 4 emociones soportadas.
- **Registro histórico**: Guarda un historial de detecciones con fecha, hora, nombre y emoción en un archivo CSV.

## Instalación

### Requisitos previos
- Python 3.8 o superior.
- Recomendable uso de GPU (CUDA) para mayor rendimiento, aunque funciona en CPU.

### Pasos
1.  Clona el repositorio.
2.  (Opcional) Crea y activa un entorno virtual.
3.  Instala las dependencias necesarias ejecutando:
    ```bash
    pip install -r requirements.txt
    ```

## Estructura de Directorios
El proyecto consta de los siguientes archivos y carpetas:

### Raíz
- `EmotionRecord.ipynb`: Notebook principal para la ejecución del sistema.
- `requirements.txt`: Lista de dependencias del proyecto.
- `LICENSE`: Licencia del software.
- `resultados.csv`: Archivo generado con el historial de detecciones.

### Carpetas
- `src/`: Código fuente auxiliar.
    - `EmotionModelTraining.ipynb`: Notebook para el entrenamiento del modelo.
    - `model.py`: Definición de la arquitectura del modelo (`MyModel`).
    - `dataset.py`: Gestión del dataset.
    - `utils.py`: Funciones de utilidad.
- `models/`: 
    - `model.pth`: Archivo del modelo entrenado.
- `data/`:
    - `identidades/`: Carpeta donde se guardan las caras nuevas detectadas.
    - `input/`: Carpeta para colocar las imágenes nuevas para procesar en lote (formato: `Nombre.jpg`).

## Uso

El proyecto se ejecuta principalmente a través del notebook `EmotionRecord.ipynb`.

### 1. Configuración Inicial
Importante ejecutar las dos primeras celdas del notebook para:
- Cargar las librerías.
- Inicializar el modelo de emociones (`MyModel`).
- Cargar las caras conocidas desde `data/identidades` usando la función `iniciarCaras` (segunda celda).

### 2. Modos de Ejecución

#### A. Reconocimiento en Lote (Tercera celda)
Procesa todas las imágenes en la carpeta `data/input`.
- Detecta caras en cada imagen.
- Identifica si la persona es conocida. Si no, la registra como nueva en `data/identidades`.
- Predice la emoción.
- Guarda el resultado en `resultados.csv` y muestra la salida en pantalla.

#### B. Reconocimiento en Tiempo Real (Cuarta celda)
Activa la cámara web para detección en vivo.
- Muestra el video con recuadros sobre las caras, indicando nombre (si no esta identificada se nombra desconocido), emoción y confianza.
- Registra el estado emocional en el CSV cada 5 segundos (configurable) si la persona es identificada, en caso contrario no se registra ninguna información.
- Presiona `q` para detener la captura.

## Salida
El archivo `resultados.csv` contendrá:
- **Nombre**: Identidad detectada.
- **Fecha**: Timestamp de la detección.
- **Emoción**: Emoción clasificada.
