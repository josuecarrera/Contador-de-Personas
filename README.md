# Contador-de-Personas
El siguiente proyecto es un contador de personas usando Python, este proyecto surge para mejorar nuestros conocimentos de Deep Learning with Python
# Detección de Movimiento con Python utilizando OpenCV y Tkinter

Este proyecto tiene como objetivo implementar un sistema de conteo y de detección de movimiento utilizando Python y las siguientes librerías:

- `tkinter`: Para la creación de interfaces gráficas.
- `PIL`: Para trabajar con imágenes.
- `collections`: Para utilizar el tipo de datos `defaultdict`.
- `pathlib`: Para trabajar con rutas de archivos.
- `OpenCV (cv2)`: Para el procesamiento de imágenes.
- `numpy`: Para cálculos numéricos.
- `shapely`: Para manipulación de geometría plana.
- `ultralytics`: Para la detección de objetos con YOLO.

## Descripción del Proyecto

El sistema desarrollado es capaz de detectar movimiento en una secuencia de video utilizando el algoritmo de substracción de fondo. Cuando se detecta movimiento, se resalta la región donde ocurrió en tiempo real.

## Instalación de Dependencias

Es necesario instalar las siguientes librerías de Python para ejecutar el proyecto:

```bash
pip install numpy opencv-python-headless pillow shapely 
```

Para la instalacion de ultralytics ejecutar los siguiente comandos
Install the ultralytics package from PyPI
```bash
pip install ultralytics
```
Install the ultralytics package from GitHub
```bash
pip install git+https://github.com/ultralytics/ultralytics.git@main
```
Para mayor información de las librerias consulte

[ultralytics](https://docs.ultralytics.com/es/quickstart/#install-ultralytics)

# Estructura del Proyecto

El proyecto consta de los siguientes archivos:

   - `main.py`: Contiene la lógica principal del programa.
# Ejecución del Programa

Para ejecutar el programa, simplemente ejecute el script main.py:
```bash
  python main.py
```
# Funcionamiento del Programa

Al ejecutar el programa, se abrirá una ventana de interfaz gráfica que mostrará el video en tiempo real. El algoritmo de detección de movimiento se aplicará automáticamente al video. Las regiones donde se detecta movimiento se resaltarán en la ventana de visualización.
