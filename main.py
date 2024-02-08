import tkinter as tk # Importamos el módulo tkinter para crear interfaces gráficas
from tkinter import ttk # Importamos el módulo tkinter para crear interfaces gráficas
from PIL import Image, ImageTk # Importamos las clases Image e ImageTk del módulo PIL para trabajar con imágenes
from collections import defaultdict # Importamos el tipo de datos defaultdict del módulo collections
from pathlib import Path # Importamos la clase Path del módulo pathlib para trabajar con rutas de archivos
import cv2 # Importamos la biblioteca OpenCV para procesamiento de imágenes
import numpy as np # Importamos numpy para cálculos numéricos
from shapely.geometry import Polygon, Point # Importamos las clases Polygon y Point del módulo geometry en la biblioteca shapely
from ultralytics import YOLO # Importamos la clase YOLO y utilidades de trazado de la biblioteca ultralytics
from ultralytics.utils.plotting import Annotator, colors

# Diccionario para mantener el historial de seguimiento de cada objeto
track_history = defaultdict(list)

# Definición de regiones de conteo con polígonos, nombres, contadores, etc.
counting_regions = [
    {
        "name": "ZONA 1", #Nombre de la region 
        "polygon": Polygon([(10, 240), (10, 170), (450, 200), (440, 350), (200, 260)]),  #Definición del polígono que delimita la región
        "counts": 0,  #Inicialización del contador de objetos en la región
        "dragging": False, # Estado de arrastre de la región
        "region_color": (255, 42, 4),  # Valor RGB  de la region
        "text_color": (255, 255, 255),  # Color RGB del texto que indica la región
    },
    {
        "name": "ZONA 2", #Nombre de la region 
        "polygon": Polygon([(440, 350), (700, 250), (755, 280), (755, 480)]),  #Definición del polígono que delimita la región
        "counts": 0, #Inicialización del contador de objetos en la región
        "dragging": False,# Estado de arrastre de la región
        "region_color": (37, 255, 225),  # Valor RGB de la region
        "text_color": (0, 0, 0),  # Color RGB del texto que indica la región
    },
    {
        "name": "ZONA 3", #Nombre de la region 
        "polygon": Polygon([(755, 150), (755, 70), (450, 200), (700, 250), (680, 210)]),  #Definición del polígono que delimita la región
        "counts": 0, #Inicialización del contador de objetos en la región
        "dragging": False,# Estado de arrastre de la región
        "region_color": (100, 200, 50),  # Valor RGB de la region
        "text_color": (255, 255, 255),  # Color RGB del texto que indica la región
    },
    {
        "name": "ZONA 4", #Nombre de la region 
        "polygon": Polygon([(455, 205), (685, 250), (445, 345)]),  #Definición del polígono que delimita la región
        "counts": 0, #Inicialización del contador de objetos en la región
        "dragging": False,# Estado de arrastre de la región
        "region_color": (80, 130, 180),  # Valor RGB de la region
        "text_color": (255, 255, 255),  # Color RGB del texto que indica la región
    },
]

# Función para obtener la región que contiene un punto
def get_region_at_point(point):
    # Iteramos sobre cada región en la lista counting_regions
    for region in counting_regions:
        # Verificamos si el polígono de la región contiene el punto dado
        if region["polygon"].contains(Point(point)):
        # Si la región contiene el punto, devolvemos la región
            return region
        # Si ninguna región contiene el punto, devolvemos None
    return None

# Función para actualizar el conteo de una región
def update_region_count(region, increment=True):
    # Si se debe incrementar el conteo
    if increment:
        # Incrementa el conteo de la región en 1
        region["counts"] += 1
    else:
        # Si no se debe incrementar, decrementa el conteo de la región en 1, asegurándose de que el conteo no sea negativo
        region["counts"] = max(0, region["counts"] - 1)

# La siguiente funcion permite la ejecución del video y muestra el procesamiento por medio de la deteccion de las imagenes
def run_video():
    # Especificaciones para la detección de objetos en el video
    weights = "yolov8n.pt"
    source = "video.mp4"
    device = "cpu"
    view_img = True
    classes = [0]
    # Contador de fotogramas del video
    vid_frame_count = 0

    # Comprueba la existencia de la ruta de origen
    if not Path(source).exists():
        raise FileNotFoundError(f"No existe '{source}'")

    # Configura el modelo YOLO
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extrae los nombres de las clases
    names = model.model.names

    # Configuración del video
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Itera sobre los fotogramas del video
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extrae los resultados de la detección de objetos
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            # Extrae los cuadros delimitadores de los objetos detectados
            boxes = results[0].boxes.xyxy.cpu()
            # Extrae los IDs de seguimiento de los objetos detectados
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # Extrae las clases de los objetos detectados
            clss = results[0].boxes.cls.cpu().tolist()

            person_counter = 1

            annotator = Annotator(frame, line_width=2, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):

                if cls == 0:  # Suponiendo que la clase '0' corresponde a personas
                    if track_id not in track_history:
                        track_history[track_id] = []
                    if track_id not in track_history:
                        track_history[track_id].append(person_counter)
                        person_counter += 1                
                annotator.box_label(box, f"{str(names[cls])}a ID: {track_id}", color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Centro del cuadro delimitador

                track = track_history[track_id]  # Trama del historial de seguimiento
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=2)

                # Comprueba si la detección está dentro de alguna región
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Dibuja las regiones (Polígonos/Rectángulos)  y cuenta los objetos detectados dentro de ellas
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=2)
        # Muestra el video procesado si view_img es True
        if view_img:
            cv2.imshow("Procesamiento del Video", frame)

        for region in counting_regions:  # Reinicializa el contador de objetos para cada región
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"): # Termina el procesamiento si se presiona la tecla 'q'
            break
    # Limpieza y liberación de recursos
    del vid_frame_count
    videocapture.release()
    cv2.destroyAllWindows()

def create_ui():
    # Crear la ventana principal de la interfaz gráfica
    root = tk.Tk()
    root.title("Proyecto de Deteccion de Personas") # Título de la ventana
    # Crear un control de pestañas para organizar la interfaz
    tab_control = ttk.Notebook(root)
    # Crear los frames para cada pestaña
    tab1 = ttk.Frame(tab_control)
    tab2 = ttk.Frame(tab_control)
    # Agregar las pestañas al control de pestañas
    tab_control.add(tab1, text='PORTADA') # Pestaña de la portada del proyecto
    tab_control.add(tab2, text='PROCESAMIENTO')  # Agregar el nuevo tab para el video
    # Empaquetar el control de pestañas para mostrarlo en la ventana
    tab_control.pack(expand=1, fill='both')

    # Cargar y mostrar la imagen en el primer tab
    image = Image.open("portada.jpg")
    photo = ImageTk.PhotoImage(image)
    label1 = tk.Label(tab1, image=photo)
    label1.image = photo  # Mantener una referencia para evitar que la imagen sea eliminada por el recolector de basura
    label1.pack(padx=10, pady=10)

    # Botón para iniciar la reproducción del video en el tercer tab
    button1 = tk.Button(tab2, text="Deteccion de Personas", command=run_video)
    button1.pack(padx=10, pady=10)

    # Reproductor de video en el tercer tab
    video_frame = tk.Frame(tab2)
    video_frame.pack(padx=10, pady=10)

    video_player = cv2.VideoCapture("video.mp4")
    video_label = tk.Label(video_frame)
    video_label.pack()
    # Función para mostrar el video en el reproductor
    def show_video():
        ret, frame = video_player.read()
        if ret:
             # Convertir el fotograma de BGR a RGB y crear una imagen de Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))  # Ajustar el tamaño del video si es necesario
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            video_label.img = img
            video_label.config(image=img)
            video_label.after(10, show_video)  # Mostrar el siguiente fotograma
        else:
            video_player.release() # Liberar el recurso del reproductor de video
            video_label.config(image=None) # Limpiar la etiqueta del video al finalizar

    # Cuando se selecciona el tab del video, comenzar a mostrar el video
    tab_control.bind("<<NotebookTabChanged>>", lambda event: show_video() if tab_control.index("current") == 1 else None)
    # Iniciar el bucle principal de la interfaz gráfica
    root.mainloop()

if __name__ == "__main__":
    create_ui()