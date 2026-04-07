from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

# 📁 Carpetas de almacenamiento
UPLOAD_FOLDER = "videos"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 📺 Servir videos procesados para que se puedan ver en el navegador
app.mount("/output", StaticFiles(directory=OUTPUT_FOLDER), name="output")

# 🧠 Configuración estable de MediaPipe para Python 3.11
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 📐 Función matemática para calcular el ángulo de la rodilla
def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return int(angle)

# --- 🎨 INTERFAZ WEB PROFESIONAL ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analizador de Patinaje</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
            .card { background: white; padding: 2.5rem; border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); text-align: center; width: 90%; max-width: 450px; }
            .icon { font-size: 4rem; margin-bottom: 10px; }
            h1 { color: #1a2a6c; margin-bottom: 10px; font-size: 1.8rem; }
            p { color: #555; line-height: 1.6; margin-bottom: 25px; }
            .file-input { margin-bottom: 25px; }
            .btn { background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d); color: white; border: none; padding: 15px 30px; border-radius: 30px; cursor: pointer; font-size: 1.1rem; font-weight: bold; width: 100%; transition: transform 0.2s; }
            .btn:hover { transform: scale(1.02); }
            .footer { margin-top: 30px; font-size: 0.8rem; color: #888; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="icon">🛼</div>
            <h1>Analizador Pro</h1>
            <p>Sube el video de entrenamiento para obtener el análisis biomecánico de la rodilla.</p>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <div class="file-input">
                    <input name="file" type="file" accept="video/*" required>
                </div>
                <button type="submit" class="btn">INICIAR ANÁLISIS</button>
            </form>
            <div class="footer">Sistema Optimizado para Clubes de Patinaje</div>
        </div>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # 📥 Guardar el video original
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🎥 Procesamiento del video con OpenCV
    cap = cv2.VideoCapture(file_path)
    output_path = os.path.join(OUTPUT_FOLDER, f"procesado_{file.filename}")
    
    # Configurar el formato de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frames_analizados = 0
    angulos = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            frames_analizados += 1
            landmarks = result.pose_landmarks.landmark

            # 🔥 Puntos clave: Cadera (24), Rodilla (26), Tobillo (28)
            hip = [landmarks[24].x, landmarks[24].y]
            knee = [landmarks[26].x, landmarks[26].y]
            ankle = [landmarks[28].x, landmarks[28].y]

            angulo = calcular_angulo(hip, knee, ankle)
            angulos.append(angulo)

            # Dibujar esqueleto y ángulo en el video
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f'Angulo: {angulo} deg', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
        
        out.write(frame)

    cap.release()
    if out:
        out.release()

    # 📊 Resultados finales
    promedio = int(sum(angulos) / len(angulos)) if angulos else 0
    
    # URL para ver el video (Render usa HTTPS)
    BASE_URL = "https://analizador-patinaje.onrender.com"
    video_url = f"{BASE_URL}/output/procesado_{file.filename}"

    return {
        "estado": "Éxito",
        "deportista": file.filename,
        "angulo_promedio_rodilla": promedio,
        "video_analizado": video_url
    }
