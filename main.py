from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

# 📁 Configuración de Carpetas
UPLOAD_FOLDER = "videos"
OUTPUT_FOLDER = "output"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# 🔗 Montar carpetas para que sean accesibles desde la web
# Esto permite que la web vea el logo y los videos procesados
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_FOLDER), name="output")

# 🧠 Configuración MediaPipe (Estable para Python 3.11)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return int(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

# --- 🏁 INTERFAZ IDÉNTICA A LA APP DE FLUTTER ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🏁 Analizador de Patinaje</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background-color: #fcfaff; margin: 0; display: flex; justify-content: center; align-items: flex-start; min-height: 100vh; padding-top: 40px; }
            .container { width: 90%; max-width: 400px; text-align: center; }
            
            /* Título superior con la bandera */
            .header-text { font-size: 1.3rem; color: #333; margin-bottom: 25px; font-weight: 400; display: flex; align-items: center; justify-content: center; gap: 10px; }
            
            /* Caja del Logo (GitHub static/logo.jpeg) */
            .logo-box { width: 100%; background: white; border-radius: 10px; padding: 10px; margin-bottom: 15px; }
            .logo-box img { width: 100%; height: auto; border-radius: 5px; display: block; }
            
            /* Nombre del Club */
            h1.club-title { font-size: 2rem; font-weight: bold; color: #1a1a1a; margin-top: 0; margin-bottom: 40px; }

            /* Estilo de botones redondeados lila (como en tu celular) */
            .btn-flutter { 
                background-color: #f3f0f7; 
                color: #5d4d8a; 
                border: 1px solid #e0daeb;
                padding: 14px 25px; 
                border-radius: 30px; 
                margin-bottom: 15px; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                gap: 15px;
                font-size: 1.1rem;
                cursor: pointer;
                width: 100%;
                text-decoration: none;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: background 0.2s;
            }
            .btn-flutter:hover { background-color: #ede9f2; }
            
            /* Iconos simulados con emoji para rapidez */
            .icon { font-size: 1.2rem; }

            input[type="file"] { display: none; }
            .file-label { cursor: pointer; }

            .footer-info { margin-top: 20px; font-size: 0.8rem; color: #aaa; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-text">🏁 Analizador de Patinaje</div>
            
            <div class="logo-box">
                <img src="/static/logo.jpeg" alt="Club Skate Wheels">
            </div>
            
            <h1 class="club-title">Club Skate Wheels</h1>

            <form action="/upload" enctype="multipart/form-data" method="post">
                <label class="btn-flutter file-label">
                    <span class="icon">📁</span>
                    <span>Elegir video</span>
                    <input name="file" type="file" accept="video/*" required>
                </label>
                
                <button type="submit" class="btn-flutter">
                    <span class="icon">☁️</span>
                    <span>Analizar</span>
                </button>
            </form>

            <div class="footer-info">Sistema de Análisis Biomecánico v1.0</div>
        </div>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # 📥 1. Guardar el video que llega
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🎥 2. Procesar con OpenCV y MediaPipe
    cap = cv2.VideoCapture(file_path)
    output_path = os.path.join(OUTPUT_FOLDER, f"procesado_{file.filename}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    angulos = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            # Puntos: Cadera(24), Rodilla(26), Tobillo(28)
            hip = [landmarks[24].x, landmarks[24].y]
            knee = [landmarks[26].x, landmarks[26].y]
            ankle = [landmarks[28].x, landmarks[28].y]

            ang = calcular_angulo(hip, knee, ankle)
            angulos.append(ang)

            # Dibujar el esqueleto
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f'Angulo: {ang} deg', (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
        out.write(frame)

    cap.release()
    if out: out.release()

    # 📊 3. Calcular promedio y responder
    promedio = int(sum(angulos) / len(angulos)) if angulos else 0
    video_url = f"https://analizador-patinaje.onrender.com/output/procesado_{file.filename}"

    return {
        "status": "ready",
        "club": "Club Skate Wheels",
        "average_angle": promedio,
        "video_url": video_url
    }
