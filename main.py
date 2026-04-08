import cv2
import mediapipe as mp
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI()

# 1. PERMITIR CONEXIONES (Para que el celular no falle)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CONFIGURACIÓN DE MEDIAPIPE
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 3. CARPETAS DE SALIDA
OUTPUT_DIR = "static/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- INTERFAZ HTML PARA EL PC (LO QUE HABÍAMOS QUITADO) ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Club Skate Wheels - Analizador</title>
            <style>
                body { font-family: 'Arial', sans-serif; text-align: center; padding: 40px; background-color: #F8F7FF; }
                .card { background: white; padding: 30px; border-radius: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); display: inline-block; max-width: 500px; }
                h2 { color: #5D4D8A; }
                .upload-btn { background: #5D4D8A; color: white; border: none; padding: 12px 25px; border-radius: 30px; cursor: pointer; font-weight: bold; }
                input[type="file"] { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="card">
                <img src="/static/logo.jpeg" style="height: 100px; margin-bottom: 20px;" onerror="this.style.display='none'">
                <h2>🏁 Analizador Biomecánico</h2>
                <p>Club Skate Wheels - Proyecto de Grado</p>
                <form action="/analyze" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept="video/*"><br>
                    <button type="submit" class="upload-btn">ANALIZAR VIDEO EN PC</button>
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1]
    input_path = f"temp_{uuid.uuid4()}.{file_extension}"
    output_filename = f"analyzed_{uuid.uuid4()}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    angles = []
    frames_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames_count += 1
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # Puntos clave: Cadera(24), Rodilla(26), Tobillo(28)
            h = [landmarks[24].x, landmarks[24].y]
            k = [landmarks[26].x, landmarks[26].y]
            a = [landmarks[28].x, landmarks[28].y]

            ang = calculate_angle(h, k, a)
            angles.append(ang)

            cv2.putText(image, f"{int(ang)} deg", 
                        tuple(np.multiply(k, [width, height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(image)

    cap.release()
    out.release()
    if os.path.exists(input_path): os.remove(input_path)

    # DATOS PARA FLUTTER Y PC
    return {
        "average_angle": round(np.mean(angles), 1) if angles else 0,
        "max_angle": round(np.max(angles), 1) if angles else 0,
        "min_angle": round(np.min(angles), 1) if angles else 0,
        "duration": round(frames_count / fps, 2),
        "video_url": f"/static/videos/{output_filename}"
    }
