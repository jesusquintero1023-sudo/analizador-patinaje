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

# 1. PERMITIR CONEXIONES (Importante para Flutter)
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

# --- INTERFAZ WEB CON EL DISEÑO LILA DE FLUTTER ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Club Skate Wheels - Web</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background-color: #FCFAFF; 
                    margin: 0; 
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    min-height: 100vh; 
                }
                .container { 
                    background: white; 
                    padding: 40px; 
                    border-radius: 30px; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.05); 
                    text-align: center; 
                    width: 90%; 
                    max-width: 400px;
                    border: 1px solid #E0DAEB;
                }
                .logo-container {
                    background: white;
                    padding: 10px;
                    border-radius: 20px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                    display: inline-block;
                    margin-bottom: 20px;
                }
                h1 { color: #1A1A1A; font-size: 24px; margin-bottom: 30px; }
                .btn { 
                    background: #F3F0F7; 
                    color: #5D4D8A; 
                    border: 1px solid #E0DAEB; 
                    padding: 15px 30px; 
                    border-radius: 30px; 
                    cursor: pointer; 
                    font-weight: bold; 
                    width: 100%;
                    font-size: 16px;
                    transition: 0.3s;
                    text-decoration: none;
                    display: inline-block;
                }
                .btn:hover { background: #EBE4FF; }
                .btn-primary {
                    background: #5D4D8A;
                    color: white;
                    margin-top: 15px;
                }
                input[type="file"] { margin: 20px 0; font-size: 14px; color: #5D4D8A; }
                p { color: #666; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="logo-container">
                    <img src="/static/logo.jpeg" height="120" alt="Logo">
                </div>
                <h1>Club Skate Wheels</h1>
                <p>Analizador Biomecánico de Patinaje</p>
                
                <form action="/analyze" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept="video/*" required>
                    <button type="submit" class="btn btn-primary">INICIAR ANÁLISIS EN PC</button>
                </form>
                
                <div style="margin-top: 30px;">
                    <p style="font-size: 10px; color: #BBB;">Sistema de Análisis v1.5 - UTS 2026</p>
                </div>
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

    # AJUSTE PARA VIDEO VERTICAL (Rotación y dimensiones)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (height, width)) 

    angles_rodilla = []
    angles_sentadilla = []
    frames_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames_count += 1
        
        # CORRECCIÓN DE ROTACIÓN PARA CELULARES
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark
            
            # Puntos: Hombro(12), Cadera(24), Rodilla(26), Tobillo(28)
            shoulder = [lm[12].x, lm[12].y]
            hip = [lm[24].x, lm[24].y]
            knee = [lm[26].x, lm[26].y]
            ankle = [lm[28].x, lm[28].y]

            ang_r = calculate_angle(hip, knee, ankle)
            ang_s = calculate_angle(shoulder, hip, knee)
            
            angles_rodilla.append(ang_r)
            angles_sentadilla.append(ang_s)

            # Dibujado de grados en el video procesado
            cv2.putText(image, f"Rodilla: {int(ang_r)} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Postura: {int(ang_s)} deg", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        out.write(image)

    cap.release()
    out.release()
    if os.path.exists(input_path): os.remove(input_path)

    return {
        "average_angle": round(np.mean(angles_rodilla), 1) if angles_rodilla else 0,
        "squat_angle": round(np.mean(angles_sentadilla), 1) if angles_sentadilla else 0,
        "max_angle": round(np.max(angles_rodilla), 1) if angles_rodilla else 0,
        "min_angle": round(np.min(angles_rodilla), 1) if angles_rodilla else 0,
        "duration": round(frames_count / fps, 2),
        "video_url": f"/static/videos/{output_filename}"
    }
