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

# Permitir conexiones externas (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de MediaPipe para Biomecánica
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Carpetas de almacenamiento
OUTPUT_DIR = "static/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- VISTA DE INICIO CON CARGA ANIMADA ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Club Skate Wheels</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                body { font-family: 'Segoe UI', sans-serif; background-color: #FCFAFF; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
                .card { background: white; padding: 40px; border-radius: 35px; box-shadow: 0 15px 35px rgba(0,0,0,0.05); text-align: center; width: 90%; max-width: 400px; }
                .btn { background: #F3F0F7; color: #5D4D8A; border: none; padding: 15px; border-radius: 30px; cursor: pointer; font-weight: bold; width: 100%; margin-bottom: 12px; display: flex; align-items: center; justify-content: center; gap: 10px; text-decoration: none; transition: 0.3s; }
                .btn-primary { background: #5D4D8A; color: white; margin-top: 15px; box-shadow: 0 4px 12px rgba(93, 77, 138, 0.2); }
                input[type="file"] { display: none; }
                h1 { font-size: 24px; margin: 15px 0; color: #1A1A1A; }
                #overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255, 255, 255, 0.9); z-index: 1000; flex-direction: column; justify-content: center; align-items: center; }
                .spinner { width: 50px; height: 50px; border: 5px solid #F3F0F7; border-top: 5px solid #5D4D8A; border-radius: 50%; animation: spin 1s linear infinite; }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            </style>
        </head>
        <body>
            <div id="overlay">
                <div class="spinner"></div>
                <p style="color: #5D4D8A; font-weight: bold; margin-top: 15px;">Analizando técnica...</p>
            </div>
            <div class="card">
                <img src="/static/logo.jpeg" height="100" style="border-radius: 20px;">
                <h1>Club Skate Wheels</h1>
                <p style="color: #888; font-size: 14px;">Analizador Biomecánico</p>
                <form action="/analyze" enctype="multipart/form-data" method="post" onsubmit="document.getElementById('overlay').style.display='flex'">
                    <label class="btn" for="u"><i class="fas fa-file-video"></i> Seleccionar Video</label>
                    <input id="u" name="file" type="file" accept="video/*" required>
                    <button type="submit" class="btn btn-primary"><i class="fas fa-magic"></i> INICIAR ANÁLISIS</button>
                </form>
            </div>
        </body>
    </html>
    """

# --- PROCESAMIENTO Y RESULTADOS COLORIDOS ---
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_video(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1]
    input_path = f"temp_{uuid.uuid4()}.{file_extension}"
    output_filename = f"analyzed_{uuid.uuid4()}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Invertimos dimensiones para el video vertical
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width)) 

    angles_rodilla, angles_sentadilla = [], []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Rotar a vertical
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Hombro(12), Cadera(24), Rodilla(26), Tobillo(28)
            p = [[lm[i].x, lm[i].y] for i in [12, 24, 26, 28]]
            ang_r = calculate_angle(p[1], p[2], p[3])
            ang_s = calculate_angle(p[0], p[1], p[2])
            angles_rodilla.append(ang_r)
            angles_sentadilla.append(ang_s)
            cv2.putText(frame, f"RODILLA: {int(ang_r)} deg", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

    cap.release(); out.release()
    if os.path.exists(input_path): os.remove(input_path)

    avg_r = round(np.mean(angles_rodilla), 1) if angles_rodilla else 0
    avg_s = round(np.mean(angles_sentadilla), 1) if angles_sentadilla else 0
    dur = round(len(angles_rodilla)/fps, 2) if angles_rodilla else 0
    video_url = f"/static/videos/{output_filename}"

    return f"""
    <html>
        <head>
            <title>Resultados</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #FCFAFF; padding: 20px; display: flex; justify-content: center; }}
                .res-card {{ background: white; padding: 30px; border-radius: 35px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); max-width: 450px; text-align: center; border: 1px solid #F0EBF9; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }}
                .item {{ background: #F3F0F7; padding: 15px; border-radius: 20px; border: 1px solid #E6DFF3; }}
                .item i {{ color: #5D4D8A; font-size: 20px; }}
                .val {{ font-size: 20px; font-weight: bold; color: #5D4D8A; display: block; }}
                .lbl {{ font-size: 10px; color: #888; font-weight: bold; }}
                video {{ width: 100%; border-radius: 20px; margin-top: 20px; border: 2px solid #5D4D8A; }}
                .back {{ color: #5D4D8A; text-decoration: none; font-weight: bold; display: block; margin-top: 20px; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="res-card">
                <h2 style="color: #5D4D8A;">INFORME TÉCNICO</h2>
                <div class="grid">
                    <div class="item"><i class="fas fa-microscope"></i><span class="lbl">RODILLA</span><span class="val">{avg_r}°</span></div>
                    <div class="item"><i class="fas fa-accessibility"></i><span class="lbl">SENTADILLA</span><span class="val">{avg_s}°</span></div>
                    <div class="item"><i class="fas fa-clock"></i><span class="lbl">DURACIÓN</span><span class="val">{dur}s</span></div>
                    <div class="item"><i class="fas fa-check-circle"></i><span class="lbl">ESTADO</span><span class="val">OK</span></div>
                </div>
                <video controls autoplay loop><source src="{video_url}" type="video/mp4"></video>
                <a href="/" class="back"><i class="fas fa-arrow-left"></i> Volver a analizar</a>
            </div>
        </body>
    </html>
    """
