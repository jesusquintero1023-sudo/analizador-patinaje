import os
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI()

# 1. CORS PARA APP MÓVIL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CONFIGURACIÓN DE CARPETAS
STATIC_DIR = "static"
VIDEOS_DIR = os.path.join(STATIC_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# CONFIGURACIÓN MEDIAPIPE (Modelo liviano para Render)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Club Skate Wheels | Analizador</title>
        <style>
            :root { --primary: #5D4D8A; --bg: #FCFAFF; }
            body { font-family: 'Segoe UI', sans-serif; background: var(--bg); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 20px; }
            .card { background: white; padding: 40px; border-radius: 30px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); text-align: center; width: 100%; max-width: 400px; }
            .logo-container { width: 100px; height: 100px; margin: 0 auto 20px; border-radius: 50%; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border: 3px solid white; }
            .logo-container img { width: 100%; height: 100%; object-fit: cover; }
            h1 { color: #111; font-size: 24px; margin: 0; }
            p { color: #777; font-size: 14px; margin-bottom: 30px; }

            /* Estilo Input File */
            .file-input-wrapper { position: relative; margin-bottom: 20px; }
            input[type="file"] { position: absolute; left: 0; top: 0; opacity: 0; width: 100%; height: 100%; cursor: pointer; }
            .custom-file-btn { background: #F3F0F7; color: var(--primary); padding: 15px; border-radius: 15px; display: block; font-weight: bold; border: 2px dashed #D1C8E0; transition: 0.3s; }
            
            #status-msg { font-size: 13px; color: #2D9B58; font-weight: bold; margin-bottom: 15px; display: none; }

            /* Botón Analizar con Animaciones */
            .btn-submit { background: var(--primary); color: white; border: none; padding: 18px; width: 100%; border-radius: 20px; font-weight: bold; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 10px; transition: 0.3s; }
            .btn-submit:disabled { background: #CCC; cursor: not_allowed; }
            
            .spinner { display: none; width: 18px; height: 18px; border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: #fff; animation: spin 1s linear infinite; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .dots:after { content: '.'; animation: dots 1.5s steps(5, end) infinite; }
            @keyframes dots { 0%, 20% { content: '.'; } 40% { content: '..'; } 60% { content: '...'; } 80%, 100% { content: ''; } }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="logo-container"><img src="/static/logo.jpeg" alt="Logo"></div>
            <h1>Club Skate Wheels</h1>
            <p>Análisis Biomecánico Híbrido</p>
            <form action="/analyze" method="post" enctype="multipart/form-data" id="uploadForm">
                <div class="file-input-wrapper">
                    <span class="custom-file-btn" id="file-label">📁 Seleccionar Video</span>
                    <input type="file" name="file" id="video-file" accept="video/*" required>
                </div>
                <div id="status-msg">✓ Video listo para analizar</div>
                <button type="submit" class="btn-submit" id="btn-ana" disabled>
                    <span id="btn-text">ANALIZAR EN WEB</span>
                    <div class="spinner" id="btn-spinner"></div>
                </button>
            </form>
        </div>
        <script>
            const fileInput = document.getElementById('video-file');
            const fileLabel = document.getElementById('file-label');
            const statusMsg = document.getElementById('status-msg');
            const btnAna = document.getElementById('btn-ana');
            const btnText = document.getElementById('btn-text');
            const btnSpinner = document.getElementById('btn-spinner');

            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) {
                    fileLabel.innerText = "🎥 Video Cargado";
                    statusMsg.style.display = "block";
                    btnAna.disabled = false;
                }
            });

            document.getElementById('uploadForm').addEventListener('submit', function() {
                btnAna.disabled = true;
                btnText.innerHTML = 'PROCESANDO<span class="dots"></span>';
                btnSpinner.style.display = 'inline-block';
            });
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    unique_id = uuid.uuid4().hex
    in_path = f"temp_{unique_id}.mp4"
    out_filename = f"out_{unique_id}.mp4"
    out_path = os.path.join(VIDEOS_DIR, out_filename)

    with open(in_path, "wb") as buffer:
        buffer.write(await file.read())

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # CODEC AVC1 (H.264) - EL ÚNICO COMPATIBLE CON WEB Y MÓVIL
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    angles_r, angles_s = [], []
    frame_count = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=0) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # DIBUJAR ESQUELETO
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                lm = results.pose_landmarks.landmark
                p_hip = [lm[24].x, lm[24].y]
                p_knee = [lm[26].x, lm[26].y]
                p_ankle = [lm[28].x, lm[28].y]
                p_shoulder = [lm[12].x, lm[12].y]

                ang_r = calculate_angle(p_hip, p_knee, p_ankle)
                ang_s = calculate_angle(p_shoulder, p_hip, p_knee)
                
                angles_r.append(ang_r)
                angles_s.append(ang_s)

                cv2.putText(frame, f"R: {int(ang_r)}", (50, 50), 1, 2, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()
    if os.path.exists(in_path): os.remove(in_path)

    avg_r = round(np.mean(angles_r), 1) if angles_r else 0
    avg_s = round(np.mean(angles_s), 1) if angles_s else 0
    duration = round(frame_count / fps, 2)
    video_url = f"/static/videos/{out_filename}"

    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe Técnico - Skate Wheels</title>
        <style>
            :root {{ --primary: #5D4D8A; --bg: #F8F9FE; }}
            body {{ font-family: 'Segoe UI', sans-serif; background: var(--bg); margin: 0; display: flex; justify-content: center; padding: 20px; }}
            .container {{ background: white; width: 100%; max-width: 450px; border-radius: 30px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center; }}
            .logo {{ width: 70px; margin-bottom: 10px; border-radius: 50%; }}
            h2 {{ color: var(--primary); margin-bottom: 25px; text-transform: uppercase; font-size: 18px; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px; }}
            .card {{ background: #F3F0F7; padding: 15px; border-radius: 20px; }}
            .card-label {{ font-size: 11px; color: #888; text-transform: uppercase; font-weight: bold; display: block; }}
            .card-value {{ font-size: 18px; color: var(--primary); font-weight: bold; }}
            video {{ width: 100%; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); background: #000; margin-bottom: 20px; }}
            .btn-back {{ display: inline-block; text-decoration: none; color: var(--primary); font-weight: bold; font-size: 14px; padding: 10px 20px; border-radius: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <img src="/static/logo.jpeg" class="logo">
            <h2>Informe de Análisis</h2>
            <div class="grid">
                <div class="card"><span class="card-label">Rodilla</span><span class="card-value">{avg_r}°</span></div>
                <div class="card"><span class="card-label">Sentadilla</span><span class="card-value">{avg_s}°</span></div>
                <div class="card"><span class="card-label">Tiempo</span><span class="card-value">{duration}s</span></div>
                <div class="card"><span class="card-label">Estado</span><span class="card-value">Éxito</span></div>
            </div>
            <video controls playsinline autoplay>
                <source src="{video_url}" type="video/mp4">
            </video>
            <br><a href="/" class="btn-back">← Nuevo Análisis</a>
        </div>
    </body>
    </html>
    """)
