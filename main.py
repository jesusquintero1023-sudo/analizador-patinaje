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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de carpetas estáticas
STATIC_DIR = "static"
VIDEOS_DIR = os.path.join(STATIC_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Inicialización de MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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
        <title>Skate Wheels | Analizador</title>
        <style>
            :root { --primary: #5D4D8A; --bg: #FCFAFF; }
            body { font-family: 'Segoe UI', sans-serif; background: var(--bg); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 20px; }
            .card { background: white; padding: 40px; border-radius: 30px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); text-align: center; width: 100%; max-width: 400px; }
            .logo-container { width: 90px; height: 90px; margin: 0 auto 20px; border-radius: 50%; overflow: hidden; border: 3px solid var(--primary); }
            .logo-container img { width: 100%; height: 100%; object-fit: cover; }
            .btn-submit { background: var(--primary); color: white; border: none; padding: 18px; width: 100%; border-radius: 20px; font-weight: bold; cursor: pointer; transition: 0.3s; display: flex; align-items: center; justify-content: center; gap: 10px; }
            .btn-submit:disabled { background: #ccc; }
            .spinner { display: none; width: 18px; height: 18px; border: 3px solid #ffffff55; border-radius: 50%; border-top-color: #fff; animation: spin 1s linear infinite; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .file-label { display: block; padding: 15px; border: 2px dashed #D1C8E0; border-radius: 15px; margin-bottom: 20px; cursor: pointer; color: var(--primary); font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="logo-container"><img src="/static/logo.jpeg"></div>
            <h1>Analizador Híbrido</h1>
            <p>Club Skate Wheels</p>
            <form action="/analyze" method="post" enctype="multipart/form-data" id="uploadForm">
                <label for="v" class="file-label" id="L">📁 Seleccionar Video</label>
                <input type="file" name="file" id="v" accept="video/*" hidden required>
                <button type="submit" class="btn-submit" id="B" disabled>
                    <span id="T">INICIAR ANÁLISIS</span>
                    <div class="spinner" id="S"></div>
                </button>
            </form>
        </div>
        <script>
            const v = document.getElementById('v');
            const B = document.getElementById('B');
            const L = document.getElementById('L');
            v.onchange = () => { if(v.files[0]){ L.innerText="🎥 Video Listo"; B.disabled=false; } };
            document.getElementById('uploadForm').onsubmit = () => { 
                B.disabled=true; document.getElementById('T').innerText="PROCESANDO..."; 
                document.getElementById('S').style.display="block"; 
            };
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    unique_id = uuid.uuid4().hex
    in_p = f"in_{unique_id}.mp4"
    out_f = f"out_{unique_id}.mp4"
    out_p = os.path.join(VIDEOS_DIR, out_f)

    with open(in_p, "wb") as f: f.write(await file.read())

    cap = cv2.VideoCapture(in_p)
    f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # CODEC COMPATIBLE CON WEB (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(out_p, fourcc, fps, (f_w, f_h))

    angles_r, angles_s = [], []
    
    with mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                # DIBUJAR PUNTOS (Lo que quieres ver)
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                lm = res.pose_landmarks.landmark
                # Puntos UTS: Hombro(12), Cadera(24), Rodilla(26), Tobillo(28)
                p12 = [lm[12].x, lm[12].y]; p24 = [lm[24].x, lm[24].y]
                p26 = [lm[26].x, lm[26].y]; p28 = [lm[28].x, lm[28].y]

                ar = calculate_angle(p24, p26, p28) # Rodilla
                as_ = calculate_angle(p12, p24, p26) # Sentadilla/Tronco
                angles_r.append(ar); angles_s.append(as_)

                cv2.putText(frame, f"Ang. Rodilla: {int(ar)}", (50, 50), 1, 2, (0,255,0), 2)

            out.write(frame)

    cap.release(); out.release()
    if os.path.exists(in_p): os.remove(in_p)

    avg_r = round(np.mean(angles_r), 1) if angles_r else 0
    avg_s = round(np.mean(angles_s), 1) if angles_s else 0
    v_url = f"/static/videos/{out_f}"

    return HTMLResponse(content=f"""
    <div style="font-family:sans-serif; text-align:center; padding:20px; background:#F8F9FE; min-height:100vh;">
        <img src="/static/logo.jpeg" style="width:80px; border-radius:50%; margin-bottom:10px;">
        <h2 style="color:#5D4D8A;">Resultados del Análisis</h2>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; max-width:400px; margin:0 auto 20px;">
            <div style="background:white; padding:15px; border-radius:15px;"><b>Rodilla</b><br><span style="font-size:20px; color:#5D4D8A;">{avg_r}°</span></div>
            <div style="background:white; padding:15px; border-radius:15px;"><b>Sentadilla</b><br><span style="font-size:20px; color:#5D4D8A;">{avg_s}°</span></div>
        </div>
        <video controls autoplay muted playsinline style="width:100%; max-width:400px; border-radius:20px; box-shadow:0 10px 20px rgba(0,0,0,0.1);">
            <source src="{v_url}" type="video/mp4">
        </video>
        <br><br><a href="/" style="color:#5D4D8A; font-weight:bold; text-decoration:none;">← Analizar otro video</a>
    </div>
    """)
