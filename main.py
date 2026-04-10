import os
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = "static"
VIDEOS_DIR = os.path.join(STATIC_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
        <title>Club Skate Wheels</title>
        <style>
            :root { --primary: #5D4D8A; --bg: #FCFAFF; }
            body { font-family: 'Segoe UI', sans-serif; background: var(--bg); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 20px; }
            .card { background: white; padding: 40px; border-radius: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); text-align: center; width: 100%; max-width: 400px; }
            .logo-img { width: 180px; margin-bottom: 20px; }
            h1 { font-size: 26px; font-weight: bold; color: #111; margin: 0; }
            .sub { color: #888; font-size: 14px; margin-bottom: 35px; }
            .file-label { display: block; background: #F3F0F7; color: var(--primary); padding: 18px; border-radius: 20px; font-weight: bold; cursor: pointer; margin-bottom: 20px; }
            .btn-submit { background: var(--primary); color: white; border: none; padding: 18px; width: 100%; border-radius: 25px; font-weight: bold; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 10px; }
            .btn-submit:disabled { background: #ccc; cursor: not-allowed; }
            .spinner { display: none; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: white; animation: spin 1s linear infinite; }
            @keyframes spin { to { transform: rotate(360deg); } }
            #check-msg { color: #2D9B58; font-weight: bold; font-size: 14px; margin-bottom: 15px; display: none; }
        </style>
    </head>
    <body>
        <div class="card">
            <img src="/static/logo.jpeg" class="logo-img">
            <h1>Club Skate Wheels</h1>
            <p class="sub">Análisis Biomecánico Híbrido</p>
            <form action="/analyze" method="post" enctype="multipart/form-data" id="uploadForm">
                <label for="v" class="file-label" id="L">📁 Seleccionar Video</label>
                <input type="file" name="file" id="v" accept="video/*" hidden required>
                <div id="check-msg">✓ Video listo para analizar</div>
                <button type="submit" class="btn-submit" id="B" disabled>
                    <span id="T">ANALIZAR EN WEB</span>
                    <div class="spinner" id="S"></div>
                </button>
            </form>
        </div>
        <script>
            const v = document.getElementById('v');
            const B = document.getElementById('B');
            const L = document.getElementById('L');
            const T = document.getElementById('T');
            const S = document.getElementById('S');
            const M = document.getElementById('check-msg');

            v.onchange = () => {
                if(v.files[0]){
                    L.innerText = "🎥 Video Cargado";
                    M.style.display = "block";
                    B.disabled = false;
                }
            };

            document.getElementById('uploadForm').onsubmit = () => {
                B.disabled = true;
                T.innerText = 'PROCESANDO...';
                S.style.display = "block";
            };
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    unique_id = uuid.uuid4().hex
    in_p, out_f = f"in_{unique_id}.mp4", f"out_{unique_id}.mp4"
    out_p = os.path.join(VIDEOS_DIR, out_f)

    with open(in_p, "wb") as f: f.write(await file.read())
    
    cap = cv2.VideoCapture(in_p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # IMPORTANTE: Codec MP4V es a veces más compatible con el guardado directo de OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(out_p, fourcc, fps, (w, h))

    angles_r, angles_s = [], []
    
    # Reducimos complejidad para que Render no se sature
    with mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                # Dibujar esqueleto
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                lm = res.pose_landmarks.landmark
                p12 = [lm[12].x, lm[12].y]; p24 = [lm[24].x, lm[24].y]
                p26 = [lm[26].x, lm[26].y]; p28 = [lm[28].x, lm[28].y]
                
                angles_r.append(calculate_angle(p24, p26, p28))
                angles_s.append(calculate_angle(p12, p24, p26))

            out.write(frame)

    cap.release()
    out.release()
    if os.path.exists(in_p): os.remove(in_p)

    avg_r = round(np.mean(angles_r), 1) if angles_r else 0
    avg_s = round(np.mean(angles_s), 1) if angles_s else 0
    max_r = round(max(angles_r), 1) if angles_r else 0
    dur = round(len(angles_r)/fps, 2)

    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root {{ --primary: #5D4D8A; --orange: #F39C12; --bg: #F0F2F5; }}
            body {{ font-family: 'Segoe UI', sans-serif; background: var(--bg); margin: 0; display: flex; justify-content: center; padding: 20px; }}
            .report-card {{ background: white; width: 100%; max-width: 450px; border-radius: 35px; overflow: hidden; box-shadow: 0 15px 40px rgba(0,0,0,0.1); text-align: center; }}
            .status-header {{ background: linear-gradient(to bottom, #FFF5E6, #FFE0B2); padding: 35px 20px; border-bottom: 1px solid #FFCC80; }}
            .status-title {{ color: var(--orange); font-weight: bold; font-size: 24px; margin: 0; }}
            .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 30px; border-bottom: 1px solid #EEE; }}
            .metric {{ display: flex; flex-direction: column; align-items: center; }}
            .m-label {{ font-size: 10px; color: #999; text-transform: uppercase; font-weight: bold; margin-bottom: 5px; }}
            .m-value {{ font-size: 24px; color: var(--primary); font-weight: bold; }}
            .video-section {{ padding: 25px; background: #FAF9FF; }}
            video {{ width: 100%; border-radius: 25px; background: #000; box-shadow: 0 10px 25px rgba(0,0,0,0.15); }}
            .btn-back {{ display: block; padding: 25px; color: var(--primary); text-decoration: none; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="report-card">
            <div class="status-header"><h2 class="status-title">TÉCNICA ACEPTABLE</h2></div>
            <div class="metrics-grid">
                <div class="metric"><span class="m-label">Rodilla</span><span class="m-value">{avg_r}°</span></div>
                <div class="metric"><span class="m-label">Sentadilla</span><span class="m-value">{avg_s}°</span></div>
                <div class="metric"><span class="m-label">Extensión Máx.</span><span class="m-value">{max_r}°</span></div>
                <div class="metric"><span class="m-label">Duración</span><span class="m-value">{dur}s</span></div>
            </div>
            <div class="video-section">
                <div style="color:#888; font-weight:bold; font-size:14px; margin-bottom:15px;">🔴 VIDEO PROCESADO</div>
                <video controls playsinline>
                    <source src="/static/videos/{out_f}" type="video/mp4">
                </video>
            </div>
            <a href="/" class="btn-back">← Analizar otro video</a>
        </div>
    </body>
    </html>
    """)
