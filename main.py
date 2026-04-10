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
            body { font-family: 'Segoe UI', sans-serif; background: var(--bg); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
            .card { background: white; padding: 40px; border-radius: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); text-align: center; width: 100%; max-width: 400px; }
            .logo-img { width: 180px; margin-bottom: 20px; }
            h1 { font-size: 26px; margin: 0; color: #111; font-weight: bold; }
            .sub { color: #888; font-size: 14px; margin-bottom: 35px; }
            .file-label { display: block; background: #F3F0F7; color: var(--primary); padding: 18px; border-radius: 20px; font-weight: bold; cursor: pointer; margin-bottom: 20px; border: none; }
            .btn-submit { background: var(--primary); color: white; border: none; padding: 18px; width: 100%; border-radius: 25px; font-weight: bold; cursor: pointer; font-size: 16px; box-shadow: 0 5px 15px rgba(93, 77, 138, 0.3); }
            .btn-submit:disabled { background: #ccc; box-shadow: none; }
        </style>
    </head>
    <body>
        <div class="card">
            <img src="/static/logo.jpeg" class="logo-img">
            <h1>Club Skate Wheels</h1>
            <p class="sub">Análisis Biomecánico Híbrido</p>
            <form action="/analyze" method="post" enctype="multipart/form-data" id="f">
                <label for="v" class="file-label" id="L">📁 Seleccionar Video</label>
                <input type="file" name="file" id="v" accept="video/*" hidden required onchange="document.getElementById('L').innerText='🎥 Video Cargado'; document.getElementById('B').disabled=false;">
                <button type="submit" class="btn-submit" id="B" disabled>ANALIZAR EN WEB</button>
            </form>
        </div>
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
    fps, w, h = cap.get(cv2.CAP_PROP_FPS) or 30, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ARREGLO PARA EDGE: Usamos 'avc1' y forzamos que el archivo se cierre correctamente
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(out_p, fourcc, fps, (w, h))

    angles = []
    with mp_pose.Pose(model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = res.pose_landmarks.landmark
                # Calculamos con los puntos que definimos (Cadera, Rodilla, Tobillo)
                ang = calculate_angle([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
                angles.append(ang)
            out.write(frame)

    cap.release()
    out.release()
    if os.path.exists(in_p): os.remove(in_p)

    avg = round(np.mean(angles), 1) if angles else 0
    max_ext = round(max(angles), 1) if angles else 0
    min_open = round(min(angles), 1) if angles else 0
    dur = round(len(angles)/fps, 2)

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
            .status-title {{ color: var(--orange); font-weight: bold; font-size: 24px; letter-spacing: 1px; margin: 0; }}
            
            .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 30px; border-bottom: 1px solid #EEE; }}
            .metric {{ display: flex; flex-direction: column; align-items: center; }}
            .m-label {{ font-size: 10px; color: #999; text-transform: uppercase; font-weight: bold; margin-top: 5px; }}
            .m-value {{ font-size: 24px; color: var(--primary); font-weight: bold; }}
            .icon {{ font-size: 20px; margin-bottom: 5px; opacity: 0.8; }}

            .video-section {{ padding: 25px; background: #FAF9FF; }}
            .video-tag {{ display: flex; align-items: center; justify-content: center; gap: 8px; color: #888; font-weight: bold; font-size: 14px; margin-bottom: 15px; text-transform: uppercase; }}
            video {{ width: 100%; border-radius: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); background: #000; }}
            
            .btn-back {{ display: block; padding: 25px; color: var(--primary); text-decoration: none; font-weight: bold; font-size: 15px; }}
        </style>
    </head>
    <body>
        <div class="report-card">
            <div class="status-header">
                <h2 class="status-title">TÉCNICA ACEPTABLE</h2>
            </div>
            
            <div class="metrics-grid">
                <div class="metric">
                    <span class="icon">📊</span>
                    <span class="m-label">Promedio</span>
                    <span class="m-value">{avg}°</span>
                </div>
                <div class="metric">
                    <span class="icon">↑</span>
                    <span class="m-label">Extensión Máx.</span>
                    <span class="m-value">{max_ext}°</span>
                </div>
                <div class="metric">
                    <span class="icon">↓</span>
                    <span class="m-label">Apertura (Min)</span>
                    <span class="m-value">{min_open}°</span>
                </div>
                <div class="metric">
                    <span class="icon">⏱️</span>
                    <span class="m-label">Duración</span>
                    <span class="m-value">{dur}s</span>
                </div>
            </div>

            <div class="video-section">
                <div class="video-tag">🔴 Video Procesado</div>
                <video controls playsinline preload="auto">
                    <source src="/static/videos/{out_f}" type="video/mp4">
                    Tu navegador no soporta el video.
                </video>
            </div>
            
            <a href="/" class="btn-back">← Analizar otro video</a>
        </div>
    </body>
    </html>
    """)
