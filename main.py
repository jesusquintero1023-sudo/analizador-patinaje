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
            .logo-img { width: 160px; margin-bottom: 20px; }
            h1 { font-size: 24px; font-weight: bold; color: #111; margin: 0; }
            .sub { color: #888; font-size: 13px; margin-bottom: 30px; }
            .file-label { display: block; background: #F3F0F7; color: var(--primary); padding: 18px; border-radius: 20px; font-weight: bold; cursor: pointer; margin-bottom: 20px; }
            .btn-submit { background: var(--primary); color: white; border: none; padding: 18px; width: 100%; border-radius: 25px; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 10px; }
            .btn-submit:disabled { background: #ccc; }
            .spinner { display: none; width: 18px; height: 18px; border: 3px solid #ffffff55; border-radius: 50%; border-top-color: #fff; animation: spin 1s linear infinite; }
            @keyframes spin { to { transform: rotate(360deg); } }
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
                <button type="submit" class="btn-submit" id="B" disabled>
                    <span id="T">ANALIZAR EN WEB</span>
                    <div class="spinner" id="S"></div>
                </button>
            </form>
        </div>
        <script>
            document.getElementById('f').onsubmit = () => {
                document.getElementById('B').disabled = true;
                document.getElementById('T').innerText = 'PROCESANDO...';
                document.getElementById('S').style.display = 'block';
            };
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    unique_id = uuid.uuid4().hex
    in_p = f"in_{unique_id}.mp4"
    out_f = f"out_{unique_id}.webm"
    out_p = os.path.join(VIDEOS_DIR, out_f)

    with open(in_p, "wb") as buffer:
        buffer.write(await file.read())
    
    cap = cv2.VideoCapture(in_p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    out = cv2.VideoWriter(out_p, fourcc, fps, (w, h))

    angles_r, angles_s = [], []
    
    with mp_pose.Pose(model_complexity=0) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = res.pose_landmarks.landmark
                p12, p24, p26, p28 = [lm[12].x, lm[12].y], [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
                angles_r.append(calculate_angle(p24, p26, p28))
                angles_s.append(calculate_angle(p12, p24, p26))
            out.write(frame)

    cap.release(); out.release()
    if os.path.exists(in_p): os.remove(in_p)

    avg_r = round(np.mean(angles_r), 1) if angles_r else 0
    avg_s = round(np.mean(angles_s), 1) if angles_s else 0
    max_r = round(max(angles_r), 1) if angles_r else 0
    dur = round(len(angles_r)/fps, 2)
    v_url = f"https://analizador-patinaje.onrender.com/static/videos/{out_f}"

    # RESPUESTA PARA FLUTTER
    if request.headers.get("Accept") == "application/json":
        return {
            "avg_rodilla": avg_r,
            "avg_sentadilla": avg_s,
            "max_ext": max_r,
            "duracion": dur,
            "video_url": v_url
        }

    # RESPUESTA PARA NAVEGADOR
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
            .status-header {{ background: linear-gradient(to bottom, #FFF5E6, #FFE0B2); padding: 30px 20px; border-bottom: 1px solid #FFCC80; }}
            .status-title {{ color: var(--orange); font-weight: bold; font-size: 22px; margin: 0; }}
            .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 25px; border-bottom: 1px solid #EEE; }}
            .metric {{ display: flex; flex-direction: column; align-items: center; }}
            .m-label {{ font-size: 10px; color: #999; text-transform: uppercase; font-weight: bold; }}
            .m-value {{ font-size: 22px; color: var(--primary); font-weight: bold; }}
            .video-section {{ padding: 20px; background: #FAF9FF; }}
            video {{ width: 100%; border-radius: 25px; background: #000; }}
            .btn-back {{ display: block; padding: 20px; color: var(--primary); text-decoration: none; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="report-card">
            <div class="status-header"><h2 class="status-title">TÉCNICA ACEPTABLE</h2></div>
            <div class="metrics-grid">
                <div class="metric"><span class="m-label">Rodilla</span><span class="m-value">{avg_r}°</span></div>
                <div class="metric"><span class="m-label">Sentadilla</span><span class="m-value">{avg_s}°</span></div>
                <div class="metric"><span class="m-label">Extensión</span><span class="m-value">{max_r}°</span></div>
                <div class="metric"><span class="m-label">Duración</span><span class="m-value">{dur}s</span></div>
            </div>
            <div class="video-section">
                <video controls playsinline autoplay muted loop>
                    <source src="{v_url}" type="video/webm">
                </video>
            </div>
            <a href="/" class="btn-back">← Volver</a>
        </div>
    </body>
    </html>
    """)
