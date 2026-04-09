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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización ligera de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

OUTPUT_DIR = "static/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Club Skate Wheels</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                body { font-family: 'Segoe UI', sans-serif; background: #FCFAFF; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
                .card { background: white; padding: 40px; border-radius: 35px; box-shadow: 0 15px 35px rgba(0,0,0,0.05); text-align: center; width: 90%; max-width: 400px; }
                .btn { background: #F3F0F7; color: #5D4D8A; border: none; padding: 15px; border-radius: 30px; cursor: pointer; font-weight: bold; width: 100%; margin-bottom: 12px; display: flex; align-items: center; justify-content: center; gap: 10px; text-decoration: none; }
                .btn-primary { background: #5D4D8A; color: white; margin-top: 15px; }
                input[type="file"] { display: none; }
                #overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255, 255, 255, 0.9); z-index: 1000; flex-direction: column; justify-content: center; align-items: center; }
                .spinner { width: 40px; height: 40px; border: 4px solid #F3F0F7; border-top: 4px solid #5D4D8A; border-radius: 50%; animation: spin 1s linear infinite; }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            </style>
        </head>
        <body>
            <div id="overlay"><div class="spinner"></div><p style="color:#5D4D8A; font-weight:bold; margin-top:15px;">Analizando técnica...</p></div>
            <div class="card">
                <img src="/static/logo.jpeg" height="100" style="border-radius: 20px;">
                <h1 style="font-size: 22px;">Club Skate Wheels</h1>
                <form action="/analyze" enctype="multipart/form-data" method="post" onsubmit="document.getElementById('overlay').style.display='flex'">
                    <label class="btn" for="u"><i class="fas fa-video"></i> Elegir Video</label>
                    <input id="u" name="file" type="file" accept="video/*" required>
                    <button type="submit" class="btn btn-primary">INICIAR ANÁLISIS EN PC</button>
                </form>
                <p style="font-size: 10px; color: #BBB; margin-top: 20px;">Sistema v1.5 - UTS 2026</p>
            </div>
        </body>
    </html>
    """

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_video(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    in_p = f"temp_{uuid.uuid4()}.{ext}"
    out_f = f"analyzed_{uuid.uuid4()}.mp4"
    out_p = os.path.join(OUTPUT_DIR, out_f)

    with open(in_p, "wb") as b: b.write(await file.read())

    cap = cv2.VideoCapture(in_p)
    f_w, f_h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 30
    out = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, (f_h, f_w))

    angles_r, angles_s = [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            p = [[lm[i].x, lm[i].y] for i in [12, 24, 26, 28]]
            ar, asat = calculate_angle(p[1], p[2], p[3]), calculate_angle(p[0], p[1], p[2])
            angles_r.append(ar); angles_s.append(asat)
            cv2.putText(frame, f"R: {int(ar)}", (40, 60), 1, 2, (0, 255, 0), 2)
        out.write(frame)
    cap.release(); out.release()
    if os.path.exists(in_p): os.remove(in_p)

    # Variables para el informe
    avg_r = round(np.mean(angles_r), 1) if angles_r else 0
    avg_s = round(np.mean(angles_s), 1) if angles_s else 0
    dur = round(len(angles_r)/fps, 2)
    v_url = f"/static/videos/{out_f}"

    return f"""
    <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #FCFAFF; display: flex; justify-content: center; padding: 20px; }}
                .res-card {{ background: white; padding: 30px; border-radius: 35px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); max-width: 400px; text-align: center; border: 1px solid #F0EBF9; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }}
                .item {{ background: #F3F0F7; padding: 15px; border-radius: 20px; }}
                .val {{ font-size: 22px; font-weight: bold; color: #5D4D8A; display: block; }}
                .lbl {{ font-size: 10px; color: #888; font-weight: bold; }}
                video {{ width: 100%; border-radius: 20px; margin-top: 15px; border: 2px solid #5D4D8A; }}
            </style>
        </head>
        <body>
            <div class="res-card">
                <h2 style="color:#5D4D8A; margin-top:0;">INFORME TÉCNICO</h2>
                <div class="grid">
                    <div class="item"><span class="lbl">RODILLA</span><span class="val">{avg_r}°</span></div>
                    <div class="item"><span class="lbl">SENTADILLA</span><span class="val">{avg_s}°</span></div>
                    <div class="item"><span class="lbl">DURACIÓN</span><span class="val">{dur}s</span></div>
                    <div class="item"><span class="lbl">ESTADO</span><span class="val">OK</span></div>
                </div>
                <video controls autoplay loop><source src="{v_url}" type="video/mp4"></video>
                <a href="/" style="display:block; margin-top:20px; color:#5D4D8A; text-decoration:none; font-weight:bold;">← Volver</a>
            </div>
        </body>
    </html>
    """
