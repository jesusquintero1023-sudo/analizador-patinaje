import cv2
import mediapipe as mp
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI()

# Permite que el celular se conecte sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5)

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
            <style>
                body { font-family: sans-serif; background: #FCFAFF; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
                .card { background: white; padding: 40px; border-radius: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center; }
                .btn { background: #5D4D8A; color: white; padding: 15px 30px; border-radius: 25px; border: none; cursor: pointer; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Club Skate Wheels</h1>
                <form action="/analyze" enctype="multipart/form-data" method="post">
                    <input type="file" name="file" accept="video/*" required><br><br>
                    <button type="submit" class="btn">ANALIZAR EN WEB</button>
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/analyze")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    in_p = f"temp_{uuid.uuid4()}.{ext}"
    out_f = f"analyzed_{uuid.uuid4()}.mp4"
    out_p = os.path.join(OUTPUT_DIR, out_f)

    with open(in_p, "wb") as b: b.write(await file.read())

    cap = cv2.VideoCapture(in_p)
    orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 30
    out = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_h, orig_w))

    angles_r, angles_s = [], []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if count % 2 == 0:
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                p = [[lm[i].x, lm[i].y] for i in [12, 24, 26, 28]]
                ar, asat = calculate_angle(p[1], p[2], p[3]), calculate_angle(p[0], p[1], p[2])
                angles_r.append(ar); angles_s.append(asat)
                cv2.putText(frame, f"R: {int(ar)}", (40, 60), 1, 2, (0, 255, 0), 2)
        out.write(frame)
        count += 1
    cap.release(); out.release()
    if os.path.exists(in_p): os.remove(in_p)

    avg_r = round(np.mean(angles_r), 1) if angles_r else 0
    avg_s = round(np.mean(angles_s), 1) if angles_s else 0
    dur = round(count / fps, 2)
    v_url = f"https://{request.base_url.hostname}/static/videos/{out_f}"

    # RESPUESTA PARA EL APK (JSON)
    if "text/html" not in request.headers.get("accept", ""):
        return JSONResponse({
            "avg_rodilla": avg_r,
            "avg_sentadilla": avg_s,
            "duracion": dur,
            "video_url": v_url
        })

    # RESPUESTA PARA EL PC (HTML)
    return HTMLResponse(content=f"<h2>Resultado: {avg_r}°</h2><video controls width='300'><source src='{v_url}'></video>")
