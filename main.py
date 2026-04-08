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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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
        <head><title>Club Skate Wheels</title></head>
        <body style="font-family:sans-serif; text-align:center; padding:50px; background:#F8F7FF;">
            <div style="background:white; padding:30px; border-radius:20px; display:inline-block; box-shadow:0 10px 20px rgba(0,0,0,0.1);">
                <h2>🏁 Analizador Biomecánico - PC</h2>
                <form action="/analyze" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept="video/*"><br><br>
                    <button type="submit" style="background:#5D4D8A; color:white; border:none; padding:10px 20px; border-radius:10px;">Analizar Video</button>
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
    # Detectar si el video es vertical u horizontal para la salida
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Ajustamos dimensiones si vamos a rotar (Vertical)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (height, width)) 

    angles_rodilla = []
    angles_sentadilla = []
    frames_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames_count += 1
        
        # --- CORRECCIÓN DE ROTACIÓN ---
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

            # Dibujar info en video
            cv2.putText(image, f"R: {int(ang_r)} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"S: {int(ang_s)} deg", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

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
