from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import mediapipe as mp

app = FastAPI()

# 📁 Carpetas
UPLOAD_FOLDER = "videos"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 📡 Servir archivos (para que Flutter pueda verlos)
app.mount("/output", StaticFiles(directory=OUTPUT_FOLDER), name="output")

# 🧠 MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    # 📁 Guardar video original
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🎥 Abrir video
    cap = cv2.VideoCapture(input_path)

    # 📊 Contador
    frames_analizados = 0

    # 🎬 Configuración video salida
    output_path = os.path.join(OUTPUT_FOLDER, f"procesado_{file.filename}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 🔥 PROCESAR VIDEO
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            frames_analizados += 1

            # 🎯 Dibujar puntos en el frame
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # 💾 Guardar frame procesado
        out.write(frame)

    cap.release()
    out.release()

    # 🌐 URL del video para Flutter
    video_url = f"https://TU-APP.onrender.com/output/procesado_{file.filename}"

    return {
        "mensaje": "Video analizado",
        "frames_con_postura": frames_analizados,
        "video_url": video_url
    }