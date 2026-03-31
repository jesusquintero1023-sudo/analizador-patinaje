from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import numpy as np

# 🔥 NUEVO IMPORT (CORREGIDO)
from mediapipe.python.solutions import pose as mp_pose

app = FastAPI()

# 📁 Carpetas
UPLOAD_FOLDER = "videos"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 📺 Permitir ver videos en la web
app.mount("/output", StaticFiles(directory=OUTPUT_FOLDER), name="output")

# 🧠 Inicializar modelo
pose = mp_pose.Pose()

# 📐 FUNCIÓN PARA ÁNGULO
def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cos_angle))

    return int(angle)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # 📥 Guardar video
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🎥 Leer video
    cap = cv2.VideoCapture(file_path)

    # 📺 Crear video procesado
    output_path = os.path.join(OUTPUT_FOLDER, f"procesado_{file.filename}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frames_analizados = 0
    angulos = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            frames_analizados += 1

            landmarks = result.pose_landmarks.landmark

            # 🔥 Coordenadas pierna derecha
            hip = [landmarks[24].x, landmarks[24].y]
            knee = [landmarks[26].x, landmarks[26].y]
            ankle = [landmarks[28].x, landmarks[28].y]

            angulo = calcular_angulo(hip, knee, ankle)
            angulos.append(angulo)

            # 📐 Dibujar ángulo
            cv2.putText(frame,
                        f'Rodilla: {angulo}',
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)

            # 🎯 Dibujar puntos
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # 📹 Inicializar writer
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

        out.write(frame)

    cap.release()
    if out:
        out.release()

    # 📊 Promedio
    angulo_promedio = int(sum(angulos) / len(angulos)) if angulos else 0

    # 🌐 URL pública (IMPORTANTE)
    video_url = f"https://analizador-patinaje.onrender.com/output/procesado_{file.filename}"

    return {
        "mensaje": "Video analizado",
        "frames_con_postura": frames_analizados,
        "angulo_rodilla": angulo_promedio,
        "video_url": video_url
    }
