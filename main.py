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

# 1. PERMITIR CONEXIÓN DESDE LA APP MÓVIL (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. MONTAR CARPETA STATIC (FUNDAMENTAL PARA EL LOGO)
# Esto le dice a Render que use tu carpeta 'static'
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuración de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5)

OUTPUT_DIR = "static/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        <title>Club Skate Wheels | Análisis</title>
        <style>
            :root { --primary: #5D4D8A; --bg: #FCFAFF; }
            body { font-family: 'Segoe UI', sans-serif; background: var(--bg); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
            
            .card { background: white; padding: 40px; border-radius: 30px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); text-align: center; width: 90%; max-width: 400px; }
            
            /* Logo Circular */
            .logo-container { width: 100px; height: 100px; margin: 0 auto 20px; border-radius: 50%; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border: 3px solid white; }
            .logo-container img { width: 100%; height: 100%; object-fit: cover; }
            
            h1 { color: #111; font-size: 24px; margin: 0; }
            p { color: #777; font-size: 14px; margin-bottom: 30px; }

            /* Botón de Elegir Archivo Personalizado */
            .file-input-wrapper { position: relative; margin-bottom: 20px; }
            input[type="file"] { position: absolute; left: 0; top: 0; opacity: 0; width: 100%; height: 100%; cursor: pointer; }
            
            .custom-file-btn { background: #F3F0F7; color: var(--primary); padding: 15px; border-radius: 15px; display: block; font-weight: bold; border: 2px dashed #D1C8E0; transition: 0.3s; }
            .custom-file-btn:hover { background: #EAE5F2; border-color: var(--primary); }

            /* Check de confirmación */
            #status-msg { font-size: 13px; color: #2D9B58; font-weight: bold; margin-bottom: 15px; display: none; }

            /* Botón Analizar */
            .btn-submit { background: var(--primary); color: white; border: none; padding: 18px; width: 100%; border-radius: 20px; font-weight: bold; font-size: 16px; cursor: pointer; box-shadow: 0 8px 20px rgba(93, 77, 138, 0.3); transition: 0.3s; }
            .btn-submit:hover { background: #4A3C71; transform: translateY(-2px); }
            .btn-submit:disabled { background: #CCC; box-shadow: none; cursor: not_allowed; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="logo-container">
                <img src="/static/logo.jpeg" alt="Logo">
            </div>
            <h1>Club Skate Wheels</h1>
            <p>Análisis Biomecánico Híbrido</p>

            <form action="/analyze" method="post" enctype="multipart/form-data" id="uploadForm">
                <div class="file-input-wrapper">
                    <span class="custom-file-btn" id="file-label">📁 Seleccionar Video</span>
                    <input type="file" name="file" id="video-file" accept="video/*" required>
                </div>
                
                <div id="status-msg">✓ Video listo para analizar</div>

                <button type="submit" class="btn-submit" id="btn-ana" disabled>ANALIZAR EN WEB</button>
            </form>
        </div>

        <script>
            const fileInput = document.getElementById('video-file');
            const fileLabel = document.getElementById('file-label');
            const statusMsg = document.getElementById('status-msg');
            const btnAna = document.getElementById('btn-ana');

            fileInput.addEventListener('change', function() {
                if (this.files && this.files.length > 0) {
                    fileLabel.innerText = "🎥 Video Cargado";
                    fileLabel.style.borderStyle = "solid";
                    statusMsg.style.display = "block";
                    btnAna.disabled = false;
                }
            });

            document.getElementById('uploadForm').addEventListener('submit', function() {
                btnAna.innerText = "PROCESANDO...";
                btnAna.style.opacity = "0.7";
            });
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    # Generar nombres únicos para evitar conflictos
    unique_id = uuid.uuid4().hex
    in_path = f"temp_{unique_id}.mp4"
    out_filename = f"out_{unique_id}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_filename)

    with open(in_path, "wb") as buffer:
        buffer.write(await file.read())

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Grabador de video (usamos mp4v para compatibilidad total)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (height, width))

    angles_r = []
    angles_s = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Rotar si es necesario (celulares suelen grabar en vertical)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Procesar cada 2 frames para velocidad
        if frame_count % 2 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Puntos: cadera(24), rodilla(26), tobillo(28), hombro(12)
                p_hip = [lm[24].x, lm[24].y]
                p_knee = [lm[26].x, lm[26].y]
                p_ankle = [lm[28].x, lm[28].y]
                p_shoulder = [lm[12].x, lm[12].y]

                ang_r = calculate_angle(p_hip, p_knee, p_ankle)
                ang_s = calculate_angle(p_shoulder, p_hip, p_knee)
                
                angles_r.append(ang_r)
                angles_s.append(ang_s)

                cv2.putText(frame, f"Rodilla: {int(ang_r)}", (30, 50), 1, 2, (0,255,0), 2)
        
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    if os.path.exists(in_path): os.remove(in_path)

    avg_r = round(np.mean(angles_r), 1) if angles_r else 0
    avg_s = round(np.mean(angles_s), 1) if angles_s else 0
    duration = round(frame_count / fps, 2)
    video_url = f"https://{request.base_url.hostname}/static/videos/{out_filename}"

    # RESPUESTA HÍBRIDA
    if "text/html" not in request.headers.get("accept", ""):
        return JSONResponse({
            "avg_rodilla": avg_r,
            "avg_sentadilla": avg_s,
            "duracion": duration,
            "video_url": video_url
        })

    return HTMLResponse(content=f"""
        <div style="font-family:sans-serif; text-align:center; padding:40px;">
            <img src="/static/logo.jpeg" width="80" style="border-radius:50%">
            <h2>Análisis Completado</h2>
            <p>Promedio Rodilla: <b>{avg_r}°</b></p>
            <video controls width="300" style="border-radius:15px; box-shadow:0 5px 15px rgba(0,0,0,0.2);">
                <source src="{video_url}" type="video/mp4">
            </video><br><br>
            <a href="/" style="text-decoration:none; color:#5D4D8A; font-weight:bold;">← Volver a analizar</a>
        </div>
    """)
