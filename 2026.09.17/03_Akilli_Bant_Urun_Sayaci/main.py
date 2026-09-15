import cv2
import numpy as np
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="Akıllı Bant Ürün ve Kargo Sayacı (Resim & Canlı Kamera)")

# YOLOv8 nano modeli
model = YOLO("yolov8n.pt")

@app.get("/")
def anasayfa():
    return {
        "mesaj": "Akıllı Bant Sayım Servisi Aktif!",
        "kullanim_1": "Resim yükleyerek analiz için: /kargo-say (POST)",
        "kullanim_2": "Canlı kamera akışı için: /canli-kamera (GET)"
    }

# 1. YÖNTEM: Bilgisayardan Resim/Fotoğraf Yükleyerek Analiz
@app.post("/kargo-say")
async def kargo_say(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame)
    kargo_sayisi = 0
    
    for r in results:
        for box in r.boxes:
            kargo_sayisi += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Toplam Kargo: {kargo_sayisi}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    _, encoded_img = cv2.imencode('.jpg', frame)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

# 2. YÖNTEM: Canlı Kamera Akışı Üzerinden Anlık Sayım
def kamera_akisi():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        results = model(frame)
        kargo_sayisi = 0
        
        for r in results:
            for box in r.boxes:
                kargo_sayisi += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Canli Nesne Sayisi: {kargo_sayisi}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.get("/canli-kamera")
def canli_kamera():
    return StreamingResponse(kamera_akisi(), media_type="multipart/x-mixed-replace; boundary=frame")