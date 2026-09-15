from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io

app = FastAPI(title="Üretim Hattı Çatlak ve Kusur Dedektörü")

@app.get("/")
def anasayfa():
    return {
        "mesaj": "Üretim Hattı Çatlak Dedektörü Aktif!",
        "otomatik_test_icin": "http://127.0.0.1:8000/otomatik-test adresine git.",
        "resim_yuklemek_icin": "http://127.0.0.1:8000/docs adresinden /catlak-kontrol kullan."
    }

@app.get("/otomatik-test")
def otomatik_test():
    # 1. Elimizde resim yoksa, sanal bir metal plaka ve üzerinde çatlak (çizgi) üretelim
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
    
    # Plaka üzerine sanal bir çatlak çizgisi çiziyoruz
    cv2.line(frame, (150, 100), (300, 350), (50, 50, 50), 3)
    cv2.line(frame, (300, 350), (450, 200), (50, 50, 50), 3)
    cv2.putText(frame, "Sanal Urun Yuzeyi ve Catlak", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    # 2. OpenCV Canny Kenar / Çatlak Tespiti
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # 3. Tespit edilen çatlak hatlarını Kırmızı renge boyama
    frame[edges != 0] = [0, 0, 255] # BGR -> Kırmızı

    # 4. Sonucu tarayıcıya resim olarak fırlatma
    _, encoded_img = cv2.imencode('.jpg', frame)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

@app.post("/catlak-kontrol")
async def catlak_kontrol(file: UploadFile = File(...)):
    # Bilgisayardan yüklenen gerçek görseli okuma
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Gri tonlama ve Canny kenar tespiti
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # Kusurlu hatları kırmızıya boyama
    image[edges != 0] = [0, 0, 255]

    _, encoded_img = cv2.imencode('.jpg', image)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")