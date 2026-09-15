from fastapi import FastAPI
import numpy as np
from sklearn.linear_model import LinearRegression

# FastAPI uygulamamızı başlatıyoruz
app = FastAPI()

# 1. Makine öğrenmesi modelimizi hazırlayalım ve eğitelim
X_egitim = np.array([
    [80, 120, 1.0],
    [100, 150, 1.1],
    [120, 180, 0.9],
    [150, 220, 1.2],
    [90, 130, 1.0]
])
y_egitim = np.array([115, 148, 175, 215, 125])

model = LinearRegression()
model.fit(X_egitim, y_egitim)

# 2. Ana sayfa (Tarayıcıdan girince görünecek karşılama mesajı)
@app.get("/")
def anasayfa():
    return {"mesaj": "Dinamik Fiyatlandırma Motoru API'si Çalışıyor!"}

# 3. Fiyat hesaplama adresi (API Endpoint)
@app.get("/fiyat-hesapla")
def fiyat_tahmin(maliyet: float, rakip_fiyat: float, talep: float):
    # Dışarıdan gelen parametrelerle tahmin yapalım
    yeni_urun = np.array([[maliyet, rakip_fiyat, talep]])
    tahmin_edilen_fiyat = model.predict(yeni_urun)
    son_fiyat = round(tahmin_edilen_fiyat[0], 2)
    
    return {
        "girilen_maliyet": maliyet,
        "girilen_rakip_fiyat": rakip_fiyat,
        "talep_orani": talep,
        "yapay_zeka_onerilen_fiyat": son_fiyat
    }