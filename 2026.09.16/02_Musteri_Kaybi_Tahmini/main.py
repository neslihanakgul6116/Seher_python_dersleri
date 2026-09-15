import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="Müşteri Kaybı (Churn) Tahmin ve Analiz Servisi")

# 1. Pandas ile Yapılandırılmış Eğitim Veri Seti (Tablo Yapısı)
veri = {
    "Yas": [25, 45, 35, 50, 23, 40, 29, 55],
    "AylikOdeme": [120, 50, 90, 40, 140, 60, 110, 45],
    "HizmetSuresi": [2, 24, 6, 36, 1, 18, 4, 48],
    "Churn": [1, 0, 1, 0, 1, 0, 1, 0]  # 1: Kayıp, 0: Sadık
}

df = pd.DataFrame(veri)

# Model için verileri ayırma
X_egitim = df[["Yas", "AylikOdeme", "HizmetSuresi"]]
y_egitim = df["Churn"]

# 2. Makine Öğrenmesi Modeli (Sınıflandırma)
model = LogisticRegression()
model.fit(X_egitim, y_egitim)

@app.get("/")
def anasayfa():
    return {"mesaj": "Pandas ve Seaborn destekli Müşteri Kaybı Tahmin Motoru aktif!"}

@app.get("/churn-tahmin")
def churn_tahmin(yas: float, aylik_odeme: float, hizmet_suresi: float):
    # Yeni müşteriyi Pandas DataFrame formatında modele veriyoruz
    yeni_musteri = pd.DataFrame([[yas, aylik_odeme, hizmet_suresi]], columns=["Yas", "AylikOdeme", "HizmetSuresi"])
    
    tahmin_sonucu = int(model.predict(yeni_musteri)[0])
    olasiliklar = model.predict_proba(yeni_musteri)
    ayrilma_ihtimali = float(olasiliklar[0][1]) * 100
    
    durum = "Yüksek Kayıp Riski (Churn)" if tahmin_sonucu == 1 else "Sadık Müşteri (Kalacak)"
    
    return {
        "girilen_yas": yas,
        "girilen_aylik_odeme": aylik_odeme,
        "hizmet_suresi_ay": hizmet_suresi,
        "tahmin_durumu": durum,
        "ayrilma_olasiligi_yuzde": round(ayrilma_olasiligi, 2)
    }

@app.get("/grafik-olustur")
def grafik_olustur():
    # Seaborn ile veri analizi ve görselleştirme
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="AylikOdeme", y="HizmetSuresi", hue="Churn", palette="Set1", s=100)
    plt.title("Müşteri Aylık Ödeme ve Hizmet Süresi Dağılımı")
    plt.xlabel("Aylık Ödeme (TL)")
    plt.ylabel("Hizmet Süresi (Ay)")
    
    # Grafiği diske kaydetmek yerine hafızadaki akışa (buffer) aktarıyoruz
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    
    # Doğrudan tarayıcıya resim olarak döndürüyoruz
    return StreamingResponse(buf, media_type="image/png")