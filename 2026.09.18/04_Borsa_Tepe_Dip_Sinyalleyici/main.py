from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go

app = FastAPI(title="Borsa Tepe/Dip Sinyalleyici ve Algoritmik Ticaret Servisi")

@app.get("/")
def anasayfa():
    return {
        "mesaj": "Borsa Tepe/Dip Sinyalleyici Aktif!",
        "grafik_icin": "Tarayıcınıza http://127.0.0.1:8000/borsa-grafik yazın."
    }

@app.get("/borsa-grafik", response_class=HTMLResponse)
def borsa_grafik():
    # 1. Yapay Borsa Fiyat Verisi Üretme (Gürültülü Sinüs Dalgası)
    np.random.seed(42)
    x = np.linspace(0, 50, 100)
    fiyatlar = 100 + np.sin(x) * 20 + np.random.normal(0, 2, 100) # Gürültülü fiyatlar
    
    df = pd.DataFrame({"Zaman": range(100), "Fiyat": fiyatlar})

    # 2. SciPy ile Tepe (Zirve) ve Dip (Çukur) Noktalarını Bulma
    # distance: tepeler arası minimum mesafe, prominence: tepe belirginliği
    tepeler, _ = find_peaks(df["Fiyat"], distance=10, prominence=2)
    dipler, _ = find_peaks(-df["Fiyat"], distance=10, prominence=2)

    # 3. Plotly ile İnteraktif Finansal Grafik Oluşturma
    fig = go.Figure()

    # Fiyat Çizgisi
    fig.add_trace(go.Scatter(
        x=df["Zaman"], y=df["Fiyat"],
        mode="lines+markers",
        name="Borsa Fiyatı",
        line=dict(color="royalblue", width=2)
    ))

    # Tepe Noktaları (Sat Sinyali - Kırmızı)
    fig.add_trace(go.Scatter(
        x=tepeler, y=df["Fiyat"].iloc[tepeler],
        mode="markers",
        name="Tepe Noktası (SAT Sinyali)",
        marker=dict(color="red", size=12, symbol="triangle-down")
    ))

    # Dip Noktaları (Al Sinyali - Yeşil)
    fig.add_trace(go.Scatter(
        x=dipler, y=df["Fiyat"].iloc[dipler],
        mode="markers",
        name="Dip Noktası (AL Sinyali)",
        marker=dict(color="green", size=12, symbol="triangle-up")
    ))

    fig.update_layout(
        title="Algoritmik Borsa Tepe ve Dip Sinyalleri (SciPy & Plotly)",
        xaxis_title="Zaman / Periyot",
        yaxis_title="Fiyat (TL)",
        template="plotly_dark"
    )

    # HTML olarak tarayıcıya döndürme
    return fig.to_html(include_plotlyjs='cdn')