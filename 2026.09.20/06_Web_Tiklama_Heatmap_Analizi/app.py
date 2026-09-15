from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = FastAPI(title="Web Tıklama Heatmap Analizi")

@app.get("/")
def anasayfa():
    return {"mesaj": "Web Tıklama Heatmap Analizi Aktif!", "endpoint": "/heatmap"}

@app.get("/heatmap")
def heatmap_olustur():
    np.random.seed(42)
    x = np.concatenate([
        np.random.normal(300, 50, 500),
        np.random.normal(700, 80, 800),
        np.random.normal(500, 150, 1200)
    ])
    y = np.concatenate([
        np.random.normal(200, 30, 500),
        np.random.normal(200, 40, 800),
        np.random.normal(600, 100, 1200)
    ])

    fig, ax = plt.subplots(figsize=(10, 6))
    hb = ax.hexbin(x, y, gridsize=40, cmap='inferno', mincnt=1)
    
    ax.set_title("Web Sitesi Kullanici Tiklama Isı Haritası", fontsize=14, color="darkblue")
    ax.set_xlabel("X Koordinati")
    ax.set_ylabel("Y Koordinati")
    ax.set_ylim(800, 0)
    ax.set_xlim(0, 1000)
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Yogunluk")

    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight', dpi=150)
    img_io.seek(0)
    plt.close(fig)

    return StreamingResponse(img_io, media_type="image/png")