from fastapi import FastAPI ,HTTPException
from pydantic import BaseModel
app=FastAPI(title="E-Ticaret & Sepet API'si")

class Product(BaseModel):
    id:int
    name:str
    price:float
    stock:int=10

class CartItem(BaseModel):
    product_id:int
    quantity:int=1

urunler_db=[
    Product(id=1, name="Kablosuz Kulaklık", price=1200.0, stock=15),
    Product(id=2, name="Akıllı Saat", price=3500.0, stock=5),
]
sepet_db = []

# --- 3. ENDPOINT'LER (ADRESLER) 📍 ---

@app.get("/products", summary="Tüm Ürünleri Listele")
def urunleri_getir():
    return urunler_db

@app.post("/products", summary="Yeni Ürün Ekle")
def urun_ekle(product: Product):
    urunler_db.append(product)
    return {"mesaj": "Ürün başarıyla eklendi!", "urun": product}

@app.post("/cart/add", summary="Sepete Ürün Ekle")
def sepete_ekle(item: CartItem):
    # Ürün veritabanında var mı kontrol edelim 🔍
    urun = next((u for u in urunler_db if u.id == item.product_id), None)
    if not urun:
        raise HTTPException(status_code=404, detail="Ürün bulunamadı!")
    
    sepet_db.append(item)
    return {"mesaj": f"{urun.name} sepete eklendi!", "sepet": sepet_db}

@app.get("/cart", summary="Sepeti ve Toplam Tutarı Göster")
def sepeti_getir():
    toplam = 0.0
    for item in sepet_db:
        urun = next((u for u in urunler_db if u.id == item.product_id), None)
        if urun:
            toplam += urun.price * item.quantity
            
    return {"sepet_icerigi": sepet_db, "toplam_tutar": toplam}