import cv2
import numpy as np
import streamlit as st
import tempfile
import os
from fpdf import FPDF

# Sayfa Yapılandırması
st.set_page_config(page_title="Profesyonel Akıllı Belge Tarayıcı", layout="wide")

st.title("📄 Profesyonel Akıllı Belge Tarayıcı (Document Scanner)")
st.write("Görüntü işleme teknikleri ile belgelerinizi otomatik algılayın, perspektifini düzeltin, filtreleyin ve PDF olarak indirin!")

# Sidebar (Yan Menü) Ayarları
st.sidebar.header("Ayarlar ve İşlem Modu")
secenek = st.sidebar.selectbox("Mod Seçin", ["Fotoğraf Tarama", "Video Akışı Tarama"])

# 1. Köşe Sıralama Fonksiyonu (Perspektif için)
def sirala_kose_noktalari(kose_noktalari):
    dikdortgen = np.zeros((4, 2), dtype="float32")
    toplam = kose_noktalari.sum(axis=1)
    dikdortgen[0] = kose_noktalari[np.argmin(toplam)] # Sol üst
    dikdortgen[2] = kose_noktalari[np.argmax(toplam)] # Sağ alt
    
    fark = np.diff(kose_noktalari, axis=1)
    dikdortgen[1] = kose_noktalari[np.argmin(fark)]   # Sağ üst
    dikdortgen[3] = kose_noktalari[np.argmax(fark)]   # Sol alt
    return dikdortgen

# 2. Perspektif Düzeltme ve Filtreleme Fonksiyonu
def belgeyi_isle(image):
    gri = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bulanik = cv2.GaussianBlur(gri, (5, 5), 0)
    kenar = cv2.Canny(bulanik, 75, 200)
    
    konturlar, _ = cv2.findContours(kenar.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)[:5]
    
    kose_noktalari = None
    for c in konturlar:
        alan = cv2.contourArea(c)
        cevre = cv2.arcLength(c, True)
        yaklasik = cv2.approxPolyDP(c, 0.02 * cevre, True)
        
        if len(yaklasik) == 4 and alan > 1000:
            kose_noktalari = yaklasik.reshape(4, 2)
            break
            
    if kose_noktalari is not None:
        dikdortgen = sirala_kose_noktalari(kose_noktalari)
        (tl, tr, br, bl) = dikdortgen
        
        genislikA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        genislikB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maks_genislik = max(int(genislikA), int(genislikB))
        
        yukseklikA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        yukseklikB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maks_yukseklik = max(int(yukseklikA), int(yukseklikB))
        
        hedef_noktalar = np.array([
            [0, 0],
            [maks_genislik - 1, 0],
            [maks_genislik - 1, maks_yukseklik - 1],
            [0, maks_yukseklik - 1]], dtype="float32")
            
        matris = cv2.getPerspectiveTransform(dikdortgen, hedef_noktalar)
        duzeltilmis = cv2.warpPerspective(image, matris, (maks_genislik, maks_yukseklik))
        
        # Tarayıcı Filtresi (Adaptive Thresholding)
        gri_duzeltilmis = cv2.cvtColor(duzeltilmis, cv2.COLOR_BGR2GRAY)
        filtrelenmis = cv2.adaptiveThreshold(
            gri_duzeltilmis, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 10
        )
        return filtrelenmis, True
    else:
        return image, False

# 3. PDF Oluşturma Fonksiyonu
def resmi_pdf_yap(islenmis_resim):
    gecici_dizin = tempfile.gettempdir()
    pdf_yolu = os.path.join(gecici_dizin, "taranmis_belge.pdf")
    img_yolu = os.path.join(gecici_dizin, "gecici_belge.jpg")
    
    # Resmi geçici olarak kaydet (Eğer siyah-beyazsa RGB'ye çevir)
    if len(islenmis_resim.shape) == 2:
        islenmis_resim = cv2.cvtColor(islenmis_resim, cv2.COLOR_GRAY2BGR)
        
    cv2.imwrite(img_yolu, islenmis_resim)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.image(img_yolu, x=15, y=15, w=180)
    pdf.output(pdf_yolu)
    return pdf_yolu

# FOTOĞRAF MODU
if secenek == "Fotoğraf Tarama":
    st.subheader("📷 Belge Fotoğrafı Yükle")
    yuklenen_dosya = st.file_uploader("Bir belge veya kağıt fotoğrafı seçin...", type=["jpg", "jpeg", "png"])
    
    if yuklenen_dosya is not None:
        dosya_bytes = np.asarray(bytearray(yuklenen_dosya.read()), dtype=np.uint8)
        orijinal_resim = cv2.imdecode(dosya_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Orijinal Görüntü")
            st.image(cv2.cvtColor(orijinal_resim, cv2.COLOR_BGR2RGB), use_column_width=True)
            
        # Belgeyi işle
        islenmis, basarili = belgeyi_isle(orijinal_resim)
        
        with col2:
            st.subheader("Taranmış & Filtrelenmiş Sonuç")
            if len(islenmis.shape) == 2:
                st.image(islenmis, channels="GRAY", use_column_width=True)
            else:
                st.image(cv2.cvtColor(islenmis, cv2.COLOR_BGR2RGB), use_column_width=True)
                
        if basarili:
            st.success("Kağıt başarıyla tespit edildi, perspektif düzeltildi ve tarayıcı filtresi kullanıldı!")
            
            # PDF İndirme Butonu
            pdf_dosya_yolu = resmi_pdf_yap(islenmis)
            with open(pdf_dosya_yolu, "rb") as f:
                st.download_button(
                    label="📥 Taranan Belgeyi PDF Olarak İndir",
                    data=f,
                    file_name="akilli_belge.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Kağıt tam olarak algılanamadı. Lütfen arka planı kontrast bir zemin yapın.")

# VİDEO MODU
elif secenek == "Video Akışı Tarama":
    st.subheader("🎬 Belge Videosu Yükle")
    video_dosya = st.file_uploader("Bir video dosyası seçin...", type=["mp4", "avi", "mov"])
    
    if video_dosya is not None:
        t_file = tempfile.NamedTemporaryFile(delete=False)
        t_file.write(video_dosya.read())
        
        cap = cv2.VideoCapture(t_file.name)
        st_frame = st.empty()
        
        son_islenmis_kare = None 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            islenmis_kare, basarili = belgeyi_isle(frame)
            if basarili:
                son_islenmis_kare = islenmis_kare # Başarılı taranan kareyi hafızada tut
                
            if len(islenmis_kare.shape) == 2:
                gosterilecek = cv2.cvtColor(islenmis_kare, cv2.COLOR_GRAY2RGB)
            else:
                gosterilecek = cv2.cvtColor(islenmis_kare, cv2.COLOR_BGR2RGB)
                
            st_frame.image(gosterilecek, channels="RGB", use_container_width=True)
            
        cap.release()
        st.success("Video işleme tamamlandı!")
        
        # Eğer videoda belge yakalandıysa, siyah ekran yerine son taranan net belgeyi ekranda sabit göster ve PDF butonu ver
        if son_islenmis_kare is not None:
            if len(son_islenmis_kare.shape) == 2:
                st.image(son_islenmis_kare, channels="GRAY", caption="Videodan Yakalanan Son Taranmış Belge", use_container_width=True)
            else:
                st.image(cv2.cvtColor(son_islenmis_kare, cv2.COLOR_BGR2RGB), caption="Videodan Yakalanan Son Taranmış Belge", use_container_width=True)
                
            pdf_dosya_yolu = resmi_pdf_yap(son_islenmis_kare)
            with open(pdf_dosya_yolu, "rb") as f:
                st.download_button(
                    label="📥 Videoda Yakalanan Belgeyi PDF Olarak İndir",
                    data=f,
                    file_name="video_taranan_belge.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Videoda net bir kağıt/belge algılanamadı.")