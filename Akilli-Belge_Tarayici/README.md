# 📄 Profesyonel Akıllı Belge Tarayıcı (Smart Document Scanner)

YOLO veya hazır yapay zeka ağırlık dosyaları (weights) kullanılmadan, tamamen **saf matematiksel geometri, kontur analizi ve görüntü işleme** teknikleriyle geliştirilmiş akıllı bir belge tarayıcı uygulamasıdır.

---

## 🚀 Projenin Amacı ve Özellikleri
Bu proje, telefon veya web kamerasından alınan eğri/açılı belge fotoğraflarını algılayarak onları kusursuz bir şekilde **kuşbakışı (bird's-eye view)** formata getirir ve fotokopi kalitesinde siyah-beyaz belge çıktısına dönüştürür.

* **📷 Fotoğraf Yükleme Desteği:** Bilgisayarınızdan `.jpg`, `.png` formatındaki belgeleri yükleyip tarayabilme.
* **📹 Canlı Kamera Desteği:** Canlı video akışı üzerinden anlık belge yakalama.
* **📐 Perspektif Düzeltme (Warp Perspective):** Belgenin 4 köşesini otomatik algılayıp ekrana dikleştirme.
* **✨ Adaptif Eşikleme (Adaptive Threshold):** Gölgelemeri silerek belgelere net fotokopi görünümü kazandırma.
* **📥 PDF Çıktısı:** Taranan belgeleri tek tuşla `PDF` formatında indirebilme.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* **Python**
* **OpenCV (`cv2`)**: Görüntü işleme, kontur bulma ve perspektif dönüşümleri için.
* **NumPy**: Matris ve geometri hesaplamaları için.
* **Streamlit**: Modern ve hızlı web arayüzü oluşturmak için.
* **FPDF2**: Taranan belgeleri PDF belgesine dönüştürmek için.

---

## ⚙️ Kurulum ve Çalıştırma

1. Proje klasörüne gidin:
   ```bash
   cd Akilli-Belge_Tarayici