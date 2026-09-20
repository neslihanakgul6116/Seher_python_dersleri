# Endüstriyel Yapay Zeka ve Veri Bilimi Portföyü 🚀

Bu repository, Python, FastAPI, OpenCV, NumPy ve Matplotlib gibi modern teknolojiler kullanılarak geliştirilmiş **6 farklı endüstriyel yapay zeka ve veri analizi projesini** içermektedir. Projeler, üretim, e-ticaret, finans ve web analitiği gibi gerçek dünya senaryolarına çözüm üretmek amacıyla tasarlanmıştır.

---

## 🛠️ Proje Listesi ve İçerikler

### 1. Dinamik Fiyatlandırma (Dynamic Pricing)
* **Açıklama:** Pazar talebine ve rekabet koşullarına göre ürün fiyatlarını otomatik optimize eden algoritma.
* **Teknolojiler:** Python, FastAPI, NumPy, SciPy

### 2. Müşteri Kaybı Tahmini (Customer Churn Prediction)
* **Açıklama:** Müşterilerin şirketten ayrılma (churn) olasılıklarını analiz ederek erken uyarı üreten model.
* **Teknolojiler:** Python, Veri Analizi, FastAPI

### 3. Akıllı Bant Ürün Sayacı (Smart Conveyor Counting)
* **Açıklama:** Konveyör bant üzerindeki ürünleri bilgisayarlı görü ve nesne tespitiyle otomatik sayan sistem.
* **Teknolojiler:** Python, OpenCV, YOLOv8 (`yolov8n.pt`)

### 4. Borsa Tepe / Dip Sinyalleyicisi (Stock Peak & Dip Detector)
* **Açıklama:** Finansal zaman serisi verilerinde tepe ve dip noktalarını tespit ederek alım/satım sinyalleri üreten modül.
* **Teknolojiler:** Python, NumPy, Matplotlib, FastAPI

### 5. Üretim Hattı Çatlak Dedektörü (Production Line Crack Detector)
* **Açıklama:** Üretim hattından geçen malzemelerin yüzeyindeki hataları ve çatlakları Canny kenar tespitiyle saptayan sistem.
* **Teknolojiler:** Python, OpenCV, FastAPI

### 6. Web Tıklama Isı Haritası Analizi (Web Click Heatmap Analysis)
* **Açıklama:** Kullanıcıların web sitesi üzerinde en çok tıkladığı alanları görselleştirerek UI/UX optimizasyonu sağlayan ısı haritası aracı.
* **Teknolojiler:** Python, FastAPI, Matplotlib, NumPy

## 7. Profesyonel Akıllı Belge Tarayıcı (Document Scanner)

* **Açıklama:** OpenCV ve Streamlit kullanarak gerçek zamanlı belge algılama, perspektif düzeltme, tarayıcı filtresi uygulama ve PDF olarak dışa aktarma özelliklerine sahip masaüstü web uygulaması.
* **Teknolojiler:** Python, OpenCV, NumPy, Streamlit, FPDF2

---

## 🚀 Kurulum ve Çalıştırma

Depoyu yerel bilgisayarınıza klonladıktan sonra ilgili proje klasörüne gidip FastAPI sunucularını Uvicorn ile ayağa kaldırabilirsiniz:

```bash
# Projeyi klonlayın
git clone [https://github.com/neslihanakgul6116/Seher_python_dersleri.git](https://github.com/neslihanakgul6116/Seher_python_dersleri.git)

# İlgili proje klasörüne gidin (Örn: 6. Proje)
cd 2026.09.20/06_Web_Tiklama_Heatmap_Analizi

# Uvicorn ile sunucuyu başlatın
python -m uvicorn app:app --reload

