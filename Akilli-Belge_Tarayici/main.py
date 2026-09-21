import os
import cv2
import numpy as np
import streamlit as st
from fpdf import FPDF

st.set_page_config(
    page_title="Akıllı Belge Tarayıcı", page_icon="📄", layout="wide"
)

# --- YARDIMCI FONKSİYONLAR (GÖRÜNTÜ İŞLEME) ---


def order_points(pts):
  """Köşe noktalarını sıralar: sol-üst, sağ-üst, sağ-alt, sol-alt"""
  rect = np.zeros((4, 2), dtype="float32")
  s = pts.sum(axis=1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
  diff = np.diff(pts, axis=1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  return rect


def four_point_transform(image, pts):
  """Kuşbakışı (Perspective Transform) dönüşümü yapar"""
  rect = order_points(pts)
  (tl, tr, br, bl) = rect

  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))

  dst = np.array(
      [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
      dtype="float32",
  )

  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  return warped


def process_document(image):
  """Belgeyi algılar, köşe bulur ve fotokopi efekti uygular"""
  orig = image.copy()
  ratio = image.shape[0] / 500.0
  image_resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

  gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  edged = cv2.Canny(blurred, 75, 200)

  cnts, _ = cv2.findContours(
      edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
  )
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

  screenCnt = None
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
      screenCnt = approx
      break

  if screenCnt is not None:
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
  else:
    warped = orig  # Bulamazsa orijinalini döndür

  # Fotokopi Efekti (Adaptive Threshold)
  warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
  tresh = cv2.adaptiveThreshold(
      warped_gray,
      255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY,
      11,
      10,
  )
  return tresh


def save_as_pdf(image):
  """Taranan resmi PDF formatına dönüştürür"""
  temp_img_path = "temp_scanned.jpg"
  cv2.imwrite(temp_img_path, image)

  pdf = FPDF()
  pdf.add_page()
  pdf.image(temp_img_path, x=10, y=10, w=190)
  pdf_output_path = "taranmis_belge.pdf"
  pdf.output(pdf_output_path)

  if os.path.exists(temp_img_path):
    os.remove(temp_img_path)
  return pdf_output_path


# --- ARAYÜZ (STREAMLIT) ---
st.title("📄 Profesyonel Akıllı Belge Tarayıcı")
st.markdown(
    "YOLO kullanmadan saf matematiksel geometri ve kontur analiziyle"
    " belgelerinizi tarayın, PDF olarak kaydedin."
)

menu = st.sidebar.selectbox(
    "Mod Seçin", ["Fotoğraf Yükle", "Kamera / Video Akışı"]
)

if menu == "Fotoğraf Yükle":
  st.subheader("📷 Fotoğraf Üzerinden Belge Tarama")
  uploaded_file = st.file_uploader(
      "Bir belge fotoğrafı seçin...", type=["jpg", "jpeg", "png"]
  )

  if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
      st.image(
          cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB),
          caption="Orijinal Görüntü",
          use_container_width=True,
      )

    if st.button("Belgeyi Tara ve Düzelt"):
      with st.spinner("Belge taranıyor ve perspektif ayarlanıyor..."):
        scanned_image = process_document(opencv_image)

      with col2:
        st.image(
            scanned_image,
            caption="Taranmış Belge (Fotokopi Modu)",
            use_container_width=True,
            clamp=True,
        )

      # PDF İndirme Butonu
      pdf_path = save_as_pdf(scanned_image)
      with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📥 PDF Olarak İndir",
            data=pdf_file,
            file_name="akilli_belge.pdf",
            mime="application/pdf",
        )

elif menu == "Kamera / Video Akışı":
  st.subheader("📹 Canlı Kamera ile Belge Yakalama")
  run_camera = st.checkbox("Kamerayı Aç")

  camera_placeholder = st.empty()
  capture_button = st.button("Anlık Görüntü Yakala ve Tara")

  if run_camera:
    cap = cv2.VideoCapture(0)
    while run_camera:
      ret, frame = cap.read()
      if not ret:
        st.error("Kamera açılamadı!")
        break
      camera_placeholder.image(
          cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB"
      )

      if capture_button:
        scanned_image = process_document(frame)
        st.image(
            scanned_image,
            caption="Yakalanan ve Taranan Belge",
            use_container_width=True,
            clamp=True,
        )
        pdf_path = save_as_pdf(scanned_image)
        with open(pdf_path, "rb") as pdf_file:
          st.download_button(
              label="📥 Taranan Belgeyi PDF İndir",
              data=pdf_file,
              file_name="kamera_belge.pdf",
              mime="application/pdf",
          )
        break
    cap.release()