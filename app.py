import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
import math

st.set_page_config(layout="wide")
st.title("AI Stamp Rebuilder V6 - RING OCR ENGINE")

CONF_THRESHOLD = 0.70

# =====================================
# UTIL
# =====================================

def resize_for_analysis(img, max_dim=1200):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def detect_circle(gray):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=80,
        maxRadius=2000
    )
    if circles is not None:
        return circles[0][0]
    return None

def enhance_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return gray

def process_result(result):
    if not result:
        return "", 0.0

    best_text = ""
    best_conf = 0.0

    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            text = item[1]
            conf = item[2]
            if len(text) > len(best_text):
                best_text = text
                best_conf = conf

    return best_text, best_conf

def generate_clean_stamp(text_top, text_mid, text_bot, diameter_cm=5):
    dpi = 300
    px = int((diameter_cm / 2.54) * dpi)

    img = Image.new("RGBA", (px, px), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    center = px // 2

    draw.ellipse((20,20,px-20,px-20), outline="blue", width=10)

    try:
        font = ImageFont.truetype("arial.ttf", int(px*0.08))
    except:
        font = ImageFont.load_default()

    draw.text((center, center-80), text_top, fill="blue", font=font, anchor="mm")
    draw.text((center, center), text_mid, fill="blue", font=font, anchor="mm")
    draw.text((center, center+80), text_bot, fill="blue", font=font, anchor="mm")

    return img

# =====================================
# MAIN
# =====================================

uploaded = st.file_uploader("Upload Foto Stempel", type=["png","jpg","jpeg"])

if uploaded:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_analysis = resize_for_analysis(img)

    gray = cv2.cvtColor(img_analysis, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9,9), 2)

    circle = detect_circle(gray)

    if circle is None:
        st.warning("Lingkaran tidak terdeteksi → Mode Manual")
    else:
        x, y, r = map(int, circle)

        # 🔥 RING MASK
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x,y), r, 255, -1)
        cv2.circle(mask, (x,y), int(r*0.75), 0, -1)

        ring = cv2.bitwise_and(img_analysis, img_analysis, mask=mask)

        st.subheader("Ring Area for OCR")
        st.image(ring, channels="BGR")

        enhanced = enhance_for_ocr(ring)
        st.image(enhanced, channels="GRAY")

        reader = easyocr.Reader(['id','en'], gpu=False)
        result = reader.readtext(enhanced, detail=1)

        text_detected, conf = process_result(result)

        st.write(f"OCR Confidence: {round(conf,2)}")

        text_top = st.text_input("Edit Teks Atas:", text_detected)
        text_mid = st.text_input("Edit Teks Tengah:")
        text_bot = st.text_input("Edit Teks Bawah:")

        if conf < CONF_THRESHOLD:
            st.error("Confidence rendah. Koreksi sebelum generate.")
        else:
            if st.button("Generate Clean"):
                clean = generate_clean_stamp(text_top, text_mid, text_bot)
                st.image(clean)
