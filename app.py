import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
import math

st.set_page_config(layout="wide")
st.title("AI Stamp Rebuilder V3 - STABLE OCR ENGINE")

CONF_THRESHOLD = 0.80

# =====================================
# UTIL FUNCTIONS
# =====================================

def resize_for_analysis(img, max_dim=1200):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def detect_circle(gray):
    attempts = [
        {"dp":1.2, "param2":30},
        {"dp":1.0, "param2":20},
        {"dp":1.5, "param2":40},
    ]

    for p in attempts:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=p["dp"],
            minDist=100,
            param1=50,
            param2=p["param2"],
            minRadius=80,
            maxRadius=2000
        )
        if circles is not None:
            return circles[0][0]

    return None

# 🔥 PREPROCESSING UPGRADE
def enhance_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE (auto contrast pintar)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Sharpen ringan
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Adaptive threshold (lebih stabil dari global threshold)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    return thresh

def generate_clean_stamp(text_top, text_mid, text_bot, diameter_cm=5):
    dpi = 300
    px = int((diameter_cm / 2.54) * dpi)

    img = Image.new("RGBA", (px, px), (255,255,255,0))
    draw = ImageDraw.Draw(img)

    center = px // 2
    radius = px // 2 - 20

    draw.ellipse((20,20,px-20,px-20), outline="blue", width=10)

    try:
        font_top = ImageFont.truetype("arial.ttf", int(px*0.07))
        font_mid = ImageFont.truetype("arial.ttf", int(px*0.09))
    except:
        font_top = ImageFont.load_default()
        font_mid = ImageFont.load_default()

    draw.text((center, center-70), text_top, fill="blue", font=font_top, anchor="mm")
    draw.text((center, center), text_mid, fill="blue", font=font_mid, anchor="mm")
    draw.text((center, center+70), text_bot, fill="blue", font=font_top, anchor="mm")

    return img

def process_result(result):
    if len(result) == 0:
        return "", 0
    text = " ".join([r[1] for r in result])
    conf = np.mean([r[2] for r in result])
    return text, conf

# =====================================
# MAIN FLOW
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

        text_top = st.text_input("Teks Atas")
        text_mid = st.text_input("Teks Tengah")
        text_bot = st.text_input("Teks Bawah")

        if st.button("Generate Clean (Manual Mode)"):
            clean = generate_clean_stamp(text_top, text_mid, text_bot)
            st.image(clean)

    else:
        x, y, r = map(int, circle)

        mask = np.zeros_like(gray)
        cv2.circle(mask, (x,y), r, 255, -1)
        isolated = cv2.bitwise_and(img_analysis, img_analysis, mask=mask)

        st.subheader("Area Stempel Terdeteksi")
        st.image(isolated, channels="BGR")

        # 🔥 PREPROCESS SEBELUM OCR
        enhanced = enhance_for_ocr(isolated)

        st.subheader("Enhanced for OCR")
        st.image(enhanced, channels="GRAY")

        h, w = enhanced.shape[:2]
        top_part = enhanced[0:h//2, :]
        bottom_part = enhanced[h//2:h, :]

        reader = easyocr.Reader(['id','en'], gpu=False)

        result_top = reader.readtext(top_part)
        result_bottom = reader.readtext(bottom_part)

        text_top_detected, conf_top = process_result(result_top)
        text_bottom_detected, conf_bottom = process_result(result_bottom)

        avg_conf = (conf_top + conf_bottom) / 2

        st.write(f"OCR Confidence: {round(avg_conf,2)}")

        text_top = st.text_input("Edit Teks Atas:", text_top_detected)
        text_mid = st.text_input("Edit Teks Tengah:")
        text_bot = st.text_input("Edit Teks Bawah:", text_bottom_detected)

        if avg_conf < CONF_THRESHOLD:
            st.error("Confidence rendah. Wajib koreksi teks sebelum generate.")
            allow_generate = False
        else:
            allow_generate = True

        if allow_generate and st.button("Generate Clean"):
            clean = generate_clean_stamp(text_top, text_mid, text_bot)
            st.image(clean)
