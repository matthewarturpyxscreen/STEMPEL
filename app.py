import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from io import BytesIO
import math

st.set_page_config(layout="wide")
st.title("AI Stamp Rebuilder V2 - STRICT MODE")

CONF_THRESHOLD = 0.85

# ==============================
# UTIL FUNCTIONS
# ==============================

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

def polar_unwrap(image, center, radius):
    height = radius
    width = int(2 * math.pi * radius)

    polar = cv2.warpPolar(
        image,
        (width, height),
        center,
        radius,
        cv2.WARP_POLAR_LINEAR
    )

    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return polar

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

# ==============================
# MAIN FLOW
# ==============================

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

        unwrap = polar_unwrap(img_analysis, (x,y), r)

        st.subheader("Hasil Polar Unwrap")
        st.image(unwrap, channels="BGR")

        h = unwrap.shape[0]
        top_part = unwrap[:h//2, :]
        bottom_part = unwrap[h//2:, :]

        reader = easyocr.Reader(['id','en'])

        result_top = reader.readtext(top_part)
        result_bottom = reader.readtext(bottom_part)

        def process_result(result):
            if len(result) == 0:
                return "", 0
            text = " ".join([r[1] for r in result])
            conf = np.mean([r[2] for r in result])
            return text, conf

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
