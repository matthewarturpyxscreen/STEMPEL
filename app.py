import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
import math

st.set_page_config(layout="wide")
st.title("AI Stamp Rebuilder V7 - FULL AUTO")

CONF_THRESHOLD = 0.60

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
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

# 🔥 RING UNWRAP
def unwrap_ring(image, center, outer_r, inner_r):
    flags = cv2.WARP_POLAR_LINEAR
    max_radius = outer_r

    polar = cv2.warpPolar(
        image,
        (int(2 * math.pi * max_radius), outer_r),
        center,
        max_radius,
        flags
    )

    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

    ring_height = outer_r - inner_r
    ring_only = polar[inner_r:outer_r, :]

    return ring_only

def process_result(result):
    if not result:
        return "", 0.0

    texts = []
    confs = []

    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            texts.append(item[1])
            confs.append(item[2])

    if not texts:
        return "", 0.0

    full_text = " ".join(texts)
    avg_conf = np.mean(confs)

    return full_text, avg_conf

def generate_clean_stamp(text_top, text_mid, text_bot, diameter_cm=5):
    dpi = 300
    px = int((diameter_cm / 2.54) * dpi)

    img = Image.new("RGBA", (px, px), (255,255,255,0))
    draw = ImageDraw.Draw(img)

    center = px // 2
    radius = px // 2 - 20

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
        st.error("Lingkaran tidak terdeteksi.")
    else:
        x, y, r = map(int, circle)

        outer_r = r
        inner_r = int(r * 0.70)

        ring_strip = unwrap_ring(img_analysis, (x,y), outer_r, inner_r)

        st.subheader("Unwrapped Ring")
        st.image(ring_strip, channels="BGR")

        enhanced = enhance_for_ocr(ring_strip)
        st.image(enhanced, channels="GRAY")

        h = enhanced.shape[0]
        top_strip = enhanced[:h//2, :]
        bottom_strip = enhanced[h//2:, :]

        reader = easyocr.Reader(['id','en'], gpu=False)

        result_top = reader.readtext(top_strip, detail=1)
        result_bottom = reader.readtext(bottom_strip, detail=1)

        text_top, conf_top = process_result(result_top)
        text_bottom, conf_bottom = process_result(result_bottom)

        avg_conf = (conf_top + conf_bottom) / 2

        st.write(f"OCR Confidence: {round(avg_conf,2)}")

        text_top = st.text_input("Teks Atas:", text_top)
        text_mid = st.text_input("Teks Tengah:")
        text_bot = st.text_input("Teks Bawah:", text_bottom)

        if avg_conf >= CONF_THRESHOLD:
            if st.button("Generate Clean"):
                clean = generate_clean_stamp(text_top, text_mid, text_bot)
                st.image(clean)
        else:
            st.warning("Confidence rendah, koreksi manual disarankan.")
