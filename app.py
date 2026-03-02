import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import easyocr
import math
import io

st.set_page_config(page_title="Stamp Rebuilder Pro", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,.stApp{background:#0f1117!important;color:#f1f5f9;font-family:'Inter',sans-serif}
.stFileUploader>div{border:2px dashed #334155!important;border-radius:12px!important;background:#1e293b!important}
.stTextInput input{background:#1e293b!important;color:#f1f5f9!important;border:1px solid #334155!important;border-radius:8px!important;font-family:'JetBrains Mono',monospace!important}
.stButton button{background:#4f46e5!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:10px 20px!important}
.stButton button:hover{background:#4338ca!important}
.block{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;margin-bottom:14px}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace}
.badge-green{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.3)}
.badge-red{background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3)}
.badge-yellow{background:rgba(234,179,8,.15);color:#fbbf24;border:1px solid rgba(234,179,8,.3)}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:20px 24px;background:linear-gradient(135deg,#3730a3,#6366f1);
  border-radius:14px;margin-bottom:24px;display:flex;align-items:center;gap:16px">
  <div style="font-size:32px">🔏</div>
  <div>
    <div style="font-size:20px;font-weight:700;color:#fff">Stamp Rebuilder Pro</div>
    <div style="font-size:12px;color:rgba(255,255,255,.65)">Upload foto stempel → deteksi otomatis → rebuild bersih</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_reader():
    return easyocr.Reader(['id', 'en'], gpu=False)


def preprocess_stamp(img_bgr):
    """Isolasi stempel: hapus background kertas, pertajam tinta."""
    # Scale jika terlalu besar
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1400:
        scale = 1400 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))

    # Pisahkan channel biru (stempel biru lebih dominan di B)
    b, g, r = cv2.split(img_bgr)
    # Stempel biru: B tinggi, R rendah
    blue_mask = cv2.subtract(b.astype(np.int16), r.astype(np.int16))
    blue_mask = np.clip(blue_mask, 0, 255).astype(np.uint8)

    # Juga coba grayscale + CLAHE
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Gabungkan: ambil max dari keduanya
    combined = cv2.max(blue_mask, cv2.subtract(255, gray_eq))

    # Denoise + threshold
    denoised = cv2.fastNlMeansDenoising(combined, h=10)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology untuk menutup gap
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return img_bgr, cleaned


def detect_circle_robust(gray_clean):
    """Deteksi lingkaran dengan multiple param, ambil yang terbaik."""
    blurred = cv2.GaussianBlur(gray_clean, (9, 9), 2)

    best = None
    for dp in [1.0, 1.2, 1.5]:
        for p2 in [25, 35, 45]:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=dp, minDist=50,
                param1=50, param2=p2,
                minRadius=60, maxRadius=2000
            )
            if circles is not None:
                # Ambil lingkaran terbesar (kemungkinan border stempel)
                c = sorted(circles[0], key=lambda x: -x[2])[0]
                if best is None or c[2] > best[2]:
                    best = c
    return best


def crop_stamp_region(img_bgr, circle):
    """Crop area stempel dengan padding."""
    x, y, r = map(int, circle)
    pad = int(r * 0.12)
    x1 = max(0, x - r - pad)
    y1 = max(0, y - r - pad)
    x2 = min(img_bgr.shape[1], x + r + pad)
    y2 = min(img_bgr.shape[0], y + r + pad)
    return img_bgr[y1:y2, x1:x2], (x - x1, y - y1, r)


def unwrap_ring(img_bgr, center, outer_r, inner_r):
    """Polar unwrap untuk baca teks melingkar."""
    cx, cy = center
    out_w = int(2 * math.pi * outer_r)
    out_h = outer_r

    flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS
    polar = cv2.warpPolar(img_bgr, (out_w, out_h), (cx, cy), outer_r, flags)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

    ring = polar[inner_r:outer_r, :]
    return ring


def enhance_for_ocr(img):
    """Siapkan gambar untuk OCR: grayscale + sharpen + threshold adaptif."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    eq = clahe.apply(gray)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(eq, -1, kernel)

    # Adaptive threshold
    ada = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 8
    )
    return ada


def run_ocr(reader, img_gray):
    """Jalankan OCR dengan beberapa rotasi, ambil confidence terbaik."""
    best_text, best_conf = "", 0.0

    for angle in [0, 90, 180]:
        if angle > 0:
            M = cv2.getRotationMatrix2D(
                (img_gray.shape[1]//2, img_gray.shape[0]//2), angle, 1)
            rotated = cv2.warpAffine(img_gray, M, (img_gray.shape[1], img_gray.shape[0]))
        else:
            rotated = img_gray

        result = reader.readtext(rotated, detail=1, paragraph=False)
        texts, confs = [], []
        for item in result:
            if len(item) >= 3 and item[2] > 0.3:
                texts.append(str(item[1]).strip())
                confs.append(item[2])

        if texts:
            avg_conf = float(np.mean(confs))
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_text = " ".join(texts)

    return best_text, best_conf


def extract_center_text(img_bgr, center, inner_r):
    """Crop area tengah stempel untuk OCR teks di dalam."""
    cx, cy = center
    r = inner_r
    pad = int(r * 0.1)
    x1 = max(0, cx - r + pad)
    y1 = max(0, cy - r + pad)
    x2 = min(img_bgr.shape[1], cx + r - pad)
    y2 = min(img_bgr.shape[0], cy + r - pad)
    return img_bgr[y1:y2, x1:x2]


def generate_stamp(lines, stamp_type, size_cm, color, border_thick):
    """Generate stempel bersih berbasis parameter."""
    dpi = 300
    px = int((size_cm / 2.54) * dpi)
    pad = 30

    img = Image.new("RGBA", (px, px), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Warna
    color_map = {
        "Biru": (0, 0, 180),
        "Merah": (180, 0, 0),
        "Hitam": (0, 0, 0),
        "Ungu": (80, 0, 140),
    }
    c = color_map.get(color, (0, 0, 180))

    cx, cy = px // 2, px // 2
    r_outer = px // 2 - pad
    r_inner = int(r_outer * 0.72)

    # Lingkaran luar
    draw.ellipse(
        (cx-r_outer, cy-r_outer, cx+r_outer, cy+r_outer),
        outline=c, width=border_thick
    )
    # Lingkaran dalam (opsional untuk stempel dinas)
    if stamp_type in ["Dinas", "Instansi"]:
        draw.ellipse(
            (cx-r_inner, cy-r_inner, cx+r_inner, cy+r_inner),
            outline=c, width=max(2, border_thick//2)
        )

    # Font sizes
    def try_font(size):
        for fname in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
            try:
                return ImageFont.truetype(fname, size)
            except Exception:
                pass
        return ImageFont.load_default()

    # Teks per baris
    n = len([l for l in lines if l.strip()])
    spacing = int((r_inner * 1.6) / max(n, 1))
    font_sz = max(18, min(52, int(px * 0.07)))

    font_main = try_font(font_sz)
    font_ring  = try_font(max(16, int(font_sz * 0.8)))

    # Teks ring atas (baris pertama)
    if len(lines) > 0 and lines[0].strip():
        _draw_arc_text(draw, lines[0].strip(), cx, cy, (r_outer+r_inner)//2,
                       start_angle=-60, end_angle=60, font=font_ring, color=c, top=True)

    # Teks ring bawah (baris terakhir jika >2 baris)
    if len(lines) > 2 and lines[-1].strip():
        _draw_arc_text(draw, lines[-1].strip(), cx, cy, (r_outer+r_inner)//2,
                       start_angle=120, end_angle=240, font=font_ring, color=c, top=False)

    # Teks tengah (baris 2 dst kecuali terakhir)
    mid_lines = lines[1:-1] if len(lines) > 2 else (lines[1:] if len(lines) > 1 else [])
    if not mid_lines and len(lines) == 1:
        mid_lines = lines

    total_h = len(mid_lines) * (font_sz + 8)
    start_y = cy - total_h // 2

    for i, line in enumerate(mid_lines):
        if not line.strip():
            continue
        y_pos = start_y + i * (font_sz + 8)
        draw.text((cx, y_pos), line.strip(), fill=c, font=font_main, anchor="mm")

    return img


def _draw_arc_text(draw, text, cx, cy, radius, start_angle, end_angle, font, color, top=True):
    """Gambar teks mengikuti busur lingkaran."""
    n = len(text)
    if n == 0:
        return
    arc_span = math.radians(end_angle - start_angle)
    char_angle = arc_span / max(n, 1)

    mid_angle = math.radians((start_angle + end_angle) / 2)

    for i, ch in enumerate(text):
        angle = mid_angle - arc_span/2 + i * char_angle + char_angle/2
        # top arc: text faces outward
        x = cx + radius * math.sin(angle)
        y = cy - radius * math.cos(angle)
        rot = math.degrees(angle) if top else math.degrees(angle) + 180

        # Buat char image kecil
        ch_img = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
        ch_draw = ImageDraw.Draw(ch_img)
        ch_draw.text((30, 30), ch, fill=color, font=font, anchor="mm")
        ch_rot = ch_img.rotate(-rot, expand=False)

        draw.bitmap((int(x)-30, int(y)-30), ch_rot.split()[3], fill=color)


def pil_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "📎 Upload foto stempel (PNG/JPG)", type=["png","jpg","jpeg"],
    help="Foto stempel bisa miring, berbayang, atau kasar — sistem akan memprosesnya"
)

if not uploaded:
    st.markdown(f"""
    <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;
      padding:32px;text-align:center;color:#64748b;margin-top:12px">
      <div style="font-size:40px;margin-bottom:10px">🔏</div>
      <div style="font-size:14px;font-weight:600;color:#94a3b8;margin-bottom:6px">Belum ada gambar</div>
      <div style="font-size:12px">Upload foto stempel di atas untuk memulai</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── LOAD IMAGE ─────────────────────────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("❌ Gagal membaca gambar.")
    st.stop()

col_orig, col_proc = st.columns(2)
with col_orig:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.markdown("**📷 Foto Asli**")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── PREPROCESSING ──────────────────────────────────────────────────────────────
with st.spinner("🔍 Mendeteksi stempel..."):
    img_proc, cleaned = preprocess_stamp(img_bgr)
    circle = detect_circle_robust(cleaned)

if circle is None:
    # fallback: coba langsung di grayscale asli
    gray_raw = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    circle   = detect_circle_robust(gray_raw)

with col_proc:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.markdown("**🔵 Deteksi Lingkaran**")
    vis = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB).copy()
    if circle is not None:
        x, y, r = map(int, circle)
        cv2.circle(vis, (x,y), r, (99,102,241), 3)
        cv2.circle(vis, (x,y), int(r*0.70), (99,102,241), 2)
        cv2.circle(vis, (x,y), 4, (239,68,68), -1)
        conf_circle = "✅ Terdeteksi"
        badge = "badge-green"
    else:
        conf_circle = "⚠️ Tidak terdeteksi"
        badge = "badge-red"
    st.image(vis, use_container_width=True)
    st.markdown(f'<span class="badge {badge}">{conf_circle}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if circle is None:
    st.error("❌ Lingkaran stempel tidak terdeteksi. Coba foto dengan pencahayaan lebih jelas.")
    st.stop()

x, y, r = map(int, circle)

# ── OCR ────────────────────────────────────────────────────────────────────────
with st.spinner("🤖 Menjalankan OCR..."):
    reader = get_reader()

    # Crop stamp
    cropped, (cx, cy, cr) = crop_stamp_region(img_proc, circle)

    outer_r = cr
    inner_r = int(cr * 0.68)

    # Ring unwrap untuk teks melingkar
    ring = unwrap_ring(cropped, (cx, cy), outer_r, inner_r)
    ring_enh = enhance_for_ocr(ring)

    # Teks tengah
    center_crop = extract_center_text(cropped, (cx, cy), inner_r)
    center_enh  = enhance_for_ocr(center_crop)

    text_ring,   conf_ring   = run_ocr(reader, ring_enh)
    text_center, conf_center = run_ocr(reader, center_enh)

# ── HASIL OCR ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📝 Hasil Deteksi Teks")

c1, c2, c3 = st.columns([2,2,1])
with c1:
    st.image(ring_enh, caption="Ring (teks melingkar)", use_container_width=True)
with c2:
    st.image(center_enh, caption="Tengah (teks utama)", use_container_width=True)
with c3:
    def conf_badge(c):
        if c >= 0.7: return "badge-green", f"✅ {round(c*100)}%"
        if c >= 0.45: return "badge-yellow", f"⚠️ {round(c*100)}%"
        return "badge-red", f"❌ {round(c*100)}%"

    b1, l1 = conf_badge(conf_ring)
    b2, l2 = conf_badge(conf_center)
    st.markdown(f"**Ring:** <span class='badge {b1}'>{l1}</span>", unsafe_allow_html=True)
    st.markdown(f"**Tengah:** <span class='badge {b2}'>{l2}</span>", unsafe_allow_html=True)

# ── EDITOR TEKS ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ✏️ Edit & Generate Stempel")

# Parse hasil OCR jadi baris
default_lines = []
if text_ring.strip():
    default_lines.append(text_ring.strip())
for part in text_center.split("  "):
    p = part.strip()
    if p:
        default_lines.append(p)
while len(default_lines) < 4:
    default_lines.append("")

col_edit, col_opt = st.columns([3, 2])

with col_edit:
    st.markdown("**Teks Stempel** (edit sesuai kebutuhan)")
    lines = []
    labels = ["Baris 1 (Ring Atas)", "Baris 2 (Tengah)", "Baris 3 (Tengah)", "Baris 4 (Ring Bawah)"]
    for i in range(4):
        val = default_lines[i] if i < len(default_lines) else ""
        lines.append(st.text_input(labels[i], value=val, key=f"line_{i}"))

with col_opt:
    st.markdown("**Pengaturan Stempel**")
    stamp_type   = st.selectbox("Tipe", ["Dinas", "Instansi", "Notaris", "Custom"])
    stamp_color  = st.selectbox("Warna", ["Biru", "Merah", "Hitam", "Ungu"])
    stamp_size   = st.slider("Ukuran (cm)", 3.0, 8.0, 5.0, 0.5)
    border_thick = st.slider("Tebal Border (px)", 4, 20, 10)

if st.button("🔏 Generate Stempel Bersih", use_container_width=True):
    active_lines = [l for l in lines if l.strip()]
    if not active_lines:
        st.warning("⚠️ Isi minimal satu baris teks.")
    else:
        with st.spinner("Generating..."):
            result_img = generate_stamp(
                lines, stamp_type, stamp_size, stamp_color, border_thick
            )

        st.markdown("---")
        st.markdown("### 🎉 Hasil Stempel")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            # White background preview
            bg = Image.new("RGB", result_img.size, "white")
            bg.paste(result_img, mask=result_img.split()[3])
            st.image(bg, caption="Preview (latar putih)", use_container_width=True)

        with col_r2:
            # Transparent preview
            st.image(result_img, caption="Preview (transparan)", use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download PNG (Transparan)",
                data=pil_to_bytes(result_img, "PNG"),
                file_name="stempel_bersih.png",
                mime="image/png",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                "⬇️ Download PNG (Putih)",
                data=pil_to_bytes(bg, "PNG"),
                file_name="stempel_putih.png",
                mime="image/png",
                use_container_width=True
            )
