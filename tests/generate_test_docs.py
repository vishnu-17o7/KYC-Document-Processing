"""Generate synthetic test documents for KYC pipeline testing."""

import io
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_docs")


def _get_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def _add_degradation(img: Image.Image) -> Image.Image:
    """Apply scan degradation: skew, blur, noise, low contrast."""
    arr = np.array(img)

    arr = arr.astype(np.float32)

    rows, cols = arr.shape[:2]
    M = np.float32([[1, 0, random.uniform(-10, 10)], [0, 1, random.uniform(-8, 8)]])
    arr = np.array(Image.fromarray(arr.astype(np.uint8)).transform(
        (cols, rows), Image.AFFINE, M.flatten()[:6], Image.BILINEAR, fillcolor=255
    ), dtype=np.float32)

    noise = np.random.normal(0, random.uniform(8, 20), arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    contrast = random.uniform(0.4, 0.7)
    arr = (arr - 128) * contrast + 128
    arr = np.clip(arr, 0, 255)

    blurred = Image.fromarray(arr.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(random.uniform(0.5, 2.0))
    )
    return blurred


def _draw_text_block(draw, x, y, lines, font, fill="black"):
    for i, line in enumerate(lines):
        draw.text((x, y + i * 28), line, fill=fill, font=font)


def generate_aadhaar(degraded: bool = False) -> bytes:
    img = Image.new("RGB", (1200, 1700), "white")
    draw = ImageDraw.Draw(img)
    h_font = _get_font(22)
    font = _get_font(18)
    s_font = _get_font(14)

    draw.rectangle([30, 30, 1170, 1670], outline="black", width=2)

    draw.text((450, 60), "Government of India", fill="red", font=_get_font(24))
    draw.text((430, 95), "Unique Identification Authority of India", fill="blue", font=_get_font(20))
    draw.text((480, 130), "AADHAAR CARD", fill="darkgreen", font=_get_font(28))

    draw.rectangle([850, 60, 1100, 250], outline="black", width=1)
    draw.text((870, 140), "[PHOTO]", fill="gray", font=_get_font(20))

    draw.text((60, 300), "To", fill="black", font=s_font)
    rows = [
        "Priya Sharma",
        "S/O: Rajesh Sharma",
        "123, MG Road, Indiranagar",
        "Bengaluru, Karnataka - 560038",
        "Mobile: 9876543210",
    ]
    for i, row in enumerate(rows):
        draw.text((80, 325 + i * 26), row, fill="black", font=font)

    draw.text((800, 300), "Enrolment No:", fill="gray", font=s_font)
    draw.text((1000, 300), "1234/56789/01234", fill="black", font=s_font)
    draw.text((800, 330), "Aadhaar No:", fill="gray", font=s_font)
    draw.text((1000, 330), "9876 5432 1098", fill="black", font=_get_font(22))

    fields = [
        ("Name:", "Priya Sharma", 120),
        ("Date of Birth:", "12/04/1990", 120),
        ("Gender:", "Female", 120),
        ("Address:", "123, MG Road, Indiranagar, Bengaluru, Karnataka - 560038", 120),
    ]
    y_pos = 460
    for label, value, indent in fields:
        draw.text((indent, y_pos), label, fill="gray", font=font)
        indent_x = indent + draw.textlength(label, font=font) + 10
        draw.text((indent_x, y_pos), value, fill="black", font=font)
        y_pos += 35

    draw.text((250, 690), "www.uidai.gov.in", fill="blue", font=s_font)

    draw.rectangle([400, 800, 800, 1200], outline="black", width=1)
    draw.text((520, 980), "[QR CODE]", fill="gray", font=_get_font(20))

    if degraded:
        img = _add_degradation(img)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_pan(degraded: bool = False) -> bytes:
    img = Image.new("RGB", (1100, 700), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(18)
    b_font = _get_font(20)

    draw.rectangle([20, 20, 1080, 680], outline="darkblue", width=3)

    draw.text((100, 40), "INCOME TAX DEPARTMENT", fill="darkblue", font=_get_font(22))
    draw.text((100, 70), "GOVT. OF INDIA", fill="darkblue", font=_get_font(16))
    draw.text((360, 40), "PERMANENT ACCOUNT NUMBER", fill="red", font=_get_font(18))
    draw.text((400, 70), "PAN CARD", fill="red", font=_get_font(24))

    draw.rectangle([750, 40, 1020, 220], outline="black", width=1)
    draw.text((820, 120), "[PHOTO]", fill="gray", font=_get_font(20))

    draw.text((180, 160), "ABCDE1234F", fill="black", font=_get_font(28))

    fields = [
        ("Name:", "Rahul Kumar"),
        ("Father's Name:", "Suresh Kumar"),
        ("Date of Birth:", "15/08/1985"),
    ]
    y_pos = 250
    for label, value in fields:
        draw.text((140, y_pos), label, fill="gray", font=font)
        draw.text((380, y_pos), value, fill="black", font=font)
        y_pos += 35

    small_font = _get_font(14)
    draw.rectangle([100, 400, 1000, 440], fill="lightgray", outline="black")
    draw.text((400, 405), "Signature", fill="black", font=small_font)

    if degraded:
        img = _add_degradation(img)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_dl(degraded: bool = False) -> bytes:
    img = Image.new("RGB", (1100, 750), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(18)
    b_font = _get_font(20)

    draw.rectangle([20, 20, 1080, 730], outline="darkgreen", width=2)

    draw.text((100, 35), "MINISTRY OF ROAD TRANSPORT & HIGHWAYS", fill="darkgreen", font=b_font)
    draw.text((150, 65), "DRIVING LICENCE", fill="darkgreen", font=_get_font(26))

    draw.rectangle([750, 35, 1020, 220], outline="black", width=1)
    draw.text((820, 115), "[PHOTO]", fill="gray", font=_get_font(20))

    fields = [
        ("DL No:", "KA05 20190001234"),
        ("Name:", "Amit Patel"),
        ("Date of Birth:", "22/06/1992"),
        ("Valid From:", "15/01/2019"),
        ("Valid To:", "14/01/2039"),
        ("Vehicle Class:", "MCWG, LMV, TRANS"),
    ]
    y_pos = 270
    for label, value in fields:
        draw.text((140, y_pos), label, fill="gray", font=font)
        draw.text((380, y_pos), value, fill="black", font=font)
        y_pos += 35

    draw.text((140, y_pos + 10), "Address:", fill="gray", font=font)
    draw.text((380, y_pos + 10), "45, Park Street, Mumbai - 400001", fill="black", font=font)

    if degraded:
        img = _add_degradation(img)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, gen_func in [
        ("aadhaar_clean", lambda: generate_aadhaar(degraded=False)),
        ("aadhaar_degraded", lambda: generate_aadhaar(degraded=True)),
        ("pan_clean", lambda: generate_pan(degraded=False)),
        ("dl_clean", lambda: generate_dl(degraded=False)),
    ]:
        data = gen_func()
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        with open(path, "wb") as f:
            f.write(data)
        print(f"Generated: {path} ({len(data)} bytes)")
