"""Text-field localisation + optional OCR field extraction."""
import re
from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np


def detect_text_boxes(img, max_side=1000, min_w=40, min_h=9):
    """Pure-OpenCV text line detector (morphological gradient + closing).
    Returns [(x, y, w, h)] in original-image coordinates."""
    H, W = img.shape[:2]
    s = min(1.0, max_side / max(H, W))
    small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2)))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h or h > 0.09 * sh or w > 0.9 * sw:
            continue
        if w / h < 2.5:
            continue
        fill = cv2.countNonZero(bw[y:y + h, x:x + w]) / float(w * h)
        if fill < 0.35:
            continue
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        boxes.append((int(x0 / s), int(y0 / s),
                      int((w + 2 * pad) / s), int((h + 2 * pad) / s)))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


@lru_cache(maxsize=1)
def tesseract_available():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_words(img, min_conf=50):
    """Word-level OCR via Tesseract. Returns list of dicts(text, conf, box)."""
    import pytesseract
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    d = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        try:
            conf = float(d["conf"][i])
        except (ValueError, TypeError):
            continue
        if t and conf >= min_conf:
            out.append({"text": t, "conf": conf,
                        "box": (int(d["left"][i]), int(d["top"][i]),
                                int(d["width"][i]), int(d["height"][i]))})
    return out


def get_boxes(img, use_ocr=False):
    """Field boxes: OCR word boxes if requested/available, else OpenCV detector."""
    if use_ocr and tesseract_available():
        try:
            b = [w["box"] for w in ocr_words(img, 50)
                 if w["box"][3] >= 10 and len(w["text"]) >= 3]
            if len(b) >= 3:
                return b
        except Exception:
            pass
    return detect_text_boxes(img)


_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b")


def validate_dates(words):
    """Heuristic OCR sanity check: flag date-looking strings that are not valid dates."""
    flags = []
    for w in words:
        m = _DATE_RE.search(w["text"])
        if not m:
            continue
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        y = int(y) if len(y) == 4 else 2000 + int(y)
        try:
            datetime(y, mo, d)
        except ValueError:
            flags.append({"text": w["text"], "reason": "invalid calendar date"})
    return flags