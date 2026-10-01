"""Synthetic forgery generation on genuine ID document images."""
import string
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONTS = ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
          "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf"]


@lru_cache(maxsize=64)
def load_font(size):
    for f in _FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _clip_box(box, W, H):
    x, y, w, h = box
    x, y = max(0, int(x)), max(0, int(y))
    return x, y, min(int(w), W - x), min(int(h), H - y)


def _random_box(W, H, rng):
    w = int(W * rng.uniform(0.12, 0.35))
    h = max(10, int(H * rng.uniform(0.04, 0.08)))
    x = int(rng.integers(0, max(1, W - w)))
    y = int(rng.integers(0, max(1, H - h)))
    return _clip_box((x, y, w, h), W, H)


def _bg_color(img, box, pad=4):
    H, W = img.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    roi = img[y0:y1, x0:x1]
    m = np.ones(roi.shape[:2], bool)
    m[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    px = roi[m] if m.any() else roi.reshape(-1, 3)
    return np.median(px, axis=0)


def _fg_color(img, box):
    x, y, w, h = box
    roi = img[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    idx = gray <= np.percentile(gray, 10)
    if not idx.any():
        return np.array([30, 30, 30], np.float32)
    return roi[idx].mean(axis=0)


def _feather_paste(dst, patch, x, y, feather=3):
    h, w = patch.shape[:2]
    f = max(1, min(feather, h // 4, w // 4))
    alpha = np.zeros((h, w), np.float32)
    alpha[f:h - f, f:w - f] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), f / 1.5)[..., None]
    roi = dst[y:y + h, x:x + w].astype(np.float32)
    dst[y:y + h, x:x + w] = (alpha * patch + (1 - alpha) * roi).clip(0, 255).astype(np.uint8)


def _erase_text(img, box, pad=6):
    """Remove text in `box` with Telea inpainting (in place)."""
    H, W = img.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    inner = np.zeros_like(m)
    inner[y - y0:y - y0 + h, x - x0:x - x0 + w] = 255
    m = cv2.bitwise_and(m, inner)
    if m.mean() / 255.0 > 0.5:
        m = cv2.bitwise_and(cv2.bitwise_not(m), inner)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)
    img[y0:y1, x0:x1] = cv2.inpaint(crop, m, 3, cv2.INPAINT_TELEA)


def _random_text(n, rng):
    mode = int(rng.integers(0, 3))
    alphabet = [string.digits, string.ascii_uppercase,
                string.ascii_uppercase + string.digits][mode]
    return "".join(rng.choice(list(alphabet), n))


def text_replace(img, box, rng):
    x, y, w, h = box
    bg, fg = _bg_color(img, box), _fg_color(img, box)
    if rng.random() < 0.5:
        _erase_text(img, box)
    else:
        img[y:y + h, x:x + w] = bg.astype(np.uint8)
    fs = max(8, int(h * 1.05))
    n = max(2, int(w / (0.6 * fs)))
    text = _random_text(n, rng)
    pil = Image.fromarray(cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    font = load_font(fs)
    while fs > 6:
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        if r - l <= w and b - t <= h * 1.1:
            break
        fs -= 1
        font = load_font(fs)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    fill = tuple(int(c) for c in fg[::-1])
    d.text((-l, (h - (b - t)) // 2 - t), text, font=font, fill=fill)
    img[y:y + h, x:x + w] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def copy_move(img, box, boxes, rng):
    H, W = img.shape[:2]
    x, y, w, h = box
    others = [b for b in boxes if b != box]
    if others:
        sx, sy, sw, sh = others[int(rng.integers(len(others)))]
    else:
        sw, sh = w, h
        sx, sy = int(rng.integers(0, max(1, W - sw))), int(rng.integers(0, max(1, H - sh)))
    patch = img[sy:sy + sh, sx:sx + sw].copy()
    if patch.size == 0:
        return
    patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
    _feather_paste(img, patch, x, y)


def splice(img, donor, rng):
    """Paste a region of ANOTHER document (e.g. swapped photo / field block)."""
    H, W = img.shape[:2]
    rw = min(max(24, int(W * rng.uniform(0.08, 0.25))), W - 2)
    rh = min(max(24, int(H * rng.uniform(0.10, 0.30))), H - 2)
    rx, ry = int(rng.integers(0, W - rw)), int(rng.integers(0, H - rh))
    src = cv2.resize(donor if donor is not None else img, (W, H))
    dx, dy = int(rng.integers(0, W - rw)), int(rng.integers(0, H - rh))
    patch = src[dy:dy + rh, dx:dx + rw].copy()
    if rng.random() < 0.5:
        try:
            mask = np.full(patch.shape[:2], 255, np.uint8)
            res = cv2.seamlessClone(patch, img, mask, (rx + rw // 2, ry + rh // 2),
                                    cv2.NORMAL_CLONE)
            img[:] = res
            return
        except cv2.error:
            pass
    _feather_paste(img, patch, rx, ry, feather=4)


def local_blur(img, box, rng):
    H, W = img.shape[:2]
    x, y, w, h = _clip_box((box[0] - 4, box[1] - 4, box[2] + 8, box[3] + 8), W, H)
    roi = img[y:y + h, x:x + w].copy()
    if rng.random() < 0.5:
        k = int(rng.choice([5, 7, 9]))
        out = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        f = int(rng.integers(2, 4))
        small = cv2.resize(roi, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    _feather_paste(img, out, x, y)


def tamper_image(img, rng, kind, boxes, donor=None):
    """Return a tampered copy of `img`. rng: np.random.Generator."""
    out = img.copy()
    H, W = out.shape[:2]
    boxes = [_clip_box(b, W, H) for b in boxes]
    boxes = [b for b in boxes if b[2] >= 12 and b[3] >= 8]
    if not boxes:
        boxes = [_random_box(W, H, rng)]

    def pick(k):
        idx = rng.choice(len(boxes), size=min(k, len(boxes)), replace=False)
        return [boxes[i] for i in idx]

    if kind == "text_replace":
        for b in pick(int(rng.integers(1, 4))):
            text_replace(out, b, rng)
    elif kind == "erase":
        for b in pick(int(rng.integers(1, 4))):
            _erase_text(out, b)
    elif kind == "copy_move":
        for b in pick(int(rng.integers(1, 3))):
            copy_move(out, b, boxes, rng)
    elif kind == "splice":
        splice(out, donor, rng)
    elif kind == "local_blur":
        for b in pick(int(rng.integers(1, 3))):
            local_blur(out, b, rng)
    else:
        raise ValueError(f"unknown tamper type: {kind}")
    return out