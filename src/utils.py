import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_images(root, include=None, exclude=None):
    """Recursively list images. include/exclude are substring filters on the path."""
    files = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        s = p.as_posix().lower()
        if include and not any(i.lower() in s for i in include):
            continue
        if exclude and any(e.lower() in s for e in exclude):
            continue
        files.append(p)
    return sorted(files)


def group_key(path, root):
    """Frames of one MIDV-500 clip (e.g. CA01_01, CA01_02) share a group;
    MIDV-2020 images (00.jpg, 01.jpg ...) are their own group."""
    rel = Path(path).relative_to(root)
    stem = re.sub(r"_\d+$", "", rel.stem)
    return rel.parent.as_posix() + "/" + stem


def load_image(path, max_side=1600, min_side=200):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        return None
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def resize_fixed(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def compute_metrics(y_true, prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, prob))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float((pred == y_true).mean()),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "auc": auc,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "false_accept_rate": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y_true, prob):
    try:
        fpr, tpr, thr = roc_curve(y_true, prob)
    except ValueError:
        return 0.5
    t = thr[int(np.argmax(tpr - fpr))]
    return float(np.clip(t, 0.0, 1.0))