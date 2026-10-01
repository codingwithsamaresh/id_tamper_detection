"""Score one image or a folder; optional Grad-CAM heatmap and OCR sanity checks.

python -m src.infer --image path/to/id.jpg --cam --ocr
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import IMG_H, IMG_W, OUT_DIR
from src.dataset import bgr_to_tensor, normalize
from src.model import build_model
from src.text_regions import ocr_words, tesseract_available, validate_dates
from src.utils import find_images, get_device


def gradcam(model, x):
    store = {}

    def fh(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h = model.layer4.register_forward_hook(fh)
    model.zero_grad()
    logit = model(x)[:, 0]
    logit.sum().backward()
    h.remove()
    a, g = store["a"], store["g"]
    w = g.mean((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, float(torch.sigmoid(logit).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--ckpt", default=str(OUT_DIR / "best_model.pt"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--cam", action="store_true")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--out_dir", default=str(OUT_DIR / "infer"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    paths = [Path(args.image)] if args.image else find_images(args.dir)
    if not paths:
        raise SystemExit("Provide --image or --dir")
    device = get_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    thr = args.threshold if args.threshold is not None else float(ckpt.get("threshold", 0.5))
    os.makedirs(args.out_dir, exist_ok=True)
    do_ocr = args.ocr and tesseract_available()
    if args.ocr and not do_ocr:
        print("[warn] Tesseract/pytesseract not available; skipping OCR")

    results = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        small = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = normalize(bgr_to_tensor(small)).unsqueeze(0).to(device)
        if args.cam:
            cam, prob = gradcam(model, x)
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.out_dir, f"{p.stem}_cam.jpg"),
                        cv2.addWeighted(small, 0.55, heat, 0.45, 0))
        else:
            with torch.no_grad():
                prob = float(torch.sigmoid(model(x)).item())
        r = {"file": str(p), "prob_tampered": round(prob, 4),
             "verdict": "TAMPERED" if prob >= thr else "GENUINE", "threshold": round(thr, 3)}
        if do_ocr:
            words = ocr_words(img, 40)
            r["ocr_fields"] = [{"text": w["text"], "conf": w["conf"]} for w in words][:60]
            r["ocr_flags"] = validate_dates(words)   # heuristic only, not evaluated
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "ocr_fields"}))

    with open(os.path.join(args.out_dir, "inference.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()