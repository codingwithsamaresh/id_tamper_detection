"""Generate fake ID-like cards to smoke-test the pipeline WITHOUT the real dataset.
python scripts/make_demo_data.py --n 200
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tamper import load_font  # noqa: E402

NAMES = ["ANNA", "JOHN", "MARIA", "LI", "OMAR", "SVEN", "PRIYA", "IVAN", "LUCIA", "KENJI"]
SURN = ["SMITH", "KUMAR", "ROSSI", "NOVAK", "TANAKA", "GARCIA", "ANDERSEN", "PETROV"]


def make_card(rng):
    W, H = 1000, 640
    base = np.array([rng.randint(200, 245), rng.randint(200, 245), rng.randint(200, 245)])
    arr = np.tile(base, (H, W, 1)).astype(np.float32)
    arr += np.linspace(0, 25, W)[None, :, None] + np.random.randn(H, W, 1) * 3
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 90], fill=(rng.randint(20, 90), rng.randint(40, 120), rng.randint(90, 180)))
    d.text((30, 25), "IDENTITY CARD", font=load_font(40), fill=(255, 255, 255))
    px, py = 40, 130
    d.rectangle([px, py, px + 260, py + 340], fill=tuple(rng.randint(90, 200) for _ in range(3)))
    d.ellipse([px + 70, py + 40, px + 190, py + 160], fill=(230, 200, 180))
    d.rectangle([px + 40, py + 190, px + 220, py + 330], fill=(60, 70, 90))
    f = load_font(32)
    y = 140
    for lab, val in [("SURNAME", rng.choice(SURN)), ("GIVEN NAME", rng.choice(NAMES)),
                     ("DATE OF BIRTH", f"{rng.randint(1,28):02d}.{rng.randint(1,12):02d}.{rng.randint(1960,2004)}"),
                     ("DOC NO", "".join(rng.choice("0123456789ABCDEF") for _ in range(9))),
                     ("EXPIRES", f"{rng.randint(1,28):02d}.{rng.randint(1,12):02d}.{rng.randint(2026,2034)}")]:
        d.text((340, y), lab, font=load_font(20), fill=(90, 90, 90))
        d.text((340, y + 24), val, font=f, fill=(20, 20, 20))
        y += 95
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/raw/demo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for i in range(args.n):
        make_card(rng).save(out / f"{i:04d}.jpg", quality=92)
    print(f"wrote {args.n} demo cards to {out}")


if __name__ == "__main__":
    main()