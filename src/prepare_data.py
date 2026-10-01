"""Build the genuine/tampered dataset from raw MIDV images.

python -m src.prepare_data --raw_dir data/raw --include scan_upright --max_images 1200
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_PROC, DATA_RAW, IMG_H, IMG_W, SEED, TAMPER_TYPES
from src.tamper import tamper_image
from src.text_regions import get_boxes
from src.utils import find_images, group_key, load_image, resize_fixed


def _process(job):
    cv2.setNumThreads(1)
    (i, path, donor_path, split, kind, out_dir, W, H, quality, seed, use_ocr, max_side) = job
    rng = np.random.default_rng(seed + i)
    img = load_image(path, max_side)
    if img is None:
        return None
    boxes = get_boxes(img, use_ocr)
    donor = load_image(donor_path, max_side) if donor_path else None
    tam = tamper_image(img, rng, kind, boxes, donor)

    out_dir = Path(out_dir)
    g_rel = f"{split}/genuine/{i:05d}.jpg"
    t_rel = f"{split}/tampered/{i:05d}_{kind}.jpg"
    for rel in (g_rel, t_rel):
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    enc = [cv2.IMWRITE_JPEG_QUALITY, quality]   # identical encoding for both classes
    cv2.imwrite(str(out_dir / g_rel), resize_fixed(img, W, H), enc)
    cv2.imwrite(str(out_dir / t_rel), resize_fixed(tam, W, H), enc)
    src = Path(path).name
    return [
        dict(filepath=g_rel, label=0, tamper_type="none", split=split, source=src, n_boxes=len(boxes)),
        dict(filepath=t_rel, label=1, tamper_type=kind, split=split, source=src, n_boxes=len(boxes)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(DATA_RAW))
    ap.add_argument("--out_dir", default=str(DATA_PROC))
    ap.add_argument("--include", nargs="*", default=None,
                    help="substring filters, e.g. scan_upright photo")
    ap.add_argument("--exclude", nargs="*", default=["template"])
    ap.add_argument("--max_images", type=int, default=10000,
                    help="source images (each yields 1 genuine + 1 tampered)")
    ap.add_argument("--max_per_group", type=int, default=0,
                help="maximum images per group; 0 = use all")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--width", type=int, default=IMG_W)
    ap.add_argument("--height", type=int, default=IMG_H)
    ap.add_argument("--max_side", type=int, default=1600)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--tamper_types", nargs="*", default=TAMPER_TYPES)
    ap.add_argument("--use_ocr", action="store_true", help="use Tesseract word boxes if installed")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    files = find_images(raw, args.include, args.exclude)
    if not files:
        sys.exit(f"No images found in {raw} (include={args.include}). "
                 "Download MIDV data first or run scripts/make_demo_data.py")
    print(f"Found {len(files)} candidate images")

    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for f in files:
        groups[group_key(f, raw)].append(f)
    flat = []
    for k in sorted(groups):
        v = groups[k][:]
        rng.shuffle(v)

        if args.max_per_group > 0:
            v = v[:args.max_per_group]

        flat += [(k, f) for f in v]
    rng.shuffle(flat)
    flat = flat[:args.max_images]

    sizes = Counter(k for k, _ in flat)
    keys = sorted(sizes)
    rng.shuffle(keys)
    n = len(flat)
    n_test, n_val = int(n * args.test_frac), int(n * args.val_frac)
    split_of, cum = {}, 0
    for k in keys:
        if cum < n - n_val - n_test:
            split_of[k] = "train"
        elif cum < n - n_test:
            split_of[k] = "val"
        else:
            split_of[k] = "test"
        cum += sizes[k]

    by_split = defaultdict(list)
    for k, f in flat:
        by_split[split_of[k]].append(f)
    for s in ("train", "val", "test"):
        if not by_split[s]:
            sys.exit(f"Split '{s}' is empty - use more images or smaller val/test fractions.")

    jobs, i = [], 0
    for s in ("train", "val", "test"):
        for j, f in enumerate(by_split[s]):
            kind = args.tamper_types[j % len(args.tamper_types)]
            donor = rng.choice(by_split[s])
            jobs.append((i, str(f), str(donor), s, kind, args.out_dir, args.width,
                         args.height, args.quality, args.seed, args.use_ocr, args.max_side))
            i += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(_process, jobs), total=len(jobs), desc="building"):
                if r:
                    rows += r
    else:
        for j in tqdm(jobs, desc="building"):
            r = _process(j)
            if r:
                rows += r

    df = pd.DataFrame(rows).sort_values("filepath").reset_index(drop=True)
    df.to_csv(out / "manifest.csv", index=False)
    with open(out / "prepare_args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(df[df.label == 1].groupby(["split", "tamper_type"]).size().unstack(fill_value=0))
    print(f"Saved manifest -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()