"""One-shot runner: prepare -> train -> evaluate.

python run_all.py --include scan_upright --max_images 1200 --epochs 8
python run_all.py --include scan_upright --holdout_type splice     # unseen-forgery test
"""
import argparse
import subprocess
import sys


def run(module, *a):
    cmd = [sys.executable, "-m", module, *map(str, a)]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--max_images", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--time_budget_min", type=float, default=45)
    ap.add_argument("--holdout_type", nargs="*", default=[])
    ap.add_argument("--use_ocr", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    prep = ["--raw_dir", a.raw_dir, "--max_images", a.max_images]
    if a.include:
        prep += ["--include", *a.include]
    if a.use_ocr:
        prep += ["--use_ocr"]
    run("src.prepare_data", *prep)

    tr = ["--epochs", a.epochs, "--time_budget_min", a.time_budget_min, "--device", a.device]
    if a.holdout_type:
        tr += ["--holdout_type", *a.holdout_type]
    run("src.train", *tr)
    run("src.evaluate", "--device", a.device)


if __name__ == "__main__":
    main()