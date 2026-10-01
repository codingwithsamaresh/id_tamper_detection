from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

IMG_W, IMG_H = 480, 320          # fixed network input (W x H)
SEED = 42
TAMPER_TYPES = ["text_replace", "erase", "copy_move", "splice", "local_blur"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)