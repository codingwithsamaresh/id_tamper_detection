"""python -m src.evaluate"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DATA_PROC, OUT_DIR
from src.dataset import IDTamperDataset
from src.model import build_model
from src.utils import compute_metrics, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DATA_PROC))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.ckpt or os.path.join(args.out_dir, "best_model.pt"), map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    thr_val = float(ckpt.get("threshold", 0.5))

    ds = IDTamperDataset(args.data_dir, args.split, train=False)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)
    probs = []
    with torch.no_grad():
        for x, _ in tqdm(dl, desc=f"eval {args.split}"):
            probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
    probs = np.concatenate(probs)
    df = ds.df.copy()
    df["prob_tampered"] = probs
    y = df.label.values

    res = {"n_samples": int(len(df)),
           "metrics_thr_0.5": compute_metrics(y, probs, 0.5),
           "metrics_thr_val_tuned": compute_metrics(y, probs, thr_val)}

    per_type = {}
    pred = (probs >= thr_val).astype(int)
    for t in sorted(df[df.label == 1].tamper_type.unique()):
        m = (df.tamper_type == t).values
        per_type[t] = {"n": int(m.sum()), "detection_rate": float(pred[m].mean())}
    res["per_tamper_type_detection_rate_at_val_thr"] = per_type
    g = (df.label == 0).values
    res["genuine_accept_rate_at_val_thr"] = float(1 - pred[g].mean())

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, f"predictions_{args.split}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"results_{args.split}.json"), "w") as f:
        json.dump(res, f, indent=2)

    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, label=f"AUC={res['metrics_thr_0.5']['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=130); plt.close()

    cm = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center")
    plt.xticks([0, 1], ["genuine", "tampered"]); plt.yticks([0, 1], ["genuine", "tampered"])
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"Confusion (thr={thr_val:.2f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=130); plt.close()

    if per_type:
        plt.figure(figsize=(6, 4))
        plt.bar(list(per_type), [v["detection_rate"] for v in per_type.values()])
        plt.ylim(0, 1); plt.ylabel("detection rate"); plt.xticks(rotation=25)
        plt.title("Per tamper-type detection"); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "per_type_detection.png"), dpi=130); plt.close()

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()