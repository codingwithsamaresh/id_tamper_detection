"""python -m src.train --epochs 8 --time_budget_min 45"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DATA_PROC, OUT_DIR, SEED
from src.dataset import IDTamperDataset
from src.model import build_model
from src.utils import best_threshold, compute_metrics, get_device, seed_everything


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, use_amp=False):
    train = optimizer is not None
    model.train(train)
    losses, ys, ps = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                enabled=use_amp):
                logit = model(x).squeeze(1)
                loss = criterion(logit.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.append(loss.item() * len(y))
            ys.append(y.detach().cpu().numpy())
            ps.append(torch.sigmoid(logit.float()).detach().cpu().numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return sum(losses) / len(ys), ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DATA_PROC))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--holdout_type", nargs="*", default=[],
                    help="tamper types excluded from train/val (test them as unseen)")
    ap.add_argument("--limit_train", type=int, default=None, help="smoke test")
    ap.add_argument("--time_budget_min", type=float, default=45)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    use_amp = device.type == "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device}  holdout={args.holdout_type}")

    tr = IDTamperDataset(args.data_dir, "train", True, args.holdout_type, args.limit_train, args.seed)
    va = IDTamperDataset(args.data_dir, "val", False, args.holdout_type)
    print(f"train={len(tr)}  val={len(va)}")
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    vl = DataLoader(va, args.batch_size * 2, shuffle=False, num_workers=args.workers)

    model = build_model(not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hist, best_auc, best_loss = [], -1.0, 1e9
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        te = time.time()
        trl, ty, tp = run_epoch(model, tl, device, criterion, opt, scaler, use_amp)
        val_l, vy, vp = run_epoch(model, vl, device, criterion, use_amp=False)
        sched.step()
        tm, vm = compute_metrics(ty, tp), compute_metrics(vy, vp)
        row = dict(epoch=ep, train_loss=trl, train_acc=tm["accuracy"], val_loss=val_l,
                   val_acc=vm["accuracy"], val_auc=vm["auc"], val_f1=vm["f1"],
                   epoch_sec=time.time() - te)
        hist.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

        auc = vm["auc"] if not np.isnan(vm["auc"]) else 0.0
        if auc > best_auc or (auc == best_auc and val_l < best_loss):
            best_auc, best_loss = auc, val_l
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": auc,
                        "threshold": best_threshold(vy, vp), "args": vars(args)},
                       os.path.join(args.out_dir, "best_model.pt"))
            print(f"  saved best (val_auc={auc:.4f})")

        elapsed = time.time() - t0
        if elapsed + (time.time() - te) > args.time_budget_min * 60 and ep < args.epochs:
            print(f"Time budget reached after epoch {ep}; stopping.")
            break

    df = pd.DataFrame(hist)
    df.to_csv(os.path.join(args.out_dir, "history.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df.epoch, df.train_loss, label="train"); ax[0].plot(df.epoch, df.val_loss, label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(df.epoch, df.train_acc, label="train acc"); ax[1].plot(df.epoch, df.val_acc, label="val acc")
    ax[1].plot(df.epoch, df.val_auc, label="val AUC"); ax[1].set_title("Accuracy / AUC"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=130)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Best val AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()