from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def bgr_to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return x


def normalize(x):
    return (x - _MEAN) / _STD


class IDTamperDataset(Dataset):
    def __init__(self, processed_dir, split, train=False, exclude_types=(), limit=None, seed=42):
        self.root = Path(processed_dir)
        df = pd.read_csv(self.root / "manifest.csv")
        df = df[df.split == split]
        if exclude_types:
            df = df[~df.tamper_type.isin(list(exclude_types))]
        if limit and limit < len(df):
            df = df.sample(limit, random_state=seed)
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def _augment(self, x):
        # mild photometric jitter only: geometric ops would destroy tamper traces
        b = np.random.uniform(0.85, 1.15)
        c = np.random.uniform(0.85, 1.15)
        m = x.mean()
        x = ((x - m) * c + m) * b
        if np.random.rand() < 0.5:
            x = x + torch.randn_like(x) * 0.01
        return x.clamp(0, 1)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(str(self.root / r.filepath), cv2.IMREAD_COLOR)
        x = bgr_to_tensor(img)
        if self.train:
            x = self._augment(x)
        return normalize(x), torch.tensor(float(r.label))