import torch.nn as nn
from torchvision import models


def build_model(pretrained=True):
    m = None
    if pretrained:
        try:
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as e:  # offline etc.
            print(f"[warn] could not load pretrained weights ({e}); using random init")
    if m is None:
        m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    return m