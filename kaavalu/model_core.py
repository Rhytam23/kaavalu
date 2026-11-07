"""The exportable part of torchvision's SSDLite: backbone + head, no post-processing.

Box decoding and NMS are awkward to export, so the .pte returns raw class
logits and box offsets and kaavalu/decode.py finishes the job in NumPy.
Needs torch + torchvision (laptop for export; also the PyTorch baseline on the Pi).
"""

import numpy as np
import torch
import torchvision
from torchvision.models.detection.image_list import ImageList

MODELS = {
    "ssdlite320_mobilenet_v3_large": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
}


class Core(torch.nn.Module):
    def __init__(self, det):
        super().__init__()
        self.backbone = det.backbone
        self.head = det.head

    def forward(self, x):
        feats = list(self.backbone(x).values())
        out = self.head(feats)
        return out["cls_logits"], out["bbox_regression"]


def load_torchvision(name="ssdlite320_mobilenet_v3_large"):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name](weights="DEFAULT").eval()


def normalisation(det):
    t = det.transform
    return list(map(float, t.image_mean)), list(map(float, t.image_std))


@torch.no_grad()
def compute_anchors(det, size):
    """Anchors (N, 4) xyxy in model-input pixels for a size x size input."""
    x = torch.zeros(1, 3, size, size)
    feats = list(det.backbone(x).values())
    images = ImageList(x, [(size, size)])
    anchors = det.anchor_generator(images, feats)[0]
    return anchors.cpu().numpy().astype(np.float32)
