"""Small binary segmenter, conservative centroid tracking, and metric primitives."""
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Block(3, 12)
        self.mid = Block(12, 24)
        self.dec = Block(36, 12)
        self.head = nn.Conv2d(12, 1, 1)

    def forward(self, x):
        skip = self.enc(x)
        low = self.mid(F.max_pool2d(skip, 2))
        up = F.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec(torch.cat([skip, up], 1)))


def loss_fn(logits, target):
    p = logits.sigmoid()
    dice = (2 * (p * target).sum((1, 2, 3)) + 1) / (
        p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - dice).mean()


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def centroid(mask):
    y, x = np.nonzero(mask)
    return None if len(x) == 0 else np.array([x.mean(), y.mean()])


def scores(p, y):
    pred, truth = p >= .5, y > .5
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    total = pred.sum() + truth.sum()
    # Both-empty scores are one; report empty frames separately in downstream studies.
    return {"iou": float(intersection / union) if union else 1.,
            "dice": float(2 * intersection / total) if total else 1.,
            "brier": float(np.mean((p - y) ** 2))}


class Tracker:
    """One foreground track, no identity association; resets on rejected observations.

    EMA is intentionally simple and unwarped: motion can produce lag/ghost masks.
    Rejected frames return raw segmentation but never a held or invented position.
    """
    def __init__(self, alpha=.7, max_entropy=.65, max_jump=.3, min_pixels=8):
        self.alpha = alpha
        self.max_entropy = max_entropy
        self.max_jump = max_jump
        self.min_pixels = min_pixels
        self.reset()

    def reset(self):
        self.previous = None
        self.position = None

    def update(self, p):
        candidate = p >= .5
        region = p > .1  # Avoid a background-dominated confidence average.
        uncertainty = float(entropy(p)[region].mean()) if region.any() else 1.
        pos = centroid(candidate)
        reason = None
        if candidate.sum() < self.min_pixels:
            reason = "insufficient_foreground"
        elif uncertainty > self.max_entropy:
            reason = "high_entropy"
        elif self.position is not None and np.linalg.norm(pos - self.position) > self.max_jump * np.hypot(*p.shape):
            reason = "centroid_jump"
        if reason:
            self.reset()
            return p, {"reliable": False, "reason": reason, "entropy": uncertainty, "centroid_xy": None}
        smooth = p if self.previous is None else self.alpha * p + (1 - self.alpha) * self.previous
        self.previous = smooth.copy()
        self.position = pos
        point = centroid(smooth >= .5)
        return smooth, {"reliable": True, "reason": None, "entropy": uncertainty,
                        "centroid_xy": point.tolist() if point is not None else None}
