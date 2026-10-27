"""Small dual-motion adaptation of Zhao et al., MICCAI 2020 (not a reproduction).

Flows are SOURCE-grid forward displacements in PIXELS: positive dx moves right.
Forward splatting jointly transports image and label with the identical operator.
CPU scatter_add is differentiable; no pretrained weights or CUDA extensions needed.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .core import Block


def splat(image, flow, source_support=None):
    """Normalized bilinear forward splat, returning image and clipped support.

    image: B,C,H,W; flow: B,2,H,W with (dx,dy). Out-of-bounds contributions
    are discarded, holes are zero, many-to-one collisions are averaged. Support
    is geometric mass, NOT visibility probability or an occlusion detector.
    """
    if image.ndim != 4 or flow.shape != (image.shape[0], 2, *image.shape[-2:]):
        raise ValueError("Expected BCHW image and B2HW pixel flow")
    batch, channels, height, width = image.shape
    if source_support is None:
        source_support = torch.ones_like(image[:, :1])
    elif source_support.shape != (batch, 1, height, width):
        raise ValueError("Expected B1HW source support")
    yy, xx = torch.meshgrid(torch.arange(height, device=image.device, dtype=image.dtype),
                            torch.arange(width, device=image.device, dtype=image.dtype), indexing="ij")
    x = xx[None] + flow[:, 0]
    y = yy[None] + flow[:, 1]
    x0, y0 = x.floor(), y.floor()
    total = image.new_zeros(batch, channels, height * width)
    mass = image.new_zeros(batch, 1, height * width)
    for xi, yi, weight in (
        (x0, y0, (1 - (x - x0)) * (1 - (y - y0))),
        (x0 + 1, y0, (x - x0) * (1 - (y - y0))),
        (x0, y0 + 1, (1 - (x - x0)) * (y - y0)),
        (x0 + 1, y0 + 1, (x - x0) * (y - y0)),
    ):
        inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        weight = (weight * inside).reshape(batch, 1, -1) * source_support.flatten(2)
        index = (yi.clamp(0, height - 1) * width + xi.clamp(0, width - 1)).long().reshape(batch, 1, -1)
        total = total.scatter_add(2, index.expand(-1, channels, -1), image.flatten(2) * weight)
        mass = mass.scatter_add(2, index, weight)
    normalized = total / mass.clamp_min(1e-6)
    return normalized.reshape_as(image), mass.reshape(batch, 1, height, width).clamp(0, 1)


class SmallFlowNet(nn.Module):
    def __init__(self, inputs=6, outputs=2, bound=8.):
        super().__init__()
        self.enc = Block(inputs, 12)
        self.mid = Block(12, 24)
        self.dec = Block(36, 12)
        self.head = nn.Conv2d(12, outputs, 3, padding=1)
        self.bound = bound
        # Small nonzero initialization avoids a permanently identity solution.
        nn.init.normal_(self.head.weight, std=.001)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skip = self.enc(x)
        low = self.mid(F.max_pool2d(skip, 2))
        up = F.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.bound * torch.tanh(self.head(self.dec(torch.cat([skip, up], 1))))


class FlowPredictor(SmallFlowNet):
    """Pair-conditioned flow, rather than paper's history+FlowNet2 conditioning."""
    def forward(self, source, target):
        return super().forward(torch.cat([source, target], 1))


def intermediate_flows(forward, backward, time=.5):
    """Equation (4) initialization; approximate grids, not exact flow composition.

    Return ordered (0->i, i->1, 1->i, i->0) for the four-leg cycle.
    """
    if not 0 < time < 1:
        raise ValueError("Interpolation time must lie strictly between zero and one")
    i0 = -(1 - time) * time * forward + time ** 2 * backward
    i1 = (1 - time) ** 2 * forward - time * (1 - time) * backward
    return (forward - i1, i1, backward - i0, i0)


def cycle_frames(source, flows):
    images, supports = [], []
    image = source
    support = torch.ones_like(source[:, :1])
    for flow in flows:
        # Carry prior support through the same warp to avoid silently filling holes.
        image, support = splat(image, flow, source_support=support)
        images.append(image)
        supports.append(support)
    return images, supports


class FlowCompensator(nn.Module):
    """Refine four intermediate flows from endpoints, initial flows and warps."""
    def __init__(self):
        super().__init__()
        self.refiner = SmallFlowNet(inputs=26, outputs=8, bound=2.)

    def forward(self, source, target, forward, backward):
        initial = intermediate_flows(forward, backward)
        frames, _ = cycle_frames(source, initial)
        features = torch.cat([source, target, *initial, *frames], 1)
        residuals = self.refiner(features).chunk(4, 1)
        return tuple(flow + delta for flow, delta in zip(initial, residuals))


def masked_l1(prediction, target, support):
    # Detach mask so lowering support cannot directly optimize the error weight.
    weight = support.detach()
    return ((prediction - target).abs() * weight).sum() / (weight.sum() * prediction.shape[1]).clamp_min(1)


def smoothness(flow):
    return (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean() + (
        flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()


def predictor_loss(source, target, flow):
    warped, support = splat(source, flow)
    reconstruction = masked_l1(warped, target, support)
    # A multi-scale RGB loss is NOT a learned perceptual-feature loss.
    coarse = F.l1_loss(F.avg_pool2d(warped, 4), F.avg_pool2d(target, 4))
    loss = .7 * reconstruction + .2 * coarse + .1 * smoothness(flow) + .05 * (1 - support).mean()
    return loss


def compensation_loss(source, target, flows, cycle=True):
    frames, support = cycle_frames(source, flows)
    mid0, end1, mid1, end0 = frames
    endpoint = 2 * masked_l1(end1, target, support[1])
    cycle_term = masked_l1(end0, source, support[3]) + .8 * masked_l1(mid0, mid1, support[0] * support[2])
    penalty = .01 * sum(smoothness(flow) for flow in flows) + .05 * sum((1 - s).mean() for s in support)
    return endpoint + (cycle_term if cycle else 0) + penalty
