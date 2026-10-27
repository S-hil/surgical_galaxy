import numpy as np
import pytest
import torch

from surgtrack.core import seed_all
from surgtrack.data import synthetic
from surgtrack.milestone2 import (VideoStore, joint_pool, compensated_pool, lighting_corruption,
                                  segmentation_loss, MotionTracker)
from surgtrack.motion import (splat, intermediate_flows, cycle_frames, FlowPredictor,
                              FlowCompensator, predictor_loss, compensation_loss)


def test_forward_splat_identity_translation_and_border_holes():
    image = torch.arange(30, dtype=torch.float32).reshape(1, 1, 5, 6)
    flow = torch.zeros(1, 2, 5, 6)
    result, support = splat(image, flow)
    assert torch.equal(result, image) and torch.equal(support, torch.ones_like(support))
    flow[:, 0] = 1  # Source pixels move right, rather than backward sampling.
    result, support = splat(image, flow)
    assert torch.equal(result[..., 1:], image[..., :-1])
    assert not support[..., 0].any()
    assert torch.equal(support[..., 1:], torch.ones_like(support[..., 1:]))


def test_forward_splat_joint_alignment_and_fractional_gradients():
    seed_all(7)
    image = torch.rand(1, 1, 4, 4, dtype=torch.double, requires_grad=True)
    flow = torch.full((1, 2, 4, 4), .23, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda im, f: splat(im, f)[0], (image, flow), eps=1e-6, atol=1e-4)
    duplicate = torch.cat([image.detach(), image.detach()], 1)
    result, _ = splat(duplicate, flow.detach())
    assert torch.equal(result[:, 0], result[:, 1])


def test_forward_splat_does_not_mix_unsupported_source_pixels():
    image = torch.tensor([[[[10., 100.], [20., 100.]]]])
    support = torch.tensor([[[[1., 0.], [1., 0.]]]])
    flow = torch.zeros(1, 2, 2, 2)
    flow[:, 0] = .5
    result, mass = splat(image, flow, source_support=support)
    assert torch.allclose(result, torch.tensor([[[[10., 10.], [20., 20.]]]]))
    assert torch.allclose(mass, torch.full_like(mass, .5))


def test_equation_four_constant_motion_and_four_leg_cycle():
    forward = torch.zeros(1, 2, 16, 16)
    forward[:, 0] = 2
    flows = intermediate_flows(forward, -forward)
    assert [f[0, 0, 8, 8].item() for f in flows] == [1, 1, -1, -1]
    image = torch.rand(1, 3, 16, 16)
    frames, support = cycle_frames(image, flows)
    assert torch.allclose(frames[-1][..., 2:-2], image[..., 2:-2], atol=1e-6)
    assert (support[-1] < 1).any()  # Border information is genuinely lost.
    with pytest.raises(ValueError):
        intermediate_flows(forward, -forward, time=1.)


def test_flow_and_compensation_have_finite_nonzero_gradients():
    seed_all(7)
    source, target = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    predictor = FlowPredictor()
    forward, backward = predictor(source, target), predictor(target, source)
    loss = predictor_loss(source, target, forward)
    loss.backward()
    assert torch.isfinite(loss) and predictor.head.weight.grad.abs().sum() > 0
    compensator = FlowCompensator()
    flows = compensator(source, target, forward.detach(), backward.detach())
    cycle = compensation_loss(source, target, flows)
    cycle.backward()
    assert torch.isfinite(cycle) and compensator.refiner.head.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in compensator.parameters())


def test_sparse_pool_never_opens_withheld_masks(tmp_path):
    seed_all(7)
    synthetic(tmp_path / "data", frames=6, size=32)
    data = VideoStore(tmp_path / "data", "train", size=32, stride=3)
    assert len(data.allowed) == 10
    withheld = [i for i in range(len(data.images)) if i not in data.allowed]
    # Deliberately invalid files: any accidental image read would now fail.
    for i in withheld:
        (data.root / data.records[i]["mask"]).write_bytes(b"POISONED WITHHELD LABEL")
    with pytest.raises(PermissionError):
        data.mask(withheld[0])
    source, target = data.batch_pairs(torch.Generator().manual_seed(7), batch=2)
    predictor = FlowPredictor().eval()
    predictor_loss(source, target, predictor(source, target)).backward()
    assert not data.accessed_masks
    pool = joint_pool(data, predictor)
    assert len(pool) == len(data.images)
    assert data.accessed_masks == data.allowed
    x, y, support = pool[withheld[0]]
    assert x.shape == (3, 32, 32) and y.shape == support.shape == (1, 32, 32)
    compensated = compensated_pool(data, pool, predictor, FlowCompensator().eval())
    assert len(compensated) == len(data.pairs)
    assert data.accessed_masks == data.allowed


def test_masked_segmentation_ignores_holes():
    logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
    support = torch.ones_like(logits)
    support[..., :8] = 0
    target = torch.zeros_like(logits)
    first = segmentation_loss(logits, target, support)
    target[..., :8] = 1
    second = segmentation_loss(logits, target, support)
    assert torch.equal(first, second)
    second.backward()
    assert not logits.grad[..., :8].any()


def test_lighting_corruptions_are_bounded_and_do_not_use_labels():
    image = torch.full((3, 32, 32), .4)
    for kind in ("wet_specular", "nearby_bloom", "instrument_shadow", "wet_combined"):
        result = lighting_corruption(image, kind, 8)
        assert result.min() >= 0 and result.max() <= 1
        assert torch.equal(result, lighting_corruption(image, kind, 8))
        assert not torch.equal(result, image)
    assert (lighting_corruption(image, "wet_specular", 8) == 1).any()
    assert lighting_corruption(image, "nearby_bloom", 8).mean() > image.mean()
    assert lighting_corruption(image, "instrument_shadow", 8).min() < .05
    assert torch.equal(image, torch.full_like(image, .4))


def test_motion_tracker_rejection_clears_history():
    seed_all(7)
    tracker = MotionTracker(FlowPredictor().eval())
    image = torch.rand(3, 32, 32)
    p = np.full((32, 32), .01, dtype="float32")
    p[5:12, 5:12] = .99
    assert tracker.update(image, p)[1]["reliable"]
    assert tracker.previous_image is not None
    _, status = tracker.update(image, np.full_like(p, .5))
    assert status["centroid_xy"] is None and tracker.previous_image is None
    assert tracker.update(image, p)[1]["motion_support_fraction"] == 0
