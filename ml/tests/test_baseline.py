import json

import numpy as np
from PIL import Image
import pytest
import torch

from surgtrack.core import TinyUNet, Tracker, loss_fn, scores, seed_all
from surgtrack.data import Frames, degrade, synthetic, validate_manifest
from surgtrack.cli import prepare
from types import SimpleNamespace


def test_model_gradients_and_odd_size():
    seed_all(7)
    model = TinyUNet()
    x = torch.rand(2, 3, 33, 35)
    y = torch.zeros(2, 1, 33, 35)
    y[:, :, 10:20, 10:20] = 1
    output = model(x)
    assert output.shape == y.shape
    loss = loss_fn(output, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_metrics_known_cases():
    y = np.array([[1., 0.], [0., 0.]])
    assert scores(y, y) == dict(iou=1., dice=1., brier=0.)
    assert scores(1 - y, y)["dice"] == 0
    assert scores(np.zeros((2, 2)), np.zeros((2, 2)))["dice"] == 1


def test_tracker_rejects_unknown_no_stale_position_and_recovers():
    tracker = Tracker()
    p = np.full((32, 32), .01)
    p[4:10, 5:11] = .99
    _, status = tracker.update(p)
    assert status["reliable"]
    _, status = tracker.update(np.full_like(p, .5))
    assert not status["reliable"] and status["centroid_xy"] is None
    assert tracker.previous is None
    _, status = tracker.update(p)
    assert status["reliable"]
    moved = np.full_like(p, .01)
    moved[24:30, 24:30] = .99
    _, status = tracker.update(moved)
    assert status["reason"] == "centroid_jump"
    tracker.reset()
    assert tracker.update(moved)[1]["reliable"]


def test_dataset_reproducibility_split_guard_and_mask_resize(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    synthetic(a, frames=2)
    synthetic(b, frames=2)
    assert (a / "images/seq00/00000.png").read_bytes() == (b / "images/seq00/00000.png").read_bytes()
    ds = Frames(a, "test", 33)
    x, y = ds[0]
    assert x.shape == (3, 33, 33) and set(y.unique().tolist()) <= {0., 1.}
    manifest = validate_manifest(a)
    manifest["records"][1]["split"] = "test"
    (a / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(a)


def test_degradations_do_not_modify_input_or_leak_clean_pixels():
    x = torch.rand(3, 32, 32)
    original = x.clone()
    for kind in ["clean", "occlusion", "haze", "opaque_fluid"]:
        assert degrade(x, kind, .5).shape == x.shape
    opaque = degrade(x, "opaque_fluid", 1.)
    assert torch.allclose(opaque, opaque[:, :1, :1].expand_as(opaque))
    assert torch.equal(x, original)
    assert torch.equal(degrade(x, "occlusion", 0.), x)


def test_endovis_binary_union_and_missing_mask(tmp_path):
    seq = tmp_path / "raw/instrument_dataset_1"
    (seq / "left_frames").mkdir(parents=True)
    Image.fromarray(np.zeros((16, 16, 3), dtype="uint8")).save(seq / "left_frames/frame000.png")
    for i in range(2):
        folder = seq / "ground_truth" / f"instrument{i}"
        folder.mkdir(parents=True)
        mask = np.zeros((16, 16), dtype="uint8")
        mask[i, i] = 10
        Image.fromarray(mask).save(folder / "frame000.png")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [seq.name]}))
    prepare(SimpleNamespace(input=tmp_path / "raw", out=tmp_path / "converted", splits=splits))
    assert Frames(tmp_path / "converted", "train", 16)[0][1].sum() == 2
    for p in (seq / "ground_truth").rglob("*.png"):
        p.unlink()
    with pytest.raises(ValueError, match="Missing masks"):
        prepare(SimpleNamespace(input=tmp_path / "raw", out=tmp_path / "failed", splits=splits))
