"""Sequence-disjoint disk data and deterministic toy instrument sequences."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def synthetic(root, seed=7, size=64, frames=16):
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError("Dataset exists; use a new output directory")
    records = []
    for seq in range(8):
        rng = np.random.default_rng(seed + seq)
        phase = rng.uniform(0, 6)
        split = "train" if seq < 5 else "val" if seq == 5 else "test"
        for frame in range(frames):
            noise = rng.normal(0, 7, (size, size, 3))
            image = np.clip(np.array([95, 42, 48]) + noise, 0, 255).astype("uint8")
            mask = Image.new("L", (size, size))
            draw = ImageDraw.Draw(mask)
            x = int(size * (.5 + .22 * np.sin(frame / 7 + phase)))
            y = int(size * (.35 + .15 * np.cos(frame / 8 + phase)))
            draw.line([(x - size // 5, size - 1), (x, y)], fill=255, width=max(3, size // 10))
            draw.line([(x, y), (x - 4, y - 7)], fill=255, width=3)
            draw.line([(x, y), (x + 4, y - 7)], fill=255, width=3)
            fg = np.array(mask) > 0
            image[fg] = np.clip(np.array([180, 190, 195]) + noise[fg], 0, 255)
            stem = f"seq{seq:02d}/{frame:05d}"
            ip, mp = f"images/{stem}.png", f"masks/{stem}.png"
            for path in [ip, mp]:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(root / ip)
            mask.save(root / mp)
            records.append(dict(image=ip, mask=mp, sequence=f"seq{seq:02d}", frame=frame, split=split))
    write_json(root / "manifest.json", {"source": "SYNTHETIC_TOY_NOT_SURGICAL_DATA", "seed": seed, "records": records})


def validate_manifest(root):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    seen, keys = {}, set()
    for r in manifest["records"]:
        if r["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        if r["sequence"] in seen and seen[r["sequence"]] != r["split"]:
            raise ValueError("Sequence leakage across splits")
        seen[r["sequence"]] = r["split"]
        key = (r["sequence"], r["frame"])
        if key in keys:
            raise ValueError("Duplicate sequence/frame")
        keys.add(key)
        for field in ["image", "mask"]:
            path = (Path(root) / r[field]).resolve()
            if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
                raise ValueError(f"Invalid {field} path")
    return manifest


class Frames(Dataset):
    def __init__(self, root, split, size=64):
        self.root, self.size = Path(root), size
        self.manifest = validate_manifest(root)
        self.records = sorted([r for r in self.manifest["records"] if r["split"] == split],
                              key=lambda r: (r["sequence"], r["frame"]))
        if not self.records:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with Image.open(self.root / r["image"]) as im:
            x = np.asarray(im.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)).copy() / 255.
        with Image.open(self.root / r["mask"]) as im:
            y = (np.asarray(im.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)) > 0).astype("float32")
        return torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32), torch.from_numpy(y[None])


def obscure(x, severity):
    """Deterministic central opaque band; retain clean target as amodal stress target."""
    x = x.clone()
    width = round(x.shape[-1] * severity)
    start = (x.shape[-1] - width) // 2
    x[..., start:start + width] = .15
    return x


def degrade(x, kind, severity):
    if kind == "clean":
        return x.clone()
    if kind == "occlusion":
        return obscure(x, severity)
    if kind == "haze":
        tint = x.new_tensor([.42, .18, .12])[:, None, None]
        return (1 - severity) * x + severity * tint
    if kind == "opaque_fluid":
        out = x.clone()
        height = round(x.shape[-2] * severity)
        if height:
            out[:, -height:, :] = x.new_tensor([.3, .08, .06])[:, None, None]
        return out
    raise ValueError(kind)
