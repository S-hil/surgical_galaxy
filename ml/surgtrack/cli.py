"""Run with python -m surgtrack.cli --help. CPU-first reproducible baseline."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .core import TinyUNet, Tracker, centroid, loss_fn, scores, seed_all
from .data import Frames, degrade, synthetic, write_json


def load(path):
    # Load only checkpoints you created/trust; older PyTorch uses pickle here.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyUNet()
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


def predict(model, x):
    with torch.inference_mode():
        return model(x[None]).sigmoid()[0, 0].numpy()


def train(a):
    seed_all(a.seed)
    train_ds = Frames(a.data, "train", a.size)
    val_ds = Frames(a.data, "val", a.size)
    model = TinyUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val_dice = float(np.mean([scores(predict(model, x), y[0].numpy())["dice"] for x, y in val_ds]))
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), val_dice=val_dice))
        print(json.dumps(history[-1]), flush=True)
        if val_dice > best:
            best = val_dice
            torch.save(dict(model=model.state_dict(), size=a.size, seed=a.seed,
                            epoch=epoch + 1, source=train_ds.manifest["source"]), out / "best.pt")
    write_json(out / "training.json", dict(history=history, seed=a.seed, torch=torch.__version__,
               python=platform.python_version(), device="cpu", threads=1,
               manifest_sha256=hashlib.sha256((Path(a.data) / "manifest.json").read_bytes()).hexdigest()))


def evaluate(a):
    seed_all(7)
    model, ckpt = load(a.checkpoint)
    ds = Frames(a.data, a.split, ckpt["size"])
    rows, summaries = [], []
    conditions = [("clean", 0), ("occlusion", .25), ("occlusion", .5),
                  ("haze", .4), ("haze", .8), ("opaque_fluid", .5), ("opaque_fluid", 1.)]
    for kind, severity in conditions:
        tracker, previous_seq, previous_raw, previous_smooth = Tracker(), None, None, None
        group = []
        for i, (x, y) in enumerate(ds):
            record = ds.records[i]
            if previous_seq != record["sequence"]:
                tracker.reset()
                previous_raw = previous_smooth = None
            previous_seq = record["sequence"]
            p = predict(model, degrade(x, kind, severity))
            smooth, status = tracker.update(p)
            target = y[0].numpy()
            raw_scores, sm_scores = scores(p, target), scores(smooth, target)
            point, truth = status["centroid_xy"], centroid(target > .5)
            error = float(np.linalg.norm(np.array(point) - truth)) if point is not None and truth is not None else None
            row = dict(sequence=record["sequence"], frame=record["frame"], condition=kind,
                       severity=severity, raw=raw_scores, temporal=sm_scores, **status,
                       centroid_error_px=error,
                       raw_probability_change=None if previous_raw is None else float(np.abs(p - previous_raw).mean()),
                       temporal_probability_change=None if previous_smooth is None else float(np.abs(smooth - previous_smooth).mean()))
            previous_raw, previous_smooth = p, smooth
            group.append(row)
            rows.append(row)
        accepted = [r for r in group if r["reliable"]]
        errors = [r["centroid_error_px"] for r in accepted if r["centroid_error_px"] is not None]
        summaries.append(dict(condition=kind, severity=severity, frames=len(group),
            raw_dice=float(np.mean([r["raw"]["dice"] for r in group])),
            temporal_dice=float(np.mean([r["temporal"]["dice"] for r in group])),
            raw_iou=float(np.mean([r["raw"]["iou"] for r in group])),
            brier=float(np.mean([r["raw"]["brier"] for r in group])),
            coverage=len(accepted) / len(group),
            accepted_dice=float(np.mean([r["temporal"]["dice"] for r in accepted])) if accepted else None,
            accepted_centroid_error_px=float(np.mean(errors)) if errors else None,
            unreliable_but_accepted=sum(r["reliable"] and r["raw"]["dice"] < .5 for r in group),
            raw_probability_change=mean_present(group, "raw_probability_change"),
            temporal_probability_change=mean_present(group, "temporal_probability_change")))
    write_json(a.out, dict(source=ds.manifest["source"], split=a.split,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        target="clean full instrument mask retained under simulated degradation; amodal stress test",
        warning="No real-fluid, tissue-plane, or clinical validation. Heuristics are uncalibrated.",
        tracker=dict(alpha=.7, max_entropy=.65, max_jump=.3, min_pixels=8), summaries=summaries, frames=rows))
    print(f"Saved {len(rows)} frame evaluations to {a.out}")


def mean_present(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def frame_stream(path):
    path = Path(path)
    if path.is_dir():
        paths = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not paths:
            raise ValueError("No frames found")
        for p in paths:
            with Image.open(p) as im:
                yield p.name, im.convert("RGB").copy()
    else:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("Install .[video] to decode video files") from e
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        try:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield f"{index:06d}.png", Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                index += 1
            if index == 0:
                raise ValueError("Video contained no decoded frames")
        finally:
            cap.release()


def infer(a):
    seed_all(7)
    model, ckpt = load(a.checkpoint)
    tracker, records = Tracker(), []
    if a.motion_checkpoint:
        from .milestone2 import MotionTracker
        from .motion import FlowPredictor
        motion_checkpoint = torch.load(a.motion_checkpoint, map_location="cpu", weights_only=True)
        if motion_checkpoint["size"] != ckpt["size"]:
            raise ValueError("Motion and segmentation checkpoint sizes differ")
        predictor = FlowPredictor()
        predictor.load_state_dict(motion_checkpoint["model"])
        tracker = MotionTracker(predictor.eval())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, image) in enumerate(frame_stream(a.input)):
        image = image.resize((ckpt["size"], ckpt["size"]), Image.Resampling.BILINEAR)
        rgb = np.asarray(image).copy()
        x = torch.tensor(rgb.transpose(2, 0, 1) / 255., dtype=torch.float32)
        raw = predict(model, x)
        p, status = tracker.update(x, raw) if a.motion_checkpoint else tracker.update(raw)
        mask = p >= .5
        overlay = rgb.copy()
        overlay[mask] = (.5 * overlay[mask] + .5 * np.array([20, 220, 80])).astype("uint8")
        if not status["reliable"]:
            overlay[:4] = [255, 120, 0]
        Image.fromarray(overlay).save(out / f"{i:06d}_overlay.png")
        Image.fromarray((mask * 255).astype("uint8")).save(out / f"{i:06d}_mask.png")
        records.append(dict(input=name, **status))
    write_json(out / "tracks.json", dict(coordinates="resized image pixels", size=ckpt["size"], frames=records))


def profile(a):
    seed_all(7)
    model, ckpt = load(a.checkpoint)
    x = torch.zeros(1, 3, ckpt["size"], ckpt["size"])
    tracker = Tracker()
    times = []
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        for _ in range(a.iterations):
            start = time.perf_counter()
            p = model(x).sigmoid()[0, 0].numpy()
            tracker.update(p)
            times.append((time.perf_counter() - start) * 1000)
    write_json(a.out, dict(device="cpu", threads=1, torch=torch.__version__, platform=platform.platform(),
        size=ckpt["size"], iterations=a.iterations, warmup=10, parameters=sum(p.numel() for p in model.parameters()),
        median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
        scope="model + sigmoid + tracker only; excludes decode, resize, I/O; zero tensor input"))


def export(a):
    seed_all(7)
    model, ckpt = load(a.checkpoint)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 3, ckpt["size"], ckpt["size"])
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError("Install .[export] for ONNX export and numerical parity check") from e
    torch.onnx.export(model, x, a.out, input_names=["rgb"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"rgb": {0: "batch"}, "logits": {0: "batch"}})
    onnx.checker.check_model(onnx.load(a.out))
    session = ort.InferenceSession(str(a.out), providers=["CPUExecutionProvider"])
    errors = []
    for batch in (1, 2):
        inp = torch.rand(batch, 3, ckpt["size"], ckpt["size"])
        with torch.inference_mode():
            expected = model(inp).numpy()
        actual = session.run(None, {"rgb": inp.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        errors.append(float(np.abs(actual - expected).max()))
    write_json(str(a.out) + ".json", dict(opset=17, batch_sizes=[1, 2], max_abs_errors=errors,
        preprocessing="RGB float32 NCHW [0,1], square resize to checkpoint size", output="logits; sigmoid required",
        excluded="temporal state, reliability gate, decoding and preprocessing"))


def prepare(a):
    """Convert extracted EndoVis 2017 left_frames + per-instrument masks."""
    source, dest = Path(a.input), Path(a.out)
    if dest.exists():
        raise FileExistsError("Use a new output directory")
    splits = json.loads(Path(a.splits).read_text())
    seen, records = set(), []
    for split, sequences in splits.items():
        if split not in {"train", "val", "test"}:
            raise ValueError("Expected train/val/test keys")
        for sequence in sequences:
            if sequence in seen:
                raise ValueError("Sequence leakage")
            seen.add(sequence)
            seq = source / sequence
            frames = sorted((seq / "left_frames").glob("*.png"))
            if not frames:
                raise ValueError(f"No PNG left_frames for {sequence}")
            for index, frame in enumerate(frames):
                masks = sorted((seq / "ground_truth").rglob(frame.name))
                if not masks:
                    raise ValueError(f"Missing masks for {frame}; do not treat missing labels as background")
                with Image.open(frame) as im:
                    rgb = im.convert("RGB")
                union = np.zeros((rgb.height, rgb.width), dtype=bool)
                for mask in masks:
                    with Image.open(mask) as im:
                        arr = np.asarray(im)
                    if arr.ndim == 3:
                        arr = arr.max(axis=2)
                    if arr.shape != union.shape:
                        raise ValueError(f"Mask/image size mismatch: {mask}")
                    union |= arr > 0
                ip, mp = f"images/{sequence}/{frame.name}", f"masks/{sequence}/{frame.name}"
                for path in (ip, mp):
                    (dest / path).parent.mkdir(parents=True, exist_ok=True)
                rgb.save(dest / ip)
                Image.fromarray((union * 255).astype("uint8")).save(dest / mp)
                records.append(dict(image=ip, mask=mp, sequence=sequence, frame=index, split=split))
    write_json(dest / "manifest.json", dict(source="EndoVis2017_local_user_supplied_binary_union", records=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--frames", type=int, default=16)
    p.set_defaults(func=lambda a: synthetic(a.out, a.seed, a.size, a.frames))
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.set_defaults(func=train)
    for command, function in [("evaluate", evaluate), ("infer", infer), ("profile", profile), ("export", export)]:
        p = sub.add_parser(command)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--out", required=True)
        p.set_defaults(func=function)
        if command == "evaluate":
            p.add_argument("--data", required=True)
            p.add_argument("--split", choices=["val", "test"], default="test")
        if command == "infer":
            p.add_argument("--input", required=True)
            p.add_argument("--motion-checkpoint", help="Milestone 2 predictor.pt; enables motion-aware temporal inference")
        if command == "profile":
            p.add_argument("--iterations", type=int, default=100)
    p = sub.add_parser("prepare-endovis")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", required=True)
    p.set_defaults(func=prepare)
    a = parser.parse_args()
    for field in ("size", "frames", "epochs", "iterations"):
        if hasattr(a, field) and getattr(a, field) < (16 if field == "size" else 1):
            parser.error(f"--{field} is too small")
    a.func(a)


if __name__ == "__main__":
    main()
