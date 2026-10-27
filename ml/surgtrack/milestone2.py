"""CPU-runnable sparse-label dual-motion experiment: python -m surgtrack.milestone2.

Original implementation of a simplified paper-inspired method, not authors' code.
Only training RGB reaches motion learning; only selected masks reach sparse models.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .core import TinyUNet, Tracker, centroid, scores, seed_all
from .data import degrade, validate_manifest, write_json
from .motion import FlowPredictor, FlowCompensator, splat, predictor_loss, compensation_loss


METHODS = ("sparse", "joint", "dual_no_cycle", "dual_cycle", "full")
CONDITIONS = ("clean", "occlusion50", "haze40", "haze80", "fluid50", "fluid100", "transient_fluid",
              "wet_specular", "nearby_bloom", "instrument_shadow", "wet_combined")


class VideoStore:
    """Separate RGB and guarded annotation access; never preload withheld masks."""
    def __init__(self, root, split, size=64, stride=5, allow_all=False):
        self.root, self.size = Path(root), size
        self.manifest = validate_manifest(root)
        self.records = sorted([r for r in self.manifest["records"] if r["split"] == split],
                              key=lambda r: (r["sequence"], r["frame"]))
        if not self.records:
            raise ValueError(f"Empty split {split}")
        self.sequences = defaultdict(list)
        self.images, self.accessed_masks = [], set()
        self.allowed = set()
        for index, record in enumerate(self.records):
            sequence = self.sequences[record["sequence"]]
            # Reset annotation schedule at gaps; propagate only across consecutive IDs.
            new_run = not sequence or record["frame"] != self.records[sequence[-1]]["frame"] + 1
            if new_run:
                run_start = len(sequence)
            if allow_all or (len(sequence) - run_start) % stride == 0:
                self.allowed.add(index)
            sequence.append(index)
            with Image.open(self.root / record["image"]) as image:
                array = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).copy()
            self.images.append(torch.tensor(array.transpose(2, 0, 1) / 255., dtype=torch.float32))
        self.pairs = [(i, j) for indices in self.sequences.values() for i, j in zip(indices, indices[1:])
                      if self.records[j]["frame"] == self.records[i]["frame"] + 1]

    def mask(self, index):
        if index not in self.allowed:
            raise PermissionError("Withheld mask access is forbidden in sparse training")
        self.accessed_masks.add(index)
        with Image.open(self.root / self.records[index]["mask"]) as mask:
            array = np.asarray(mask.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST))
        return torch.tensor((array > 0).astype("float32"))[None]

    def batch_pairs(self, generator, batch=8):
        if not self.pairs:
            raise ValueError("Motion learning requires consecutive frames within sequences")
        indices = torch.randint(len(self.pairs), (batch,), generator=generator).tolist()
        return (torch.stack([self.images[self.pairs[i][0]] for i in indices]),
                torch.stack([self.images[self.pairs[i][1]] for i in indices]))


def learn_motion(data, seed, steps, out):
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    predictor = FlowPredictor()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=.001)
    history = []
    for step in range(steps):
        source, target = data.batch_pairs(generator)
        optimizer.zero_grad()
        loss = (predictor_loss(source, target, predictor(source, target)) +
                predictor_loss(target, source, predictor(target, source))) / 2
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    predictor.eval().requires_grad_(False)
    torch.save({"model": predictor.state_dict(), "size": data.size, "seed": seed}, out / "predictor.pt")
    compensators = {}
    for use_cycle in (False, True):
        # Identical starts and pair batches isolate the cycle-loss ablation.
        seed_all(seed + 1000)
        generator = torch.Generator().manual_seed(seed + 1000)
        compensator = FlowCompensator()
        optimizer = torch.optim.Adam(compensator.parameters(), lr=.001)
        losses = []
        for step in range(steps):
            source, target = data.batch_pairs(generator)
            with torch.no_grad():
                forward, backward = predictor(source, target), predictor(target, source)
            optimizer.zero_grad()
            flows = compensator(source, target, forward, backward)
            loss = compensation_loss(source, target, flows, cycle=use_cycle)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        name = "cycle" if use_cycle else "no_cycle"
        compensator.eval().requires_grad_(False)
        torch.save({"model": compensator.state_dict(), "size": data.size, "seed": seed}, out / f"compensator_{name}.pt")
        compensators[name] = compensator
        write_json(out / f"motion_{name}.json", dict(losses=losses, cycle=use_cycle, steps=steps))
    write_json(out / "motion_predictor.json", dict(losses=history, steps=steps, training_rgb=len(data.images),
                                                   annotation_reads=len(data.accessed_masks)))
    return predictor, compensators


def supervised_pool(data):
    return [(data.images[i], data.mask(i), torch.ones(1, data.size, data.size)) for i in sorted(data.allowed)]


@torch.no_grad()
def joint_pool(data, predictor):
    """Causal multistep joint propagation; anchor again at every selected annotation."""
    rearranged = {}
    for indices in data.sequences.values():
        previous = None
        for index in indices:
            if index in data.allowed:
                sample = (data.images[index], data.mask(index), torch.ones(1, data.size, data.size))
            else:
                if previous is None or data.records[index]["frame"] != data.records[previous]["frame"] + 1:
                    raise ValueError("Unanchored temporal run")
                image, mask, support = rearranged[previous]
                flow = predictor(data.images[previous][None], data.images[index][None])
                transported, mass = splat(torch.cat([image, mask])[None], flow, source_support=support[None])
                sample = (transported[0, :3], transported[0, 3:4], mass[0])
            rearranged[index] = sample
            previous = index
    return rearranged


@torch.no_grad()
def compensated_pool(data, rearranged, predictor, compensator):
    samples = []
    for i, j in data.pairs:
        image, mask, support = rearranged[i]
        target = rearranged[j][0]
        forward, backward = predictor(image[None], target[None]), predictor(target[None], image[None])
        flow = compensator(image[None], target[None], forward, backward)[0]
        transported, mass = splat(torch.cat([image, mask])[None], flow, source_support=support[None])
        samples.append((transported[0, :3], transported[0, 3:4], mass[0]))
    return samples


def segmentation_loss(logits, targets, support):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * support).sum() / support.sum().clamp_min(1)
    p = logits.sigmoid()
    numerator = 2 * (p * targets * support).sum((1, 2, 3)) + 1
    denominator = ((p + targets) * support).sum((1, 2, 3)) + 1
    return bce + (1 - numerator / denominator).mean()


@torch.no_grad()
def validation_dice(model, data):
    return float(np.mean([scores(model(x[None]).sigmoid()[0, 0].numpy(), data.mask(i)[0].numpy())["dice"]
                          for i, x in enumerate(data.images)]))


def learn_segmenter(pool, validation, seed, steps, out, source):
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    model = TinyUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    best, history = -1., []
    for step in range(1, steps + 1):
        indices = torch.randint(len(pool), (8,), generator=generator).tolist()
        x, y, support = [torch.stack([pool[i][k] for i in indices]) for k in range(3)]
        model.train()
        optimizer.zero_grad()
        loss = segmentation_loss(model(x), y, support)
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == steps:
            model.eval()
            dice = validation_dice(model, validation)
            history.append(dict(step=step, loss=float(loss.item()), validation_dice=dice))
            if dice > best:
                best = dice
                torch.save(dict(model=model.state_dict(), size=validation.size, seed=seed, source=source,
                                step=step, milestone=2), out)
    write_json(str(out) + ".json", dict(history=history, steps=steps, batch_size=8, pool_size=len(pool)))


class MotionTracker:
    """Optional inference extension, separate from the paper's training method.

    Retains probability history only when the frame-level heuristic accepts it.
    Photo-error and round-trip checks are pixel heuristics, not safety guarantees.
    """
    def __init__(self, predictor):
        self.predictor = predictor
        self.gate = Tracker(alpha=1.)
        self.reset()

    def reset(self):
        self.previous_image = self.previous_probability = None
        self.gate.reset()

    @torch.no_grad()
    def update(self, image, probability):
        _, status = self.gate.update(probability)
        if not status["reliable"]:
            self.reset()
            return probability, {**status, "motion_support_fraction": 0.}
        result, fraction = probability, 0.
        if self.previous_image is not None:
            source = self.previous_image[None]
            target = image[None]
            forward = self.predictor(source, target)
            backward = self.predictor(target, source)
            old = torch.from_numpy(self.previous_probability)[None, None]
            transported, support = splat(torch.cat([source, old, forward], 1), forward)
            photo_error = (transported[:, :3] - target).abs().mean(1, keepdim=True)
            roundtrip = torch.linalg.vector_norm(transported[:, 4:6] + backward, dim=1, keepdim=True)
            trusted = ((support > .5) & (photo_error < .12) & (roundtrip < 1.5))[0, 0].numpy()
            warped_p = transported[0, 3].numpy()
            result = np.where(trusted, .7 * probability + .3 * warped_p, probability)
            fraction = float(trusted.mean())
        self.previous_image, self.previous_probability = image.clone(), result.copy()
        point = centroid(result >= .5)
        return result, {**status, "centroid_xy": None if point is None else point.tolist(),
                        "motion_support_fraction": fraction}


def condition_frame(image, condition, ordinal):
    if condition == "clean":
        return image.clone(), 0., "clean"
    if condition in {"wet_specular", "nearby_bloom", "instrument_shadow", "wet_combined"}:
        return lighting_corruption(image, condition, ordinal), 1., "lighting"
    settings = dict(occlusion50=("occlusion", .5), haze40=("haze", .4), haze80=("haze", .8),
                    fluid50=("opaque_fluid", .5), fluid100=("opaque_fluid", 1.))
    if condition == "transient_fluid":
        if 7 <= ordinal <= 11:
            return degrade(image, "opaque_fluid", 1.), 1., "obscured"
        return image.clone(), 0., "recovery" if ordinal >= 12 else "before"
    kind, severity = settings[condition]
    return degrade(image, kind, severity), severity, "static"


def lighting_corruption(image, kind, ordinal):
    """2D stress operators only: no fluid optics, tissue geometry or light transport.

    Coordinates depend on frame index, never clean labels. A moving saturated
    spot models lost detail, a larger halo models nearby overexposure, and a
    moving wedge models illumination loss (distinct from foreground occlusion).
    """
    height, width = image.shape[-2:]
    y, x = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    center_x = .5 + .15 * np.sin(ordinal / 4)
    center_y = .4 + .1 * np.cos(ordinal / 5)
    distance = (x - center_x) ** 2 + (y - center_y) ** 2
    result = image.clone()
    if kind in {"nearby_bloom", "wet_combined"}:
        halo = torch.exp(-distance / (.23 ** 2))
        result = (result + .95 * halo[None]).clamp(0, 1)
    if kind in {"wet_specular", "wet_combined"}:
        result[:, distance < .13 ** 2] = 1.
    if kind in {"instrument_shadow", "wet_combined"}:
        # Abstract widening projected shadow; it does not follow instrument masks.
        axis = .3 + .18 * np.sin(ordinal / 6) + .25 * y
        wedge = ((x - axis).abs() < (.055 + .12 * y)).float()
        result = result * (1 - .94 * wedge[None])
    return result.clamp(0, 1)


def average(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return float(np.mean(values)) if values else None


@torch.no_grad()
def evaluate(model, predictor, data, seed, method):
    rows = []
    for condition in CONDITIONS:
        gates = {"raw": Tracker(alpha=1.), "ema": Tracker(), "motion": MotionTracker(predictor)}
        for sequence, indices in data.sequences.items():
            for gate in gates.values():
                gate.reset()
            previous_id = None
            previous_masks = {}
            for ordinal, i in enumerate(indices):
                frame = data.records[i]["frame"]
                if previous_id is not None and frame != previous_id + 1:
                    for gate in gates.values():
                        gate.reset()
                    previous_masks = {}
                previous_id = frame
                image, severity, phase = condition_frame(data.images[i], condition, ordinal)
                truth = data.mask(i)[0].numpy()
                probability = model(image[None]).sigmoid()[0, 0].numpy()
                for mode, gate in gates.items():
                    start = time.perf_counter()
                    p, status = gate.update(image, probability) if mode == "motion" else gate.update(probability)
                    if mode == "raw":
                        p = probability
                    target_point = centroid(truth > .5)
                    point = status["centroid_xy"]
                    error = float(np.linalg.norm(np.array(point) - target_point)) if point is not None and target_point is not None else None
                    rows.append(dict(seed=seed, method=method, mode=mode, condition=condition, sequence=sequence,
                        frame=frame, ordinal=ordinal, severity=severity, phase=phase, **scores(p, truth), **status,
                        centroid_error_px=error, temporal_processing_ms=(time.perf_counter() - start) * 1000,
                        saturated_pixel_fraction=float((image.min(0).values >= .98).float().mean()),
                        dark_pixel_fraction=float((image.max(0).values <= .08).float().mean()),
                        probability_change=None if mode not in previous_masks else float(np.abs(p - previous_masks[mode]).mean())))
                    previous_masks[mode] = p.copy()
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["method"], row["mode"], row["condition"], row["phase"])].append(row)
    summaries = []
    for (seed, method, mode, condition, phase), group in sorted(grouped.items()):
        accepted = [r for r in group if r["reliable"]]
        summaries.append(dict(seed=seed, method=method, mode=mode, condition=condition, phase=phase,
            frames=len(group), dice=average(group, "dice"), iou=average(group, "iou"), brier=average(group, "brier"),
            coverage=len(accepted) / len(group), accepted_dice=average(accepted, "dice"),
            accepted_centroid_error_px=average(accepted, "centroid_error_px"),
            bad_but_accepted=sum(r["reliable"] and r["dice"] < .5 for r in group),
            probability_change=average(group, "probability_change")))
    return summaries


@torch.no_grad()
def preview(model, predictor, data, out):
    """Actual dual-cycle predictions under transient synthetic coverage."""
    indices = next(iter(data.sequences.values()))
    gate = MotionTracker(predictor)
    frames = []
    for ordinal, i in enumerate(indices):
        observed, _, phase = condition_frame(data.images[i], "transient_fluid", ordinal)
        p = model(observed[None]).sigmoid()[0, 0].numpy()
        p, status = gate.update(observed, p)
        rgb = (observed.permute(1, 2, 0).numpy() * 255).astype("uint8")
        overlay = rgb.copy()
        selected = p >= .5
        overlay[selected] = (.5 * overlay[selected] + .5 * np.array([20, 220, 80])).astype("uint8")
        if not status["reliable"]:
            overlay[:3] = [255, 120, 0]
        image = Image.new("RGB", (512, 312), "black")
        image.paste(Image.fromarray(rgb).resize((256, 256), Image.Resampling.NEAREST), (0, 56))
        image.paste(Image.fromarray(overlay).resize((256, 256), Image.Resampling.NEAREST), (256, 56))
        draw = ImageDraw.Draw(image)
        draw.text((8, 5), "SYNTHETIC - educational instrument demo; no anatomy", fill="white")
        draw.text((8, 21), f"Frame {ordinal:02d}: {phase} | track {'accepted' if status['reliable'] else 'rejected'}", fill="white")
        draw.text((8, 39), "Observed input", fill="white")
        draw.text((264, 39), "Dual-cycle + motion overlay", fill="white")
        frames.append(image)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=160, loop=0)


@torch.no_grad()
def lighting_preview(model, predictor, data, out):
    conditions = ("wet_specular", "nearby_bloom", "instrument_shadow", "wet_combined")
    gates = {name: MotionTracker(predictor) for name in conditions}
    frames = []
    for ordinal, index in enumerate(next(iter(data.sequences.values()))):
        canvas = Image.new("RGB", (512, 370), "black")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), "SYNTHETIC lighting stress - input | prediction", fill="white")
        draw.text((8, 20), "Nonclinical. Green = instrument; not a tissue plane.", fill="white")
        for panel, name in enumerate(conditions):
            image = lighting_corruption(data.images[index], name, ordinal)
            probability = model(image[None]).sigmoid()[0, 0].numpy()
            p, status = gates[name].update(image, probability)
            rgb = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
            overlay = rgb.copy()
            selected = p >= .5
            overlay[selected] = (.5 * overlay[selected] + .5 * np.array([20, 220, 80])).astype("uint8")
            if not status["reliable"]:
                overlay[:3] = [255, 120, 0]
            left, top = (panel % 2) * 256, 40 + (panel // 2) * 165
            draw.text((left + 5, top), name.replace("_", " "), fill="white")
            draw.text((left + 5, top + 15), "track accepted" if status["reliable"] else "track rejected", fill="white")
            for offset, array in ((0, rgb), (128, overlay)):
                canvas.paste(Image.fromarray(array).resize((128, 128), Image.Resampling.NEAREST), (left + offset, top + 32))
        frames.append(canvas)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=160, loop=0)


@torch.no_grad()
def benchmark(model, predictor, image, iterations=50):
    results = {}
    for mode in ("raw", "ema", "motion"):
        gate = MotionTracker(predictor) if mode == "motion" else Tracker(alpha=1. if mode == "raw" else .7)
        times, accepted, motion_updates = [], 0, 0
        for index in range(iterations + 5):
            has_history = isinstance(gate, MotionTracker) and gate.previous_image is not None
            start = time.perf_counter()
            probability = model(image[None]).sigmoid()[0, 0].numpy()
            _, status = gate.update(image, probability) if mode == "motion" else gate.update(probability)
            elapsed = (time.perf_counter() - start) * 1000
            if index >= 5:
                times.append(elapsed)
                accepted += int(status["reliable"])
                motion_updates += int(has_history and status["reliable"])
        results[mode] = dict(median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                             accepted_frames=accepted, motion_updates=motion_updates)
    return dict(iterations=iterations, warmup=5, cpu_threads=1, modes=results,
                scope="Segmentation + sigmoid + temporal processing on repeated clean RGB; excludes decode, resize, metrics and I/O")


def run(a):
    out = Path(a.out)
    if out.exists():
        raise FileExistsError("Use a new run directory; existing experiments are preserved")
    out.mkdir(parents=True)
    seed_all(a.seeds[0])
    training = VideoStore(a.data, "train", a.size, a.stride)
    validation = VideoStore(a.data, "val", a.size, allow_all=True)
    testing = VideoStore(a.data, "test", a.size, allow_all=True)
    full = VideoStore(a.data, "train", a.size, allow_all=True)
    # The full reference uses all annotations in its own explicitly separate store.
    protocol = dict(size=a.size, stride=a.stride, seeds=a.seeds, motion_steps=a.motion_steps,
        segmentation_steps=a.steps, batch_size=8, source=training.manifest["source"],
        sparse_annotation_count=len(training.allowed), training_frames=len(training.images),
        actual_label_fraction=len(training.allowed) / len(training.images),
        sparse_annotation_ids=[dict(sequence=training.records[i]["sequence"], frame=training.records[i]["frame"])
                               for i in sorted(training.allowed)],
        manifest_sha256=hashlib.sha256((Path(a.data) / "manifest.json").read_bytes()).hexdigest(),
        python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
        platform=platform.platform(), cpu_threads=1, data_seed=training.manifest.get("seed"),
        caveat="Paper-inspired adaptation; synthetic tests do not validate surgical or fluid performance.")
    write_json(out / "protocol.json", protocol)
    all_rows = []
    for seed in a.seeds:
        run_dir = out / f"seed{seed}"
        run_dir.mkdir()
        print(f"seed={seed}: learning motion using training RGB only", flush=True)
        predictor, compensators = learn_motion(training, seed, a.motion_steps, run_dir)
        rearranged = joint_pool(training, predictor)
        joint = list(rearranged.values())
        pools = {"sparse": supervised_pool(training), "joint": joint,
                 "dual_no_cycle": joint + compensated_pool(training, rearranged, predictor, compensators["no_cycle"]),
                 "dual_cycle": joint + compensated_pool(training, rearranged, predictor, compensators["cycle"]),
                 "full": supervised_pool(full)}
        for method in METHODS:
            print(f"seed={seed}: training {method}, pool={len(pools[method])}, steps={a.steps}", flush=True)
            checkpoint = run_dir / f"{method}.pt"
            learn_segmenter(pools[method], validation, seed, a.steps, checkpoint, training.manifest["source"])
            model = TinyUNet()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
            model.eval()
            rows = evaluate(model, predictor, testing, seed, method)
            all_rows.extend(rows)
            write_json(run_dir / f"{method}_evaluation.json", dict(frames=rows, summaries=summarize(rows)))
            if method == "dual_cycle" and seed == a.seeds[0]:
                preview(model, predictor, testing, out / "preview.gif")
                lighting_preview(model, predictor, testing, out / "lighting_preview.gif")
                write_json(out / "profile.json", benchmark(model, predictor, testing.images[0]))
        write_json(run_dir / "annotation_audit.json", dict(
            sparse_allowed=sorted(training.allowed), sparse_accessed=sorted(training.accessed_masks),
            full_reference_accessed=sorted(full.accessed_masks),
            note="Indices refer to protocol's sorted training records. Full reference masks never enter sparse pools."))
    write_json(out / "comparison.json", dict(protocol=protocol, summaries=summarize(all_rows)))
    print(f"Completed: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--motion-steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    a = parser.parse_args()
    if a.size < 16 or a.stride < 2 or a.steps < 1 or a.motion_steps < 1:
        parser.error("size>=16, stride>=2, steps>=1 and motion-steps>=1 required")
    if len(set(a.seeds)) != len(a.seeds):
        parser.error("Seeds must be distinct")
    run(a)


if __name__ == "__main__":
    main()
