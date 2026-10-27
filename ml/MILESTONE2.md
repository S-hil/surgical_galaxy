# Milestone 2 — sparse-label dual-motion learning

**Nonclinical educational research prototype. No tissue-plane inference, surgery guidance, surgical automation, or validation on actual fluid-obscured surgery.** This extends the instrument baseline; instrument masks remain distinct from tissue fibers or dissection planes.

## Paper relationship

Reference: [Zhao et al., MICCAI 2020, Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video](https://arxiv.org/abs/2007.02501). [Full method](https://arxiv.org/html/2007.02501v1), [authors' implementation](https://github.com/zxzhaoeric/Semi-InstruSeg).

The paper combines learned image/label propagation and intermediate-flow compensation, then trains segmentation using generated pairs. Its intermediate-flow initialization and four-leg cycle appear in equations (4)–(6). The authors use pretrained FlowNet2, learned perceptual losses, U-Net11/VGG11 segmentation, and an optional ConvLSTM. Their evaluation uses real EndoVis data and a different protocol.

This is an original, **paper-inspired adaptation**, not a faithful reproduction, authors' code port, or replication of published scores. No third-party model weights or code are redistributed.

## Implemented method

`surgtrack/motion.py` contains a pair-conditioned flow predictor, a four-flow residual compensator, and a differentiable normalized bilinear forward splat. `surgtrack/milestone2.py` trains and evaluates the experiment. Code intentionally runs on CPU without downloads or custom extensions.

1. Train the predictor using only adjacent training RGB pairs in both directions. Its objective combines masked L1 reconstruction, coarse RGB reconstruction, spatial flow smoothness, and support loss.
2. Freeze the predictor. Initialize intermediate flows at time 0.5, construct the four-leg image cycle, and train two identically initialized compensators: endpoint-only and endpoint + return/midpoint cycle objectives.
3. Reveal every fifth training annotation. Jointly propagate the current image, soft mask, and support along a sequence, re-anchoring at the next revealed annotation. Never pair a displaced mask with an unwarped real image.
4. Add compensated intermediate image-mask pairs to the rearranged training sequence. Train binary segmenters with support-weighted BCE + soft Dice.

Flow conventions are explicit: source-grid forward `(dx,dy)` in resized-image pixels; positive dx moves content right. Bilinear splatting discards out-of-bounds contributions, averages collisions, and leaves holes unsupported. Unsupported source pixels contribute no mass, avoiding dark hole pixels contaminating subsequent warps. Geometric support is not probability of clinical visibility; no depth-aware ordering or inpainting is performed.

Differences from the paper are material: our predictor sees an image pair rather than history plus pretrained optical flows; small randomly initialized networks replace the pretrained components; coarse RGB error replaces perceptual features; masks remain soft; interpolation uses one midpoint; propagation is forward from sparse anchors; regularization weights, training schedule, and evaluation differ; there is no ConvLSTM. The compensator is trained on raw pairs and used on reconstructed pairs, a distribution shift that can cause errors. The arithmetic intermediate-flow initialization is approximate on spatially varying flow fields.

## Run

From the project directory, after the baseline installation:

```bash
python -m pip install -e '.[dev]'
python -m surgtrack.cli synthetic --out milestone2/data --seed 29 --frames 20
python -m surgtrack.milestone2 --data milestone2/data --out milestone2/run --seeds 7 11 --motion-steps 100 --steps 160
python -m pytest -q
```

The bundled data/run already exist. To reproduce, use **new output directories** and point `--data` at that new dataset. The runner refuses to overwrite experiments. A small wiring check can use `--seeds 7 --motion-steps 2 --steps 2`; this verifies execution, not learning quality. The full run can take several minutes on CPU.

For locally obtained, converted EndoVis data:

```bash
python -m surgtrack.milestone2 --data data/endovis --out runs/endovis-motion --size 256 --stride 5 --motion-steps 1000 --steps 2000 --seeds 7 11
```

This is an example invocation, **not a tuned real-data recipe**. It has not been run on real EndoVis data. Follow the baseline README for legal access, annotation conversion, and procedure/patient separation. The in-memory experiment caches RGB and generated pairs; a large dataset/resolution may need a streaming dataset implementation. Flow bounds are in resized pixels and require validation for fast motion or larger images. Frame IDs must be consecutive integers for adjacent-frame motion training; the runner resets propagation at gaps.

## Baselines and controls

Five models are trained per seed:

- `sparse`: revealed annotations only.
- `joint`: sparse annotations + jointly propagated frame-mask pairs.
- `dual_no_cycle`: joint pairs + intermediate pairs using the endpoint-trained compensator.
- `dual_cycle`: joint pairs + intermediate pairs using the cycle-trained compensator.
- `full`: all training annotations, as a fully supervised reference (not a guaranteed upper bound).

Every segmentation model starts from identical weights per seed, receives the same 160 optimizer updates and batch size 8, uses the same optimizer, and is selected by clean validation Dice at fixed 20-step checkpoints. Generated-pair methods consume extra RGB and motion-training compute; this is equal **segmentation-update** budget, not equal total compute. Pools differ in size, so individual anchors are sampled with different frequency. The full reference uses a separate annotation store and never supplies masks to sparse methods. Two seeds demonstrate run-to-run variation; they do not establish statistical significance.

The shipped synthetic training split contains 100 frames and 20 revealed masks (20%). The actual fraction is computed and recorded; a stride of five need not imply exactly 20% for arbitrary sequence lengths or gaps. `VideoStore.mask` rejects withheld-mask access, and tests deliberately poison those files to expose accidental reads. Motion training opens RGB only. Validation and test masks are available solely to evaluation/selection in separate stores; test scores never select checkpoints. Test masks are evaluated for every frame.

`protocol.json` records sizes, seeds, budgets, annotation identities, environment, source label, and manifest hash. Individual checkpoint histories record selected validation performance. Annotation audit files list accessed training-mask indices. Fixed seeds and CPU deterministic operations support reruns on the tested environment; bitwise equality across library versions is not guaranteed.

## Temporal inference

Each segmenter is evaluated in three modes: frame-only (`raw`), baseline EMA (`ema`), and `motion`. Motion mode is an **additional engineering extension**, separate from the paper's generated-data training method. It forward-warps the previous accepted probability map and blends it with the current prediction. Pixels must pass support, photometric-error, and approximate round-trip checks. A rejected frame clears history and emits no position. The original entropy/area/jump gate is retained and remains uncalibrated.

```bash
python -m surgtrack.cli infer --input milestone2/data/images/seq06 --checkpoint milestone2/run/seed7/dual_cycle.pt --motion-checkpoint milestone2/run/seed7/predictor.pt --out milestone2/inference
```

The same command accepts a local video with `.[video]` installed. Output coordinates/masks use the square model resolution. A centroid of a binary instrument union is not an anatomical landmark, tool-tip detector, instance identity, or tissue plane. Motion compensation can fail during deformation, occlusion, illumination changes, out-of-frame motion, and disocclusion. A cycle-consistent warp can still be wrong.

## Wet-scene edge cases

All methods are tested on clean input and ten degraded scenarios:

- Central opaque occlusion; moderate/severe tinted haze; half/full-frame opaque fluid coverage.
- Transient full coverage at frames 7–11, with separate before, obscured, and recovery summaries.
- A moving white saturated spot approximating mirror-like wet-surface glare.
- A larger bright halo approximating overexposure blooming into nearby regions.
- A moving dark wedge approximating an instrument blocking illumination, distinct from directly occluding the camera view.
- Combined glare, nearby bloom, and shadow.

Lighting operators use image coordinates and frame index, not labels. They are **2D stress tests**, not physically faithful fluid/light transport, patient tissue, or a rendered light source/occluder simulation. The shadow is not geometrically tied to the rendered instrument. Saturation erases source detail; darkness reduces signal. Preserving accurate hidden anatomy is not a justified expectation. All corruptions retain the original full instrument mask for an amodal stress target, including hidden pixels. No artificial plane labels are generated.

Reports preserve Dice, IoU, Brier, accepted-frame error/coverage, probability changes, and failed-but-accepted counts. Per-frame logs record saturated/dark pixel fractions. Whole-image metrics can hide errors in small critical regions; mild saturation does not imply correct local perception. Thresholds were fixed before this run and are not calibrated clinical safety criteria. Partial visibility may produce confident accepted errors; this milestone does not solve that failure. Lower flicker or higher coverage is not automatically better. Recovery summaries are frame averages, not a validated time-to-recovery metric.

In the bundled predictions, bright specular spots and bloom can be classified as instruments while the track is accepted. The animations intentionally preserve these errors. This is a failure-analysis baseline, not a demonstration that learned motion makes wet-scene perception reliable. No post-test threshold tuning was performed to hide these cases.

## Outputs

`milestone2/run/comparison.json` contains per-seed/method/mode/scenario summaries. Each seed directory contains predictor/compensator/segmenter checkpoints, training histories, detailed frame evaluations, and annotation audits. `preview.gif` shows transient opacity; `lighting_preview.gif` shows the requested lighting edge cases, using actual seed-7 dual-cycle model predictions with motion inference. Orange stripes indicate rejected tracks; green indicates predicted instruments, never safe dissection regions.

`profile.json` measures warm CPU segmentation + temporal processing, with median/p95 and a count of motion updates actually executed. Input is a repeated clean frame. Decode, resizing, display, metric calculation, and disk I/O are excluded. Per-frame evaluation timings are diagnostics that include evaluation overhead; use the separate profile for inference timing. Neither is an end-to-end deployment guarantee.

The new segmenter checkpoint is compatible with the existing ONNX path:

```bash
python -m pip install -e '.[export]'
python -m surgtrack.cli export --checkpoint milestone2/run/seed7/dual_cycle.pt --out milestone2/segmenter.onnx
```

Only segmentation is exported. Flow prediction, forward splatting, compensator, gates, temporal state, and preprocessing remain PyTorch/Python; no claim of a fully exported temporal system is made.

## Interview discussion

Explain why joint propagation aligns a generated image and mask, while a warped label paired with an unchanged image can introduce contradictory supervision. Walk through forward versus backward flow, holes, collisions, and support-weighted losses. Explain what the cycle ablation actually isolates and why cycle consistency alone cannot establish correct motion. Discuss label budget versus total compute, the fully supervised reference, validation-only model selection, and procedure-level splits.

For the tissue-plane direction, the next prerequisite is authorized expert annotations of visible connective-tissue/plane cues, unknown visibility, and annotator disagreement. Neither binary instruments nor synthetic lighting effects establish the ability to identify a safe dissection plane under liquid.
