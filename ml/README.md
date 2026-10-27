# Surgical video perception: milestone 1

**Milestone 2 is now available:** see [the motion-learning experiment](MILESTONE2.md) for sparse-label joint propagation, learned intermediate-flow compensation, cycle ablations, and wet-scene glare/bloom/shadow tests. The baseline commands below remain supported.

**Nonclinical educational prototype. No surgical guidance, automation, robot control, or real-world validation.** Inspired by surgical perception engineering, without affiliation with Andromeda Surgical or claims about its internal methods.

The longer-term question is recognizing tissue dissection planes when visibility degrades. This milestone deliberately establishes an instrument-perception pipeline first. **Instrument masks are not tissue planes.** The demo does not detect anatomy, estimate a 3D plane, see through opaque liquid, or identify safe places to dissect.

## What runs

- A small, randomly initialized PyTorch U-Net-style binary segmenter with BCE + soft Dice loss.
- Deterministic toy instrument videos, split by sequence (5 training, 1 validation, 2 testing).
- Binary instrument-mask adapter for locally obtained EndoVis 2017 data.
- Ordered frame/video inference with masks, overlays, and one foreground-centroid track.
- Stateful probability EMA (unwarped) and entropy/area/jump rejection. Rejection clears state and returns no position; the raw segmentation remains visible and is marked with an orange stripe.
- Clean versus central occlusion, tinted haze, and opaque lower-frame coverage stress tests; Dice, IoU, Brier, coverage, accepted-frame Dice, centroid error, probability change, and failed-but-accepted counts.
- CPU profiling and optional ONNX export with runtime parity checks.

## Quick start

Run from this project directory. Python 3.10–3.11 is a conservative choice.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m surgtrack.cli synthetic --out demo/data
python -m surgtrack.cli train --data demo/data --out demo/run --epochs 12
python -m surgtrack.cli evaluate --data demo/data --checkpoint demo/run/best.pt --out demo/evaluation.json
python -m surgtrack.cli infer --input demo/data/images/seq06 --checkpoint demo/run/best.pt --out demo/predictions
python -m surgtrack.cli profile --checkpoint demo/run/best.pt --out demo/profile.json
python -m pytest -q
```

Use a new synthetic output directory to avoid overwriting data. All operations use CPU, one PyTorch thread, and fixed seeds; checkpoint selection uses only validation Dice. The manifest and checkpoint hashes are recorded. Exact reproducibility across PyTorch versions/platforms is not promised. `demo/environment.txt` records the environment used for the bundled run.

The shipped `demo/` contains an actual synthetic-only run, not benchmark results on real surgery. Predictions are resized to model coordinates (64×64 by default). Green indicates predicted foreground, not a safe surgical region. `tracks.json` stores centroid coordinates and rejection reasons.

### Video files

```bash
python -m pip install -e '.[video]'
python -m surgtrack.cli infer --input /path/to/local/video.mp4 --checkpoint demo/run/best.pt --out demo/video_predictions
```

Or pass a directory of zero-padded PNG/JPEG frames in chronological order. Each invocation processes one sequence and starts fresh. Decoding stops at the end of the stream; corrupted-stream recovery and original-resolution overlays are outside this milestone.

## EndoVis 2017

Official discovery: [EndoVis dataset index](https://opencas.dkfz.de/endovis/datasetspublications/), [2017 challenge data page](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/), [challenge paper](https://arxiv.org/abs/1902.06426). Obtain data through the organizers and comply with their access conditions and terms. No dataset is redistributed here; access/download was not completed in this run. Publicly described availability is not an assumption of unrestricted licensing.

Extract training data to the following layout (preserve exact frame names):

```text
raw/instrument_dataset_1/left_frames/frame000.png
raw/instrument_dataset_1/ground_truth/<instrument-subfolder>/frame000.png
...
```

The adapter unions nonzero pixels across per-instrument masks, preserving background as zero. It discards instrument identity and parts, uses the left camera only, fails on absent labels, and checks dimensions. Inspect converted overlays before training; naming/layout differences require adapting the importer. Do not mix incompatible annotation encodings or right-camera masks.

```bash
python -m surgtrack.cli prepare-endovis --input /path/to/raw --splits splits.example.json --out data/endovis
python -m surgtrack.cli train --data data/endovis --out runs/endovis --size 256 --epochs 30
python -m surgtrack.cli evaluate --data data/endovis --checkpoint runs/endovis/best.pt --out runs/endovis/test.json
```

The example split is an educational held-out-sequence protocol, **not the official challenge test protocol**. Verify directory names locally. Keep correlated clips from the same procedure/patient together; sequence separation alone does not establish patient independence. Freeze the split and thresholds before test evaluation. For another dataset, supply `manifest.json` with `source` and records containing relative `image`, `mask`, `sequence`, integer `frame`, and `split` (`train`, `val`, `test`). Masks must be binary PNGs, zero background/nonzero foreground, matching RGB dimensions.

## Evaluation interpretation

Every corruption retains the **clean full mask** as target: this is an amodal robustness stress test, including hidden pixels. It is not a visible-only segmentation score, a physically realistic fluid simulation, or a labeled real-fluid benchmark. The opaque-full condition removes all source image content. Good performance there must not be interpreted as seeing hidden anatomy.

Reports compare raw and temporal outputs frame by frame. Lower probability change can arise from lag and is not proof of correctness; it includes real object/camera motion. Centroid error is measured only for accepted tracks against the full-mask centroid and must be read together with coverage. Multi-instrument unions can put the centroid between instruments: there is no identity association or true landmark annotation here. Both-empty Dice/IoU are defined as one. Scores are macro frame averages; frames are correlated, so these are not independent-sample confidence intervals.

Entropy is predictive ambiguity, not calibrated epistemic uncertainty or a reliable out-of-distribution detector. Confident errors can pass the gate. Brier score includes background pixels and can hide foreground failure. `unreliable_but_accepted` counts accepted frames with raw Dice <0.5; this diagnostic is defined for this experiment, not a clinical failure threshold. Gate defaults are hand chosen and uncalibrated. No claim that EMA improves accuracy is made. Next evaluate thresholds on validation-only data, risk versus coverage, and procedure-level bootstrap intervals using enough independent procedures.

## ONNX and profiling

```bash
python -m pip install -e '.[export]'
python -m surgtrack.cli export --checkpoint demo/run/best.pt --out demo/model.onnx
```

Export uses opset 17, dynamic batch, fixed checkpoint height/width; it checks ONNX structure and compares PyTorch/ONNX Runtime logits for batches 1 and 2. Input: RGB float32 NCHW, scaled to [0,1], square-resized. Output: logits; sigmoid is external. Tracking/preprocessing are not exported. Only load trusted checkpoints. If optional dependencies are absent, the command reports installation instructions.

Profiling reports warm CPU median/p95 for model + sigmoid + tracker on a zero tensor. It excludes decoding, resizing, disk output, and display. It is not end-to-end video throughput or a real-time deployment guarantee. Memory and accelerator performance are not profiled in milestone 1.

## Research direction and interview discussion

**Task reference:** [Kumazu et al., 2021: Automated segmentation of loose connective tissue fibers](https://www.nature.com/articles/s41598-021-00557-3). It uses U-Net to segment expert-annotated fiber cues associated with dissection planes. This is closer to tissue-plane perception than instrument segmentation; it does not establish performance through opaque fluid. Access to the authors' annotated training data has not been verified. Our tiny architecture, toy labels, loss, and EMA are not a reproduction of their study.

**Milestone 2 reference:** [Zhao et al., MICCAI 2020: Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video](https://arxiv.org/abs/2007.02501). The paper learns frame-label propagation via a flow predictor and a compensator with cycle learning, evaluated on EndoVis 2017. The new milestone implements a small original adaptation of both ideas, with documented differences; it does not reproduce the published architecture or scores. See [MILESTONE2.md](MILESTONE2.md).

A manageable experiment: establish the real-data fully supervised baseline; fix a sparse-label protocol; implement the paper's motion components; compare against frame-only and EMA baselines at equal label budgets; ablate cycle learning; report degraded-view performance, rejection coverage, failure cases, and latency. Do not compare our toy scores to published surgical scores.

For a later tissue-plane milestone, obtain authorized expert annotations of visible fibers/planes, visibility/unknown regions, inter-rater disagreement, and clinically meaningful boundaries. Use procedure/patient-level splits. A plane cue on a 2D image is not a verified safe 3D dissection surface. Under opaque coverage, output unknown rather than hallucinating an anatomical reconstruction. This project makes no clinical claims or surgical recommendations.

Interview talking points:

1. Why adjacent-frame random splits leak visual information and inflate scores.
2. Why temporal smoothing can reduce flicker while worsening latency/ghosting, and how motion-compensated methods differ.
3. Why high confidence is compatible with incorrect predictions; how to calibrate and evaluate abstention.
4. Why instrument, tissue-fiber, and safe-plane labels are distinct supervision problems.
5. Why simulated haze/occlusion cannot substitute for real-fluid testing and expert annotation.
6. How label budgets, ablations, reproducible splits, ONNX parity, and measured latency support a credible engineering result.

Tests cover gradient flow, metric edge cases, sequence leakage, reproducible synthesis, degradation information removal, tracker rejection/reset/recovery, and EndoVis mask union/missing-label handling.

## Verification of the bundled run

Completed locally: 12 training epochs; held-out synthetic evaluation over 224 frame/condition pairs; 16 ordered inference frames; CPU profiling; six passing tests; ONNX structural validation and runtime parity for batches 1 and 2. The EndoVis adapter was tested on miniature fixture files only. Real EndoVis training and video-file decoding were not exercised; directory-frame inference was exercised.

The retained failure case matters: half-frame opaque-fluid coverage produced poor segmentation while every frame passed the heuristic reliability gate. This exposes confident errors rather than demonstrating robust fluid handling. Raw and temporal predictions, rejection/coverage diagnostics, and full per-frame results are preserved in `demo/evaluation.json`. Do not tune on these held-out results and continue calling them an untouched test set.
