# MSL — Mongolian Sign Language → Mongolian text

Two separable stages:

- **Stage A — Recognition:** video → gloss sequence (uppercase Mongolian labels).
- **Stage B — Translation:** gloss sequence → grammatical Mongolian, with case
  suffixes and vowel harmony.

See [msl-recognition-spec.md](msl-recognition-spec.md) for the full spec.

## Status: Milestone 0 — synthetic-only skeleton

What exists: the dataset in `--synthetic` mode, three sequence backbones, the
training loop, checkpointing, and ONNX export. **No video, no MediaPipe, no
camera.** Everything runs on random tensors of the correct shape.

Nothing here produces a meaningful accuracy number — synthetic labels are random
by construction, and per spec §3a only a proxy corpus (AUTSL, Milestone 4)
yields numbers worth reporting.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned in the project
venv; your system Python is untouched.

```bash
uv sync --group dev
```

PyTorch comes from the CUDA 13.0 wheel index, not the default CPU wheel — see
the `[[tool.uv.index]]` block in `pyproject.toml`. The smoke test asserts
`torch.cuda.is_available()` and fails loudly rather than falling back to CPU.

## Running

```bash
uv run pytest -v                                    # full suite
uv run python -m msl.train.train_isolated --synthetic --arch transformer
uv run python -m msl.infer.export_onnx --benchmark
```

`--device` accepts `auto` (default; CUDA if present), `cuda`, or `cpu`.
`--arch` accepts `transformer`, `bilstm`, or `tcn`. Training is step-based, not
epoch-based: "epoch" is not a useful unit at 20 clips per class.

## Layout

```
msl/
  config.py            typed config, loaded from configs/*.yaml
  runtime.py           device resolution, seeding, RSS reporting
  data/
    landmarks.py       which keypoints we keep; no MediaPipe import
    extract.py         video -> normalised landmark tensors, cached
    dataset.py         synthetic source; real path stubbed to Milestone 2
    vocab.py           gloss <-> id
  models/
    encoder.py         transformer / BiLSTM / TCN backbones; ST-GCN stub
    heads.py           classification head; CTC head stub
    build.py           assembly, atomic checkpointing, backbone transfer
  train/train_isolated.py
  infer/export_onnx.py
tests/
```

Backbones emit a pooled embedding and know nothing about class count. The
classifier is a separate module with its own state-dict subtree, so a backbone
pretrained on AUTSL can be reloaded under a Mongolian vocabulary of a different
size (`load_backbone_only`).

## Landmark extraction

`extract.py` splits into a pure NumPy core (normalisation, packing, caching) and
a MediaPipe layer that is imported lazily — so the test suite runs with no model
bundle and no video present, and `import msl.data.extract` stays cheap.

**MediaPipe 1.0 removed `mediapipe.solutions` entirely.** The legacy `Holistic`
API is gone, not merely deprecated, so any tutorial code built on it fails
outright. We use the Tasks API `HolisticLandmarker`, which additionally assigns
left/right hand from the body model rather than leaving us to disambiguate
handedness ourselves.

One-off download of the model bundle (13.7 MB, gitignored):

```bash
mkdir -p assets
curl -L -o assets/holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

Set `data.data_root` (and optionally `data.cache_dir`, `data.assets_dir`) in a
config to put video and the `.npy` cache on an external drive — AUTSL plus its
extracted cache runs to tens of GB.

Cache keys hash the video's *contents*, not its path, and fold in the settings
that change output shape. A re-record under the same filename therefore misses
the cache, and a `use_z=False` run is never served a `use_z=True` tensor.

### Data source

Recognition is validated against **AUTSL** (Turkish, 226 signs, 43 signers).
It ships an official signer-independent split, which is the reason it was chosen
over WLASL's larger vocabulary — the classifier head is discarded on transfer, so
vocabulary breadth buys nothing, while clips-per-class and a real signer split
buy everything. Record the exact provenance of the copy used here, and verify any
mirror preserves the official split and `signer_id` column before training on it.

## What does not go in git

Raw video and cached `.npy` landmarks are excluded by `.gitignore` and belong on
an external drive with an offsite copy. A repo that has once held large binaries
stays bloated permanently, since the blobs survive in history.

Versioned: code, configs, and the small text files whose history is the point —
`manifest.csv`, `signers.csv`, and the gloss prompt list.

## Vocabulary

This codebase never generates, invents, or suggests sign vocabulary. Glosses
come from an operator-supplied list sourced from native signers or a published
MSL dictionary. Synthetic-mode labels are named `SYNTH_00…` precisely so they
cannot be mistaken for signs.

## Next

Milestone 1: `extract.py` + manifest, verified by rendering the landmark overlay
on a real video — the only way to catch a silently broken MediaPipe setup.
