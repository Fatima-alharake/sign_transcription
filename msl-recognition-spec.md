# PROJECT SPEC — Mongolian Sign Language → Mongolian text

You are working in this repository as a coding agent. Build the system described below.
Work incrementally: implement one milestone at a time, run it, show me the output, then stop and wait for my go-ahead before the next milestone. Do not scaffold all files at once.

---

## 1. Goal

Take a video of a person signing in Mongolian Sign Language (Монгол дохионы хэл / MSL) and output Mongolian text in Cyrillic.

Split this into two separable stages. Do not conflate them:

- **Stage A — Recognition:** video → sequence of *glosses* (uppercase Mongolian labels for individual signs, e.g. `БИ`, `НЭР`, `ЮУ`).
- **Stage B — Translation:** gloss sequence → grammatical Mongolian sentence. MSL word order and morphology are not Mongolian's; MSL has no case suffixes, drops copulas, and uses topic-comment ordering. Stage B must add agglutinative suffixes with correct vowel harmony.

Milestone 1 targets **isolated sign recognition only** (one clip = one sign). Continuous recognition comes later.

## 2. Hard constraints

- Python 3.11, PyTorch, `uv` for dependency management.
- **Landmark-based, not raw-pixel.** Extract pose/hand/face keypoints with MediaPipe and train on the keypoint sequences. Raw-video 3D CNNs need far more data than we will have. Note that MediaPipe's legacy `Holistic` solution is deprecated — check the installed MediaPipe version and use the Tasks API (`HandLandmarker`, `PoseLandmarker`, `FaceLandmarker`, or `HolisticLandmarker` if available) rather than assuming the old API works.
- Must run inference on CPU at ≥15 FPS. Export to ONNX.
- **I currently have no data at all, and I do not know MSL.** No public MSL dataset exists. Assume a small corpus long-term (100–500 clips per class is optimistic; design for 20–50), and assume zero clips right now. Everything must be developable and testable before any real signing data exists — see section 3a.
- Any placeholder gestures I record are arbitrary and meaningless. Never treat them as real signs, and never generate or invent MSL vocabulary — real glosses must come from native signers or a published MSL dictionary.
- All code typed, all modules importable and testable in isolation. Pytest for anything with non-obvious logic (normalisation, suffix generation, CTC decoding).

## 3. Repository layout

```
msl/
  data/
    capture.py         # session-based recording tool with quality gates
    prompts.py         # gloss prompt list + per-signer coverage tracking
    manifest.py        # atomic append-only manifest and signer registry
    extract.py         # video → landmark tensors (.npy), cached
    dataset.py         # torch Dataset, augmentation, train/val/test split by SIGNER
    vocab.py           # gloss vocabulary management
  models/
    encoder.py         # transformer encoder over landmark sequences
    heads.py           # classification head (isolated) + CTC head (continuous)
  train/
    train_isolated.py
    evaluate.py
  translate/
    gloss_to_text.py   # Stage B
    morphology.py      # vowel harmony + suffix attachment
  infer/
    realtime.py        # webcam → predictions, sliding window
    export_onnx.py
  configs/             # yaml, one per experiment
tests/
```

## 3a. Developing with no data

Three data sources, used in this order. Do not skip ahead.

1. **Synthetic tensors.** A `--synthetic` flag on the dataset that emits random arrays of the correct shape `(batch, 64, n_features)` with random labels and a fixed seed. This proves shapes flow through the model, loss computes, the training loop runs, checkpointing works, and ONNX export succeeds — all without a single video file. Every module must be runnable in this mode. Add a smoke test that trains 5 steps on synthetic data and asserts loss is finite.

2. **Placeholder gestures.** I will record ~20 clips each of 3 arbitrary distinct hand movements, labelled `A`, `B`, `C`. These are not signs and carry no meaning. Their only purpose is to prove MediaPipe extraction, normalisation, and the realtime loop work on real camera input. The model will overfit completely; that is expected and is not a result.

3. **Proxy sign-language dataset.** Add a loader for **AUTSL** (Turkish, 226 signs, pre-split, manageable size) and optionally **WLASL** (American, ~2,000 signs). Training on one of these gives genuine accuracy numbers and validates the architecture on real signing. Later, Mongolian data fine-tunes from these weights rather than training from scratch — low-level motion features transfer across sign languages even though vocabularies do not. Keep the encoder's classifier head separable from the backbone so the head can be swapped when the vocabulary changes.

Report results from sources 1 and 2 as pipeline checks only. Only source 3 produces numbers worth putting in a report.

## 4. Landmark extraction (`data/extract.py`)

Per frame, extract and concatenate:

- 33 pose landmarks → keep only upper body (indices 0, 11–24). Signing happens above the waist.
- 21 left-hand + 21 right-hand landmarks. These carry most of the signal.
- A **subset** of face mesh points, not all 468: lips outline, eyebrows, eye corners. Mouthing and eyebrow position are grammatically meaningful in sign languages (eyebrow raise marks polar questions), so do not discard the face entirely — but 468 points is mostly noise.

Normalisation, applied per frame, in this order:

1. Translate so the mid-shoulder point is the origin.
2. Scale by inter-shoulder distance, so the model is invariant to distance from camera.
3. Keep x, y, z; z is unreliable in MediaPipe but adds some depth cue — make it a config flag so I can ablate it.

Handle missing hands: MediaPipe returns nothing when a hand leaves frame. Emit zeros **plus a per-hand visibility mask channel** so the model can distinguish "hand absent" from "hand at origin".

Cache extracted landmarks as `.npy` keyed by a hash of the video file, so re-training does not re-run MediaPipe.

## 4a. Data collection tool (`data/capture.py`)

This is the most important deliverable in the project. Everything else is worthless without data, and the dataset outlives any model I train on it. Build it as a standalone tool that works with no model present.

### Recording loop

Prompted, session-based capture:

1. Operator enters a signer ID and session metadata once at session start.
2. Tool displays the target gloss on screen, with a reference video or image if one is available for that gloss.
3. Countdown (3 seconds, on-screen), then record a fixed window (default 3s, configurable) with a visible recording indicator.
4. Immediately replay the take.
5. Operator accepts, retakes, or skips. Retakes are frequent — make accept/retake single keypresses, not menu navigation.
6. Advance to the next prompt.

Track counts per (gloss, signer) and prompt for the least-covered glosses first, so the dataset stays balanced rather than deep in whatever was recorded first. Sessions must be resumable: if the tool crashes or the laptop dies, restarting continues from where it stopped without losing accepted takes.

### Quality gates at capture time

Run landmark extraction on each take immediately and reject or warn before the operator moves on. Catching a bad clip six months later is useless. Check:

- Both hands detected in a reasonable fraction of frames (configurable threshold).
- Signer fully in frame — no landmarks clipped at the edges.
- The sign does not start on frame 0 or end on the final frame, which means the recording window clipped it.
- Adequate brightness and no severe motion blur.
- Detected FPS matches the requested FPS.

Show the landmark overlay live in the preview window during recording. The operator should be able to see MediaPipe tracking the hands in real time — this catches lighting and background problems instantly.

### Metadata (required, not optional)

Written to `manifest.csv` on every accepted take:

`clip_id, path, gloss, gloss_id, signer_id, take_number, timestamp, duration_frames, fps, resolution, dominant_hand, consent_scope, source, quality_flags, notes`

Signer-level metadata in a separate `signers.csv`: `signer_id, dominant_hand, is_native_signer, consent_scope, notes`. Whether a signer is a native/fluent signer or a learner is a critical field — models trained on learners' signing do not generalise to native signing.

Writes must be atomic and append-only. Never rewrite the whole manifest in place; a crash mid-write would destroy the entire collection record.

### Storage

Store raw video **and** cached landmark `.npy` alongside it. Raw video must be kept: better landmark extractors will exist later, and re-extracting from video is possible while recovering video from landmarks is not. Warn me about disk usage — 3-second 1080p clips add up fast. Provide a config for capture resolution and a separate archival copy.

### Consent and ethics

- `consent_scope` is a required field with explicit values: `internal-only`, `research-publication`, `public-release`. Default to the most restrictive.
- The tool must record that consent was obtained before the first take of a session and refuse to record without it.
- Sign language video cannot be anonymised. The face carries grammatical information — eyebrow position marks questions, mouthing disambiguates signs — so blurring it destroys the data. Signers are therefore always identifiable, and consent must be informed on exactly that point.
- Include a `--export` mode that filters the dataset by consent scope, so a publishable subset can be produced without manual filtering.
- Add a `CONSENT.md` in the repo with a plain-language consent form template in Mongolian and English for me to adapt.

### Non-negotiable

The tool must never generate, invent, or suggest sign vocabulary. Glosses come from a prompt list I supply, sourced from native signers or a published MSL dictionary.

## 5. Dataset and augmentation (`data/dataset.py`)

- Resample every clip to a fixed length (default 64 frames) by linear interpolation over the time axis.
- **Split by signer, not by clip.** If the same person appears in train and test, the reported accuracy is meaningless. Make signer ID a required field in the manifest.
- Augmentations: temporal resampling (0.8×–1.2× speed), small rotations about the camera axis, per-joint Gaussian noise, random frame dropout, random affine scaling.
- **Do not mirror horizontally by default.** Sign languages have dominant-hand asymmetry, and mirroring turns a right-handed signer into a left-handed one, which may or may not be the same sign. Provide it as a flag with a comment explaining the risk.

Manifest format: a single `manifest.csv` with columns `path, gloss, signer_id, split, source, notes`.

## 6. Model (`models/encoder.py`)

Baseline: a small Transformer encoder.

- Linear projection of the per-frame feature vector to `d_model=128`.
- Learned positional embeddings (sequences are short and fixed-length).
- 4 layers, 4 heads, FFN 256, dropout 0.3 — this is a small-data regime, regularise hard.
- Mean-pool over time → linear classifier.

Provide a `--arch` flag with a BiLSTM alternative as a second baseline, and stub an ST-GCN variant (spatio-temporal graph conv over the hand skeleton) for later. Report all three in the eval table.

Training: AdamW, cosine schedule with warmup, label smoothing 0.1, early stopping on val top-1. Log to TensorBoard.

## 7. Evaluation (`train/evaluate.py`)

- Top-1 and top-5 accuracy, per-class F1.
- **Confusion matrix as a first-class output.** Signs that differ only in handshape or only in movement will collapse into each other — I need to see which pairs.
- Held-out signer accuracy reported separately and prominently.

## 8. Stage B — gloss to Mongolian (`translate/`)

Start rule-based, not neural. There will not be enough parallel gloss↔text data to train a seq2seq model initially.

`morphology.py` must implement:
- Vowel harmony classification of a stem (masculine а/о/у, feminine э/ө/ү, neutral и) and selection of the harmonising suffix allomorph.
- Case suffix attachment: nominative, genitive (-ын/-ийн/-ы/-н), accusative (-ыг/-ийг/-г), dative-locative (-д/-т), ablative (-аас/-ээс/-оос/-өөс), instrumental (-аар/-ээр/-оор/-өөр), comitative (-тай/-тэй/-той).
- Plural, and verb tense endings.

Write this as a table-driven module with a comprehensive pytest suite, because the rules have many exceptions.

`gloss_to_text.py` maps gloss sequences to sentence templates, reorders to Mongolian SOV, inserts the copula and required particles, then calls `morphology.py`. Leave a clearly marked interface where an LLM or a trained seq2seq model can later replace the rules.

## 9. What I want you to flag rather than guess

Ask me before deciding:
- Whether output should also be produced in traditional Mongolian script (I have an existing Cyrillic↔traditional converter that could be plugged in here).
- The initial gloss vocabulary size and which signs to cover first.
- Whether to target realtime webcam use or offline file processing first.

## 10. Milestone order

0. **Synthetic-only skeleton.** `dataset.py` in `--synthetic` mode, the transformer encoder, the training loop, checkpointing, ONNX export. No video, no MediaPipe, no camera. Ends with a passing smoke test that trains 5 steps and exports. This milestone requires nothing from me.
1. `extract.py` + manifest. Verify against any video file. Visualise extracted landmarks overlaid on the video to prove extraction is correct — this is the only way to catch a silently broken MediaPipe setup.
1b. **`capture.py`, the full data collection tool** (section 4a): prompts, countdown, replay, accept/retake, quality gates, live landmark overlay, atomic manifest writes, resumable sessions, consent recording. Test it by recording a throwaway session of myself. This milestone is the one that must be solid — it is what I will hand to other people.
2. `dataset.py` real-data path + augmentation + unit tests on normalisation.
3. Train on my 3 placeholder gestures. Expect ~100% accuracy and treat it as a plumbing check, not a result.
4. AUTSL loader; train the transformer, BiLSTM, and TCN baselines; produce the comparison table, confusion matrix, and held-out-signer numbers.
5. `realtime.py` sliding-window inference, running off the AUTSL-trained checkpoint.
6. `morphology.py` with tests.
7. `gloss_to_text.py`.
8. CTC head for continuous signing.

Start with milestone 0. Print the plan for it, then implement.
