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

0. **Aspect correction, before anything else.** MediaPipe normalises x by frame width and y by frame height, so on any non-square frame the two axes carry different physical units — a circular hand motion arrives as an ellipse. Convert x into units of image height (multiply by width/height) before centring or scaling. Without this, the inter-shoulder divisor (almost entirely horizontal) is wrong relative to hand motion (substantially vertical) by the aspect ratio. **A pixel-space overlay cannot detect this failure** — it renders correctly either way — so it must be guarded by a test asserting that the same physical pose at 1:1 and 16:9 yields identical features.
1. Translate so the mid-shoulder point is the origin.
2. Scale by inter-shoulder distance, so the model is invariant to distance from camera. Compute this divisor from x and y only — including the unreliable z channel (§4.3) would corrupt x and y as well, not merely z.
3. Keep x, y, z; z is unreliable in MediaPipe but adds some depth cue — make it a config flag so I can ablate it.

**§4.3 UPDATED FROM MEASUREMENT. `use_z` now defaults to `false`.** The ablation
was run on WLASL100 (American Sign Language, 55.3%-complete corpus, 3 seeds,
fixed 2400-step budget, signer-disjoint test): dropping z gives **+7.10pp top-1
on the transformer and +11.34pp on the TCN**, with non-overlapping seed ranges
on both. It is the largest effect measured in the project.

**Why it is harmful: SCALE WAS TESTED AND IS NOT THE MAIN CAUSE.** An earlier
revision of this section asserted that z was harmful for being unscaled rather
than uninformative. That was a plausible mechanism recorded as a fact before it
was tested, and the direct test refutes it. Rescaling z onto x and y's standard
deviation (`z_scale = 0.6953`, fitted on train only) recovers **only 19.1% of
the gain from dropping z**, and scaled-z does not even separate from raw z
(0.5000 +/- 0.0165 against 0.4783 +/- 0.0085, overlapping ranges), while
dropping z remains decisively better than both (0.5917 +/- 0.0263, disjoint
ranges against each). **Most of the harm is z's content, not its loudness.**

The scale observations below are real measurements and remain true; what was
wrong was the causal weight placed on them:

- z is 91 of 275 features — **33.1% of the per-frame vector**.
- Its mean absolute magnitude in the cached tensors is **1.5× that of x and y**.
- That is a direct consequence of §4.2, which — correctly — computes the
  inter-shoulder divisor from x and y only, since letting the unreliable z into
  the divisor would corrupt every coordinate rather than just the z channel.
  The divisor is therefore not z's natural scale, and z is left larger than the
  signal it accompanies.

A third of the input is therefore both loud *and* uninformative, and only the
second part turns out to matter much. **Discard z; do not try to rescue it by
rescaling.** The scaling path is implemented (`FeatureConfig.z_scale`,
`landmarks.scale_z`) and measured, so it does not need re-testing.

The residual possibility this does not rule out: MediaPipe's z may be usable for
*pose* while being noise for the 42 hand landmarks, which are inferred from a 2D
image with weak depth priors. A per-block z ablation would test that. It has not
been run, and nothing here should be read as evidence about it either way.

Storage is a separate decision from consumption. **The landmark cache always
stores z** (`extract.CACHE_USE_Z`) and `use_z` is applied when loading: a cached
tensor with z can always be narrowed, while one without z can only be restored
by re-running MediaPipe over the whole corpus. `cache_key` therefore ignores
`use_z`, so changing the flag never invalidates the cache.

**Aspect ratio must be recorded per clip in the manifest, and applied from the live frame dimensions at inference.** Training on 16:9 clips and inferring from a 4:3 webcam without re-deriving the correction reintroduces exactly this distortion at the point where it is hardest to notice. `realtime.py` must read the actual camera resolution rather than assuming the training aspect.

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

**Video and `.npy` files must never enter git.** Add `.gitignore` rules for the data directories from the start. `manifest.csv`, `signers.csv`, and the gloss prompt list *are* version-controlled — they are small, text, and the part whose history matters. Recorded media needs a separate backup strategy (external drive plus one offsite copy); git is not that strategy, and a repo that has ever contained large binaries stays bloated forever.

### Consent and ethics

- `consent_scope` is a required field with explicit values: `internal-only`, `research-publication`, `public-release`. Default to the most restrictive.
- The tool must record that consent was obtained before the first take of a session and refuse to record without it.
- Sign language video cannot be anonymised. The face carries grammatical information — eyebrow position marks questions, mouthing disambiguates signs — so blurring it destroys the data. Signers are therefore always identifiable, and consent must be informed on exactly that point.
- Include a `--export` mode that filters the dataset by consent scope, so a publishable subset can be produced without manual filtering.
- Add a `CONSENT.md` in the repo with a plain-language consent form template in Mongolian and English for me to adapt.

### Non-negotiable

The tool must never generate, invent, or suggest sign vocabulary. Glosses come from a prompt list I supply, sourced from native signers or a published MSL dictionary.

## 4b. The Ember dictionary as gloss source and reference set

I volunteer for Ember, which operates a verified MSL dictionary at `sign.ember.mn` — around 1,236 signs, each with a video and structured metadata (initial letter, topic, difficulty level, age band, one-handed vs two-handed).

**Access via an official data export from Ember, not by scraping the site.** Do not write a crawler. If an internal API or database export is the route, I will obtain it; your job is to consume whatever format I supply.

### Primary use: prompt list and reference videos

This replaces the placeholder gloss list in `prompts.py`. `capture.py` displays the dictionary video for the target gloss before each take, so signers reproduce the canonical form rather than their own variant. Build `prompts.py` to load an Ember export with fields at least: `gloss, gloss_id, topic, level, age_band, handedness, reference_video_path`.

Support filtering the prompt list by topic and level, so a collection session can cover one coherent semantic domain rather than an alphabetical slice.

### Secondary use: free quality signal

The `handedness` field is a validation gate, not just metadata. If the dictionary marks a sign one-handed but extraction shows sustained two-hand activity in a recorded take, either the take is wrong or the extractor is mislabelling hands. Surface this as a `quality_flags` value at capture time. This catches the left/right identity swap that is otherwise silent.

### What the dictionary is NOT

- **Not a training set.** It is a single signer with roughly one clip per sign. Training a classifier on it would learn that one person, and there is no way to measure signer generalisation from it — the signer-independent split in §5 becomes impossible. Never report accuracy from dictionary-only training.
- **Not natural signing.** Dictionary entries are citation form: isolated, deliberate, evenly paced. Continuous signing has coarticulation and compressed transitions. A model tuned on citation form degrades on real signing, which matters from Milestone 8 onward.
- **Not a substitute for recruiting signers.** It defines *what* to collect. Collection still requires multiple signers.

### Later: one-shot recognition (do not build before Milestone 4)

Because there is one reference clip per sign, a metric-learning approach fits well: train an embedding on AUTSL, then classify by nearest neighbour against dictionary embeddings rather than a fixed classifier head. Adding vocabulary then costs one video instead of a retrain. Note this as a design option in the README; do not implement it early.

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

Provide a `--arch` flag with two further baselines: a **BiLSTM**, and a **TCN** (dilated 1D convolutions over the time axis). The TCN is the one to beat — its locality prior makes it strong in exactly the small-data regime this project starts in, and it often outperforms the transformer below a few thousand clips. Report all three in the eval table.

Separately, stub an **ST-GCN** variant (spatio-temporal graph convolution over the hand and body skeleton, using true joint adjacency rather than array-index adjacency) as a fourth architecture for later. Stub only — do not implement it before Milestone 4.

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

## 9a. Known issues carried forward

- **ONNX export uses the deprecated TorchScript path** (`dynamo=False`). It works on torch 2.13 and output is numerically verified against PyTorch, but it is the legacy exporter. Related: a TracerWarning in `encoder.py` means the sequence-length guard is frozen into the traced graph as a constant, so the exported model will not raise on a wrong sequence length. Harmless while the time axis is fixed at 64 frames. **This must be revisited at Milestone 5**, when `realtime.py` feeds variable-length sliding windows — either move to the dynamo exporter with a dynamic time axis, or assert the window length before every call.

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
