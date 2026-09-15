# MSL — Mongolian Sign Language → Mongolian text

Two separable stages:

- **Stage A — Recognition:** video → gloss sequence (uppercase Mongolian labels).
- **Stage B — Translation:** gloss sequence → grammatical Mongolian, with case
  suffixes and vowel harmony.

See [msl-recognition-spec.md](msl-recognition-spec.md) for the full spec.

## Status: Milestone 1 — extraction and overlay written, not yet verified on video

What exists: the Milestone 0 skeleton (dataset in `--synthetic` mode, three
sequence backbones, training loop, checkpointing, ONNX export), plus
`extract.py` — a pure-NumPy normalisation core with a lazily imported MediaPipe
layer — and `overlay.py`, the renderer for verifying extraction by eye. The
overlay has not yet been run on real footage, so extraction remains unverified
and the face subset stays marked PROVISIONAL.

Not yet started: the manifest (`manifest.py`), `capture.py`, and the real-data
dataset path.

Nothing here produces a meaningful accuracy number — synthetic labels are random
by construction, and per spec §3a only a proxy corpus (WLASL, Milestone 4)
yields numbers worth reporting.

## Verifying extraction

MediaPipe fails silently. A wrong face index still returns a float, a left/right
hand swap still returns two hands, and a classifier trained on either still
converges. Two independent checks guard against that.

**1. Topology guard tests.** `landmarks.py` hand-copies MediaPipe's landmark
indices so it stays importable with no MediaPipe installed.
`tests/test_topology.py` pins that copy to the installed package's own
connection tables (`FaceLandmarksConnections`, `HandLandmarksConnections`,
`PoseLandmarksConnections`) and is skipped when MediaPipe is absent. It has
already earned its place: `LEFT_EYEBROW_INDICES` and `RIGHT_EYEBROW_INDICES`
were inverted. Both held genuine eyebrow points in correct chain order, so
nothing downstream ever looked wrong — only the names were swapped. MediaPipe
names landmarks subject-relatively (`PoseLandmark.LEFT_SHOULDER` is the signer's
own left), and on an unmirrored frame that side appears on the viewer's right,
which is how the inversion got in.

**2. The overlay.** Renders the kept landmarks over real frames:

```bash
uv run python -m msl.data.overlay --video clip.mp4 --out outputs/overlay
```

Writes `overlay.mp4` plus magnified, index-labelled face crops — 34 face points
on a full-body 1080p frame are a few pixels apart and confirm nothing at full
size. It also reports per-block detection rates and a numeric check on
HolisticLandmarker's left/right hand assignment.

The overlay draws **only** the points fed to the model, never the full 478-point
mesh: a full mesh renders as a plausible face even when our subset addresses the
wrong vertices.

**What the overlay cannot check: aspect correction.** It maps normalised
coordinates back to pixels (`x*width`, `y*height`), the exact inverse of what
MediaPipe did, so the drawing lands on the signer whether or not the feature
path's aspect correction (spec §4.0) is right — or applied at all. That
correction is guarded by tests asserting the same physical pose at 1:1 and 16:9
yields identical features. An overlay that lands perfectly is not evidence the
pipeline is correct.

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
pretrained on WLASL can be reloaded under a Mongolian vocabulary of a different
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
config to put video and the `.npy` cache on an external drive. Measured
footprint is smaller than a pixel-based corpus would need, because we keep
keypoints rather than frames: WLASL100 is ~1.8 GB of video and ~84 MB of `.npy`
cache; the full WLASL2000 is ~14 GB and ~850 MB.

Cache keys hash the video's *contents*, not its path, and fold in the settings
that change output shape. A re-record under the same filename therefore misses
the cache, and a `use_z=False` run is never served a `use_z=True` tensor.

### Data source — WLASL

The backbone is pretrained on **WLASL** (Word-Level American Sign Language,
~2,000 glosses).

**WLASL is American Sign Language.** It pretrains the backbone only. Low-level
motion features transfer across sign languages; vocabulary does not. No WLASL
gloss is ever an MSL gloss, and every number derived from WLASL is an **ASL**
number that must be labelled as such everywhere it appears — this README,
figures, slides, talks and commit messages alike.

**Licence: C-UDA** (Computational Use of Data Agreement) — academic and
computational use only, no commercial use.

#### This reverses an earlier choice, for access rather than merit

**AUTSL** (Turkish Sign Language, 226 signs, 43 signers) was chosen first, for
one specific reason: it ships an **official signer-independent split**. Spec §5
is unambiguous that this is what matters — if the same person appears in train
and test, the reported accuracy is meaningless. AUTSL also carries far more
clips per class (~170 vs WLASL's ~10), which is what a transferable motion
backbone actually needs, given the classifier head is discarded on transfer and
vocabulary breadth buys nothing.

AUTSL was not reachable. Its files are encrypted, and the decryption keys are
issued through a CodaLab instance that has been read-only since January 2023,
so the documented access route may no longer work at all. WLASL is used instead
because it can actually be obtained without a key.

That substitution has a cost, and the cost is precisely the property AUTSL was
picked for.

#### WLASL's shipped split is not signer-independent

Measured from `WLASL_v0.3.json` (2,000 glosses, 21,083 instances, 119 signers):

```
train AND test  : 104 signers shared
test signers NOT in train: 1  (signer 120)
test INSTANCES whose signer also appears in train: 2877/2878 = 100.0%
```

Effectively every test clip is performed by someone the model trained on, and
the top 10 signers are 57.7% of all instances. Taken at face value the
published protocol produces exactly the number spec §5 calls meaningless.

**We therefore build our own signer-independent split** — whole signers held
out, greedily smallest-first, constrained so no gloss loses all training data,
deterministic and seeded, written to a committed CSV. On the official WLASL100
subset this holds out ~20% of instances while leaving all 100 glosses testable.

#### Reporting rule

Two numbers are reported, **both labelled ASL**:

- the **signer-independent** number, which is the headline;
- the **official-split** number, which appears *only* alongside it, marked
  signer-dependent and optimistic.

The official number never appears alone — not in this README, not in a plot
title, not in a commit message.

#### Missing source video, and how it is obtained

WLASL ships URLs, not video, and those URLs rot. Measured across a 66-URL
sample of WLASL100, three hosts are unusable — **27.5% of the subset**:

| host | clips | status |
|---|---|---|
| `www.signingsavvy.com` | 225 | `403` — refusing automated access |
| `www.handspeak.com` | 174 | `404` — gone |
| `www.aslpro.com` | 162 | `404` — gone (Flash `.swf`) |

**We do not work around the 403.** A 403 is the site explicitly refusing
automated access. Defeating bot protection is not something this project does,
regardless of whether the data is licensed for our use — the licence governs
what we may do with the data, not how we may take it.

The supported route instead: the WLASL maintainers **provide pre-processed
video for the full dataset on request**, gated on agreeing to terms of use.
`start_kit/find_missing.py` writes a `missing.txt` of absent video IDs to
request against, and the repository README directs requests to a Google Form,
quoting a roughly 7-day turnaround. That is the intended path for all three
hosts above, and it is a request made by a person, not by this code.

`aslsearch.com` is a separate hazard: it answers `200` with zero content
length, so a naive downloader writes an error page to `{video_id}.mp4` and a
resume-by-existence check then trusts it forever. The downloader here treats
zero-length as failure and validates that every file actually decodes.

#### What actually downloaded

`data/wlasl/download_log_100.csv` records one row per attempted clip:
`video_id, gloss, gloss_id, split, signer_id, source, host, status, http_code,
bytes, frames, width, height, fps, url, note`.

> **`frames` is capped at `MAX_PROBE_FRAMES = 900`.** Validation stops decoding
> there, so for the 128 clips that reach the cap the column is a **lower bound,
> not a frame count** — their true lengths run to 12,232. Those 128 are exactly
> the clips carrying a trim range, so nothing downstream depends on the capped
> value, but do not compute durations from this column. Read the true count from
> the file.

Measured from that log, not from metadata:

```
attempted        2038
usable           1127   (55.3%)
glosses          100/100 surviving, min 6 / median 11 / max 17 clips
signers            69   (from 97 in metadata), top-10 = 61.4%, signer 5 alone = 21.5%
disk             1.2 GB video + 14 MB metadata

http_error    563   signingsavvy 225, handspeak 174, aslpro 162, aslsignbank 2
ytdlp_failed  234   youtube.com 138, youtu.be 96 (all "Private video")
undecodable   114   aslsearch, all of them
```

The 96 `youtu.be` failures are a single contributor — signer 109, source
`asllex` — whose channel went private, not scattered rot.

#### Estimate post-mortem: the risk named in prose but not in the arithmetic

The pre-download projection said ~1,272 usable (62%). The actual figure is
1,127 — **short by 144 clips**. The obvious suspect was the YouTube survival
assumption. It was not the cause:

| source | predicted | actual | delta |
|---|---|---|---|
| YouTube (both hosts) | 479 | 451 | **−28** |
| `aslsearch.com` | 114 | 0 | **−114** |
| everything else | 679 | 676 | −2 |

YouTube came in at 65.8% against an assumed 70% — essentially right.
**79% of the error was aslsearch**, a host whose reachability probe returned
`200` for every sample and which was therefore scored as fully surviving. Its
bodies are ~114 bytes of non-video. The hazard had *already been identified in
prose* in the same report that projected it at 100% survival.

**The general lesson, because this will recur:** a risk written down in prose
is not a risk carried into the arithmetic. Any caveat that qualifies a number
has to change the number, or it is decoration. Where a caveat cannot be
quantified, the estimate should carry a range rather than a point value.

That the 114 clips are recorded as `undecodable` rather than sitting on disk as
plausible `.mp4` files is entirely down to the decode check — a zero-length
rule alone would have passed all of them, since 114 bytes is not zero.

#### 911 clips are missing, and the corpus is 55.3% complete

911 of 2,038 clips did not arrive. The maintainers' pre-processed-video request
route would likely recover most of them, including signer 109's 96 private
`asllex` clips, and would restore signer diversity — which link rot damaged
more than it damaged raw counts (signers per gloss fell from 10/15 min/median
to 5/9).

**Any result produced before that request lands is computed on a 55.3%-complete
corpus and must say so**, alongside the ASL label.

#### The signer-independent split, as built

```
train 789 clips / 18 signers      test 338 clips / 51 signers
holdout 30.0%                     signer overlap: none
glosses with training clips: 100/100
glosses with test clips:     100/100
```

The ~20% target was not reachable under the two-sided constraint; 30.0% is what
full coverage costs, and the achieved fraction is reported rather than the
constraint relaxed or glosses dropped.

One asymmetry to keep in view: the split trains on 18 signers and tests on 51.
Choosing test coverage greedily by "most glosses per signer" instead satisfies
the constraint with just **2** held-out signers at 28.7% — a test set measuring
whether the model recognises two particular people. Preferring many small
signers costs 1.3 points of holdout and buys 51. The cost is that training sees
fewer distinct signers, which is a consequence of one signer holding 21.5% of
the corpus, and is worth revisiting once the missing clips land.

#### Trimming, bbox, and the val split

**`frame_start`/`frame_end` must be honoured.** 132 of 1127 clips (11.7%) are
segments cut from long compilation videos (`asl5200`, `northtexas`), whose true
length runs to a median of 7,221 frames and a maximum of 12,232. The labelled
sign occupies about 1% of the file. Trimming discards 844,091 of those clips'
853,865 frames — 91.1% of every frame in the corpus. Post-trim, all 1127 clips
run 19–212 frames, median 72.

**Do not crop to `bbox`.** Tested rather than assumed, on the six clips with the
smallest bbox coverage, where cropping should help most:

```
                     pose / Lhand / Rhand / face  (detection rate)
full frame            0.76 / 0.11 / 0.30 / 0.76
cropped to bbox       0.40 / 0.03 / 0.06 / 0.02
cropped + upscaled    0.10 / 0.01 / 0.00 / 0.02
```

Cropping is not redundant, it is harmful, and upscaling the crop is worse still,
so it is not a resolution effect. The cause: **the bbox does not correspond to
the video we downloaded.** Against the region where MediaPipe actually finds the
person, the metadata bbox scores IoU ≈ 0.05–0.12, and the mismatch is a
translation rather than a scale error — the boxes are close to the right *size*
and in the wrong *place*. **A falsifiable prediction, not a conclusion.** The evidence — a translation
error at near-correct scale (IoU 0.05–0.12, boxes the right size in the wrong
place) — points to these coordinates describing the maintainers' *pre-processed*
videos rather than the raw source URLs. That is a claim that can be checked: **if
the request route lands, the bbox should agree with the pre-processed video.**
Check it then. Do not assume it will, and do not assume it will not; if it still
disagrees, the explanation is wrong and the bbox is simply unreliable.

Either way, MediaPipe localises the person itself and does it better than a stale
box, so nothing currently depends on resolving this.

`bbox` is `[x1, y1, x2, y2]`, and 9 of 1127 are out of bounds for our file
(all `aslu`, where the bbox assumes 1280x720 and we received 720x480), so it is
useful only as a weak sanity signal, never as a crop.

**Val is signer-dependent, and is never a result.** Val is a clip-level split
inside the training signers, used solely as a stopping criterion. A
signer-disjoint val is possible but costs 15 of the 18 training signers — 412
clips, 52.2% of training data — leaving training with 377 clips from 3 signers.
That trades the reported number's honesty for nothing, because test already
provides it. **Test is the honest, signer-disjoint number; val accuracy is
inflated and must never appear in a report, a plot title, or this README.**

#### Known limitations

- **`signer_id` reliability is unverified.** Our split is only as good as those
  labels. WLASL aggregates 8 separate ASL dictionary sites, so one person could
  plausibly carry different ids across sources, or one id could conflate
  several people. Confirming it means watching video, which we have not done.
  If the labels are wrong, a "signer-independent" split is signer-independent
  in name only.
- **WLASL is citation-form dictionary video, not natural signing.** Entries are
  isolated, deliberate and evenly paced. This is the same limitation spec §4b
  records for the Ember dictionary. It is acceptable for backbone pretraining,
  but the backbone will be tuned to citation form, and continuous signing has
  coarticulation and compressed transitions that it will not have seen.

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

Finish Milestone 1 by running the overlay on a real clip. The topology guard
tests prove our indices are what MediaPipe *declares* lips and eyebrows to be;
they cannot prove the bundled `facemesh2_lite` emits its 478 points in that
order. Only the lip ring landing on a real mouth closes that gap, so the face
subset stays marked PROVISIONAL in `landmarks.py` until then.

Then: `manifest.py`, and Milestone 1b — `capture.py`, the data collection tool,
which reuses `overlay.py` for its live preview.
