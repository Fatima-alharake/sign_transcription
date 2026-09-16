# MSL — Mongolian Sign Language → Mongolian text

Landmark-based sign language recognition, built toward Mongolian Sign Language
(MSL) in two separable stages:

- **Stage A — Recognition:** video → gloss sequence (uppercase Mongolian labels).
- **Stage B — Translation:** gloss sequence → grammatical Mongolian, with case
  suffixes and vowel harmony.

**There is no MSL data yet, and no MSL result in this repository.** No public MSL
corpus exists. The work so far builds the pipeline and pretrains a backbone on a
proxy corpus, so that Mongolian data — once collected — fine-tunes from trained
weights rather than from scratch. Full spec:
[msl-recognition-spec.md](msl-recognition-spec.md).

## Headline result

**Isolated sign recognition on WLASL100: 59.17% ± 2.63 top-1, 79.88% ± 1.35
top-5, macro F1 0.546 ± 0.035.**

Every one of those numbers needs its caveats attached, so here they are in the
same breath:

- **This is American Sign Language, not Mongolian.** WLASL is ASL. It is used
  for backbone pretraining only; the vocabulary does not transfer.
- **The corpus is 55.3% complete** — 1,127 of 2,038 clips. 911 are unrecoverable
  from dead hosts and private videos. Every number is computed on the half that
  survived.
- **Signer-independent test split, built by us.** WLASL's shipped split is not
  signer-independent (2,877 of 2,878 test instances share a signer with train).
  338 test clips, 100 classes, chance top-1 1.0%. Test signers appear nowhere in
  training.
- **3 seeds, reported as mean ± sample sd with ranges.** At n=3 an sd is a rough
  spread indicator, not a confidence interval. Differences are described as
  overlapping or disjoint ranges, never as statistical significance — with 3 v 3
  seeds the smallest achievable two-sided rank p is 0.1, so nothing here could
  reach p < 0.05 by construction.
- **Untuned.** No learning-rate sweep, no architecture search. Spec defaults with
  a fixed step budget.
- **Citation-form dictionary video, not natural signing**, and `signer_id`
  reliability is unverified across the 8 dictionary sites WLASL aggregates — so
  the split is only as good as those labels.

Best configuration: TCN backbone, spec §5 augmentation, MediaPipe's z channel
discarded. Full table and every variant measured: [Results](#results-backbone-pretraining-on-wlasl100-asl-not-msl).

## What this project is, in one minute

Landmark-based rather than raw-pixel: MediaPipe Holistic extracts pose, hand and
face keypoints, and the model trains on keypoint sequences. Raw-video 3D CNNs
need far more data than a from-scratch sign corpus will ever have.

- `msl/data/` — extraction with a pure-NumPy normalisation core, the WLASL
  loader, augmentation, the signer-independent split builder.
- `msl/models/` — transformer, BiLSTM and TCN backbones behind one
  `(B, T, F) -> (B, d_model)` contract, with the classifier head kept separable
  so a backbone can be reloaded under a different vocabulary.
- `msl/train/` — step-based training loop, evaluation (top-1/5, per-class F1,
  confusion matrix), and `sweep.py`, which runs a variant across seeds and
  refuses to run off the reference protocol.
- 210 tests. Runs on CPU or CUDA; ONNX export works.

## Methodology decisions worth your attention

These are the four things a reader should look at before the numbers, because
each one changes what the numbers mean.

### 1. The signer-independent split was rebuilt, because the shipped one is not

WLASL ships a split in which **2,877 of 2,878 test instances have a signer who
also appears in training**. Accuracy measured on it answers "can the model
re-recognise people it has seen", which is not the question. We build our own:
whole signers held out, greedily smallest-first, seeded, committed as CSV.

The constraint is **two-sided** — every gloss keeps both training clips *and* at
least one held-out test clip. Enforcing only the training side is a bug that
actually happened here: it left 12 of 100 glosses with no test clip while still
reporting "100 classes". Achieved: 30.0% holdout, 51 held-out signers, 100/100
glosses on both sides. Details:
[the split, as built](#the-signer-independent-split-as-built).

**Val is signer-dependent by design and is never reported.** It is a clip-level
split inside the training signers, used only as a stopping criterion. Its
accuracy is inflated and means nothing about generalisation.

### 2. An earlier set of numbers was retracted, not quietly dropped

The first protocol used early stopping on val and evaluated the best-val
checkpoint. It produced 43.00% ± 1.20 top-1 for the augmented transformer, which
was the project's reference number for a while. **It is withdrawn and is not
comparable to anything in the results table.** Two confounds:

1. **The stopping step, not the change under test, drove the score.** Val is 116
   clips with a noisy plateau, so 600 steps of patience stopped runs at
   near-arbitrary points. The stopping-step range widened exactly as a change
   perturbed val — baseline 100 steps, augmentation 650, separator trim 2,200 —
   and stopping step against test top-1 gave r = 0.949 within one variant and
   r = 0.68 across all nine early-stopped runs.
2. **Runs ended at wildly different learning rates.** With the cosine defined
   over 4,000 steps, a run stopping at 1,400 ended at lr ×0.773 while one at
   3,600 ended at ×0.027 — a 28-fold range, leaving short runs unannealed.

The replacement protocol is a **fixed 2,400-step budget, no early stopping,
cosine completing at the budget, final checkpoint evaluated** — the last part
matters, because switching off early stopping while still loading the best-val
checkpoint would have left val selecting the model anyway. The budget was chosen
from the curves, not rounded to: smoothed val reaches 98% of its plateau by step
1,600–1,700 and peaks at 1,600/1,750, so 2,400 is 1.37× the latest peak, and
beyond it training loss keeps falling while val stays flat.

Evidence the diagnosis was right: the separator-trim variant's top-1 sd fell
from 0.0577 under early stopping to 0.0030 at a fixed budget — 19-fold. The
variance was the protocol, not the data. `sweep.py` now **refuses to run** a
comparison off the reference protocol rather than warning about it.

### 3. A mechanism was recorded in the spec, then measured and corrected

Dropping MediaPipe's z channel is the largest single effect in the project:
**+7.10pp top-1 on the transformer, +11.34pp on the TCN**, disjoint ranges on
both. That much is solid.

The *explanation* was not. z is 33.1% of the feature vector at 1.5× the
magnitude of x and y — because normalisation divides every channel by the
inter-shoulder distance, which spec §4.2 correctly computes from x and y only.
"z is harmful because it is loud" was written into the spec as a fact on the
strength of that. **It was then tested and is mostly wrong:**

| z handling | top-1 | vs dropped |
|---|---|---|
| z dropped | 0.5917 ± 0.0263 | — |
| z rescaled to x,y's std | 0.5000 ± 0.0165 | −9.17pp, disjoint |
| raw z | 0.4783 ± 0.0085 | −11.34pp, disjoint |

Rescaling recovers only **19.1%** of the gain and does not even separate from
raw z. A per-block ablation then refuted the follow-up hypothesis too — that z
might be usable for the body while being noise for the 2D-inferred hands:

| z kept for | channels | top-1 | per channel |
|---|---|---|---|
| nothing | 0 | 0.5917 ± 0.0263 | — |
| pose only | 15 | 0.5434 ± 0.0061 | 0.322pp |
| hands only | 42 | 0.5138 ± 0.0146 | 0.185pp |
| everything | 91 | 0.4783 ± 0.0085 | 0.125pp |

All three are disjoint *below* z-dropped. No block helps, and **pose z is the
more harmful per channel** — pose-z-only outscores hands-z-only only because it
adds 15 channels instead of 42, not because body depth is better. Every avenue
for keeping z is now measured and closed. The spec records the correction rather
than the original claim.

### 4. Predictions were logged before each run, including the wrong ones

Each experiment was predicted in advance. Six misses, and three of them share
one root — worth more than the hit rate:

| # | prediction | actual | root cause |
|---|---|---|---|
| 1 | separator trim +0.8pp | −2.86pp | **protocol** |
| 2 | TCN +1.5pp, overlapping | +1.48pp, **disjoint** | **configuration-inheritance** |
| 3 | scaled z 0.582 | 0.5000, both calls inverted | **correlation-as-mechanism** |
| 4 | BiLSTM 0.480 | 0.3481 (−13.2pp) | **analogy-without-evidence** |
| 5 | augmentation separates on top-1 | does not, once z is dropped | **configuration-inheritance** |
| 6 | pose-z ≈ dropped, hands-z below | pose-z disjoint below too | **mechanism-from-plausibility** |

- **protocol / configuration-inheritance** (#1, #2, #5) — three of six — are all
  *the measurement conditions rather than the thing measured*. #1 changed val,
  which was the stopping criterion. #5 was measured with z present and quoted
  after z was dropped. #2 then reasoned *from* #5 and inherited its error. This
  is why the protocol is now enforced in code.
- **correlation-as-mechanism / mechanism-from-plausibility** (#3, #6) are a
  mechanism asserted from plausibility and then predicted from. A correlation
  that is real is not therefore the mechanism.
- **analogy-without-evidence** (#4): BiLSTM had no prior measurement here and
  was given a prior by analogy to the other two architectures.

One more worth stating: **no prediction was offered for the z ablation**, which
turned out to be the largest effect in the project.

## Licensing — code and data are governed separately

**Code: [MIT](LICENSE).** The MIT licence covers the code in this repository
only. It does not relicense the dataset, and it cannot.

**Data: C-UDA, unchanged.** WLASL is licensed **C-UDA** (Computational Use of
Data Agreement v1.0):
*"You will use the Data solely for Computational Use."* Academic and
computational use only, no commercial use.

Trained weights are **Results** and the C-UDA *"does not impose any restriction
with respect to the use, modification, or distribution of Results"*. Raw video
is **Data** and stays out of this repository. The split CSVs and the download
and extraction logs are published; the landmark cache is not. Each call, with
the quoted clauses and the one judgement call flagged as such:
**[DATA_LICENCE.md](DATA_LICENCE.md)**.

So: take the code under MIT and do as you like with it. Anything you do with
the WLASL-derived CSVs and logs is still bound by the C-UDA, MIT
notwithstanding.

## Status

**Milestone 4 (proxy corpus) is done. Milestone 1 is not closed.**

Done: the Milestone 0 skeleton (synthetic dataset mode, three backbones,
training loop, checkpointing, ONNX export); `extract.py` with a pure-NumPy
normalisation core; the WLASL downloader, signer-independent split, loader,
augmentation and separator trimming; the sweep harness; evaluation with
confusion matrices.

**Not closed — Milestone 1.** `overlay.py` is code-complete and tested but has
never been run on real footage, so extraction is unverified by eye and the face
landmark subset stays marked **PROVISIONAL**: the 478-point count is confirmed
from the model's own metadata, but the semantics (that index 61 is a lip corner)
are not.

Not started: `manifest.py`, `capture.py` (the data collection tool, and per spec
§4a the most important deliverable in the project), Stage B morphology and
translation, and the CTC head for continuous signing.

Open leads, unmeasured: a hand-rate quality gate alongside `min_pose_rate`; the
ST-GCN backbone; and whether augmentation helps at all now that z is dropped —
it no longer separates.

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

### Results — backbone pretraining on WLASL100 (ASL, not MSL)

**Every number below is American Sign Language**, on a corpus that is 55.3%
complete (1127 of 2038 clips; 911 lost to dead hosts and private videos). It
exists to pretrain a backbone; the vocabulary does not transfer and none of it
is an MSL result. Signer-independent split, test signers disjoint from train,
338 test clips, 100 classes, chance top-1 1.0%. The known limitations above
apply to all of it. **Val accuracy is never reported** — it is signer-dependent
and a stopping criterion only.

Protocol: **fixed budget of 2400 steps, no early stopping, cosine completing at
the budget, final checkpoint evaluated.** Three seeds (1234/2345/3456) per
variant, untuned. Mean +/- sample sd, with the min-max range, because at n=3 the
sd is a rough spread indicator rather than a confidence interval.

| variant | test top-1 | test top-5 | test macro F1 |
|---|---|---|---|
| transformer, no augmentation | 0.4241 +/- 0.0248 | 0.6982 +/- 0.0156 | 0.3998 +/- 0.0233 |
| transformer + augmentation | 0.4635 +/- 0.0075 | 0.7041 +/- 0.0185 | 0.4330 +/- 0.0200 |
| transformer + aug + separator trim | 0.4615 +/- 0.0030 | 0.7051 +/- 0.0090 | 0.4245 +/- 0.0061 |
| TCN + augmentation | 0.4783 +/- 0.0085 | 0.7219 +/- 0.0107 | 0.4442 +/- 0.0062 |
| transformer + aug, **z dropped** | 0.5345 +/- 0.0222 | 0.7998 +/- 0.0178 | 0.4923 +/- 0.0273 |
| TCN + aug, hands-z only | 0.5138 +/- 0.0146 | 0.7515 +/- 0.0180 | 0.4807 +/- 0.0199 |
| TCN + aug, z rescaled to x,y's std | 0.5000 +/- 0.0165 | 0.7367 +/- 0.0078 | 0.4602 +/- 0.0126 |
| TCN + aug, pose-z only | 0.5434 +/- 0.0061 | 0.7732 +/- 0.0181 | 0.5067 +/- 0.0052 |
| **TCN + aug, z dropped** (current best) | **0.5917 +/- 0.0263** | 0.7988 +/- 0.0135 | 0.5460 +/- 0.0354 |

**Rescaling z does not rescue it.** The obvious explanation for z's harm is that
it is loud: 33.1% of the feature vector at 1.5x the magnitude of x and y. That
explanation was tested and is mostly wrong. Rescaling z onto x and y's standard
deviation (`z_scale = 0.6953`, fitted on the training split only) recovers just
**19.1%** of the gain from dropping z, and scaled-z does not even separate from
raw z — 0.5000 +/- 0.0165 against 0.4783 +/- 0.0085, overlapping ranges — while
dropping z beats both with disjoint ranges. Most of the harm is z's *content*,
not its magnitude. Not ruled out: z may be usable for pose while being noise for
the 42 hand landmarks, which are inferred from a 2D image with weak depth priors.
A per-block ablation would test that and has not been run.

**The per-block hypothesis is refuted: no landmark block benefits from z.**
pose-z-only (0.5434 +/- 0.0061) and hands-z-only (0.5138 +/- 0.0146) are both
disjoint *below* z-dropped, and per channel pose z is the *more* harmful of the
two (0.322pp against 0.185pp) — so the idea that body depth is usable while
2D-inferred hand depth is noise does not survive measurement. Marginal harm also
falls as z is added (0.322 -> 0.185 -> 0.125 pp per channel), so admitting any z
at all carries most of the cost rather than each channel costing the same.

**Dropping MediaPipe's z channel is by far the largest effect measured** —
+7.10pp top-1 on the transformer and +11.34pp on the TCN, with non-overlapping
ranges in both cases and by a wide margin. Spec §4.3 anticipated z as
"unreliable but adds some depth cue"; on this corpus it is not a weak cue, it is
actively harmful, and the reason is measurable. z is 91 of 275 features (33.1%
of the vector) and its mean absolute magnitude in the cached tensors is **1.5x
that of x and y**. Normalisation divides every channel by the inter-shoulder
distance, which spec §4.2 deliberately computes from x and y only -- correct,
since including z would corrupt x and y too -- but the side effect is that z is
left on an arbitrary scale *larger* than the signal. A third of the input is
loud noise, and it dominates the linear input projection. Anything that consumes
these features should default to `use_z: false`.

What separates and what does not, stated as ranges rather than significance —
with 3 v 3 seeds the smallest achievable two-sided rank p is 0.1, so no result
here can reach p < 0.05 by construction:

- **Dropping z: non-overlapping by a wide margin, on both architectures.**
  Transformer .5089–.5473 against .4556–.4704; TCN .5621–.6124 against
  .4734–.4882. The clearest result in the table.
- **Augmentation vs no augmentation on top-1: non-overlapping ranges.** Baseline
  spans .4083–.4527, augmented .4556–.4704. On top-5 and macro F1 they overlap,
  so no separation there.
- **TCN vs transformer on top-1: non-overlapping, but narrowly.** With z,
  .4734–.4882 against .4556–.4704 — a 0.30pp gap between the TCN's worst seed
  and the transformer's best, which one more seed either side could close.
  Without z the gap widens to 1.48pp (.5621–.6124 against .5089–.5473). top-5
  and macro F1 overlap between the two architectures in both conditions.
- **Separator trim vs augmentation: no effect on any metric.** The trim's top-1
  range (.4586–.4645) sits entirely inside augmentation's (.4556–.4704). The
  trim does appear to *stabilise* training — its sd is the smallest of that
  group on every metric — but at n=3 an sd estimate is itself very noisy, so
  treat that as suggestive only. It is kept as a genuine data correction that
  does not move the number.

#### Spec §6 tested on its own terms

§6 says the TCN "is the one to beat — its locality prior makes it strong in
exactly the small-data regime this project starts in." That is a claim about an
*unaugmented* small-data regime, so it was tested there: no augmentation,
`use_z: false`, fixed 2400-step budget, 3 seeds.

| architecture, no augmentation | top-1 | top-5 | macro F1 | params |
|---|---|---|---|---|
| **TCN** | **0.5779 +/- 0.0061** | 0.7890 +/- 0.0074 | 0.5372 +/- 0.0033 | 695,268 |
| transformer | 0.4970 +/- 0.0178 | 0.7663 +/- 0.0030 | 0.4645 +/- 0.0152 | 574,948 |
| BiLSTM | 0.3481 +/- 0.0252 | 0.5927 +/- 0.0197 | 0.3114 +/- 0.0296 | 434,148 |

**The claim holds, and so does the reasoning behind it.** All three are pairwise
disjoint on top-1. The TCN's margin over the transformer *widens* from +5.72pp
with augmentation to **+8.09pp** without it, which is what "the locality prior
substitutes for data" predicts — the prior matters more when nothing else is
supplying variety. The TCN also has the tightest seed spread of the three
(sd 0.0061 against 0.0178 and 0.0252). BiLSTM is not competitive: 14.9pp below
the transformer and 23.0pp below the TCN.

**Augmentation does not separate once z is dropped.** The +3.94pp augmentation
result above was measured with z present. With z dropped — now the default —
augmentation's advantage overlaps for both architectures: transformer
0.4970 -> 0.5345 (+3.75pp, overlapping ranges) and TCN 0.5779 -> 0.5917
(+1.38pp, overlapping ranges). Both means are higher, so keeping augmentation is
reasonable, but at n=3 that preference is not evidence-based and should not be
described as a measured gain in the current configuration. Plausibly some of the
original augmentation benefit was compensating for z noise.

Budget fit is recorded per run in `sweep_metrics.json` so a delta can be checked
for the failure where the budget suits one arm and not the other. At 2400 steps
every variant lands at a final training loss of 0.86–0.88 with a late slope
under 0.04 per 1000 steps: converged, not truncated.

#### Earlier numbers under early stopping are NOT comparable

An earlier protocol used early stopping on val (patience 12 at eval_every 50)
and evaluated the best-val checkpoint. It produced **0.4300 +/- 0.0120 top-1 for
the augmented variant**, which was quoted as the reference for a time. **That
number is not comparable to anything in the table above and must not be placed
beside it.** Two confounds, both since fixed:

1. **The stopping step, not the change under test, drove the score.** Val is 116
   clips and its plateau is noisy, so 600 steps of patience stopped runs at
   near-arbitrary points. The stopping-step range widened exactly as a change
   perturbed val — baseline 100 steps, augmentation 650, separator trim 2200 —
   and stopping step against test top-1 gave r = 0.949 within the trim variant
   and r = 0.68 across all nine early-stopped runs. More seeds could not fix
   this: they average a confound rather than remove it.
2. **Runs ended at wildly different learning rates.** With the cosine defined
   over 4000 steps, a run stopping at 1400 ended at lr x 0.773 while one at 3600
   ended at x 0.027 — a 28-fold range, so early-stopped models were left hot and
   unannealed.

The fixed-budget protocol removes both, and the effect is visible: under early
stopping the separator trim scored 0.4014 +/- 0.0577 top-1, and at a fixed
budget the same data change scores 0.4615 +/- 0.0030 — a 19-fold reduction in
spread. The variance was the protocol, not the data. Note also that the two
protocols disagree about *which* metric separates: early stopping separated
top-5 and not top-1, the fixed budget the reverse.

Because the cosine now completes at 2400, these runs are not "the first 2400
steps" of the old schedule — the learning rate decays faster throughout, so old
numbers cannot be read as a prefix of new ones.

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
