# Ember3 — Mongolian Sign Language recognition (MSL → Cyrillic)

## Source of truth
msl-recognition-spec.md. Read it before acting — especially §4 (normalisation),
§4a (capture tool + manifest), §4b (Ember dictionary), §9a (known issues),
§10 (milestone order).

## Working method
One milestone at a time: implement, run it, show output, then STOP and wait for
go-ahead. Do not chain milestones.

## Current state — WLASL100 backbone (ASL, not MSL)

Every number here is AMERICAN SIGN LANGUAGE on a corpus that is 55.3% complete
(1127 of 2038 clips), signer-disjoint test, 338 test clips, 100 classes, chance
top-1 1.0%. Never quote one without that label. Val is never reported.

**PROTOCOL: fixed budget 2400 steps, no early stopping, cosine completing at the
budget, FINAL checkpoint evaluated.** 3 seeds (1234/2345/3456), untuned. Early
stopping is not used for variant comparisons and its numbers are not comparable
to these -- see the hard-won fact below.

| variant | top-1 | top-5 | macro F1 |
|---|---|---|---|
| transformer, no augmentation | 0.4241 +/- 0.0248 | 0.6982 +/- 0.0156 | 0.3998 +/- 0.0233 |
| transformer + augmentation | 0.4635 +/- 0.0075 | 0.7041 +/- 0.0185 | 0.4330 +/- 0.0200 |
| transformer + aug + separator trim | 0.4615 +/- 0.0030 | 0.7051 +/- 0.0090 | 0.4245 +/- 0.0061 |
| TCN + augmentation | 0.4783 +/- 0.0085 | 0.7219 +/- 0.0107 | 0.4442 +/- 0.0062 |
| transformer + aug, z dropped | 0.5345 +/- 0.0222 | 0.7998 +/- 0.0178 | 0.4923 +/- 0.0273 |
| TCN + aug, z scaled 0.6953 | 0.5000 +/- 0.0165 | 0.7367 +/- 0.0078 | 0.4602 +/- 0.0126 |
| **TCN + aug, z dropped (REFERENCE)** | **0.5917 +/- 0.0263** | 0.7988 +/- 0.0135 | 0.5460 +/- 0.0354 |

SPEC §6 TESTED ON ITS OWN TERMS (no augmentation, use_z false, fixed budget,
3 seeds). "The TCN is the one to beat below a few thousand clips" HOLDS:

| arch, NO augmentation | top-1 | range | top-5 | macro F1 | sd |
|---|---|---|---|---|---|
| TCN | **0.5779** | .5710-.5828 | 0.7890 | 0.5372 | 0.0061 |
| transformer | 0.4970 | .4793-.5148 | 0.7663 | 0.4645 | 0.0178 |
| BiLSTM | 0.3481 | .3195-.3669 | 0.5927 | 0.3114 | 0.0252 |

All three pairwise disjoint on top-1. The §6 MECHANISM is supported too, on the
criterion set before the run: the TCN's margin over the transformer WIDENS from
+5.72pp with augmentation to +8.09pp without it (+2.37pp), which is what "the
locality prior substitutes for data" predicts, and the TCN also has the tightest
seed spread of the three. BiLSTM is not in the race -- 14.9pp below the
transformer, 23.0pp below the TCN.

AUGMENTATION NO LONGER SEPARATES ONCE z IS DROPPED. This qualifies the earlier
finding, which was measured WITH z present (+3.94pp, disjoint ranges). With z
dropped -- now the default -- augmentation overlaps for both architectures:
transformer 0.4970 -> 0.5345 (+3.75pp, ranges overlap) and TCN 0.5779 -> 0.5917
(+1.38pp, ranges overlap). Keep augmentation if you like -- both means are
higher -- but do not claim it separates in the current default configuration,
and note that preferring the augmented pipeline is not an evidence-based choice
at n=3. Plausibly augmentation was partly compensating for z noise.

Separations, as non-overlapping ranges and never as significance (at 3 v 3 the
smallest achievable two-sided rank p is 0.1): dropping z separates on both
architectures by a wide margin; augmentation separates on top-1 only; TCN beats
transformer on top-1 only, by 0.30pp of range gap with z and 1.48pp without;
the separator trim shows no effect on any metric.

DROPPING MediaPipe's z IS THE LARGEST EFFECT MEASURED: +7.10pp top-1 on the
transformer, +11.34pp on the TCN. DEFAULT IS use_z: false and it should stay
that way.

SCALE IS NOT WHY, AND THIS WAS TESTED. z is 91 of 275 features (33.1%) at 1.5x
the magnitude of x and y -- both real measurements -- and it is tempting to
conclude z is harmful for being loud. It is not. Rescaling z onto x,y's std
(z_scale 0.6953, fitted on train only) recovers only 19.1% of the drop-z gain,
and scaled-z does not even separate from raw z: 0.5000 +/- 0.0165 against
0.4783 +/- 0.0085, overlapping ranges. Dropping z beats both with disjoint
ranges. MOST OF THE HARM IS z's CONTENT, NOT ITS LOUDNESS. Do not try to rescue
z by rescaling -- that path is implemented (FeatureConfig.z_scale,
landmarks.scale_z), measured, and closed. NOT ruled out: z may be usable for
pose while being noise for the 42 hand landmarks, which are inferred from a 2D
image. A per-block z ablation would test that and has not been run.

The ablation costs nothing: landmarks.drop_z slices z out of the use_z=True
cache and is exactly equivalent to extracting with use_z=False, because x and y
never depend on z. Do not re-extract for it.

THE 0.4300 +/- 0.0120 FIGURE IS DEAD. It was the early-stopping reference and is
not comparable to anything above. Do not quote it, and do not place it beside a
fixed-budget number.

THREE SEEDS MINIMUM FOR EVERY VARIANT, reported as mean and spread. A single run
is unreadable: the baseline's seed-to-seed top-1 range is still 4.4pp wide even
at a fixed budget. Use msl/train/sweep.py with --max-steps 2400
--no-early-stopping; it records the protocol in sweep_metrics.json and warns when
run under early stopping.

**Next step: not yet chosen.** Done and measured: augmentation, separator
trimming (no effect, kept behind `data.drop_separator_frames`, default false),
TCN, and the z ablation. Open leads, in rough order of expected value: re-test
the spec §6 "TCN is the one to beat" claim WITHOUT augmentation, which is the
comparison the spec actually describes and has not been run; a hand-rate gate
in place of or alongside min_pose_rate (see the signer-121 fact); and BiLSTM,
the third §6 baseline, which has never been run at all.

PREDICTION LOG, with root cause. Kept because the misses cluster.

| # | prediction | actual | root cause |
|---|---|---|---|
| 1 | separator trim +0.8pp top-1 | -2.86pp | **protocol** |
| 2 | TCN +1.5pp, ranges overlapping | +1.48pp, ranges DISJOINT | **configuration-inheritance** |
| 3 | scaled z 0.582 top-1 | 0.5000, both directional calls inverted | **correlation-as-mechanism** |
| 4 | BiLSTM 0.480 top-1 | 0.3481 (-13.2pp) | **analogy-without-evidence** |
| 5 | augmentation separates on top-1 | does not separate once z is dropped | **configuration-inheritance** |
| - | z ablation: no prediction offered | largest effect in the project | - |

What each root means, and how to not repeat it:

* **protocol** (#1): the change under test also altered the measurement
  apparatus. The trim edited val, which was the early-stopping criterion, so the
  stopping step moved and the data effect was never measured. Before predicting
  a delta, ask what the change does to the measurement, not only to the data.
* **configuration-inheritance** (#2, #5): a result measured under one
  configuration was carried forward after the configuration changed. The +3.94pp
  augmentation win was measured WITH z; it does not survive dropping z. Miss #2
  then reasoned *from* that win ("augmentation already spent the small-data
  gain, so the TCN has no headroom") and inherited its error. Every recorded
  number needs the configuration it was measured under attached, and a number
  must be re-measured, not re-quoted, once that configuration changes.
* **correlation-as-mechanism** (#3): z really is 33.1% of the vector at 1.5x the
  magnitude of x and y, and that really did look like the reason it hurt. It was
  not. A correlation that is real is not therefore the mechanism; test the
  mechanism before recording it, and never predict from an untested one.
* **analogy-without-evidence** (#4): BiLSTM had no prior result on this corpus
  and was given a prior by analogy to the transformer and TCN. Analogy is not
  evidence. An architecture with no measurement here gets a wide interval, not a
  neighbour's mean.

THREE OF THE FIVE MISSES SHARE ONE ROOT: #1, #2 and #5 are all the measurement
conditions rather than the thing measured. That is the failure mode to watch on
this project, and it is why the fixed-budget protocol is now enforced by
sweep.py refusing to run off-protocol rather than by remembering a CLI flag.

RUN ARTIFACTS DO NOT SURVIVE A FRESH CLONE. `runs/` and `checkpoints/` are both
gitignored, and zero files under them are tracked, so sweep_metrics.json, the
confusion matrices and every checkpoint are local-only. They are reproducible --
the pipeline is deterministic and seed 1234 re-runs exactly -- at roughly 15
minutes per 3-seed sweep. If a sweep result ever needs to outlive the machine,
commit its sweep_metrics.json the way the WLASL logs were committed.

## Hard-won facts — do not re-derive or contradict
- MediaPipe 1.0 REMOVED `mediapipe.solutions` entirely. Legacy Holistic does not
  exist; every tutorial built on it fails at import. Use the Tasks API
  HolisticLandmarker (it also assigns left/right hand from the body model).
- The bundled facemesh2_lite returns 478 face landmarks (468 canonical mesh + 10
  iris at 468-477), confirmed from its own tflite metadata. All FACE_INDICES are
  < 468 so they address mesh points only. They remain PROVISIONAL: the count is
  confirmed, the semantics (that index 61 is a lip corner) are NOT.
- Aspect correction is mandatory and must be derived from each frame's real
  width/height, never a constant. MediaPipe normalises x by width and y by height,
  so on non-square frames the inter-shoulder divisor is wrong relative to hand
  motion. A pixel-space overlay CANNOT detect this — it renders correctly either
  way. Guarded by tests; do not "simplify" them away.
- Absent hands are zeroed AFTER normalisation, never before, plus a visibility mask.
- np.save() appends ".npy" to any path not ending in it — this already broke atomic
  cache writes once. save_cached writes through an open file handle.

## Standing rules
- NEVER generate, invent, or suggest MSL vocabulary. Glosses come only from native
  signers or the Ember dictionary export the user supplies. Do not scrape
  sign.ember.mn.
- Never report accuracy from synthetic or placeholder data as a result.
- WLASL is American Sign Language — usable for backbone pretraining only, never as
  MSL vocabulary. Label every WLASL number as ASL everywhere it appears: README,
  figures, slides, talks, commit messages.
- WLASL is licensed C-UDA (Computational Use of Data Agreement): academic and
  computational use only, no commercial use.
- WLASL's shipped split is NOT signer-independent: 2877/2878 test instances have a
  signer who also appears in train. We build our own signer-independent split
  (whole signers held out, greedily smallest-first, no gloss losing all training
  data, seeded, committed as CSV). ALWAYS report two numbers, both labelled ASL:
  the signer-independent one as the headline, and the official-split one only
  alongside it, marked signer-dependent and optimistic. The official number must
  never appear alone -- not in the README, not in a plot title, not in a commit
  message.
- The WLASL100 corpus on disk is 55.3% complete (1127 of 2038 clips; 911 missing to
  dead hosts and private videos). Every number computed before the maintainers'
  pre-processed-video request lands must say so, alongside the ASL label.
- The signer-independent split is built by msl/data/wlasl_split.py under a TWO-SIDED
  constraint: every gloss keeps both training clips and at least one held-out test
  clip. Enforcing only the training side is a bug that already occurred -- it left
  12 of 100 glosses with no test clip while still reporting 100 classes. The ~20%
  target is soft; the constraint is hard. Achieved: 30.0% holdout, 51 held-out
  signers, 100/100 glosses on both sides.
- VAL IS SIGNER-DEPENDENT and is never a reported result. It is a clip-level split
  inside the training signers, used only as a stopping criterion. Signers in val also
  appear in train, so val accuracy is inflated and means nothing as a measure of
  generalisation. TEST is the honest number: signer-disjoint, never touched for model
  selection. Never put a val number in a report, a plot title, or a README.
- CACHED-METRIC RECOVERY RESTS ON TWO INVARIANTS OF normalise_frame, and breaks
  SILENTLY if either changes. wlasl_loader.rates_from_features() recovers pose and
  hand presence from a cached .npy alone, because (a) normalise_frame returns an
  all-zero vector EXACTLY when there is no usable pose, and (b) any frame that did
  normalise has the shoulders at +/-0.5 and so can never be all-zero. If
  normalisation ever emits a non-zero vector for a pose-less frame, or an all-zero
  vector for a valid one, every cached clip silently gets the wrong pose_rate and is
  then filtered out of training without an error. Change extract.py's normalisation
  and you must revisit rates_from_features. The failure it already caused once:
  cache hits reported pose_rate 0.0 and were dropped from the dataset on every re-run.
- MediaPipe HolisticLandmarker RETAINS STATE ACROSS detect() CALLS, even in IMAGE
  running mode (segmentation smoothing). Feeding it a different frame size than the
  previous call crashes the graph: "RET_CHECK failure ... current_mat->rows ==
  previous_mat->rows". Use one landmarker per video, or per input size. This is safe
  in extract.py today only because LandmarkExtractor is opened per video; reusing one
  across a mixed-resolution corpus (WLASL has 15 distinct resolutions) would crash.
- MediaPipe EXTRAPOLATES pose landmarks outside the frame. Hip/knee/ankle points come
  back with normalised y above 1.0 (observed up to 2.04) when the signer is framed
  from the waist up. Any bounding box or extent computed from raw pose landmarks must
  clamp to [0,1] and use the upper-body subset first, or it is garbage. An unclamped
  person-extent computation already produced a wrong answer once.
- WLASL frame_start/frame_end MUST be honoured when extracting. 132 of 1127 clips are
  segments of long compilation videos (median 7221 frames); the labelled sign occupies
  about 1% of the file. Do NOT crop to bbox: the bbox does not correspond to the raw
  video we downloaded (IoU ~0.1 against where MediaPipe finds the person) and cropping
  measurably destroys detection.
- PER-CLIP SOURCE RESOLUTION LIVES IN TWO COMMITTED LOGS, not in the cache. The
  cached .npy does not store frame dimensions, so a cache-only extraction run
  writes width=0, height=0, fps=0 for every row (verified: all 1127 rows of
  extract_log.csv). Spec §4 requires aspect ratio per clip, so both logs are
  committed under one narrow .gitignore exception walked down level by level --
  a `!` rule cannot re-admit a file under an excluded directory.
  * data/wlasl/download_log_100.csv is the PRIMARY record: keyed by video_id,
    all 2038 attempted clips, with width/height/fps/frames/bytes/source/host/url
    and the download status that explains why 911 are missing.
  * data/wlasl/extract_log_firstpass.csv is a redundant second record covering
    the 1115 rows that came from a real extraction (width>0).
  They agree EXACTLY on width, height and fps for all 1115 overlapping clips --
  0 mismatches, checked. The only column that differs is frame count, for 132
  clips, and that is correct rather than a discrepancy: the download log's
  `frames` is the whole file, the extraction log's `frames_in` is the
  frame_start..frame_end range. Do not add an aspect_ratio column to either;
  it is width/height and storing it invites the two going out of step.
  Never "refresh" either log by re-running extraction against a warm cache --
  that replaces real resolutions with zeros. Re-deriving them needs the cache
  deleted and all 1127 clips re-extracted, against a corpus whose missing half
  is unrecoverable.
- RELATED LATENT GITIGNORE BUG, not yet fixed: the `!**/manifest.csv` and
  `!**/signers.csv` whitelists are unreachable for anything under the data root,
  for the same parent-directory reason. Spec §4a requires manifest.csv and
  signers.csv to be version-controlled. If capture.py writes them under
  data_root they will be silently untracked -- confirmed empirically, not
  assumed. Settle this when open decision 1 (manifest schema) is settled.
- WLASL northtexas CLIPS CARRY BLACK SEPARATOR FRAMES INSIDE THE LABELLED RANGE.
  43 of that source's 51 clips have a contiguous near-black run within
  frame_start..frame_end, median 33.8% of the range (p90 51.9%, max 54.3%); 21
  of them start within 15 frames of the range start, so the range is offset, not
  merely wide -- one clip runs 9 pose frames, 51 black, 36 pose, meaning the
  leading frames belong to the PREVIOUS dictionary entry. This is SOURCE-specific,
  not format-specific: asl5200 is also trimmed compilation video, 81 clips, and
  has ZERO black runs. The seek is exact (POS_FRAMES after seek == requested), so
  this is WLASL's labelling, not our trimming. MediaPipe correctly finds no pose
  on a black frame and normalise_frame correctly emits all-zero, so the pipeline
  is behaving as designed.
  THE LUMINANCE THRESHOLD MUST BE 10, NOT 20. A first pass at 20 wrongly flagged
  13 asl5200 clips as "100% black": asl5200 footage is uniformly dim, with a
  per-frame mean of 18-21 even where pose tracks at rate 1.0. northtexas
  separators measure 2.3-7.3. At 10 the cut self-validates: 100% of frames called
  dark also had no pose, and dark explains 99.8% of all pose-less frames in these
  clips.
  The pose-rate gate catches only 7 of the 43; 36 remain in the dataset (30 train,
  6 val) with pose rates 0.53-1.0. TEST IS UNAFFECTED -- signers 121 and 9 are both
  entirely inside train/val.
- SIGNER 121 IS A SYSTEMATICALLY DEGRADED SIGNER and is exactly the 51 northtexas
  clips. Pose-rate median 0.705 against 1.000 for the corpus; its 44 kept clips
  average left-hand 0.165 / right 0.282 against 0.312 / 0.589 for the other 1076.
  Not empty (max hand rate 0.616, none with zero hands throughout), so they are
  not simply droppable -- but 4.5% of the corpus contributes roughly half the
  usual hand signal, all of it in train/val. Related: pose_rate measures whether
  there was an origin to centre on, NOT whether there was a hand to recognise. A
  gate on max(left,right) < 0.1 would select 9 clips (8 of them signer 121, 1
  signer 59), all train/val, 9 distinct glosses, starving nothing. The gate has
  NOT been changed; this is recorded as an option, not a decision.
- WLASL gloss_ids ARE NOT ALPHABETICAL. id 0 is "book", id 1 "drink", id 2
  "computer". Any vocabulary built by sorting gloss names is a PERMUTATION of the
  labels the dataset actually returns, because __getitem__ yields gloss_id.
  Nothing errors and accuracy is unchanged -- the permutation is consistent -- so
  the only visible symptom is that every class name on a confusion matrix, every
  per-class F1 and every confused-pair listing names the wrong sign. Build the
  vocabulary by indexing gloss_id, never by sorting. This already occurred once in
  WlaslLandmarkDataset.glosses and was caught before the first training run.
- EARLY STOPPING ON A 116-CLIP VAL SET CONFOUNDS EVERY VARIANT COMPARISON.
  patience 12 at eval_every 50 is 600 steps of patience on a plateau that is
  already noisy at 116 val clips, so the stopping step is close to arbitrary --
  and the stopping step, not the change under test, drives the score. Evidence:
  stopping-step range widens exactly as a change perturbs val more --
  baseline [2050,2100,2150] range 100 -> augmentation [1550,1900,2200] range 650
  -> separator trim [1400,1900,3600] range 2200. Stopping step against test top-1
  gives r = 0.949 within the trim variant (n=3, indicative not conclusive) and
  r = 0.68 across all nine early-stopped runs. ANY change that touches val --
  trimming its frames, altering its clips -- confounds the comparison, and more
  seeds cannot fix it: they average a confound rather than remove it.
  A SECOND, COMPOUNDING CONFOUND: with the cosine defined over max_steps=4000,
  runs that stopped early ended at wildly different learning rates -- lr x 0.773
  at step 1400 versus x 0.027 at 3600, a 28-fold range. A run stopped early ends
  hot and unannealed, which is part of why longer runs scored better.
  THE FIX, adopted: fixed step budget, no early stopping, cosine completing at
  the budget so every run ends annealed, and EVALUATE THE FINAL CHECKPOINT rather
  than the best-val one. Selecting the best-val checkpoint would leave val
  driving the comparison even with early stopping switched off.
  Numbers produced under early stopping are NOT comparable to fixed-budget
  numbers. The 0.4300 +/- 0.0120 reference is an early-stopping number.
- A PREDICTION MISS WORTH REMEMBERING: separator trimming was predicted at
  +0.8pp top-1 and came out -2.86pp. The prediction modelled the DATA change
  (5.4% of training clips lose 31% of their frames) and ignored the PROTOCOL --
  that the same trim also edits val, which is the stopping criterion. The data
  effect was never measured; the protocol effect was. This is the same class of
  error as a caveat named in prose but not carried into the arithmetic: the
  confound was stated in the commit message for the trim and still left out of
  the prediction. Before predicting a delta, ask what the change does to the
  measurement apparatus, not only to the data.
- A BiLSTM SWEEP ON CUDA REPORTS EXIT 127 DESPITE SUCCEEDING. Verify the
  SWEEP_OK marker file in the report directory, never the exit code. Scoped by
  bisection: transformer and TCN on CUDA exit 0, BiLSTM on CPU exits 0, BiLSTM on
  CUDA exits 127 with complete and valid metrics on disk. A bare cuDNN LSTM
  forward pass exits 0, so it is not simply the RNN, and an os._exit(0) at the
  end of main() is REACHED and prints while the shell still sees 127 -- the
  status is overridden below Python, so no in-process fix exists. ROOT CAUSE
  UNKNOWN and deliberately left unexplained rather than guessed at. This already
  made one three-architecture sweep look like a failure when all nine runs had
  succeeded.
- Known limitations to restate wherever WLASL numbers appear: (a) signer_id
  reliability is unverified and the split is only as good as those labels, WLASL
  aggregating 8 separate dictionary sites; (b) WLASL is citation-form dictionary
  video, not natural signing, so the backbone is tuned to citation form.
- Proxy corpus choice was REVERSED, for access not merit. AUTSL (Turkish Sign
  Language) was chosen first because it ships an official signer-independent split
  — the property spec §5 says decides whether a number means anything — and carries
  ~170 clips per class against WLASL's ~10. AUTSL proved unobtainable: encrypted
  files whose decryption keys are issued through a CodaLab instance read-only since
  January 2023. WLASL is a fallback, not an upgrade. If WLASL's shipped split is not
  signer-independent, say so plainly wherever a number appears; never let a
  signer-dependent number stand unqualified.

## Environment — already set up, do not redo
uv sync done. Python 3.11.9, torch 2.13.0+cu130, CUDA available, RTX 5090.
mediapipe 1.0.0, opencv 5.0.0, numpy 2.4.6. Model bundle at
assets/holistic_landmarker.task (13.7 MB, gitignored). Windows + PowerShell
primary; Bash also available.

The machine changed on 2026-09-15: the project moved by flash drive from an
RTX 3060 box to an RTX 5090 one, and .venv was NOT copied. `uv sync` on the new
machine reproduced the same pinned versions; 172 tests pass; the landmark cache
and raw videos transferred intact (verified: all 1127 content-digest cache keys
re-derive to the same filenames). Any earlier note in this file saying RTX 3060
is stale. The `num_workers: 0` default in the configs was chosen for the old
16GB box and is now a conservative setting rather than a necessary one --
revisit it when training, do not assume it is tuned.

## Open decisions — ask, do not choose
1. Manifest schema conflict: §4a defines 16 columns (incl. resolution, mirrored);
   §5 line ~149 defines a different 6-column set (incl. split). Aspect ratio per
   clip must land in whichever wins. msl/data/manifest.py cannot be written until
   this is settled.
2. WLASL: whether to use a shipped split or construct a signer-independent one,
   and which subset size to start from.
3. Disk budget / whether data_root points at an external drive.
