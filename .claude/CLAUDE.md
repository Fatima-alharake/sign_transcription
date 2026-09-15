# Ember3 — Mongolian Sign Language recognition (MSL → Cyrillic)

## Source of truth
msl-recognition-spec.md. Read it before acting — especially §4 (normalisation),
§4a (capture tool + manifest), §4b (Ember dictionary), §9a (known issues),
§10 (milestone order).

## Working method
One milestone at a time: implement, run it, show output, then STOP and wait for
go-ahead. Do not chain milestones.

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
uv sync done. torch 2.13.0+cu130, CUDA available, RTX 3060. mediapipe 1.0.0,
opencv 5.0.0, numpy 2.4.6. Model bundle at assets/holistic_landmarker.task
(13.7 MB, gitignored). Windows + PowerShell primary; Bash also available.

## Open decisions — ask, do not choose
1. Manifest schema conflict: §4a defines 16 columns (incl. resolution, mirrored);
   §5 line ~149 defines a different 6-column set (incl. split). Aspect ratio per
   clip must land in whichever wins. msl/data/manifest.py cannot be written until
   this is settled.
2. WLASL: whether to use a shipped split or construct a signer-independent one,
   and which subset size to start from.
3. Disk budget / whether data_root points at an external drive.
