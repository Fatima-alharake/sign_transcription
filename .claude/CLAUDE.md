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
- AUTSL is Turkish Sign Language — usable for backbone pretraining only, never as
  vocabulary. Label any AUTSL number as Turkish Sign Language everywhere it appears.
- If an AUTSL mirror is used instead of the official source, VERIFY it preserves the
  official signer-independent split and the signer_id column before training, and
  record provenance in the README.

## Environment — already set up, do not redo
uv sync done. torch 2.13.0+cu130, CUDA available, RTX 3060. mediapipe 1.0.0,
opencv 5.0.0, numpy 2.4.6. Model bundle at assets/holistic_landmarker.task
(13.7 MB, gitignored). Windows + PowerShell primary; Bash also available.

## Open decisions — ask, do not choose
1. Manifest schema conflict: §4a defines 16 columns (incl. resolution, mirrored);
   §5 line ~149 defines a different 6-column set (incl. split). Aspect ratio per
   clip must land in whichever wins. msl/data/manifest.py cannot be written until
   this is settled.
2. AUTSL access route and provenance.
3. Disk budget / whether data_root points at an external drive.
