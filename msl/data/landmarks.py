"""Landmark layout: which keypoints we keep and how they pack into a frame vector.

This module is deliberately free of MediaPipe imports. It defines *indices and
shapes only*, so that the model, the dataset and the ONNX exporter can all agree
on the per-frame feature dimension before any extraction code exists
(Milestone 0 runs with no MediaPipe installed at all).

`extract.py` (Milestone 1) will import these constants and is the only module
allowed to talk to MediaPipe.

Frame vector layout, in order:

    [ pose x,y(,z) ] [ left hand x,y(,z) ] [ right hand x,y(,z) ]
    [ face x,y(,z) ] [ left-hand visible ] [ right-hand visible ]

The two trailing scalars are the per-hand visibility mask (spec S4): MediaPipe
returns nothing when a hand leaves frame, and we emit zeros for it. Without an
explicit mask the model cannot distinguish "hand absent" from "hand resting at
the origin", which after mid-shoulder centring is a physically plausible pose.
"""

from __future__ import annotations

from typing import Final

# --- Pose ------------------------------------------------------------------
# MediaPipe Pose returns 33 landmarks. Signing happens above the waist, so we
# keep the nose (0) as a head-position anchor plus 11..24 (shoulders, elbows,
# wrists, hands, hips). Everything below the hips is discarded.
POSE_INDICES: Final[tuple[int, ...]] = (0,) + tuple(range(11, 25))

# Indices *within* POSE_INDICES of the two shoulders, used by the normalisation
# step in extract.py (translate to mid-shoulder, scale by inter-shoulder
# distance). Stored as positions in the kept subset, not raw MediaPipe indices,
# because the normaliser sees the already-subset array.
LEFT_SHOULDER_POS: Final[int] = POSE_INDICES.index(11)
RIGHT_SHOULDER_POS: Final[int] = POSE_INDICES.index(12)

# --- Hands -----------------------------------------------------------------
# Full 21-point hand skeleton, both hands. These carry most of the signal, so
# nothing is dropped here.
N_HAND_LANDMARKS: Final[int] = 21

# --- Face ------------------------------------------------------------------
# A subset of the 468-point face mesh (spec S4): 468 points is mostly noise, but
# discarding the face entirely would throw away grammar. Eyebrow raise marks
# polar questions and mouthing disambiguates otherwise-identical manual signs,
# so we keep lips, eyebrows and eye corners.
#
# The installed model returns 478 points, not 468: MediaPipe 1.0's bundle uses
# "facemesh2_lite" ("Detects 478 face landmarks" per its own tflite metadata),
# which is the 468-point canonical mesh plus 10 iris points appended at indices
# 468-477. Every index below is < 468, so all of them address canonical mesh
# points and none collide with the iris block.
#
# PROVISIONAL, still: knowing the *count* does not prove that index 61 is a lip
# corner on this model. The published topology says it is and the first 468
# points are documented as unchanged, but only rendering the overlay on a real
# face confirms the semantics. If any tuple turns out wrong, fix it here and
# nothing else changes -- every other module derives its dimensions from
# FEATURE_LAYOUT below.
LIPS_OUTER_INDICES: Final[tuple[int, ...]] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
)
LEFT_EYEBROW_INDICES: Final[tuple[int, ...]] = (70, 63, 105, 66, 107)
RIGHT_EYEBROW_INDICES: Final[tuple[int, ...]] = (300, 293, 334, 296, 336)
# Inner and outer corner of each eye -- enough to capture eye aperture and gaze
# direction shifts without carrying the full eye contour.
EYE_CORNER_INDICES: Final[tuple[int, ...]] = (33, 133, 362, 263)

FACE_INDICES: Final[tuple[int, ...]] = (
    LIPS_OUTER_INDICES
    + LEFT_EYEBROW_INDICES
    + RIGHT_EYEBROW_INDICES
    + EYE_CORNER_INDICES
)

# --- Derived sizes ---------------------------------------------------------
N_POSE: Final[int] = len(POSE_INDICES)
N_FACE: Final[int] = len(FACE_INDICES)
N_LANDMARKS: Final[int] = N_POSE + 2 * N_HAND_LANDMARKS + N_FACE

# Number of trailing mask scalars (one per hand).
N_VISIBILITY_CHANNELS: Final[int] = 2

# Ordered (name, count) blocks, for anyone slicing a frame vector apart.
FEATURE_LAYOUT: Final[tuple[tuple[str, int], ...]] = (
    ("pose", N_POSE),
    ("left_hand", N_HAND_LANDMARKS),
    ("right_hand", N_HAND_LANDMARKS),
    ("face", N_FACE),
)


def coords_per_landmark(use_z: bool) -> int:
    """Coordinates kept per landmark: x,y,z or just x,y.

    z is unreliable in MediaPipe but carries some depth cue, so the spec asks
    for it behind a config flag to make ablation cheap.
    """
    return 3 if use_z else 2


def feature_dim(use_z: bool = True) -> int:
    """Per-frame feature vector length, including the hand visibility mask."""
    return N_LANDMARKS * coords_per_landmark(use_z) + N_VISIBILITY_CHANNELS


def block_slice(block: str, use_z: bool = True) -> slice:
    """Slice selecting one named block's coordinates from a frame vector."""
    c = coords_per_landmark(use_z)
    offset = 0
    for name, count in FEATURE_LAYOUT:
        if name == block:
            return slice(offset, offset + count * c)
        offset += count * c
    raise KeyError(f"unknown feature block {block!r}")


def visibility_slice(use_z: bool = True) -> slice:
    """Slice selecting the two trailing per-hand visibility scalars."""
    total = feature_dim(use_z)
    return slice(total - N_VISIBILITY_CHANNELS, total)
