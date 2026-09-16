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
# These tuples are pinned to MediaPipe's own topology tables
# (`mediapipe.tasks.python.vision.FaceLandmarksConnections`) by
# tests/test_topology.py, which is skipped when MediaPipe is absent. Keeping the
# constants here rather than importing those tables is deliberate: this module
# must stay importable with no MediaPipe installed (Milestone 0), and config.py
# imports it. The guard test is what keeps the copy honest -- it is how the
# left/right eyebrow inversion below was found.
#
# STILL PROVISIONAL, narrowly: the tables prove these indices are what MediaPipe
# *declares* lips and eyebrows to be. They do not prove the bundled tflite emits
# its points in that order. Only the overlay on a real face closes that gap.
#
# Naming is subject-relative throughout, matching PoseLandmark (LEFT_SHOULDER is
# the signer's own left). On an unmirrored frame the signer's left appears on
# the viewer's right.

# Outer lip contour, traversed as a closed ring: the lower lip left-to-right,
# then back along the upper lip. Ring order matters for the overlay -- a correct
# ordering draws a clean lip outline and a scrambled one draws a star, which
# makes the polyline itself a test of the ordering.
LIPS_OUTER_INDICES: Final[tuple[int, ...]] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
)

# Upper contour of each eyebrow, in chain order. MediaPipe's full eyebrow group
# is 10 points (an upper and a lower chain); the upper chain alone is enough to
# track eyebrow *height*, which is the grammatical signal -- eyebrow raise marks
# polar questions (spec S4).
#
# These two were inverted until the guard test caught it: (70, 63, 105, 66, 107)
# reads as the signer's left eyebrow if you assume viewer-relative naming, but
# MediaPipe lists it under FACE_LANDMARKS_RIGHT_EYEBROW. Both tuples were always
# valid eyebrow chains, so nothing downstream ever looked broken -- exactly the
# silent left/right swap that spec S4b treats as a first-class hazard.
LEFT_EYEBROW_INDICES: Final[tuple[int, ...]] = (300, 293, 334, 296, 336)
RIGHT_EYEBROW_INDICES: Final[tuple[int, ...]] = (70, 63, 105, 66, 107)

# Two corners of each eye -- enough for eye aperture and gaze-direction shifts
# without carrying the full 16-point eye contour. Which member of each pair is
# the inner versus the outer corner is not something the connection tables
# state; it does not affect the feature vector, only how you read a label.
EYE_CORNER_INDICES: Final[tuple[int, ...]] = (362, 263, 33, 133)

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


def drop_z(features):
    """Convert a use_z=True frame tensor into its use_z=False equivalent.

    EXACT, not approximate. `normalise_frame` derives the origin and the scale
    the same way whichever setting is active -- the inter-shoulder divisor is
    computed from x and y only (spec S4.2), and z never enters it -- so the x
    and y of a normalised frame do not depend on use_z. Dropping every third
    coordinate therefore reproduces what extraction with use_z=False would have
    written, byte for byte, and `tests/test_landmarks.py` asserts exactly that
    against normalise_frame rather than trusting this comment.

    This exists so the z ablation costs nothing: the cache holds 275-dim
    tensors, and re-extracting 1127 clips through MediaPipe to remove one
    channel per landmark would take hours to produce data we can slice.
    """
    import numpy as np

    features = np.asarray(features)
    wide = feature_dim(True)
    if features.shape[-1] != wide:
        raise ValueError(
            f"expected a use_z=True tensor of width {wide}, got {features.shape[-1]}"
        )
    n_frames = features.shape[0]
    coords = features[:, : N_LANDMARKS * 3].reshape(n_frames, N_LANDMARKS, 3)
    kept = coords[:, :, :2].reshape(n_frames, N_LANDMARKS * 2)
    visibility = features[:, visibility_slice(True)]
    return np.concatenate([kept, visibility], axis=1).astype(features.dtype, copy=False)


def scale_z(features, factor: float):
    """Multiply the z channel of a use_z=True frame tensor by `factor`.

    Tests spec S4.3's refined claim. z is harmful because it is UNSCALED -- a
    third of the vector at 1.5x the magnitude of x and y -- rather than because
    it is meaningless, so putting it on x and y's scale should recover whatever
    depth cue is genuinely there. If the scaled channel still does not help,
    that is evidence about z's information content rather than its loudness.

    Zeroed blocks stay zeroed (0 * factor == 0) and the visibility flags are
    outside the coordinate region, so neither invariant is touched.
    """
    import numpy as np

    features = np.asarray(features)
    wide = feature_dim(True)
    if features.shape[-1] != wide:
        raise ValueError(
            f"expected a use_z=True tensor of width {wide}, got {features.shape[-1]}"
        )
    out = features.copy()
    coords = out[:, : N_LANDMARKS * 3].reshape(features.shape[0], N_LANDMARKS, 3)
    coords[:, :, 2] *= factor
    return out


Z_BLOCK_NAMES: Final[tuple[str, ...]] = tuple(name for name, _ in FEATURE_LAYOUT)


def feature_dim_for(keep_z: tuple[str, ...] | None) -> int:
    """Per-frame width when z is kept for `keep_z` blocks only.

    `None` means every block keeps z (the full 275). An empty tuple means none
    do, which is the same 184 as use_z=False.
    """
    if keep_z is None:
        return feature_dim(True)
    unknown = set(keep_z) - set(Z_BLOCK_NAMES)
    if unknown:
        raise KeyError(f"unknown feature block(s): {sorted(unknown)}")
    return sum(
        count * (3 if name in keep_z else 2) for name, count in FEATURE_LAYOUT
    ) + N_VISIBILITY_CHANNELS


def select_z_blocks(features, keep_z: tuple[str, ...]):
    """Keep z for some landmark blocks and drop it for the rest.

    Tests whether MediaPipe's z is usable for the body while being noise for the
    hands, whose depth is inferred from a 2D image with weak priors. That is a
    HYPOTHESIS this function exists to measure, not a mechanism to assume.

    Applied AFTER augmentation, never before. `augment_sequence` assumes a
    uniform coordinate count per landmark, which a mixed 2-and-3 layout breaks;
    running it on the full uniform tensor first keeps those invariants and their
    tests untouched. Nothing is lost by the ordering: rotation about the camera
    axis never mixes z into x or y, scaling is uniform, and per-joint noise is
    independent per coordinate, so the surviving x and y are exactly what they
    would have been.
    """
    import numpy as np

    features = np.asarray(features)
    wide = feature_dim(True)
    if features.shape[-1] != wide:
        raise ValueError(
            f"expected a use_z=True tensor of width {wide}, got {features.shape[-1]}"
        )
    n_frames = features.shape[0]
    coords = features[:, : N_LANDMARKS * 3].reshape(n_frames, N_LANDMARKS, 3)

    parts = []
    offset = 0
    for name, count in FEATURE_LAYOUT:
        block = coords[:, offset:offset + count, :]
        parts.append(block.reshape(n_frames, count * 3) if name in keep_z
                     else block[:, :, :2].reshape(n_frames, count * 2))
        offset += count
    parts.append(features[:, visibility_slice(True)])
    return np.concatenate(parts, axis=1).astype(features.dtype, copy=False)


def visibility_slice(use_z: bool = True) -> slice:
    """Slice selecting the two trailing per-hand visibility scalars."""
    total = feature_dim(use_z)
    return slice(total - N_VISIBILITY_CHANNELS, total)


# --- Drawing topology ------------------------------------------------------
# Skeleton edges for the overlay renderer (spec S10.1, and the live preview in
# S4a). Expressed as positions *within our kept subsets*, never as raw MediaPipe
# indices, because the renderer only ever sees already-subset arrays. Doing the
# raw -> subset mapping once, here, is what stops the overlay from drawing a
# plausible-looking skeleton out of the wrong points.
#
# Pinned to MediaPipe's shipped connection tables by tests/test_topology.py.

# Upper-body pose edges, in RAW MediaPipe pose indices.
_POSE_EDGES_RAW: Final[tuple[tuple[int, int], ...]] = (
    (11, 12),                                # shoulder line
    (11, 13), (13, 15),                      # left upper arm, forearm
    (12, 14), (14, 16),                      # right upper arm, forearm
    (11, 23), (12, 24), (23, 24),            # torso
    (15, 17), (15, 19), (15, 21), (17, 19),  # left hand stub
    (16, 18), (16, 20), (16, 22), (18, 20),  # right hand stub
)

# The nose (raw 0) is kept as a head-position anchor but has no edge to anything
# we retained -- MediaPipe's own face edges run to eyes and ears, which we drop.
# It renders as an isolated point, which is correct rather than an omission.

_POSE_POS: Final[dict[int, int]] = {raw: i for i, raw in enumerate(POSE_INDICES)}

POSE_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = tuple(
    (_POSE_POS[a], _POSE_POS[b]) for a, b in _POSE_EDGES_RAW
)

# Hand edges. Hands keep all 21 points, so subset position == raw index and no
# mapping is needed. Standard MediaPipe hand skeleton: wrist to each finger
# base, then along each finger, plus the knuckle line across the palm.
HAND_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (1, 5), (5, 6), (6, 7), (7, 8),            # index (joins the thumb at 1,
                                               # not the wrist -- guard test)
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm closure
)

_FACE_POS: Final[dict[int, int]] = {raw: i for i, raw in enumerate(FACE_INDICES)}


def _face_group(raw_indices: tuple[int, ...]) -> tuple[int, ...]:
    """Raw mesh indices -> their positions within FACE_INDICES."""
    return tuple(_FACE_POS[i] for i in raw_indices)


# (name, positions within FACE_INDICES, closed). `closed` draws the last point
# back to the first: the lips are a ring, the eyebrows are open chains, and the
# eye corners are isolated points with no contour between them.
FACE_GROUPS: Final[tuple[tuple[str, tuple[int, ...], bool], ...]] = (
    ("lips", _face_group(LIPS_OUTER_INDICES), True),
    ("left_eyebrow", _face_group(LEFT_EYEBROW_INDICES), False),
    ("right_eyebrow", _face_group(RIGHT_EYEBROW_INDICES), False),
    ("eye_corners", _face_group(EYE_CORNER_INDICES), False),
)

# Groups whose points are drawn but not joined by a contour.
FACE_POINT_ONLY_GROUPS: Final[frozenset[str]] = frozenset({"eye_corners"})
