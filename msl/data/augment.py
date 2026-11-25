"""Augmentation for landmark sequences (spec S5).

Temporal resampling, rotation about the camera axis, per-joint Gaussian noise,
frame dropout and random affine scaling. Horizontal mirroring is implemented
but **off by default** -- see `AugmentConfig.mirror`.

Three invariants this module must not break, all of them silent if violated:

1. **An absent hand stays exactly zero.** `extract.py` zeroes a missing hand
   after normalisation and sets its visibility flag to 0. Additive noise on a
   zeroed block would turn "hand absent" into "hand at a specific place", which
   is precisely the distinction the visibility mask exists to preserve.
   Rotation and scaling map 0 to 0 and are safe; noise is not, so noise is
   masked.
2. **A pose-less frame stays exactly all-zero.** `normalise_frame` returns an
   all-zero vector exactly when there was no usable pose. That invariant is
   what `rates_from_features` reads pose presence from, and a model that sees
   noise there learns to read a frame that carries nothing.
3. **The visibility channels are flags, never coordinates.** They are excluded
   from every geometric transform. Mirroring is the one operation that touches
   them, because swapping the hands must swap their flags too.

Every default here is a starting value, not a tuned one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from msl.data import landmarks as lm


@dataclass(frozen=True)
class AugmentConfig:
    """Augmentation parameters. All ranges are inclusive and uniform."""

    enabled: bool = False

    # Temporal. NOTE ON WHAT "SPEED" MEANS HERE: every clip is resampled to a
    # fixed seq_len downstream, so a *global* speed change is a no-op -- it
    # resamples back to the same 64 frames. Speed is therefore realised as a
    # random contiguous temporal crop covering 1/speed of the clip, which is
    # what actually varies the sign's rate and phase relative to the window.
    speed_min: float = 0.8
    speed_max: float = 1.2
    frame_dropout: float = 0.05  # fraction of frames dropped before resampling

    # Spatial.
    rotation_degrees: float = 10.0   # about the camera (z) axis
    scale_min: float = 0.9
    scale_max: float = 1.1
    joint_noise_std: float = 0.01    # in inter-shoulder units (shoulders at +/-0.5)

    # DO NOT ENABLE WITHOUT READING THIS. Sign languages have dominant-hand
    # asymmetry: mirroring turns a right-handed signer into a left-handed one,
    # which for many signs is a different production and for some may be a
    # different sign. Spec S5 requires this be opt-in with the risk stated.
    # When it is enabled the left and right hand blocks AND their visibility
    # flags must swap together; swapping only the coordinates silently
    # mislabels which hand is which.
    mirror: bool = False
    mirror_p: float = 1.0


def mirror_sequence(sequence: np.ndarray, use_z: bool) -> np.ndarray:
    out = np.array(sequence, dtype=np.float32, copy=True)
    coord_end = lm.N_LANDMARKS * lm.coords_per_landmark(use_z)
    coords = _coord_view(out[:, :coord_end], use_z)
    visibility = out[:, lm.visibility_slice(use_z)]

    coords[..., 0] *= -1.0
    left, right = _hand_block_bounds(use_z)
    c = lm.coords_per_landmark(use_z)
    lp = slice(left.start // c, left.stop // c)
    rp = slice(right.start // c, right.stop // c)
    swapped = coords.copy()
    swapped[:, lp, :] = coords[:, rp, :]
    swapped[:, rp, :] = coords[:, lp, :]
    coords[...] = swapped
    visibility[:, [0, 1]] = visibility[:, [1, 0]]
    out[:, :coord_end] = coords.reshape(out.shape[0], coord_end)
    return out


def _coord_view(frame_block: np.ndarray, use_z: bool) -> np.ndarray:
    """Reshape the coordinate region of (T, F) into (T, n_landmarks, c)."""
    c = lm.coords_per_landmark(use_z)
    return frame_block.reshape(frame_block.shape[0], lm.N_LANDMARKS, c)


def _hand_block_bounds(use_z: bool) -> tuple[slice, slice]:
    return lm.block_slice("left_hand", use_z), lm.block_slice("right_hand", use_z)


def augment_sequence(
    sequence: np.ndarray,
    cfg: AugmentConfig,
    use_z: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply spec S5 augmentation to a (T, F) landmark sequence.

    Temporal operations run first, on the raw-length sequence, because dropping
    and cropping frames only makes sense before the fixed-length resample. The
    caller resamples afterwards.
    """
    if not cfg.enabled or sequence.shape[0] == 0:
        return sequence

    out = np.array(sequence, dtype=np.float32, copy=True)

    # --- temporal ----------------------------------------------------------
    n = out.shape[0]
    if cfg.speed_max > 0 and n >= 4:
        speed = rng.uniform(cfg.speed_min, cfg.speed_max)
        window = int(round(n / speed))
        window = max(2, min(window, n))
        offset = int(rng.integers(0, n - window + 1)) if window < n else 0
        out = out[offset:offset + window]

    if cfg.frame_dropout > 0 and out.shape[0] >= 4:
        keep = rng.random(out.shape[0]) >= cfg.frame_dropout
        if keep.sum() >= 2:  # never drop a sequence below two frames
            out = out[keep]

    # --- spatial -----------------------------------------------------------
    # A frame that never normalised is all-zero and must stay that way.
    valid = np.abs(out).sum(axis=1) > 0

    coord_end = lm.N_LANDMARKS * lm.coords_per_landmark(use_z)
    coords = _coord_view(out[:, :coord_end], use_z)
    visibility = out[:, lm.visibility_slice(use_z)]

    if cfg.rotation_degrees > 0:
        theta = np.deg2rad(rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees))
        cos, sin = np.cos(theta), np.sin(theta)
        x = coords[..., 0].copy()
        y = coords[..., 1].copy()
        # Rotation about the origin, which after normalisation IS the
        # mid-shoulder point -- so this rotates about the signer, not a corner.
        # Zero stays zero under rotation, so absent blocks survive untouched.
        coords[..., 0] = cos * x - sin * y
        coords[..., 1] = sin * x + cos * y

    if cfg.scale_max > 0:
        coords *= rng.uniform(cfg.scale_min, cfg.scale_max)

    if cfg.joint_noise_std > 0:
        noise = rng.normal(0.0, cfg.joint_noise_std, size=coords.shape)
        # Mask 1: never write noise into a frame that carries no pose.
        noise *= valid[:, None, None]
        # Mask 2: never write noise into a hand that was absent. Zeroed hands
        # carry visibility 0; noise here would fabricate a hand position.
        left, right = _hand_block_bounds(use_z)
        c = lm.coords_per_landmark(use_z)
        left_points = slice(left.start // c, left.stop // c)
        right_points = slice(right.start // c, right.stop // c)
        noise[:, left_points, :] *= (visibility[:, 0] > 0)[:, None, None]
        noise[:, right_points, :] *= (visibility[:, 1] > 0)[:, None, None]
        coords += noise

    if cfg.mirror and (rng.random() < cfg.mirror_p):
        coords[..., 0] *= -1.0
        left, right = _hand_block_bounds(use_z)
        c = lm.coords_per_landmark(use_z)
        lp = slice(left.start // c, left.stop // c)
        rp = slice(right.start // c, right.stop // c)
        swapped = coords.copy()
        swapped[:, lp, :] = coords[:, rp, :]
        swapped[:, rp, :] = coords[:, lp, :]
        coords[...] = swapped
        visibility[:, [0, 1]] = visibility[:, [1, 0]]

    # Re-zero any frame that was invalid: rotation and scaling preserve zero,
    # but this is the cheap guarantee rather than the inferred one.
    out[:, :coord_end] = coords.reshape(out.shape[0], coord_end)
    out[~valid] = 0.0
    return out
