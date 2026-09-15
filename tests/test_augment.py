"""Augmentation invariants (spec S5).

These guard the three ways augmentation can corrupt a landmark tensor without
raising anything: fabricating a hand that was absent, filling a pose-less frame
with noise, and treating the visibility flags as coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest

from msl.data import landmarks as lm
from msl.data.augment import AugmentConfig, augment_sequence

USE_Z = True
F = lm.feature_dim(USE_Z)
COORD_END = lm.N_LANDMARKS * lm.coords_per_landmark(USE_Z)


def make_sequence(n_frames: int = 40, seed: int = 0) -> np.ndarray:
    """A plausible clip: both hands present, shoulders at +/-0.5, no dead frames."""
    rng = np.random.default_rng(seed)
    seq = rng.normal(0.0, 0.3, size=(n_frames, F)).astype(np.float32)
    seq[:, lm.visibility_slice(USE_Z)] = 1.0
    return seq


def full_config(**overrides) -> AugmentConfig:
    base = dict(enabled=True, speed_min=0.8, speed_max=1.2, frame_dropout=0.05,
                rotation_degrees=10.0, scale_min=0.9, scale_max=1.1,
                joint_noise_std=0.01, mirror=False)
    base.update(overrides)
    return AugmentConfig(**base)


def test_disabled_is_identity():
    seq = make_sequence()
    out = augment_sequence(seq, AugmentConfig(enabled=False), USE_Z,
                           np.random.default_rng(0))
    np.testing.assert_array_equal(out, seq)


def test_absent_hand_stays_exactly_zero():
    """A zeroed hand block with visibility 0 must survive untouched.

    This is the failure the visibility mask exists to prevent: noise on a
    zeroed block turns 'hand absent' into 'hand at a specific position'.
    """
    seq = make_sequence()
    left = lm.block_slice("left_hand", USE_Z)
    seq[:, left] = 0.0
    seq[:, lm.visibility_slice(USE_Z).start] = 0.0  # left hand flag off

    out = augment_sequence(seq, full_config(), USE_Z, np.random.default_rng(1))
    assert np.all(out[:, left] == 0.0)
    # and the hand that WAS present must actually have been transformed
    right = lm.block_slice("right_hand", USE_Z)
    assert not np.allclose(out[:, right], 0.0)


def test_poseless_frame_stays_exactly_zero():
    """An all-zero frame means 'no usable pose' and must stay all-zero.

    rates_from_features reads pose presence from exactly this property.
    """
    seq = make_sequence()
    seq[5] = 0.0
    seq[11] = 0.0
    out = augment_sequence(seq, full_config(frame_dropout=0.0, speed_min=1.0,
                                            speed_max=1.0), USE_Z,
                           np.random.default_rng(2))
    zero_rows = [i for i in range(out.shape[0]) if np.all(out[i] == 0.0)]
    assert 5 in zero_rows and 11 in zero_rows


def test_visibility_channels_are_not_transformed():
    seq = make_sequence()
    vis = lm.visibility_slice(USE_Z)
    seq[:, vis] = [1.0, 0.0]
    seq[:, lm.block_slice("right_hand", USE_Z)] = 0.0
    out = augment_sequence(seq, full_config(frame_dropout=0.0, speed_min=1.0,
                                            speed_max=1.0), USE_Z,
                           np.random.default_rng(3))
    assert set(np.unique(out[:, vis])) <= {0.0, 1.0}
    np.testing.assert_array_equal(out[:, vis][:, 0], 1.0)
    np.testing.assert_array_equal(out[:, vis][:, 1], 0.0)


def test_rotation_preserves_distance_from_origin():
    """Rotation about the camera axis is rigid in the x,y plane."""
    seq = make_sequence(n_frames=12)
    cfg = full_config(frame_dropout=0.0, speed_min=1.0, speed_max=1.0,
                      scale_min=1.0, scale_max=1.0, joint_noise_std=0.0,
                      rotation_degrees=30.0)
    out = augment_sequence(seq, cfg, USE_Z, np.random.default_rng(4))
    c = lm.coords_per_landmark(USE_Z)
    before = seq[:, :COORD_END].reshape(-1, lm.N_LANDMARKS, c)
    after = out[:, :COORD_END].reshape(-1, lm.N_LANDMARKS, c)
    r_before = np.hypot(before[..., 0], before[..., 1])
    r_after = np.hypot(after[..., 0], after[..., 1])
    np.testing.assert_allclose(r_before, r_after, atol=1e-5)
    # z is untouched by a rotation about the camera axis
    np.testing.assert_allclose(before[..., 2], after[..., 2], atol=1e-6)


def test_scaling_is_uniform_across_landmarks():
    seq = make_sequence(n_frames=8)
    cfg = full_config(frame_dropout=0.0, speed_min=1.0, speed_max=1.0,
                      rotation_degrees=0.0, joint_noise_std=0.0,
                      scale_min=1.5, scale_max=1.5)
    out = augment_sequence(seq, cfg, USE_Z, np.random.default_rng(5))
    np.testing.assert_allclose(out[:, :COORD_END], seq[:, :COORD_END] * 1.5,
                               atol=1e-5)


def test_temporal_ops_change_length_but_keep_width():
    seq = make_sequence(n_frames=60)
    out = augment_sequence(seq, full_config(), USE_Z, np.random.default_rng(6))
    assert out.shape[1] == F
    assert 2 <= out.shape[0] <= 60


def test_never_collapses_below_two_frames():
    seq = make_sequence(n_frames=4)
    for seed in range(40):
        out = augment_sequence(seq, full_config(frame_dropout=0.9), USE_Z,
                               np.random.default_rng(seed))
        assert out.shape[0] >= 2


def test_mirror_swaps_hands_and_their_flags():
    seq = make_sequence(n_frames=6)
    left = lm.block_slice("left_hand", USE_Z)
    right = lm.block_slice("right_hand", USE_Z)
    seq[:, right] = 0.0
    vis = lm.visibility_slice(USE_Z)
    seq[:, vis] = [1.0, 0.0]

    cfg = full_config(mirror=True, frame_dropout=0.0, speed_min=1.0,
                      speed_max=1.0, rotation_degrees=0.0, scale_min=1.0,
                      scale_max=1.0, joint_noise_std=0.0)
    out = augment_sequence(seq, cfg, USE_Z, np.random.default_rng(7))
    # the present hand moved into the right-hand block, and its flag with it
    assert np.all(out[:, left] == 0.0)
    assert not np.allclose(out[:, right], 0.0)
    np.testing.assert_array_equal(out[:, vis][:, 0], 0.0)
    np.testing.assert_array_equal(out[:, vis][:, 1], 1.0)


def test_mirror_negates_x():
    seq = make_sequence(n_frames=5)
    cfg = full_config(mirror=True, frame_dropout=0.0, speed_min=1.0,
                      speed_max=1.0, rotation_degrees=0.0, scale_min=1.0,
                      scale_max=1.0, joint_noise_std=0.0)
    out = augment_sequence(seq, cfg, USE_Z, np.random.default_rng(8))
    c = lm.coords_per_landmark(USE_Z)
    pose = lm.block_slice("pose", USE_Z)
    before = seq[:, pose].reshape(-1, lm.N_POSE, c)
    after = out[:, pose].reshape(-1, lm.N_POSE, c)
    np.testing.assert_allclose(after[..., 0], -before[..., 0], atol=1e-6)
    np.testing.assert_allclose(after[..., 1], before[..., 1], atol=1e-6)


def test_deterministic_for_a_given_seed():
    seq = make_sequence()
    a = augment_sequence(seq, full_config(), USE_Z, np.random.default_rng(99))
    b = augment_sequence(seq, full_config(), USE_Z, np.random.default_rng(99))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("use_z", [True, False])
def test_works_without_z(use_z):
    f = lm.feature_dim(use_z)
    rng = np.random.default_rng(0)
    seq = rng.normal(0, 0.3, size=(20, f)).astype(np.float32)
    seq[:, lm.visibility_slice(use_z)] = 1.0
    out = augment_sequence(seq, full_config(), use_z, np.random.default_rng(0))
    assert out.shape[1] == f
