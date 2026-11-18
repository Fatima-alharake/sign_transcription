from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msl.config import ExperimentConfig, ModelConfig
from msl.data import landmarks as lm
from msl.data.extract import FrameLandmarks
from msl.data.realtime import _mirror_landmarks, _should_clear, _should_quit
from msl.infer.realtime import (
    MotionGate,
    SignPrediction,
    SubtitleAccumulator,
    WlaslSignPredictor,
    prepare_window,
)
from msl.models.build import build_model, save_checkpoint


def test_mirror_landmarks_flips_only_x_and_does_not_mutate_input() -> None:
    pose = np.array([[0.2, 0.3, 0.4]])
    mirrored = _mirror_landmarks(FrameLandmarks(pose=pose))

    assert mirrored.pose is not None
    assert mirrored.pose[0].tolist() == pytest.approx([0.8, 0.3, 0.4])
    assert pose[0].tolist() == pytest.approx([0.2, 0.3, 0.4])
    assert mirrored.left_hand is None


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
def test_quit_key_accepts_q_in_both_cases_and_escape(key: int) -> None:
    assert _should_quit(key)


def test_quit_key_ignores_other_keys() -> None:
    assert not _should_quit(ord("a"))


def _visible_pose() -> FrameLandmarks:
    pose = np.full((lm.N_POSE, 3), 0.5, dtype=np.float64)
    pose[lm.LEFT_SHOULDER_POS] = (0.6, 0.4, 0.0)
    pose[lm.RIGHT_SHOULDER_POS] = (0.4, 0.4, 0.0)
    return FrameLandmarks(pose=pose)


def test_prepare_window_matches_checkpoint_feature_shape() -> None:
    cfg = ExperimentConfig()
    frames = [_visible_pose()] * cfg.feature.seq_len

    sequence = prepare_window(frames, cfg, width=640, height=480)

    assert sequence is not None
    assert sequence.shape == (cfg.feature.seq_len, cfg.feature.n_features)
    assert sequence.dtype == np.float32


def test_prepare_window_skips_windows_without_enough_pose() -> None:
    sequence = prepare_window(
        [FrameLandmarks()] * 64,
        ExperimentConfig(),
        width=640,
        height=480,
    )

    assert sequence is None


def test_wlasl_predictor_loads_checkpoint_and_returns_glosses(
    tmp_path: Path,
) -> None:
    cfg = ExperimentConfig()
    cfg = ExperimentConfig(
        feature=cfg.feature,
        model=ModelConfig(
            arch="transformer",
            d_model=16,
            n_layers=1,
            n_heads=2,
            ffn_dim=32,
            dropout=0.0,
            tcn_kernel_size=5,
        ),
        train=cfg.train,
        data=cfg.data,
        synthetic=cfg.synthetic,
        augment=cfg.augment,
    )
    model = build_model(cfg, num_classes=2)
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, model, cfg, ["BOOK", "DRINK"], 1, 0.5)
    predictor = WlaslSignPredictor(checkpoint, device="cpu", fps=30)

    predictions = predictor.predict(
        [_visible_pose()] * predictor.window_frames,
        width=640,
        height=480,
        top_k=2,
    )

    assert predictions is not None
    assert {prediction.gloss for prediction in predictions} == {"BOOK", "DRINK"}
    assert predictions[0].confidence >= predictions[1].confidence


@pytest.mark.parametrize("key", [ord(" "), ord("c"), ord("C")])
def test_clear_key_accepts_space_and_c(key: int) -> None:
    assert _should_clear(key)


def test_clear_key_ignores_other_keys() -> None:
    assert not _should_clear(ord("x"))
    assert not _should_clear(ord("q"))


def test_motion_gate_detects_activity_and_idleness() -> None:
    gate = MotionGate(motion_threshold=0.01, min_active_frames=2, idle_frames_threshold=3)

    # Frame with no hands -> idle
    empty_frame = FrameLandmarks()
    assert not gate.update(empty_frame)

    # Stationary hand at constant position (0.5, 0.5)
    static_hand = np.full((21, 3), 0.5, dtype=np.float64)
    frame_static = FrameLandmarks(right_hand=static_hand)

    # First frame initializes reference
    gate.update(frame_static)
    # Subsequent static frames have zero displacement -> idle
    for _ in range(5):
        gate.update(frame_static)
    assert not gate.is_active

    # Moving hand: displacement ~0.05 per frame
    pos = 0.5
    for i in range(5):
        pos += 0.05
        moving_hand = np.full((21, 3), pos, dtype=np.float64)
        gate.update(FrameLandmarks(right_hand=moving_hand))
    assert gate.is_active

    # Hand stops moving -> transitions to idle after idle_frames_threshold
    final_hand = np.full((21, 3), pos, dtype=np.float64)
    for _ in range(5):
        gate.update(FrameLandmarks(right_hand=final_hand))
    assert not gate.is_active


def test_subtitle_accumulator_confirms_debounces_and_clears() -> None:
    acc = SubtitleAccumulator(
        min_confidence=0.40,
        confirm_count=2,
        cooldown_seconds=1.0,
        idle_timeout_seconds=5.0,
    )
    assert acc.get_text() == ""

    # Low confidence predictions (< 0.40) should be ignored
    low_pred = [SignPrediction("BOOK", 0.35)]
    assert acc.update(low_pred, is_active=True, current_time=0.0) is None
    assert acc.get_text() == ""

    # High confidence prediction first appearance (count 1 < confirm_count 2)
    book_pred = [SignPrediction("BOOK", 0.55)]
    assert acc.update(book_pred, is_active=True, current_time=0.1) is None
    assert acc.get_text() == ""

    # Second appearance confirms "BOOK"
    committed = acc.update(book_pred, is_active=True, current_time=0.2)
    assert committed == "BOOK"
    assert acc.get_text() == "BOOK"

    # Immediate repetition during cooldown is blocked
    assert acc.update(book_pred, is_active=True, current_time=0.5) is None
    assert acc.get_text() == "BOOK"

    # Different sign "DRINK" is accepted upon confirmation without waiting for same-sign cooldown
    drink_pred = [SignPrediction("DRINK", 0.60)]
    acc.update(drink_pred, is_active=True, current_time=0.6)
    committed2 = acc.update(drink_pred, is_active=True, current_time=0.7)
    assert committed2 == "DRINK"
    assert acc.get_text() == "BOOK  DRINK"

    acc.clear()
    assert acc.get_text() == ""


def test_subtitle_accumulator_idle_timeout() -> None:
    acc = SubtitleAccumulator(
        min_confidence=0.40,
        confirm_count=1,
        idle_timeout_seconds=2.0,
    )
    acc.update([SignPrediction("HELLO", 0.8)], is_active=True, current_time=1.0)
    assert acc.get_text() == "HELLO"

    # Still within idle timeout (< 2.0s)
    acc.update(None, is_active=False, current_time=2.0)
    assert acc.get_text() == "HELLO"

    # Exceeds idle timeout (>= 2.0s since last active) -> auto clears
    acc.update(None, is_active=False, current_time=3.5)
    assert acc.get_text() == ""

