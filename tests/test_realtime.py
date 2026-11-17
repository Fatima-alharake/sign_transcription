from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msl.config import ExperimentConfig, ModelConfig
from msl.data import landmarks as lm
from msl.data.extract import FrameLandmarks
from msl.data.realtime import _mirror_landmarks, _should_quit
from msl.infer.realtime import WlaslSignPredictor, prepare_window
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
