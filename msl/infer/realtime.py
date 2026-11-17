"""Live isolated-sign predictions from a trained WLASL checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from msl.config import ExperimentConfig
from msl.data import landmarks as lm
from msl.data.extract import FrameLandmarks, normalise_frame
from msl.data.vocab import Vocab
from msl.data.wlasl_loader import DEFAULT_MIN_POSE_RATE, resample
from msl.models.build import build_model, load_checkpoint
from msl.runtime import resolve_device


@dataclass(frozen=True)
class SignPrediction:
    gloss: str
    confidence: float


def prepare_window(
    frames: Sequence[FrameLandmarks],
    cfg: ExperimentConfig,
    *,
    width: int,
    height: int,
    min_pose_rate: float = DEFAULT_MIN_POSE_RATE,
) -> np.ndarray | None:
    """Convert raw frames to the same fixed-size features used for training."""
    if not frames:
        return None
    if not 0.0 <= min_pose_rate <= 1.0:
        raise ValueError("min_pose_rate must be between 0 and 1")

    vectors: list[np.ndarray] = []
    pose_frames = 0
    for frame in frames:
        vector, pose_present = normalise_frame(
            frame,
            width=width,
            height=height,
            use_z=True,
            min_shoulder_distance=cfg.data.min_shoulder_distance,
        )
        pose_frames += int(pose_present)
        vector = vector[None, :]
        if cfg.feature.z_blocks is not None:
            vector = lm.select_z_blocks(vector, tuple(cfg.feature.z_blocks))
        elif not cfg.feature.use_z:
            vector = lm.drop_z(vector)
        elif cfg.feature.z_scale is not None:
            vector = lm.scale_z(vector, cfg.feature.z_scale)
        vectors.append(vector[0])

    if pose_frames / len(frames) < min_pose_rate:
        return None

    sequence = np.stack(vectors)
    if sequence.shape[1] != cfg.feature.n_features:
        raise ValueError(
            f"prepared feature width {sequence.shape[1]} does not match "
            f"checkpoint width {cfg.feature.n_features}"
        )
    return resample(sequence, cfg.feature.seq_len)


class WlaslSignPredictor:
    """Load a WLASL classifier and predict over a rolling webcam window."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        fps: float = 30.0,
        window_seconds: float = 2.5,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.device = resolve_device(device)
        checkpoint = load_checkpoint(checkpoint_path, map_location=str(self.device))
        raw_config = checkpoint.get("config")
        raw_glosses = checkpoint.get("glosses")
        if not isinstance(raw_config, dict):
            raise ValueError("checkpoint is missing its experiment config")
        if not isinstance(raw_glosses, list) or not all(
            isinstance(gloss, str) for gloss in raw_glosses
        ):
            raise ValueError("checkpoint is missing its ordered gloss vocabulary")

        self.cfg = ExperimentConfig.from_dict(raw_config)
        self.vocab = Vocab(raw_glosses)
        self.model = build_model(self.cfg, num_classes=len(self.vocab)).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        self.window_frames = max(
            self.cfg.feature.seq_len,
            round(fps * window_seconds),
        )
        self.prediction_interval = max(1, round(fps / 4))

    def predict(
        self,
        frames: Sequence[FrameLandmarks],
        *,
        width: int,
        height: int,
        min_pose_rate: float = DEFAULT_MIN_POSE_RATE,
        top_k: int = 3,
    ) -> list[SignPrediction] | None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if len(frames) < self.window_frames:
            return None
        sequence = prepare_window(
            frames,
            self.cfg,
            width=width,
            height=height,
            min_pose_rate=min_pose_rate,
        )
        if sequence is None:
            return None

        batch = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(batch)[0].softmax(dim=0)
            scores, indices = probabilities.topk(min(top_k, len(self.vocab)))
        return [
            SignPrediction(
                self.vocab.to_gloss(int(index)),
                float(score),
            )
            for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
        ]
