"""Live isolated-sign predictions from a trained WLASL checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
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


class MotionGate:
    """Detects whether signer is actively executing hand gestures or resting."""

    def __init__(
        self,
        *,
        motion_threshold: float = 0.008,
        min_active_frames: int = 2,
        idle_frames_threshold: int = 8,
    ) -> None:
        if motion_threshold <= 0:
            raise ValueError("motion_threshold must be positive")
        self.motion_threshold = motion_threshold
        self.min_active_frames = min_active_frames
        self.idle_frames_threshold = idle_frames_threshold

        self.last_left_hand: np.ndarray | None = None
        self.last_right_hand: np.ndarray | None = None
        self.consecutive_active: int = 0
        self.consecutive_idle: int = 0
        self.is_active: bool = False
        self.current_energy: float = 0.0

    def update(self, frame: FrameLandmarks) -> bool:
        """Update motion detector with the current frame and return whether signing is active."""
        displacements: list[float] = []

        if frame.left_hand is not None and self.last_left_hand is not None:
            diff = frame.left_hand[:, :2] - self.last_left_hand[:, :2]
            dist = np.linalg.norm(diff, axis=1).mean()
            displacements.append(float(dist))

        if frame.right_hand is not None and self.last_right_hand is not None:
            diff = frame.right_hand[:, :2] - self.last_right_hand[:, :2]
            dist = np.linalg.norm(diff, axis=1).mean()
            displacements.append(float(dist))

        self.last_left_hand = frame.left_hand.copy() if frame.left_hand is not None else None
        self.last_right_hand = frame.right_hand.copy() if frame.right_hand is not None else None

        has_hands = (frame.left_hand is not None or frame.right_hand is not None)
        if not has_hands:
            instant_energy = 0.0
        elif displacements:
            instant_energy = max(displacements)
        else:
            # Hands appeared freshly in this frame
            instant_energy = self.motion_threshold * 1.5

        # Exponential moving average smoothing for energy display
        self.current_energy = 0.5 * self.current_energy + 0.5 * instant_energy

        is_moving_now = has_hands and instant_energy >= self.motion_threshold
        if is_moving_now:
            self.consecutive_active += 1
            self.consecutive_idle = 0
            if self.consecutive_active >= self.min_active_frames:
                self.is_active = True
        else:
            self.consecutive_idle += 1
            self.consecutive_active = 0
            if self.consecutive_idle >= self.idle_frames_threshold:
                self.is_active = False

        return self.is_active


class SubtitleAccumulator:
    """Debounces top sign predictions into a running subtitle sentence."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.40,
        confirm_count: int = 2,
        cooldown_seconds: float = 1.2,
        idle_timeout_seconds: float = 5.0,
        max_words: int = 7,
    ) -> None:
        if min_confidence <= 0.0 or min_confidence > 1.0:
            raise ValueError("min_confidence must be in (0, 1]")
        if confirm_count < 1:
            raise ValueError("confirm_count must be at least 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be positive")

        self.min_confidence = min_confidence
        self.confirm_count = confirm_count
        self.cooldown_seconds = cooldown_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_words = max_words

        self.words: list[str] = []
        self._pending_candidate: str | None = None
        self._pending_count: int = 0
        self._last_committed_word: str | None = None
        self._last_committed_time: float = 0.0
        self._last_active_time: float = 0.0

    def update(
        self,
        predictions: list[SignPrediction] | None,
        is_active: bool,
        current_time: float,
    ) -> str | None:
        """Process predictions and return a newly committed word, if any."""
        if is_active:
            self._last_active_time = current_time
            if predictions:
                top = predictions[0]
                if top.confidence >= self.min_confidence:
                    if top.gloss == self._pending_candidate:
                        self._pending_count += 1
                    else:
                        self._pending_candidate = top.gloss
                        self._pending_count = 1

                    if self._pending_count >= self.confirm_count:
                        time_since_commit = current_time - self._last_committed_time
                        is_different_sign = (self._pending_candidate != self._last_committed_word)
                        if is_different_sign or time_since_commit >= self.cooldown_seconds:
                            committed = self._pending_candidate
                            self.words.append(committed)
                            if len(self.words) > self.max_words:
                                self.words.pop(0)
                            self._last_committed_word = committed
                            self._last_committed_time = current_time
                            self._pending_candidate = None
                            self._pending_count = 0
                            return committed
                else:
                    self._pending_candidate = None
                    self._pending_count = 0
        else:
            self._pending_candidate = None
            self._pending_count = 0
            if self.words and (current_time - self._last_active_time >= self.idle_timeout_seconds):
                self.clear()

        return None

    def clear(self) -> None:
        self.words.clear()
        self._pending_candidate = None
        self._pending_count = 0
        self._last_committed_word = None

    def get_text(self) -> str:
        return "  ".join(self.words)



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
