"""Typed experiment configuration, loaded from the YAML files in msl/configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from msl.data.landmarks import feature_dim


@dataclass(frozen=True)
class FeatureConfig:
    """Shape of the per-frame landmark vector."""

    # DEFAULT FALSE, from measurement (spec S4.3): dropping z is the largest
    # effect measured on WLASL100 -- +7.10pp top-1 on the transformer, +11.34pp
    # on the TCN, non-overlapping ranges on both. z is not merely uninformative,
    # it is UNSCALED: 33.1% of the feature vector at 1.5x the magnitude of x and
    # y, because S4.2 correctly excludes it from the inter-shoulder divisor.
    # The landmark cache still stores z regardless (extract.CACHE_USE_Z); this
    # flag only decides whether training consumes it.
    use_z: bool = False
    use_velocity: bool = False
    # Multiplier applied to the z channel when use_z is True. None leaves z at
    # its raw scale, which is the condition spec S4.3 measured as harmful. The
    # value is a FITTED PARAMETER and must be fitted on the training split only.
    z_scale: float | None = None
    # Keep z for these landmark blocks only ("pose", "left_hand", "right_hand",
    # "face"); None means the uniform use_z layout. Set this and z is selected
    # per block AFTER augmentation, so the augmentation invariants -- which
    # assume a uniform coordinate count -- are never exercised on a mixed layout.
    z_blocks: tuple[str, ...] | None = None
    seq_len: int = 64  # every clip is resampled to this many frames

    @property
    def n_features(self) -> int:
        if self.z_blocks is not None:
            from msl.data.landmarks import feature_dim_for

            base = feature_dim_for(tuple(self.z_blocks))
            return base + (base - 2) if self.use_velocity else base
        from msl.data.landmarks import feature_dim_with_velocity

        return feature_dim_with_velocity(self.use_z, self.use_velocity)


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "transformer"  # transformer | bilstm | tcn
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.3
    tcn_kernel_size: int = 5
    proj_dim: int = 0


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 200
    warmup_steps: int = 20
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 25
    patience: int = 5
    early_stopping: bool = True
    seed: int = 1234
    num_workers: int = 0
    contrastive_weight: float = 0.0
    contrastive_temp: float = 0.07
    out_dir: str = "checkpoints"
    log_dir: str = "runs"
    tensorboard: bool = True


@dataclass(frozen=True)
class DataConfig:
    """Where video, landmark cache and MediaPipe model bundles live.

    `data_root` is deliberately configurable rather than hardcoded: AUTSL plus
    its extracted .npy cache runs to tens of GB, which will not sit on the
    system drive of every machine this runs on. Point it at an external drive.
    """

    data_root: str = "data"
    # Defaults to <data_root>/landmark_cache when None -- see resolved_cache_dir.
    cache_dir: str | None = None
    # MediaPipe .task bundles. Not vendored: 13.7 MB of binary has no business
    # in git, and the download is a one-off (see extract.py for the URL).
    assets_dir: str = "assets"
    holistic_task: str = "holistic_landmarker.task"

    # Below this inter-shoulder distance (in normalised image units) the scale
    # divisor is treated as degenerate and the frame is marked invalid rather
    # than dividing and producing infinities. Roughly a shoulder width of 0.1%
    # of frame width -- far smaller than any real signer, so this only fires on
    # genuinely broken pose output.
    min_shoulder_distance: float = 1e-3

    # Drop WLASL's black separator frames from the labelled range at load time
    # (msl/data/separator_trim.py). A data fix, not a model change: the clip is
    # kept and only the frames go. Off by default so the reference variant stays
    # reproducible.
    drop_separator_frames: bool = False

    min_pose_confidence: float = 0.5
    min_hand_confidence: float = 0.5
    min_face_confidence: float = 0.5

    @property
    def resolved_cache_dir(self) -> Path:
        if self.cache_dir is not None:
            return Path(self.cache_dir)
        return Path(self.data_root) / "landmark_cache"

    @property
    def holistic_task_path(self) -> Path:
        return Path(self.assets_dir) / self.holistic_task


@dataclass(frozen=True)
class AugmentConfigSection:
    """Spec S5 augmentation. Off by default so an un-augmented run stays the
    baseline; see msl/data/augment.py for what each parameter does and why
    mirroring is opt-in."""

    enabled: bool = False
    speed_min: float = 0.8
    speed_max: float = 1.2
    frame_dropout: float = 0.05
    rotation_degrees: float = 10.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    joint_noise_std: float = 0.01
    mirror: bool = False
    mirror_p: float = 0.5


@dataclass(frozen=True)
class SyntheticConfig:
    """Random-tensor data source (spec S3a.1).

    Proves shapes flow through the model, loss computes, the loop runs,
    checkpointing works and ONNX export succeeds -- with no video files.
    """

    n_train: int = 128
    n_val: int = 32
    num_classes: int = 5
    seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    augment: AugmentConfigSection = field(default_factory=AugmentConfigSection)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        sections: dict[str, type] = {
            "feature": FeatureConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "data": DataConfig,
            "synthetic": SyntheticConfig,
            "augment": AugmentConfigSection,
        }
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")

        kwargs: dict[str, Any] = {}
        for name, section_cls in sections.items():
            values = raw.get(name) or {}
            if not isinstance(values, dict):
                raise TypeError(f"config section {name!r} must be a mapping")
            allowed = {f.name for f in fields(section_cls)}
            bad = set(values) - allowed
            if bad:
                raise ValueError(f"unknown key(s) in section {name!r}: {sorted(bad)}")
            kwargs[name] = section_cls(**values)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
