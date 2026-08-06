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

    use_z: bool = True
    seq_len: int = 64  # every clip is resampled to this many frames

    @property
    def n_features(self) -> int:
        return feature_dim(self.use_z)


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "transformer"  # transformer | bilstm | tcn | stgcn (stubbed)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.3  # small-data regime: regularise hard
    tcn_kernel_size: int = 5


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
    patience: int = 5  # early-stopping patience, in evaluations not steps
    seed: int = 1234
    # DataLoader workers. Machine has 16GB RAM typically ~90% committed, so each
    # worker's copy of the parent process matters; keep this low and rely on
    # lazy per-item loading rather than prefetch depth.
    num_workers: int = 0
    out_dir: str = "checkpoints"
    log_dir: str = "runs"
    tensorboard: bool = True


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
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        sections: dict[str, type] = {
            "feature": FeatureConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "synthetic": SyntheticConfig,
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
