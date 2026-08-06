"""Assembling a backbone and a head into a trainable classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from msl.config import ExperimentConfig
from msl.models.encoder import SequenceBackbone, build_backbone
from msl.models.heads import ClassificationHead


class SignClassifier(nn.Module):
    """Backbone + classification head, checkpointed as two separate subtrees.

    `state_dict()` naturally namespaces these as `backbone.*` and `head.*`,
    which is what makes `load_backbone_only` below a one-liner: fine-tuning
    from a proxy-language checkpoint keeps `backbone.*` and discards `head.*`.
    """

    def __init__(self, backbone: SequenceBackbone, head: ClassificationHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))

    def embed(self, x: Tensor) -> Tensor:
        """Pooled embedding without the classifier -- for probing and retrieval."""
        return self.backbone(x)


def build_model(cfg: ExperimentConfig, num_classes: int) -> SignClassifier:
    backbone = build_backbone(
        n_features=cfg.feature.n_features,
        cfg=cfg.model,
        seq_len=cfg.feature.seq_len,
    )
    head = ClassificationHead(
        d_model=backbone.d_model,
        num_classes=num_classes,
        dropout=cfg.model.dropout,
    )
    return SignClassifier(backbone, head)


def load_backbone_only(model: SignClassifier, checkpoint_path: str | Path) -> list[str]:
    """Load pretrained backbone weights, leaving the head randomly initialised.

    This is the fine-tuning path from spec S3a.3: train on AUTSL, then adapt to
    a Mongolian vocabulary of a different size. Returns the backbone keys that
    were missing from the checkpoint, so a silent mismatch is visible.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    backbone_state = {
        key[len("backbone.") :]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(f"no backbone.* weights found in {checkpoint_path}")
    incompatible = model.backbone.load_state_dict(backbone_state, strict=False)
    return list(incompatible.missing_keys)


def save_checkpoint(
    path: str | Path,
    model: SignClassifier,
    cfg: ExperimentConfig,
    glosses: list[str],
    step: int,
    best_val_acc: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Write a checkpoint atomically.

    Written to a temp file and renamed, so an interrupted save cannot leave a
    truncated checkpoint where a valid one used to be.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "glosses": glosses,  # a checkpoint is meaningless without its vocab order
        "step": step,
        "best_val_acc": best_val_acc,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=True)
