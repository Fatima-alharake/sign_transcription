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
    def __init__(
        self,
        backbone: SequenceBackbone,
        head: ClassificationHead,
        proj_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.proj_head = proj_head

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))

    def embed(self, x: Tensor, normalize: bool = False) -> Tensor:
        h = self.backbone(x)
        if normalize:
            return torch.nn.functional.normalize(h, p=2, dim=-1)
        return h

    def project(self, x: Tensor, normalize: bool = True) -> Tensor:
        h = self.backbone(x)
        return self.project_embedding(h, normalize=normalize)

    def project_embedding(self, h: Tensor, normalize: bool = True) -> Tensor:
        z = self.proj_head(h) if self.proj_head is not None else h
        if normalize:
            return torch.nn.functional.normalize(z, p=2, dim=-1)
        return z


def build_model(
    cfg: ExperimentConfig,
    num_classes: int,
    include_proj_head: bool = False,
) -> SignClassifier:
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
    proj_head = None
    proj_dim = getattr(cfg.model, "proj_dim", 0)
    contrastive_weight = getattr(cfg.train, "contrastive_weight", 0.0)
    if include_proj_head or (proj_dim > 0 and contrastive_weight > 0):
        from msl.models.heads import ProjectionHead

        actual_dim = proj_dim if proj_dim > 0 else backbone.d_model
        proj_head = ProjectionHead(
            d_model=backbone.d_model,
            proj_dim=actual_dim,
            dropout=cfg.model.dropout,
        )
    return SignClassifier(backbone, head, proj_head=proj_head)


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
