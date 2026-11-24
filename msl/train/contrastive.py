from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = float(base_temperature)

    def forward(
        self,
        features: Tensor,
        labels: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        device = features.device

        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        elif len(features.shape) != 3:
            raise ValueError(
                f"features must have shape (B, D) or (B, n_views, D), got {features.shape}"
            )

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask` simultaneously")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("labels length must match batch_size")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            assert mask is not None
            mask = mask.float().to(device)

        contrast_count = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=-1)

        if self.contrast_mode == "one":
            anchor_feature = F.normalize(features[:, 0], p=2, dim=-1)
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_counts = mask.sum(1)
        valid_anchors = pos_counts > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / pos_counts[valid_anchors]
        return -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()


def augment_landmarks_torch(
    x: Tensor,
    scale_range: tuple[float, float] = (0.92, 1.08),
    noise_std: float = 0.008,
) -> Tensor:
    out = x.clone()
    b, _, f = out.shape
    scale = torch.empty(b, 1, 1, device=x.device).uniform_(scale_range[0], scale_range[1])
    coord_channels = f - 2 if f > 2 else f
    out[:, :, :coord_channels] *= scale
    if noise_std > 0:
        out[:, :, :coord_channels] += torch.randn_like(out[:, :, :coord_channels]) * noise_std
    return out
