"""Task heads attached to a backbone embedding.

Heads are separate modules with their own state-dict subtree so that a backbone
pretrained on a proxy corpus (AUTSL, 226 classes) can be reloaded under a
different vocabulary without touching the pretrained weights.
"""

from __future__ import annotations

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """Isolated sign recognition: pooled embedding -> class logits."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.fc(self.dropout(embedding))


class CTCHead(nn.Module):
    """Continuous signing: per-frame logits over vocabulary + blank. Stub.

    Deferred to Milestone 8. Note this one consumes the *unpooled* `(B, T, C)`
    sequence, not the pooled embedding, so backbones will need to expose their
    pre-pooling activations when this lands.
    """

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError(
            "CTC head lands in Milestone 8 (continuous signing). Milestone 0 "
            "covers isolated recognition only."
        )
