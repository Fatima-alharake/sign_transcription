"""Sequence backbones over landmark frames.

Every backbone maps `(B, T, n_features)` to a pooled `(B, d_model)` embedding
and knows *nothing* about the number of classes. That separation is deliberate
(spec S3a.3): the plan is to pretrain on a proxy corpus like AUTSL and later
fine-tune on Mongolian data, which means swapping a 226-class head for an
MSL-sized one while keeping the backbone weights. If the classifier lived in
here, that swap would mean surgery on a state dict.

Three real baselines -- transformer, BiLSTM, TCN -- plus an ST-GCN stub that
lands in Milestone 4 when the skeleton adjacency from extract.py exists.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from msl.config import ModelConfig


class SequenceBackbone(nn.Module):
    """Base class fixing the (B, T, F) -> (B, d_model) contract."""

    d_model: int

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class TransformerBackbone(SequenceBackbone):
    """Small transformer encoder (spec S6).

    Linear projection to d_model, learned positional embeddings (sequences are
    short and fixed-length, so sinusoids buy nothing), then mean-pool over time.
    """

    def __init__(self, n_features: int, cfg: ModelConfig, seq_len: int) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_features, cfg.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.input_dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # more stable at this depth without careful warmup
        )
        # enable_nested_tensor=False: the fast path silently changes behaviour
        # under ONNX export and we always pass dense, equal-length sequences.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"expected {self.seq_len} frames, got {x.shape[1]}; clips are "
                "resampled to a fixed length before batching"
            )
        h = self.input_proj(x) + self.pos_embedding
        h = self.input_dropout(h)
        h = self.encoder(h)
        return self.norm(h.mean(dim=1))


class BiLSTMBackbone(SequenceBackbone):
    """Bidirectional LSTM baseline (spec S6, second baseline).

    Hidden size is d_model // 2 per direction so the concatenated output is
    d_model wide and heads are interchangeable across backbones.
    """

    def __init__(self, n_features: int, cfg: ModelConfig, seq_len: int) -> None:
        super().__init__()
        if cfg.d_model % 2 != 0:
            raise ValueError("d_model must be even for a bidirectional LSTM")
        self.d_model = cfg.d_model

        self.input_proj = nn.Linear(n_features, cfg.d_model)
        self.lstm = nn.LSTM(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model // 2,
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.lstm(self.input_proj(x))
        return self.norm(h.mean(dim=1))


class _TemporalBlock(nn.Module):
    """Dilated residual temporal-conv block.

    Non-causal: isolated sign classification sees the whole clip at once, so
    there is no reason to hide the future. The CTC head for continuous signing
    (Milestone 8) may want a causal variant for streaming; that is a separate
    flag when it arrives.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # 'same' length
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.dropout(self.act(self.norm1(self.conv1(x))))
        h = self.dropout(self.act(self.norm2(self.conv2(h))))
        return x + h


class TCNBackbone(SequenceBackbone):
    """Temporal convolutional network baseline (spec S6, third baseline).

    Receptive field grows exponentially through dilation 1, 2, 4, ... so a
    handful of layers covers the full 64-frame clip while staying far cheaper
    than attention.
    """

    def __init__(self, n_features: int, cfg: ModelConfig, seq_len: int) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        if cfg.tcn_kernel_size % 2 == 0:
            raise ValueError("tcn_kernel_size must be odd to preserve sequence length")

        self.input_proj = nn.Linear(n_features, cfg.d_model)
        self.blocks = nn.ModuleList(
            _TemporalBlock(cfg.d_model, cfg.tcn_kernel_size, 2**i, cfg.dropout)
            for i in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x).transpose(1, 2)  # (B, C, T) for Conv1d
        for block in self.blocks:
            h = block(h)
        return self.norm(h.mean(dim=2))


class STGCNBackbone(SequenceBackbone):
    """Spatio-temporal graph conv over the hand/body skeleton. Stub.

    Deferred to Milestone 4: it needs a real adjacency matrix over the landmark
    graph, which only exists once extract.py pins the joint ordering.
    """

    def __init__(self, n_features: int, cfg: ModelConfig, seq_len: int) -> None:
        super().__init__()
        raise NotImplementedError(
            "ST-GCN is stubbed until Milestone 4 (needs the skeleton adjacency "
            "from extract.py). Use --arch transformer, bilstm or tcn."
        )


ARCHITECTURES: dict[str, type[SequenceBackbone]] = {
    "transformer": TransformerBackbone,
    "bilstm": BiLSTMBackbone,
    "tcn": TCNBackbone,
    "stgcn": STGCNBackbone,
}


def build_backbone(n_features: int, cfg: ModelConfig, seq_len: int) -> SequenceBackbone:
    try:
        backbone_cls = ARCHITECTURES[cfg.arch]
    except KeyError:
        raise ValueError(
            f"unknown arch {cfg.arch!r}; choose from {sorted(ARCHITECTURES)}"
        ) from None
    return backbone_cls(n_features=n_features, cfg=cfg, seq_len=seq_len)
