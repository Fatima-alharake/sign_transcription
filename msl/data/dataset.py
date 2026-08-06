"""Datasets over landmark sequences.

Two sources, matching spec S3a:

* `SyntheticLandmarkDataset` -- random tensors of the correct shape, no files.
  This is the only source Milestone 0 uses.
* `LandmarkClipDataset` -- real extracted clips, lazily memory-mapped from
  cached .npy. Stubbed until Milestone 2, but the loading contract is fixed
  here so the training loop never has to change.

Memory discipline (16GB machine, typically ~90% committed): neither dataset
ever materialises the corpus. Items are produced one at a time in __getitem__
and nothing is cached across calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from msl.config import ExperimentConfig


class SyntheticLandmarkDataset(Dataset[tuple[torch.Tensor, int]]):
    """Random landmark tensors with random labels, from a fixed seed.

    Proves shapes flow through the model, loss computes, the training loop
    runs, checkpointing works and ONNX export succeeds -- with no video files
    and no MediaPipe (spec S3a.1). It teaches the model nothing; any accuracy
    number from this source is noise and must never be reported as a result.

    Samples are generated on demand from a per-index seed rather than
    preallocated, so the dataset costs O(1) memory and is identical across
    DataLoader worker counts and shuffle orders.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        n_features: int,
        num_classes: int,
        seed: int = 0,
    ) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if num_classes < 2:
            raise ValueError("need at least 2 classes")
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_features = n_features
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not 0 <= index < self.n_samples:
            raise IndexError(index)
        # Per-index RNG: reproducible independent of access order or worker id.
        rng = np.random.default_rng([self.seed, index])
        x = rng.standard_normal((self.seq_len, self.n_features), dtype=np.float32)
        y = int(rng.integers(self.num_classes))
        return torch.from_numpy(x), y


class LandmarkClipDataset(Dataset[tuple[torch.Tensor, int]]):
    """Real clips, read lazily from the .npy landmark cache.

    Not implemented until Milestone 2. The contract is fixed now:

    * one .npy per clip, path resolved from manifest.csv;
    * opened with `np.load(..., mmap_mode="r")` so only the frames actually
      touched are paged in -- never load a whole video or the whole dataset;
    * resampled to `seq_len` frames and copied to a small owned array before
      returning, so the memory map does not outlive the call;
    * augmentation applied per item inside the worker process.
    """

    def __init__(self, manifest_path: str | Path, cfg: ExperimentConfig) -> None:
        raise NotImplementedError(
            "real-data path lands in Milestone 2; Milestone 0 is synthetic-only. "
            "Use SyntheticLandmarkDataset or pass --synthetic."
        )


def build_synthetic_loaders(
    cfg: ExperimentConfig,
) -> tuple[DataLoader, DataLoader]:
    """Train and validation DataLoaders over synthetic tensors."""
    common = dict(
        seq_len=cfg.feature.seq_len,
        n_features=cfg.feature.n_features,
        num_classes=cfg.synthetic.num_classes,
    )
    train_ds = SyntheticLandmarkDataset(
        n_samples=cfg.synthetic.n_train, seed=cfg.synthetic.seed, **common
    )
    # Distinct seed so validation samples are not a subset of training ones.
    val_ds = SyntheticLandmarkDataset(
        n_samples=cfg.synthetic.n_val, seed=cfg.synthetic.seed + 10_000, **common
    )

    loader_kwargs = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if cfg.train.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2  # keep RSS down on a 16GB box

    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
    )


def infinite(loader: DataLoader):
    """Cycle a DataLoader forever, for step-based rather than epoch-based training."""
    while True:
        yield from loader
