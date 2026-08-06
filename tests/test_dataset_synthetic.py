"""Synthetic dataset: correct shapes, reproducible, and genuinely lazy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msl.config import ExperimentConfig
from msl.data.dataset import (
    LandmarkClipDataset,
    SyntheticLandmarkDataset,
    build_synthetic_loaders,
)
from msl.data.vocab import Vocab, placeholder_vocab


@pytest.fixture
def cfg() -> ExperimentConfig:
    return ExperimentConfig.from_yaml("msl/configs/synthetic.yaml")


def make_dataset(n: int = 8, seed: int = 0) -> SyntheticLandmarkDataset:
    return SyntheticLandmarkDataset(
        n_samples=n, seq_len=64, n_features=275, num_classes=5, seed=seed
    )


def test_item_shape_and_dtype() -> None:
    x, y = make_dataset()[0]
    assert x.shape == (64, 275)
    assert x.dtype == torch.float32
    assert isinstance(y, int)
    assert 0 <= y < 5


def test_same_seed_gives_identical_samples() -> None:
    a, b = make_dataset(seed=7), make_dataset(seed=7)
    for i in range(len(a)):
        xa, ya = a[i]
        xb, yb = b[i]
        assert torch.equal(xa, xb)
        assert ya == yb


def test_different_seeds_give_different_samples() -> None:
    x0, _ = make_dataset(seed=0)[0]
    x1, _ = make_dataset(seed=1)[0]
    assert not torch.equal(x0, x1)


def test_items_are_independent_of_access_order() -> None:
    """Per-index seeding, not a sequential RNG: item 3 is item 3 whether or not
    items 0..2 were drawn first. Without this, shuffling or adding DataLoader
    workers would silently change the data."""
    ds = make_dataset(n=6, seed=3)
    forward = [ds[i][0] for i in range(6)]
    backward = [ds[i][0] for i in reversed(range(6))][::-1]
    for a, b in zip(forward, backward):
        assert torch.equal(a, b)


def test_labels_are_not_all_identical() -> None:
    ds = make_dataset(n=64)
    assert len({ds[i][1] for i in range(len(ds))}) > 1


def test_dataset_holds_no_bulk_state() -> None:
    """O(1) memory: the corpus is generated on demand, never materialised.

    On a 16GB machine at ~90% commit, a dataset that preallocates is the
    difference between running and thrashing.
    """
    ds = SyntheticLandmarkDataset(
        n_samples=1_000_000, seq_len=64, n_features=275, num_classes=5
    )
    for value in vars(ds).values():
        assert isinstance(value, int), "dataset should hold only scalars"


def test_out_of_range_index_raises() -> None:
    ds = make_dataset(n=4)
    with pytest.raises(IndexError):
        ds[4]


def test_rejects_degenerate_configuration() -> None:
    with pytest.raises(ValueError):
        SyntheticLandmarkDataset(0, 64, 275, 5)
    with pytest.raises(ValueError):
        SyntheticLandmarkDataset(8, 64, 275, num_classes=1)


def test_loaders_batch_to_the_configured_shape(cfg: ExperimentConfig) -> None:
    train_loader, val_loader = build_synthetic_loaders(cfg)
    x, y = next(iter(train_loader))
    assert x.shape == (cfg.train.batch_size, cfg.feature.seq_len, cfg.feature.n_features)
    assert y.shape == (cfg.train.batch_size,)
    assert len(val_loader.dataset) == cfg.synthetic.n_val


def test_train_and_val_are_disjoint(cfg: ExperimentConfig) -> None:
    """Different seeds, so validation is not a slice of training data."""
    train_loader, val_loader = build_synthetic_loaders(cfg)
    first_train = train_loader.dataset[0][0]
    first_val = val_loader.dataset[0][0]
    assert not torch.equal(first_train, first_val)


def test_real_data_path_is_an_explicit_stub(cfg: ExperimentConfig) -> None:
    with pytest.raises(NotImplementedError, match="Milestone 2"):
        LandmarkClipDataset("manifest.csv", cfg)


# --- vocabulary ------------------------------------------------------------


def test_placeholder_glosses_cannot_be_mistaken_for_signs() -> None:
    """Spec S4a: never generate or invent MSL vocabulary. Synthetic-mode labels
    must be obviously non-lexical wherever they surface -- logs, checkpoints,
    confusion matrices."""
    vocab = placeholder_vocab(5)
    assert len(vocab) == 5
    assert all(g.startswith("SYNTH_") for g in vocab.glosses)
    assert all(g.isascii() for g in vocab.glosses)  # no Cyrillic: not a gloss


def test_vocab_roundtrips_ids_and_glosses() -> None:
    vocab = Vocab(["БИ", "НЭР", "ЮУ"])
    for i, gloss in enumerate(vocab.glosses):
        assert vocab.to_id(gloss) == i
        assert vocab.to_gloss(i) == gloss


def test_vocab_rejects_duplicates_and_empty() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Vocab(["БИ", "БИ"])
    with pytest.raises(ValueError, match="empty"):
        Vocab([])


def test_vocab_json_roundtrip_preserves_order(tmp_path) -> None:
    """Order is load-bearing: it fixes the classifier's column order."""
    vocab = Vocab(["ЮУ", "БИ", "НЭР"])
    path = tmp_path / "vocab.json"
    vocab.save_json(path)
    assert Vocab.load_json(path).glosses == ["ЮУ", "БИ", "НЭР"]


def test_vocab_from_file_skips_comments_and_blanks(tmp_path) -> None:
    path = tmp_path / "glosses.txt"
    path.write_text("# supplied by operator\n\nБИ\nНЭР\n\n", encoding="utf-8")
    assert Vocab.from_file(path).glosses == ["БИ", "НЭР"]
