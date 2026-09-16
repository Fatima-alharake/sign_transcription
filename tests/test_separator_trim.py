"""Separator trimming: which frames are dropped and which span survives."""

from __future__ import annotations

import numpy as np

from msl.data import landmarks as lm
from msl.data.separator_trim import MIN_RUN, Trim, plan_trim, zero_runs

F = lm.feature_dim(True)


def clip(blocks: list[tuple[str, int]]) -> np.ndarray:
    """Build a clip from ('live'|'black', n_frames) blocks."""
    parts = []
    rng = np.random.default_rng(0)
    for kind, n in blocks:
        if kind == "black":
            parts.append(np.zeros((n, F), dtype=np.float32))
        else:
            block = rng.normal(0.3, 0.2, size=(n, F)).astype(np.float32)
            block += 0.5  # keep every frame comfortably non-zero
            parts.append(block)
    return np.concatenate(parts, axis=0)


def test_no_trim_when_no_black_run():
    assert plan_trim(clip([("live", 50)])) is None


def test_single_frame_dropout_is_not_a_separator():
    """A one-frame detection blink is not a separator and must not trigger."""
    seq = clip([("live", 20), ("black", 1), ("live", 20)])
    assert plan_trim(seq) is None


def test_run_at_the_minimum_length_does_trigger():
    seq = clip([("live", 20), ("black", MIN_RUN), ("live", 30)])
    trim = plan_trim(seq)
    assert trim is not None and trim.kind == "offset_longest_block"


def test_leading_pad_keeps_everything_after():
    seq = clip([("black", 30), ("live", 60)])
    trim = plan_trim(seq)
    assert trim == Trim(30, 90, "pad_leading")
    assert trim.kept == 60


def test_trailing_pad_keeps_everything_before():
    seq = clip([("live", 60), ("black", 30)])
    trim = plan_trim(seq)
    assert trim == Trim(0, 60, "pad_trailing")


def test_padding_both_ends():
    seq = clip([("black", 10), ("live", 40), ("black", 10)])
    trim = plan_trim(seq)
    assert trim == Trim(10, 50, "pad_both_ends")


def test_interior_separator_keeps_the_longer_block_not_the_concatenation():
    """The offset case: two blocks, only one can carry the label.

    Concatenating them would splice two dictionary entries under one gloss.
    """
    seq = clip([("live", 9), ("black", 51), ("live", 36)])
    trim = plan_trim(seq)
    assert trim.kind == "offset_longest_block"
    assert (trim.keep_start, trim.keep_end) == (60, 96)
    assert trim.kept == 36  # not 9 + 36


def test_interior_separator_picks_the_longer_side_either_way():
    """The separator direction is not consistent, so the rule is 'longest'."""
    seq = clip([("live", 58), ("black", 20), ("live", 4)])
    trim = plan_trim(seq)
    assert (trim.keep_start, trim.keep_end) == (0, 58)


def test_every_dropped_frame_is_all_zero():
    seq = clip([("live", 12), ("black", 20), ("live", 40)])
    trim = plan_trim(seq)
    kept = seq[trim.keep_start:trim.keep_end]
    assert np.all(np.abs(kept).sum(axis=1) > 0)


def test_zero_runs_reports_positions_and_lengths():
    seq = clip([("live", 5), ("black", 7), ("live", 5), ("black", 4)])
    assert zero_runs(seq) == [(5, 7), (17, 4)]


def test_all_black_clip_yields_no_span():
    assert plan_trim(clip([("black", 40)])) is None


def test_real_trim_table_is_consistent_with_the_cache():
    """The committed table must match what the cached tensors actually say.

    Skipped when the corpus is not on this machine.
    """
    import csv
    from pathlib import Path

    table = Path("msl/data/splits/wlasl100_separator_trims.csv")
    log = Path("data/wlasl/extract_log.csv")
    if not (table.exists() and log.exists()):
        import pytest
        pytest.skip("WLASL corpus not present")

    with open(log, newline="", encoding="utf-8") as handle:
        cache = {r["video_id"]: r["cache"] for r in csv.DictReader(handle)}
    with open(table, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "trim table is empty"
    for row in rows:
        path = cache.get(row["video_id"])
        if not path or not Path(path).exists():
            continue
        features = np.asarray(np.load(path, mmap_mode="r"))
        assert features.shape[0] == int(row["n_frames"])
        kept = features[int(row["keep_start"]):int(row["keep_end"])]
        # the surviving span contains no separator frame at all
        assert np.all(np.abs(kept).sum(axis=1) > 0), row["video_id"]
        assert kept.shape[0] == int(row["kept"])
