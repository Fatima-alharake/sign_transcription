"""The two-sided constraint on the signer-independent split (spec S5).

The bug these tests exist to prevent already happened once: a greedy that
protected only the *training* side produced a split where 12 of 100 glosses had
no test clip, which would have been reported as a 100-class ASL result. Only the
training-side constraint is obvious, and enforcing only it looks fine.
"""

from __future__ import annotations

import pytest

from msl.data.wlasl_split import SplitClip, build_split, describe, write_split


def corpus(spec: dict[str, list[str]]) -> list[SplitClip]:
    """Build clips from {gloss: [signer_id, ...]}."""
    clips: list[SplitClip] = []
    for gloss_id, (gloss, signers) in enumerate(spec.items()):
        for i, signer in enumerate(signers):
            clips.append(SplitClip(f"{gloss}-{i}", gloss, gloss_id, signer))
    return clips


def assert_two_sided(result, expected_glosses: int) -> None:
    total, with_train, with_test = result.coverage()
    assert total == expected_glosses
    assert with_train == expected_glosses, "a gloss lost all training data"
    assert with_test == expected_glosses, "a gloss lost all test data"


def test_every_gloss_keeps_both_sides() -> None:
    clips = corpus({
        "book": ["a", "b", "c", "d"],
        "drink": ["a", "b", "e", "f"],
        "help": ["c", "d", "e", "f"],
    })
    result = build_split(clips, target=0.30)
    assert result.feasible
    assert_two_sided(result, 3)


def test_no_signer_appears_on_both_sides() -> None:
    """The whole point: a signer in train and test makes the number meaningless."""
    clips = corpus({
        "book": ["a", "b", "c"], "drink": ["a", "b", "d"], "help": ["c", "d", "e"],
    })
    result = build_split(clips, target=0.30)
    train_signers = {c.signer_id for c in result.train}
    test_signers = {c.signer_id for c in result.test}
    assert not (train_signers & test_signers)


def test_single_signer_corpus_is_infeasible_not_silently_wrong() -> None:
    """One signer cannot be split signer-independently. Saying so is the only
    correct behaviour; returning an empty test set would not be."""
    clips = corpus({"book": ["a", "a", "a"], "drink": ["a", "a"]})
    result = build_split(clips)
    assert not result.feasible
    assert result.uncoverable


def test_gloss_with_one_signer_blocks_that_signer_from_holdout() -> None:
    """If 'rare' only ever comes from signer z, holding z out would strip its
    training data, so z must stay in train -- and then 'rare' can have no test
    clip, making the corpus infeasible under the two-sided rule."""
    clips = corpus({
        "common": ["a", "b", "c", "d"],
        "rare": ["z"],
    })
    result = build_split(clips, target=0.5)
    assert not result.feasible
    assert "rare" in result.uncoverable


def test_prefers_many_small_signers_over_few_large_ones() -> None:
    """Covering by 'most glosses per signer' picks the biggest signers and can
    satisfy the constraint with two of them -- a test set that measures whether
    the model recognises two people. Diversity is the point of the split."""
    spec = {f"g{i}": ["big"] + [f"s{i}", f"t{i}"] for i in range(8)}
    # 'big' appears in every gloss; each gloss also has two unique small signers.
    clips = corpus(spec)
    result = build_split(clips, target=0.40)
    assert result.feasible
    assert_two_sided(result, 8)
    assert len(result.held_out_signers) > 2, "should spread across small signers"
    assert "big" not in result.held_out_signers


def test_is_deterministic() -> None:
    clips = corpus({
        "book": ["a", "b", "c", "d"], "drink": ["a", "b", "e", "f"],
        "help": ["c", "d", "e", "f"],
    })
    first = build_split(clips, target=0.3).held_out_signers
    second = build_split(clips, target=0.3).held_out_signers
    assert first == second


def test_holdout_may_exceed_target_rather_than_drop_a_gloss() -> None:
    """The target is soft; the constraint is hard. Coverage wins."""
    clips = corpus({"book": ["a", "b"], "drink": ["a", "b"], "help": ["a", "b"]})
    result = build_split(clips, target=0.01)
    assert result.feasible
    assert_two_sided(result, 3)
    assert result.holdout_fraction > 0.01


def test_describe_reports_no_signer_overlap(tmp_path) -> None:
    clips = corpus({
        "book": ["a", "b", "c"], "drink": ["a", "b", "d"], "help": ["c", "d", "e"],
    })
    summary = describe(build_split(clips, target=0.3))
    assert summary["signer_overlap"] == []
    assert summary["glosses_with_train"] == summary["glosses"]
    assert summary["glosses_with_test"] == summary["glosses"]
    assert "American Sign Language" in summary["language"]


def test_written_split_round_trips(tmp_path) -> None:
    import csv

    clips = corpus({"book": ["a", "b", "c"], "drink": ["a", "b", "d"]})
    result = build_split(clips, target=0.3)
    path = write_split(result, tmp_path / "split.csv")

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(clips)
    assert {r["split"] for r in rows} == {"train", "test"}
    assert list(tmp_path.glob("*.tmp")) == [], "atomic write left a temp file"
