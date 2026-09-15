"""Signer-independent split for WLASL (spec S5).

WLASL is **American Sign Language**; every number derived from this split is an
ASL number.

WLASL ships a split, and it is not signer-independent: 2877 of its 2878 test
instances are performed by someone who also appears in training. Spec S5 is
explicit that such a number is meaningless, so we build our own.

The constraint is two-sided, and both sides matter:

* every gloss keeps at least one **training** clip -- otherwise the model never
  sees the class it is asked to predict;
* every gloss keeps at least one **held-out test** clip -- otherwise evaluation
  silently covers fewer classes than training does, and the reported accuracy
  describes a smaller problem than the one solved.

Only the first is obvious, and enforcing only the first is the bug this module
exists to fix: a greedy that protects training alone happily produced a split
where 12 of 100 glosses had no test clip at all, which would have been reported
as a 100-class result.

The ~20% holdout target is *soft*; the two-sided constraint is *hard*. If full
coverage costs more or less than 20%, the achieved fraction is reported rather
than the constraint being relaxed or glosses being dropped.

Deterministic: no RNG. Selection is greedy with fully ordered tie-breaks, so the
same corpus always yields the same split. `seed` is accepted and recorded for
provenance, not used to randomise.
"""

from __future__ import annotations

import csv
import collections
from dataclasses import dataclass, field
from pathlib import Path

USABLE_STATUSES = ("ok", "cached")


@dataclass(frozen=True)
class SplitClip:
    video_id: str
    gloss: str
    gloss_id: int
    signer_id: str


@dataclass
class SplitResult:
    held_out_signers: set[str]
    train: list[SplitClip] = field(default_factory=list)
    test: list[SplitClip] = field(default_factory=list)
    feasible: bool = True
    uncoverable: list[str] = field(default_factory=list)
    seed: int = 0

    @property
    def holdout_fraction(self) -> float:
        total = len(self.train) + len(self.test)
        return len(self.test) / total if total else 0.0

    def coverage(self) -> tuple[int, int, int]:
        """(glosses total, with training clips, with test clips)."""
        glosses = {c.gloss for c in self.train} | {c.gloss for c in self.test}
        return (
            len(glosses),
            len({c.gloss for c in self.train}),
            len({c.gloss for c in self.test}),
        )


def load_usable_clips(log_path: str | Path) -> list[SplitClip]:
    """Clips that actually exist on disk, from the download log.

    Never the metadata: the metadata describes a corpus that is 55% present.
    """
    with open(log_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        SplitClip(r["video_id"], r["gloss"], int(r["gloss_id"]), r["signer_id"])
        for r in rows
        if r["status"] in USABLE_STATUSES
    ]


def _by_gloss(clips: list[SplitClip]) -> dict[str, collections.Counter]:
    table: dict[str, collections.Counter] = {}
    for clip in clips:
        table.setdefault(clip.gloss, collections.Counter())[clip.signer_id] += 1
    return table


def _keeps_training_everywhere(
    gloss_signers: dict[str, collections.Counter], held: set[str]
) -> bool:
    """True if every gloss still has a clip from a signer that is NOT held out."""
    return all(
        any(signer not in held for signer in counter)
        for counter in gloss_signers.values()
    )


def build_split(
    clips: list[SplitClip], target: float = 0.20, seed: int = 0
) -> SplitResult:
    """Hold out whole signers under the two-sided constraint.

    Phase 1 covers every gloss with a test clip, which is the binding
    requirement. Phase 2 tops the holdout up toward `target` only if phase 1
    came in under it, and never at the cost of the constraint.
    """
    gloss_signers = _by_gloss(clips)
    signer_size = collections.Counter(c.signer_id for c in clips)
    total = len(clips)
    held: set[str] = set()

    def admissible(signer: str) -> bool:
        return _keeps_training_everywhere(gloss_signers, held | {signer})

    # --- Phase 1a: spend small signers first -------------------------------
    # Covering greedily by "most glosses per signer" picks the *largest*
    # signers, and on this corpus that satisfies the constraint with just two
    # of them -- a test set that measures whether the model recognises two
    # particular people. Preferring small signers buys signer diversity in the
    # test set for barely any extra holdout, which is the whole point of a
    # signer-independent split.
    uncovered = {
        gloss for gloss, counter in gloss_signers.items()
        if not any(s in held for s in counter)
    }
    for signer, size in sorted(signer_size.items(), key=lambda kv: (kv[1], kv[0])):
        if signer in held or not uncovered:
            continue
        if not any(signer in gloss_signers[g] for g in uncovered):
            continue
        if (sum(signer_size[s] for s in held) + size) / total > target:
            continue
        if admissible(signer):
            held.add(signer)
            uncovered = {g for g in uncovered if signer not in gloss_signers[g]}

    # --- Phase 1b: cover whatever small signers could not reach ------------
    while uncovered:
        best: tuple[int, int, str] | None = None
        for signer in sorted(signer_size):
            if signer in held or not admissible(signer):
                continue
            gain = sum(1 for g in uncovered if signer in gloss_signers[g])
            if gain == 0:
                continue
            # Cheapest first, then most glosses covered, then id: a total order,
            # so the split is reproducible without an RNG.
            key = (signer_size[signer], -gain, signer)
            if best is None or key < best:
                best = key
        if best is None:
            # No admissible signer can cover what remains.
            return SplitResult(
                held_out_signers=held,
                train=[c for c in clips if c.signer_id not in held],
                test=[c for c in clips if c.signer_id in held],
                feasible=False,
                uncoverable=sorted(uncovered),
                seed=seed,
            )
        chosen = best[2]
        held.add(chosen)
        uncovered = {g for g in uncovered if chosen not in gloss_signers[g]}

    # --- Phase 2: approach the target, never breaking the constraint -------
    if len(clips) and sum(signer_size[s] for s in held) / total < target:
        for signer, size in sorted(signer_size.items(), key=lambda kv: (kv[1], kv[0])):
            if signer in held:
                continue
            if (sum(signer_size[s] for s in held) + size) / total > target:
                continue
            if admissible(signer):
                held.add(signer)

    return SplitResult(
        held_out_signers=held,
        train=[c for c in clips if c.signer_id not in held],
        test=[c for c in clips if c.signer_id in held],
        feasible=True,
        seed=seed,
    )


def write_split(result: SplitResult, path: str | Path) -> Path:
    """Write the split as a small, committed CSV.

    Versioned deliberately: a split that cannot be reproduced exactly makes
    every number computed against it unreproducible too.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(c, "train") for c in result.train] + [(c, "test") for c in result.test]
    rows.sort(key=lambda pair: (pair[0].gloss_id, pair[0].video_id))
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "gloss", "gloss_id", "signer_id", "split"])
        for clip, split in rows:
            writer.writerow(
                [clip.video_id, clip.gloss, clip.gloss_id, clip.signer_id, split]
            )
    tmp.replace(path)
    return path


def describe(result: SplitResult) -> dict:
    total_glosses, with_train, with_test = result.coverage()
    signers_train = {c.signer_id for c in result.train}
    signers_test = {c.signer_id for c in result.test}
    per_class_test = collections.Counter(c.gloss for c in result.test)
    return {
        "language": "American Sign Language (WLASL)",
        "feasible": result.feasible,
        "uncoverable_glosses": result.uncoverable,
        "clips_total": len(result.train) + len(result.test),
        "train_clips": len(result.train),
        "test_clips": len(result.test),
        "holdout_fraction": round(result.holdout_fraction, 4),
        "signers_held_out": len(result.held_out_signers),
        "signers_train": len(signers_train),
        "signers_test": len(signers_test),
        "signer_overlap": sorted(signers_train & signers_test),
        "glosses": total_glosses,
        "glosses_with_train": with_train,
        "glosses_with_test": with_test,
        "test_clips_per_gloss_min": min(per_class_test.values()) if per_class_test else 0,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Build a signer-independent WLASL split (American Sign Language)"
    )
    parser.add_argument("--log", type=Path, default=Path("data/wlasl/download_log_100.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("msl/data/splits/wlasl100_signer_independent.csv"))
    parser.add_argument("--target", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    clips = load_usable_clips(args.log)
    result = build_split(clips, target=args.target, seed=args.seed)
    summary = describe(result)
    if result.feasible:
        write_split(result, args.out)
        summary["written_to"] = str(args.out)
    print(json.dumps(summary, indent=2))
    return 0 if result.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
