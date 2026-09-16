"""Trim WLASL's black separator frames out of the labelled range.

WHAT THIS FIXES. 43 of the 51 `northtexas` clips have a contiguous near-black
run inside `frame_start..frame_end` -- the separator between two dictionary
entries in a compilation video. MediaPipe finds no pose on a black frame and
`normalise_frame` correctly emits an all-zero vector, so the tensor carries a
block of zeros in the middle of a labelled sign. Dropping those frames is a
data fix, not a model change; the clip is kept, only the frames go.

HOW THE FRAMES ARE IDENTIFIED. From the cached tensor alone: a frame is a
separator frame when its feature vector is all-zero, which by the
`normalise_frame` invariant means no usable pose. No video is re-read.

That this is the same set as the luminance criterion is *checked*, not assumed.
Selecting every clip in the corpus with an all-zero run of >= MIN_RUN frames
returns exactly the 43 clips the luminance-10 scan found, all of them
northtexas, with no false positives anywhere else in the 1127. The luminance
scan separately confirmed that 100% of frames below the cut had no pose and
that darkness explains 99.8% of all pose-less frames in these clips.

THE OFFSET PROBLEM, STATED RATHER THAN HIDDEN. Some ranges are merely *wide* --
the separator pads the start or the end, and everything that survives is one
contiguous block of the labelled sign. Others are *offset*: the separator sits
in the interior with real footage on both sides, which means the range straddles
an entry boundary and one of the two blocks belongs to a DIFFERENT dictionary
entry. Concatenating them would splice two signs under one label.

This module therefore keeps the LONGEST surviving block, never the
concatenation. For a padding-only clip that is the whole of the surviving
footage and is exactly right. For an offset clip it is a HEURISTIC: it assumes
the labelled sign is the larger of the two blocks. That assumption is not
verifiable from the data -- the separator direction is not consistent (one clip
has 58 frames before it and 1 after, another 35 before and 63 after), so
"whatever follows the separator" is not a better rule. 9 clips are affected this
way, 8 train and 1 val. Their labels may attach to the wrong block. TEST IS
UNAFFECTED: signer 121 appears only in train and val.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRIM_CSV = Path("msl/data/splits/wlasl100_separator_trims.csv")

# Ignore single-frame dropouts: a separator is a sustained run, and a one-frame
# detection blink is not what this is for.
MIN_RUN = 3

TRIM_COLUMNS = (
    "video_id", "gloss", "signer_id", "split", "source",
    "n_frames", "keep_start", "keep_end", "kept", "dropped", "kind",
)


@dataclass(frozen=True)
class Trim:
    keep_start: int
    keep_end: int  # exclusive
    kind: str

    @property
    def kept(self) -> int:
        return self.keep_end - self.keep_start


def zero_runs(features: np.ndarray, min_run: int = MIN_RUN) -> list[tuple[int, int]]:
    """Contiguous runs of all-zero frames, as (start, length), length >= min_run."""
    blank = np.abs(features).sum(axis=1) == 0
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(blank):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(blank) - start))
    return [r for r in runs if r[1] >= min_run]


def plan_trim(features: np.ndarray, min_run: int = MIN_RUN) -> Trim | None:
    """The span to keep, or None when the clip needs no trimming."""
    n = features.shape[0]
    runs = zero_runs(features, min_run)
    if not runs:
        return None

    keep = np.ones(n, dtype=bool)
    for start, length in runs:
        keep[start:start + length] = False

    # Contiguous surviving blocks.
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < n:
        if keep[index]:
            end = index
            while end < n and keep[end]:
                end += 1
            blocks.append((index, end))
            index = end
        else:
            index += 1
    if not blocks:
        return None

    interior = any(s > 0 and s + length < n for s, length in runs)
    best = max(blocks, key=lambda b: b[1] - b[0])
    if interior:
        kind = "offset_longest_block"
    elif runs[0][0] == 0 and runs[-1][0] + runs[-1][1] >= n:
        kind = "pad_both_ends"
    elif runs[0][0] == 0:
        kind = "pad_leading"
    else:
        kind = "pad_trailing"
    return Trim(keep_start=best[0], keep_end=best[1], kind=kind)


def load_trims(csv_path: str | Path = TRIM_CSV) -> dict[str, Trim]:
    path = Path(csv_path)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            row["video_id"]: Trim(int(row["keep_start"]), int(row["keep_end"]),
                                  row["kind"])
            for row in csv.DictReader(handle)
        }


def build_trim_table(
    extract_log: str | Path = "data/wlasl/extract_log.csv",
    download_log: str | Path = "data/wlasl/download_log_100.csv",
    out_csv: str | Path = TRIM_CSV,
    min_run: int = MIN_RUN,
) -> tuple[Path, list[dict]]:
    """Scan every cached clip and write the trim table. Deterministic."""
    with open(extract_log, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with open(download_log, newline="", encoding="utf-8") as handle:
        sources = {r["video_id"]: r.get("source", "?")
                   for r in csv.DictReader(handle)}

    out: list[dict] = []
    for row in rows:
        if not row["cache"]:
            continue
        features = np.asarray(np.load(row["cache"], mmap_mode="r"))
        trim = plan_trim(features, min_run)
        if trim is None:
            continue
        out.append({
            "video_id": row["video_id"], "gloss": row["gloss"],
            "signer_id": row["signer_id"], "split": row["split"],
            "source": sources.get(row["video_id"], "?"),
            "n_frames": features.shape[0],
            "keep_start": trim.keep_start, "keep_end": trim.keep_end,
            "kept": trim.kept, "dropped": features.shape[0] - trim.kept,
            "kind": trim.kind,
        })

    out.sort(key=lambda r: r["video_id"])
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIM_COLUMNS)
        writer.writeheader()
        writer.writerows(out)
    tmp.replace(path)
    return path, out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import collections

    parser = argparse.ArgumentParser(
        description="Build the WLASL separator-trim table (ASL)")
    parser.add_argument("--extract-log", default="data/wlasl/extract_log.csv")
    parser.add_argument("--out", default=str(TRIM_CSV))
    args = parser.parse_args(argv)

    path, out = build_trim_table(extract_log=args.extract_log, out_csv=args.out)
    print(f"clips needing a trim: {len(out)}  -> {path}")
    print("by source:", dict(collections.Counter(r["source"] for r in out)))
    print("by split :", dict(collections.Counter(r["split"] for r in out)))
    print("by kind  :", dict(collections.Counter(r["kind"] for r in out)))
    total = sum(r["n_frames"] for r in out)
    kept = sum(r["kept"] for r in out)
    print(f"frames: {total} in range, {kept} kept, {total - kept} dropped "
          f"({(total - kept) / total:.0%})")
    shortest = min(out, key=lambda r: r["kept"])
    print(f"shortest surviving clip: {shortest['video_id']} "
          f"{shortest['gloss']} -> {shortest['kept']} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
