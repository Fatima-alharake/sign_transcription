"""WLASL loader: split assignment, landmark extraction, and the torch Dataset.

WLASL is **American Sign Language**, used to pretrain a backbone only. Every
number derived from it is an ASL number, currently computed on a corpus that is
55.3% complete.

Four things this module is careful about, each for a reason established earlier:

1. **Normalisation is `extract.py`'s, unchanged.** This module never touches a
   coordinate. It hands frames to `extract_video` and stores what comes back.
2. **Trimming happens before MediaPipe runs.** 132 clips are segments of long
   compilation videos in which the labelled sign is about 1% of the file.
   `frame_start`/`frame_end` are passed down to the frame reader, not applied to
   the packed tensor afterwards.
3. **Split membership is read, never computed at runtime.** Train/test comes
   from the committed signer-independent CSV; the clip-level val assignment is
   derived once, written to a companion CSV, and read back thereafter. A split
   recomputed per run is a split nobody can reproduce.
4. **A fresh extractor per video.** HolisticLandmarker keeps state between
   detections and crashes when the frame size changes; this corpus has 15
   distinct resolutions.

**Val is signer-dependent by design.** It is a clip-level split inside the
training signers, used only as a stopping criterion. Its accuracy is inflated
and must never be reported. Test is the honest, signer-disjoint number.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from msl.config import DataConfig, ExperimentConfig, FeatureConfig
from msl.data import landmarks as lm
from msl.data.extract import (
    ClipLandmarks,
    VideoReadError,
    cache_path,
    extract_video,
    save_cached,
)

SPLIT_CSV = Path("msl/data/splits/wlasl100_signer_independent.csv")
SPLIT_WITH_VAL_CSV = Path("msl/data/splits/wlasl100_train_val_test.csv")

# A clip whose pose is detected in fewer than this fraction of frames cannot be
# trusted: with no mid-shoulder point there is no origin to centre on and no
# inter-shoulder distance to scale by, so those frames normalise to all-zero
# vectors. See `PoseFailurePolicy` below -- this is a reported decision, not a
# silent one.
DEFAULT_MIN_POSE_RATE = 0.5

EXTRACT_LOG_COLUMNS = (
    "video_id", "gloss", "gloss_id", "signer_id", "split", "status",
    "frames_in", "frames_out", "pose_rate", "left_hand_rate", "right_hand_rate",
    "width", "height", "fps", "cache", "note",
)

OK = "ok"
CACHED = "cached"
NO_POSE = "no_pose"                # pose never detected: cannot be normalised
LOW_POSE = "low_pose_rate"         # normalisable, but below the threshold
READ_ERROR = "read_error"
EMPTY_RANGE = "empty_range"
MISSING_FILE = "missing_file"
USABLE = (OK, CACHED)


def rates_from_features(features: np.ndarray, use_z: bool) -> dict[str, float]:
    """Recover pose/hand presence from a cached tensor.

    The hand rates come straight from the visibility mask channels. Pose
    presence is inferred from the frame being non-zero: `normalise_frame`
    returns an all-zero vector exactly when there is no pose to centre on, and a
    frame that did normalise always has the shoulders at +/-0.5, so it can never
    be all zero.
    """
    if features.size == 0:
        return {"pose": 0.0, "left_hand": 0.0, "right_hand": 0.0}
    visibility = features[:, lm.visibility_slice(use_z)]
    normalised = np.abs(features).sum(axis=1) > 0
    return {
        "pose": float(normalised.mean()),
        "left_hand": float((visibility[:, 0] > 0).mean()),
        "right_hand": float((visibility[:, 1] > 0).mean()),
    }


@dataclass(frozen=True)
class WlaslClip:
    video_id: str
    gloss: str
    gloss_id: int
    signer_id: str
    split: str
    path: Path
    frame_start: int = 1
    frame_end: int | None = None

    @property
    def is_trimmed(self) -> bool:
        return self.frame_end is not None


# --- Split assignment ------------------------------------------------------


def assign_val(
    split_csv: str | Path = SPLIT_CSV,
    out_csv: str | Path = SPLIT_WITH_VAL_CSV,
    val_fraction: float = 0.15,
    seed: int = 1234,
) -> Path:
    """Carve a clip-level val split out of the training signers, once.

    Stratified by gloss so every class appears in val, and guaranteed to leave
    each gloss at least one training clip. Deterministic: the per-gloss
    shuffle is seeded and the input order is sorted first, so the assignment
    does not depend on CSV row order.

    Val signers also appear in train. That is deliberate -- a signer-disjoint
    val would cost 15 of the 18 training signers and leave training with 377
    clips from 3 people -- and it is exactly why a val number is never a result.
    """
    with open(split_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_gloss: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_gloss[row["gloss"]].append(row)

    val_ids: set[str] = set()
    for gloss in sorted(by_gloss):
        clips = sorted(by_gloss[gloss], key=lambda r: r["video_id"])
        if len(clips) < 2:
            continue  # never strip a gloss of its only training clip
        n_val = max(1, round(val_fraction * len(clips)))
        n_val = min(n_val, len(clips) - 1)
        rng = random.Random(f"{seed}:{gloss}")
        val_ids.update(c["video_id"] for c in rng.sample(clips, n_val))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "gloss", "gloss_id", "signer_id", "split"])
        for row in sorted(rows, key=lambda r: (int(r["gloss_id"]), r["video_id"])):
            split = row["split"]
            if split == "train" and row["video_id"] in val_ids:
                split = "val"
            writer.writerow([row["video_id"], row["gloss"], row["gloss_id"],
                             row["signer_id"], split])
    tmp.replace(out_csv)
    return out_csv


def describe_splits(csv_path: str | Path = SPLIT_WITH_VAL_CSV) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict = {"language": "American Sign Language (WLASL)"}
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        out[split] = {
            "clips": len(part),
            "signers": len({r["signer_id"] for r in part}),
            "glosses": len({r["gloss"] for r in part}),
        }
    train_signers = {r["signer_id"] for r in rows if r["split"] == "train"}
    val_signers = {r["signer_id"] for r in rows if r["split"] == "val"}
    test_signers = {r["signer_id"] for r in rows if r["split"] == "test"}
    out["val_signers_also_in_train"] = len(val_signers & train_signers)
    out["test_signers_also_in_train"] = len(test_signers & train_signers)
    out["val_is_signer_dependent_by_design"] = True
    return out


# --- Clip discovery --------------------------------------------------------


def load_clips(
    data: DataConfig | None = None,
    splits_csv: str | Path = SPLIT_WITH_VAL_CSV,
) -> list[WlaslClip]:
    """Join the committed split to the WLASL metadata and the files on disk."""
    data = data or DataConfig()
    root = Path(data.data_root) / "wlasl"
    meta = json.loads((root / "metadata" / "WLASL_v0.3.json").read_text(encoding="utf-8"))
    by_id = {i["video_id"]: i for e in meta for i in e["instances"]}

    video_dir = root / "raw_videos"
    paths: dict[str, Path] = {}
    if video_dir.exists():
        for name in os.listdir(video_dir):
            paths.setdefault(name.split(".")[0], video_dir / name)

    with open(splits_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    clips: list[WlaslClip] = []
    for row in rows:
        vid = row["video_id"]
        inst = by_id.get(vid, {})
        end = inst.get("frame_end", -1)
        clips.append(
            WlaslClip(
                video_id=vid,
                gloss=row["gloss"],
                gloss_id=int(row["gloss_id"]),
                signer_id=row["signer_id"],
                split=row["split"],
                path=paths.get(vid, video_dir / f"{vid}.mp4"),
                frame_start=int(inst.get("frame_start", 1) or 1),
                frame_end=None if end in (-1, None) else int(end),
            )
        )
    return clips


# --- Extraction ------------------------------------------------------------


def extract_clip(
    clip: WlaslClip,
    cfg: ExperimentConfig,
    min_pose_rate: float = DEFAULT_MIN_POSE_RATE,
) -> dict:
    """Extract one clip to the .npy cache. Returns a log row.

    Pose-failure policy, stated rather than assumed:

    * **no pose in any frame** -> `no_pose`. Nothing is cached. Every frame
      normalised to an all-zero vector, so the tensor carries no signal; caching
      it would put a silent block of zeros into training wearing a real label.
    * **pose present but below `min_pose_rate`** -> `low_pose_rate`. The tensor
      *is* cached, because the frames that did normalise are genuine, but the
      clip is excluded from the dataset by default and reported separately. The
      threshold is a parameter so the cut can be moved without re-extracting.
    """
    row: dict = {
        "video_id": clip.video_id, "gloss": clip.gloss, "gloss_id": clip.gloss_id,
        "signer_id": clip.signer_id, "split": clip.split, "frames_in": 0,
        "frames_out": 0, "pose_rate": 0.0, "left_hand_rate": 0.0,
        "right_hand_rate": 0.0, "width": 0, "height": 0, "fps": 0.0,
        "cache": "", "note": "",
    }
    if not clip.path.exists():
        row.update(status=MISSING_FILE, note=str(clip.path))
        return row

    # The segment is part of the key: several WLASL clips are different frame
    # ranges of one compilation video and are byte-identical on disk.
    destination = cache_path(
        clip.path, cfg.feature, cfg.data.resolved_cache_dir,
        segment=(clip.frame_start, clip.frame_end),
    )
    if destination.exists():
        features = np.load(destination)
        # Recompute the quality metrics from the tensor rather than leaving them
        # at zero. A cache hit that reports pose_rate 0.0 is filtered out by
        # every consumer downstream, so a clip extracted successfully once would
        # be silently dropped from training on the next run -- the same
        # trust-what-is-on-disk failure the downloader had.
        rates = rates_from_features(features, cfg.feature.use_z)
        row.update(
            frames_out=int(features.shape[0]), cache=str(destination),
            pose_rate=round(rates["pose"], 4),
            left_hand_rate=round(rates["left_hand"], 4),
            right_hand_rate=round(rates["right_hand"], 4),
            note="reused cached .npy",
        )
        row["status"] = CACHED if rates["pose"] >= min_pose_rate else LOW_POSE
        return row

    if clip.frame_end is not None and clip.frame_end < clip.frame_start:
        row.update(status=EMPTY_RANGE,
                   note=f"frame_end {clip.frame_end} < frame_start {clip.frame_start}")
        return row

    try:
        landmarks: ClipLandmarks = extract_video(
            clip.path, cfg.feature, cfg.data,
            start=clip.frame_start, end=clip.frame_end,
        )
    except VideoReadError as exc:
        row.update(status=READ_ERROR, note=str(exc)[:160])
        return row
    except Exception as exc:  # MediaPipe graph failures surface here
        row.update(status=READ_ERROR, note=f"{type(exc).__name__}: {exc}"[:160])
        return row

    rates = landmarks.detection_rate()
    row.update(
        frames_out=landmarks.n_frames, width=landmarks.width,
        height=landmarks.height, fps=round(landmarks.fps, 2),
        pose_rate=round(rates["pose"], 4),
        left_hand_rate=round(rates["left_hand"], 4),
        right_hand_rate=round(rates["right_hand"], 4),
    )
    row["frames_in"] = (
        clip.frame_end - clip.frame_start + 1 if clip.frame_end else landmarks.n_frames
    )

    if rates["pose"] == 0.0:
        row.update(status=NO_POSE,
                   note="no pose in any frame: no mid-shoulder origin, no "
                        "inter-shoulder scale; tensor would be all zeros")
        return row

    save_cached(landmarks, destination)
    row["cache"] = str(destination)
    row["status"] = LOW_POSE if rates["pose"] < min_pose_rate else OK
    if row["status"] == LOW_POSE:
        row["note"] = f"pose in only {rates['pose']:.0%} of frames"
    return row


def extract_all(
    cfg: ExperimentConfig,
    clips: list[WlaslClip] | None = None,
    limit: int | None = None,
    min_pose_rate: float = DEFAULT_MIN_POSE_RATE,
    log_path: str | Path | None = None,
) -> Path:
    """Extract every clip, appending one row per clip to a log as it goes."""
    clips = clips if clips is not None else load_clips(cfg.data)
    if limit:
        clips = clips[:limit]
    log_path = Path(log_path or Path(cfg.data.data_root) / "wlasl" / "extract_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"WLASL100 (AMERICAN SIGN LANGUAGE): extracting {len(clips)} clips")
    print(f"cache -> {cfg.data.resolved_cache_dir}")
    print(f"log   -> {log_path}")

    write_header = not log_path.exists()
    counts: collections.Counter = collections.Counter()
    for index, clip in enumerate(clips, 1):
        row = extract_clip(clip, cfg, min_pose_rate=min_pose_rate)
        counts[row["status"]] += 1
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXTRACT_LOG_COLUMNS)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow({k: row.get(k, "") for k in EXTRACT_LOG_COLUMNS})
            handle.flush()
        if index % 25 == 0 or index == len(clips):
            good = counts[OK] + counts[CACHED]
            other = " ".join(f"{k}={v}" for k, v in sorted(counts.items())
                             if k not in USABLE)
            print(f"  [{index}/{len(clips)}] ok {good}  {other}", flush=True)
    return log_path


def summarise_extraction(log_path: str | Path) -> dict:
    with open(log_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable = [r for r in rows if r["status"] in USABLE]
    lengths = sorted(int(r["frames_out"]) for r in usable if r["frames_out"])
    def pct(xs, p):
        return xs[int(p * (len(xs) - 1))] if xs else 0
    left = [float(r["left_hand_rate"]) for r in usable if r["left_hand_rate"]]
    right = [float(r["right_hand_rate"]) for r in usable if r["right_hand_rate"]]
    return {
        "language": "American Sign Language (WLASL)",
        "attempted": len(rows),
        "extracted": len(usable),
        "by_status": dict(collections.Counter(r["status"] for r in rows)),
        "no_pose_clips": [r["video_id"] for r in rows if r["status"] == NO_POSE],
        "low_pose_clips": len([r for r in rows if r["status"] == LOW_POSE]),
        "seq_len": {
            "min": lengths[0] if lengths else 0, "p25": pct(lengths, 0.25),
            "median": pct(lengths, 0.5), "p75": pct(lengths, 0.75),
            "max": lengths[-1] if lengths else 0,
        },
        "mean_left_hand_rate": round(float(np.mean(left)), 4) if left else 0.0,
        "mean_right_hand_rate": round(float(np.mean(right)), 4) if right else 0.0,
        "clips_left_hand_absent_most_of_clip": sum(1 for v in left if v < 0.5),
        "clips_right_hand_absent_most_of_clip": sum(1 for v in right if v < 0.5),
        "clips_both_hands_absent_most_of_clip": sum(
            1 for r in usable
            if float(r["left_hand_rate"] or 0) < 0.5 and float(r["right_hand_rate"] or 0) < 0.5
        ),
    }


# --- Dataset ---------------------------------------------------------------


def resample(sequence: np.ndarray, seq_len: int) -> np.ndarray:
    """Linear interpolation onto `seq_len` frames (spec S5)."""
    if sequence.shape[0] == seq_len:
        return sequence.astype(np.float32, copy=False)
    source = np.linspace(0.0, 1.0, num=sequence.shape[0])
    target = np.linspace(0.0, 1.0, num=seq_len)
    out = np.empty((seq_len, sequence.shape[1]), dtype=np.float32)
    for channel in range(sequence.shape[1]):
        out[:, channel] = np.interp(target, source, sequence[:, channel])
    return out


class WlaslLandmarkDataset:
    """Cached landmark tensors for one split. Torch is imported lazily."""

    def __init__(
        self,
        split: str,
        cfg: ExperimentConfig,
        log_path: str | Path | None = None,
        min_pose_rate: float = DEFAULT_MIN_POSE_RATE,
    ) -> None:
        log_path = Path(log_path or Path(cfg.data.data_root) / "wlasl" / "extract_log.csv")
        with open(log_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.cfg = cfg
        self.split = split
        self.rows = [
            r for r in rows
            if r["split"] == split and r["status"] in USABLE and r["cache"]
            and float(r["pose_rate"] or 0) >= min_pose_rate
        ]
        self.glosses = sorted({r["gloss"] for r in rows}, key=lambda g: g)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        features = np.load(row["cache"], mmap_mode="r")
        window = resample(np.asarray(features), self.cfg.feature.seq_len)
        return torch.from_numpy(window), int(row["gloss_id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WLASL loader: assign val, extract landmarks, report (ASL)"
    )
    parser.add_argument("--config", default="msl/configs/synthetic.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-pose-rate", type=float, default=DEFAULT_MIN_POSE_RATE)
    parser.add_argument("--assign-val-only", action="store_true")
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    log_path = Path(cfg.data.data_root) / "wlasl" / "extract_log.csv"

    if not SPLIT_WITH_VAL_CSV.exists() or args.assign_val_only:
        assign_val(val_fraction=args.val_fraction, seed=args.seed)
        print(f"val assignment written -> {SPLIT_WITH_VAL_CSV}")
    print(json.dumps(describe_splits(), indent=2))
    if args.assign_val_only:
        return 0

    if not args.summarise_only:
        log_path = extract_all(cfg, limit=args.limit, min_pose_rate=args.min_pose_rate)
    print("\n=== EXTRACTION (AMERICAN SIGN LANGUAGE) ===")
    print(json.dumps(summarise_extraction(log_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
