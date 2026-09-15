"""Extraction-quality report for WLASL (American Sign Language).

Answers the questions that decide whether a later accuracy number is
explainable: does detection degrade on the low-resolution sources, and are the
unusable clips concentrated somewhere that would quietly starve a gloss of
training or test data.

Everything here reads the extraction log and the download log. Nothing is
recomputed from metadata, which describes a corpus that is 55.3% present.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from msl.data.wlasl_loader import LOW_POSE, NO_POSE, USABLE


def _pct(values: list[int], p: float) -> int:
    return values[int(p * (len(values) - 1))] if values else 0


def load(extract_log: Path, download_log: Path) -> tuple[list[dict], dict[str, dict]]:
    with open(extract_log, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with open(download_log, newline="", encoding="utf-8") as handle:
        dl = {r["video_id"]: r for r in csv.DictReader(handle)}
    return rows, dl


def report(extract_log: Path, download_log: Path, min_pose_rate: float = 0.5) -> dict:
    rows, dl = load(extract_log, download_log)
    usable = [r for r in rows if r["status"] in USABLE]

    lengths = sorted(int(r["frames_out"]) for r in usable if r["frames_out"])
    left = [float(r["left_hand_rate"] or 0) for r in usable]
    right = [float(r["right_hand_rate"] or 0) for r in usable]

    # --- hand presence by source resolution ------------------------------
    by_res: dict[str, dict] = collections.defaultdict(
        lambda: {"clips": 0, "left": 0.0, "right": 0.0, "pose": 0.0, "sources": collections.Counter()}
    )
    for r in usable:
        d = dl.get(r["video_id"], {})
        w, h = d.get("width") or r["width"], d.get("height") or r["height"]
        key = f"{w}x{h}"
        b = by_res[key]
        b["clips"] += 1
        b["left"] += float(r["left_hand_rate"] or 0)
        b["right"] += float(r["right_hand_rate"] or 0)
        b["pose"] += float(r["pose_rate"] or 0)
        b["sources"][d.get("source", "?")] += 1
    resolutions = {}
    # Sorted by pixel count, ascending: the question this table answers is
    # whether detection degrades on the low-resolution sources, and that is only
    # legible if the rows run smallest-first.
    def _pixels(key: str) -> int:
        w, _, h = key.partition("x")
        return (int(w) if w.isdigit() else 0) * (int(h) if h.isdigit() else 0)
    for key, b in sorted(by_res.items(), key=lambda kv: (_pixels(kv[0]), kv[0])):
        n = b["clips"]
        width = int(key.split("x")[0]) if key.split("x")[0].isdigit() else 0
        resolutions[key] = {
            "clips": n,
            "pixels": width * (int(key.split("x")[1]) if key.split("x")[1].isdigit() else 0),
            "pose_rate": round(b["pose"] / n, 3),
            "left_hand_rate": round(b["left"] / n, 3),
            "right_hand_rate": round(b["right"] / n, 3),
            "top_source": b["sources"].most_common(1)[0][0] if b["sources"] else "?",
        }

    # --- where are the bad clips concentrated? ---------------------------
    bad = [r for r in rows if r["status"] in (NO_POSE, LOW_POSE)]
    per_gloss_bad = collections.Counter(r["gloss"] for r in bad)
    per_signer_bad = collections.Counter(r["signer_id"] for r in bad)

    # --- would exclusion starve a gloss? ---------------------------------
    def kept(r: dict) -> bool:
        return r["status"] in USABLE and float(r["pose_rate"] or 0) >= min_pose_rate

    before = collections.Counter((r["gloss"], r["split"]) for r in rows)
    after = collections.Counter((r["gloss"], r["split"]) for r in rows if kept(r))
    glosses = sorted({r["gloss"] for r in rows})
    starved_train = []
    starved_test = []
    heavy_loss = []
    for g in glosses:
        tr_b, tr_a = before[(g, "train")], after[(g, "train")]
        te_b, te_a = before[(g, "test")], after[(g, "test")]
        if tr_b and tr_a == 0:
            starved_train.append(g)
        if te_b and te_a == 0:
            starved_test.append(g)
        if tr_b and tr_a / tr_b < 0.5:
            heavy_loss.append({"gloss": g, "train_before": tr_b, "train_after": tr_a})

    return {
        "language": "American Sign Language (WLASL)",
        "rows_in_log": len(rows),
        "extracted_usable": len(usable),
        "by_status": dict(collections.Counter(r["status"] for r in rows)),
        "no_pose_clips": [
            {"video_id": r["video_id"], "gloss": r["gloss"], "signer_id": r["signer_id"],
             "split": r["split"], "frames": r["frames_out"]}
            for r in rows if r["status"] == NO_POSE
        ],
        "low_pose_clips": [
            {"video_id": r["video_id"], "gloss": r["gloss"],
             "signer_id": r["signer_id"], "split": r["split"],
             "pose_rate": float(r["pose_rate"] or 0),
             "left_hand_rate": float(r["left_hand_rate"] or 0),
             "right_hand_rate": float(r["right_hand_rate"] or 0),
             "frames": int(r["frames_out"] or 0),
             "resolution": "{}x{}".format(
                 dl.get(r["video_id"], {}).get("width") or r["width"],
                 dl.get(r["video_id"], {}).get("height") or r["height"]),
             "source": dl.get(r["video_id"], {}).get("source", "?")}
            for r in sorted(rows, key=lambda x: x["gloss"])
            if r["status"] == LOW_POSE
        ],
        "sequence_length": {
            "min": lengths[0] if lengths else 0,
            "p10": _pct(lengths, 0.10), "p25": _pct(lengths, 0.25),
            "median": _pct(lengths, 0.50), "p75": _pct(lengths, 0.75),
            "p90": _pct(lengths, 0.90), "max": lengths[-1] if lengths else 0,
        },
        "hand_presence": {
            "mean_left": round(sum(left) / len(left), 3) if left else 0.0,
            "mean_right": round(sum(right) / len(right), 3) if right else 0.0,
            "clips_left_absent_most": sum(1 for v in left if v < 0.5),
            "clips_right_absent_most": sum(1 for v in right if v < 0.5),
            "clips_both_absent_most": sum(
                1 for r in usable
                if float(r["left_hand_rate"] or 0) < 0.5
                and float(r["right_hand_rate"] or 0) < 0.5
            ),
            "clips_neither_hand_ever": sum(
                1 for r in usable
                if float(r["left_hand_rate"] or 0) == 0.0
                and float(r["right_hand_rate"] or 0) == 0.0
            ),
        },
        "by_resolution": resolutions,
        "unusable_concentration": {
            "total_bad": len(bad),
            "worst_glosses": per_gloss_bad.most_common(10),
            "worst_signers": per_signer_bad.most_common(10),
        },
        "exclusion_impact": {
            "min_pose_rate": min_pose_rate,
            "clips_per_split": {
                split: {
                    "before": sum(v for (g, sp), v in before.items() if sp == split),
                    "after": sum(v for (g, sp), v in after.items() if sp == split),
                    "glosses_before": len({g for (g, sp) in before if sp == split}),
                    "glosses_after": len({g for (g, sp) in after
                                          if sp == split and after[(g, sp)]}),
                }
                for split in ("train", "val", "test")
            },
            "glosses_losing_all_training_clips": starved_train,
            "glosses_losing_all_test_clips": starved_test,
            "glosses_losing_over_half_of_training": heavy_loss,
            "glosses_affected_at_all": sorted(
                {g for g in glosses
                 if any(before[(g, sp)] != after[(g, sp)]
                        for sp in ("train", "val", "test"))}
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WLASL extraction quality report (ASL)")
    parser.add_argument("--extract-log", type=Path,
                        default=Path("data/wlasl/extract_log.csv"))
    parser.add_argument("--download-log", type=Path,
                        default=Path("data/wlasl/download_log_100.csv"))
    parser.add_argument("--min-pose-rate", type=float, default=0.5)
    args = parser.parse_args(argv)
    print(json.dumps(report(args.extract_log, args.download_log, args.min_pose_rate),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
