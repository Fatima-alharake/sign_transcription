"""Run one variant across several seeds and report mean and spread.

WHY THIS EXISTS. The first two WLASL runs differed by 1.78pp top-1, which
looked like an improvement and was not: paired on the same 338 test clips, 39
flipped wrong->right and 33 flipped right->wrong, exact McNemar p = 0.556. At
338 test clips the unpaired standard error on a single top-1 is about 2.7pp, so
any single-run delta smaller than that is unreadable. Every variant from here
is run at three seeds minimum and reported as mean with spread.

A delta between variants means nothing unless it is larger than the spread
within a variant. This module prints both so that comparison is possible rather
than implied.

Every number is AMERICAN SIGN LANGUAGE on a 55.3%-complete corpus. Val is never
aggregated or reported -- it is signer-dependent and a stopping criterion only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np

from msl.config import ExperimentConfig
from msl.data.vocab import Vocab
from msl.models.build import build_model, load_checkpoint
from msl.runtime import describe_device, resolve_device
from msl.train.evaluate import evaluate, plot_confusion, plot_loss_seeds
from msl.train.train_isolated import train
from msl.train.train_wlasl import CAVEAT, build_loaders

DEFAULT_SEEDS = (1234, 2345, 3456)
METRICS = ("test_top1", "test_top5", "test_macro_f1")


def run_seed(
    cfg: ExperimentConfig,
    seed: int,
    device,
    min_pose_rate: float,
    log_path: Path | None = None,
) -> tuple[dict, np.ndarray, list[float]]:
    """One full train-and-evaluate at one seed. Returns metrics, confusion, loss."""
    cfg = replace(
        cfg,
        train=replace(
            cfg.train,
            seed=seed,
            out_dir=f"{cfg.train.out_dir}/seed{seed}",
            log_dir=f"{cfg.train.log_dir}/seed{seed}",
            tensorboard=False,  # the sweep keeps its own curves
        ),
    )
    # Built after the seed is set: the dataset's augmentation RNG is seeded from
    # cfg.train.seed, so building loaders first would give every seed the same
    # augmentation stream.
    train_loader, val_loader, test_loader, vocab = build_loaders(
        cfg, min_pose_rate, log_path=log_path)

    result = train(cfg, device=device, vocab=vocab, verbose=False,
                   loaders=(train_loader, val_loader))

    model = build_model(cfg, num_classes=len(vocab)).to(device)
    ckpt = load_checkpoint(result.best_checkpoint, map_location=str(device))
    model.load_state_dict(ckpt["model"])
    if ckpt["glosses"] != vocab.glosses:
        raise SystemExit("checkpoint vocabulary does not match the split CSV")
    test = evaluate(model, test_loader, device, num_classes=len(vocab))

    return (
        {
            "seed": seed,
            "test_top1": round(test.top1, 4),
            "test_top5": round(test.top5, 4),
            "test_macro_f1": round(test.macro_f1, 4),
            "steps_run": result.steps_run,
        },
        test.confusion,
        result.losses,
    )


def aggregate(per_seed: list[dict]) -> dict:
    out: dict = {}
    for key in METRICS:
        values = [row[key] for row in per_seed]
        out[key] = {
            "mean": round(statistics.fmean(values), 4),
            # Sample sd. At n=3 this is a rough spread indicator, not a
            # confidence interval -- the min/max range is the honest summary.
            "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "values": values,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a WLASL variant across seeds (ASL)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--label", default=None,
                        help="name for this variant in the report")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-pose-rate", type=float, default=0.5)
    parser.add_argument("--extract-log", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--arch", default=None,
                        choices=["transformer", "bilstm", "tcn", "stgcn"])
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.arch:
        cfg = replace(cfg, model=replace(cfg.model, arch=args.arch))
    device = resolve_device(args.device)
    label = args.label or args.config.stem

    print("=" * 74)
    print(f"SWEEP: {label}  ({len(args.seeds)} seeds: {args.seeds})")
    print(CAVEAT)
    print("=" * 74)
    print(f"arch {cfg.model.arch}   augmentation "
          f"{'ON' if cfg.augment.enabled else 'OFF'}   use_z {cfg.feature.use_z}")
    print(f"device: {describe_device(device)}\n", flush=True)

    per_seed: list[dict] = []
    confusions: list[np.ndarray] = []
    losses_by_seed: dict[int, list[float]] = {}
    for seed in args.seeds:
        row, confusion, losses = run_seed(
            cfg, seed, device, args.min_pose_rate, log_path=args.extract_log)
        per_seed.append(row)
        confusions.append(confusion)
        losses_by_seed[seed] = losses
        print(f"  seed {seed}: top-1 {row['test_top1']:.4f}  "
              f"top-5 {row['test_top5']:.4f}  macro F1 "
              f"{row['test_macro_f1']:.4f}  ({row['steps_run']} steps)",
              flush=True)

    agg = aggregate(per_seed)
    print(f"\n  {'metric':<16}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}")
    for key in METRICS:
        a = agg[key]
        print(f"  {key:<16}{a['mean']:>8.4f}{a['sd']:>8.4f}"
              f"{a['min']:>8.4f}{a['max']:>8.4f}")

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    # Summed over seeds: a per-seed matrix at 338 clips is too sparse to read.
    total = np.sum(confusions, axis=0)
    np.save(report_dir / "confusion_test_summed.npy", total)
    _, _, _, vocab = build_loaders(cfg, args.min_pose_rate,
                                   log_path=args.extract_log)
    plot_confusion(
        total, vocab.glosses, report_dir / "confusion_test_summed.png",
        f"Test confusion summed over {len(args.seeds)} seeds, row-normalised -- "
        f"{label} (ASL, signer-disjoint, 55.3%-complete corpus)",
    )
    plot_loss_seeds(
        losses_by_seed, report_dir / "training_loss_seeds.png",
        f"WLASL100 {label} -- training loss per seed "
        f"(ASL, 55.3%-complete corpus)",
    )
    (report_dir / "sweep_metrics.json").write_text(json.dumps({
        "label": label,
        "language": "American Sign Language (WLASL100)",
        "corpus_completeness": "55.3% (1127 of 2038 clips)",
        "arch": cfg.model.arch,
        "use_z": cfg.feature.use_z,
        "augmentation": (vars(cfg.augment) if cfg.augment.enabled else None),
        "seeds": args.seeds,
        "per_seed": per_seed,
        "aggregate": agg,
        "note": "val accuracy deliberately absent: signer-dependent, "
                "stopping criterion only",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
