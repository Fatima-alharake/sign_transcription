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
import sys
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

# THE protocol every variant comparison must run under. A variant measured on a
# different protocol is not comparable to the reference, and the sweep refuses to
# produce such a number rather than emitting one that merely looks comparable.
# Early stopping on a 116-clip val set made the stopping step -- not the change
# under test -- drive the score, and that mistake is one forgotten CLI flag away.
# The configs carry these values so the default is correct; the flags remain for
# deliberate deviation via --protocol-override, which labels the output.
REFERENCE_PROTOCOL = {
    "early_stopping": False,
    "max_steps": 2400,
    "checkpoint": "final",
}


def protocol_of(cfg: ExperimentConfig) -> dict:
    return {
        "early_stopping": cfg.train.early_stopping,
        "max_steps": cfg.train.max_steps,
        "checkpoint": "best_val" if cfg.train.early_stopping else "final",
    }


def check_protocol(cfg: ExperimentConfig, override: bool) -> dict:
    """Refuse a comparison run whose protocol differs from the reference."""
    actual = protocol_of(cfg)
    differences = {
        key: (REFERENCE_PROTOCOL[key], actual[key])
        for key in REFERENCE_PROTOCOL
        if REFERENCE_PROTOCOL[key] != actual[key]
    }
    if differences and not override:
        lines = [
            "PROTOCOL MISMATCH -- refusing to run.",
            "A sweep on a different protocol is not comparable to the reference,",
            "and a number that is not comparable is worse than no number.",
            "",
        ]
        for key, (expected, got) in sorted(differences.items()):
            lines.append(f"  {key}: reference {expected!r}, this run {got!r}")
        lines += [
            "",
            "Fix the config (max_steps: 2400, early_stopping: false), or pass",
            "--protocol-override to run deliberately off-protocol. Overridden",
            "runs are labelled non-comparable in sweep_metrics.json.",
        ]
        raise SystemExit("\n".join(lines))
    return differences


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

    # With early stopping off, evaluate the FINAL model. Picking the best-val
    # checkpoint would put val back in charge of the comparison -- the exact
    # confound the fixed budget exists to remove.
    checkpoint = (result.best_checkpoint if cfg.train.early_stopping
                  else result.last_checkpoint)
    model = build_model(cfg, num_classes=len(vocab)).to(device)
    ckpt = load_checkpoint(checkpoint, map_location=str(device))
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
    parser.add_argument("--max-steps", type=int, default=None,
                        help="fixed step budget; the cosine completes here")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="train the full budget and evaluate the FINAL "
                             "checkpoint -- required for variant comparisons")
    parser.add_argument("--protocol-override", action="store_true",
                        help="run deliberately off the reference protocol; the "
                             "output is labelled non-comparable")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.arch:
        cfg = replace(cfg, model=replace(cfg.model, arch=args.arch))
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.no_early_stopping:
        overrides["early_stopping"] = False
    if overrides:
        cfg = replace(cfg, train=replace(cfg.train, **overrides))
    off_protocol = check_protocol(cfg, args.protocol_override)
    device = resolve_device(args.device)
    label = args.label or args.config.stem

    print("=" * 74)
    print(f"SWEEP: {label}  ({len(args.seeds)} seeds: {args.seeds})")
    print(CAVEAT)
    print("=" * 74)
    print(f"arch {cfg.model.arch}   augmentation "
          f"{'ON' if cfg.augment.enabled else 'OFF'}   use_z {cfg.feature.use_z}"
          f"   separator trim {cfg.data.drop_separator_frames}")
    protocol = ("early stopping (patience "
                f"{cfg.train.patience}), best-val checkpoint"
                if cfg.train.early_stopping else
                f"FIXED BUDGET {cfg.train.max_steps} steps, no early stopping, "
                "final checkpoint")
    print(f"protocol: {protocol}")
    if off_protocol:
        print("  OFF-PROTOCOL (--protocol-override): "
              f"{sorted(off_protocol)} differ from the reference. These numbers "
              "are NOT comparable to the reference table.")
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
        "protocol": {
            **protocol_of(cfg),
            "patience": cfg.train.patience if cfg.train.early_stopping else None,
            "matches_reference_protocol": not off_protocol,
            "off_protocol_fields": sorted(off_protocol) or None,
            "comparable_to": (
                "NOTHING in the reference table -- run off-protocol"
                if off_protocol else
                f"the reference table (fixed budget {cfg.train.max_steps}, "
                "final checkpoint)"
            ),
        },
        "seeds": args.seeds,
        # Budget-fit evidence, recorded so a variant comparison can be checked
        # for the failure where the budget suits one arm and not the other: a
        # run still descending steeply at the budget was truncated, one flat for
        # most of it wasted the tail. Compare final_loss and late_slope across
        # variants before reading any delta as an architecture or data effect.
        "budget_fit": {
            seed: {
                "final_loss_mean_last_100": round(float(np.mean(v[-100:])), 4),
                "loss_at_half_budget": round(float(np.mean(v[len(v) // 2 - 50:
                                                            len(v) // 2 + 50])), 4),
                "late_slope_per_1k_steps": round(
                    float((np.mean(v[-100:]) - np.mean(v[-600:-500])) / 500 * 1000), 4
                ) if len(v) >= 600 else None,
            }
            for seed, v in sorted(losses_by_seed.items())
        },
        "per_seed": per_seed,
        "aggregate": agg,
        "note": "val accuracy deliberately absent: signer-dependent, "
                "stopping criterion only",
    }, indent=2), encoding="utf-8")
    metrics_path = report_dir / "sweep_metrics.json"
    print(f"\n  -> {report_dir}")

    # DO NOT TRUST THIS PROCESS'S EXIT CODE. VERIFY THE MARKER FILE.
    #
    # A BiLSTM sweep on CUDA reports exit 127 on this machine (Windows, torch
    # 2.13+cu130, RTX 5090) even though it completes and writes valid metrics.
    # Scoped by bisection: transformer and TCN on CUDA exit 0, BiLSTM on CPU
    # exits 0, BiLSTM on CUDA exits 127. A bare cuDNN LSTM forward pass exits 0,
    # so it is not simply the RNN. An os._exit(0) placed here is REACHED and
    # prints, and the shell still sees 127 -- the status is being overridden
    # below Python, so no in-process trick can fix it. Root cause unknown and
    # left unexplained rather than guessed at.
    #
    # Why this matters enough to engineer around: an unattended sweep that exits
    # non-zero after succeeding will eventually be read as a failure -- or worse,
    # train whoever reads it to wave away a genuine failure as "just the BiLSTM
    # thing". So success is signalled by an artifact, not by $?.
    #
    # Callers: check for SWEEP_OK in the report directory, not the exit code.
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} was not written", flush=True)
        return 1
    marker = report_dir / "SWEEP_OK"
    marker.write_text(
        f"{label}\nseeds={args.seeds}\nprotocol={protocol_of(cfg)}\n",
        encoding="utf-8",
    )
    print(f"  SWEEP OK -- success marker written to {marker}")
    print("  (verify this file, not the exit code: a BiLSTM sweep on CUDA "
          "reports 127 despite succeeding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
