"""Train the isolated-sign baseline on WLASL100 -- AMERICAN SIGN LANGUAGE.

READ THE LABEL ON THE NUMBER. Nothing this produces is a Mongolian Sign
Language result. WLASL is ASL, used to pretrain a backbone whose low-level
motion features transfer; the vocabulary does not. The corpus on disk is 55.3%
complete (1127 of 2038 clips), so every figure carries that caveat too.

What is honest here and what is not:

* **TEST is the number.** Signer-disjoint: no signer in test appears in train.
* **VAL is not a number.** It is a clip-level split inside the training
  signers, used only to decide when to stop. Its accuracy is inflated and is
  never reported.
* The split is read from the committed CSV, never recomputed.

Labels are `gloss_id`, so the vocabulary must be built in `gloss_id` order.
Building it alphabetically would leave accuracy unchanged and every class name
on the confusion matrix wrong.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from msl.config import ExperimentConfig
from msl.data.vocab import Vocab
from msl.data.wlasl_loader import (
    DEFAULT_MIN_POSE_RATE,
    WlaslLandmarkDataset,
    describe_splits,
)
from msl.models.build import build_model, load_checkpoint
from msl.runtime import describe_device, resolve_device
from msl.train.evaluate import evaluate, plot_confusion, plot_curves
from msl.train.train_isolated import train

CAVEAT = (
    "AMERICAN SIGN LANGUAGE (WLASL100), corpus 55.3% complete (1127/2038 clips). "
    "Not MSL. Signer-independent split, test signers disjoint from train."
)


def build_loaders(
    cfg: ExperimentConfig,
    min_pose_rate: float,
    log_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocab]:
    splits = {
        name: WlaslLandmarkDataset(name, cfg, log_path=log_path,
                                   min_pose_rate=min_pose_rate)
        for name in ("train", "val", "test")
    }
    vocab = Vocab(splits["train"].glosses)

    def loader(name: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            splits[name],
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=cfg.train.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return (loader("train", True), loader("val", False),
            loader("test", False), vocab)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the isolated-sign baseline on WLASL100 (ASL)")
    parser.add_argument("--config", type=Path,
                        default=Path("msl/configs/wlasl100.yaml"))
    parser.add_argument("--arch", default=None,
                        choices=["transformer", "bilstm", "tcn", "stgcn"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-pose-rate", type=float,
                        default=DEFAULT_MIN_POSE_RATE)
    parser.add_argument("--report-dir", type=Path,
                        default=Path("runs/wlasl100/report"))
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.arch:
        cfg = replace(cfg, model=replace(cfg.model, arch=args.arch))
    device = resolve_device(args.device)

    train_loader, val_loader, test_loader, vocab = build_loaders(
        cfg, args.min_pose_rate)

    print("=" * 74)
    print(CAVEAT)
    print("=" * 74)
    print(json.dumps(describe_splits(), indent=2))
    print(f"clips after the pose-rate filter (>= {args.min_pose_rate}): "
          f"train {len(train_loader.dataset)}  "
          f"val {len(val_loader.dataset)}  test {len(test_loader.dataset)}")
    print(f"vocabulary: {len(vocab)} glosses, ordered by gloss_id "
          f"(first three: {vocab.glosses[:3]})")
    print(f"device: {describe_device(device)}\n")

    result = train(cfg, device=device, vocab=vocab,
                   loaders=(train_loader, val_loader))

    # Evaluate the EARLY-STOPPED checkpoint, not the final one: the last model
    # is whatever the loop happened to end on, and reporting it would quietly
    # use test data that the stopping decision never saw.
    model = build_model(cfg, num_classes=len(vocab)).to(device)
    ckpt = load_checkpoint(result.best_checkpoint, map_location=str(device))
    model.load_state_dict(ckpt["model"])
    if ckpt["glosses"] != vocab.glosses:
        raise SystemExit("checkpoint vocabulary does not match the split CSV")

    test = evaluate(model, test_loader, device, num_classes=len(vocab))

    print("\n" + "=" * 74)
    print("TEST (signer-disjoint) -- " + CAVEAT)
    print("=" * 74)
    print(f"  top-1     : {test.top1:.4f}   ({test.top1 * 100:.2f}%)")
    print(f"  top-5     : {test.top5:.4f}   ({test.top5 * 100:.2f}%)")
    print(f"  macro F1  : {test.macro_f1:.4f}")
    print(f"  clips     : {test.n_clips} over {test.n_classes} classes present")
    print(f"  chance top-1 ~ {1 / len(vocab):.4f}, top-5 ~ {5 / len(vocab):.4f}")

    print("\n  most-confused pairs (true -> predicted, count / support):")
    for true_g, pred_g, count, support in test.worst_pairs(vocab.glosses, k=12):
        print(f"    {true_g:<14} -> {pred_g:<14} {count}/{support}")

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    curves = plot_curves(
        result.losses, result.val_accuracies, cfg.train.eval_every,
        report_dir / "training_curves.png",
        f"WLASL100 {cfg.model.arch} baseline (ASL, 55.3%-complete corpus)",
    )
    matrix = plot_confusion(
        test.confusion, vocab.glosses, report_dir / "confusion_test.png",
        f"Test confusion, row-normalised -- WLASL100 {cfg.model.arch} "
        f"(ASL, signer-disjoint, 55.3%-complete corpus)",
    )
    summary = {
        "language": "American Sign Language (WLASL100)",
        "corpus_completeness": "55.3% (1127 of 2038 clips)",
        "arch": cfg.model.arch,
        "tuned": False,
        "augmentation": None,
        "test_top1": round(test.top1, 4),
        "test_top5": round(test.top5, 4),
        "test_macro_f1": round(test.macro_f1, 4),
        "test_clips": test.n_clips,
        "classes": len(vocab),
        "chance_top1": round(1 / len(vocab), 4),
        "steps_run": result.steps_run,
        "note": "val accuracy deliberately omitted: signer-dependent, "
                "stopping criterion only",
    }
    (report_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    np_path = report_dir / "confusion_test.npy"
    import numpy as np
    np.save(np_path, test.confusion)

    print(f"\n  curves  -> {curves}")
    print(f"  matrix  -> {matrix}")
    print(f"  metrics -> {report_dir / 'test_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
