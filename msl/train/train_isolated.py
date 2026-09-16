"""Isolated sign recognition training loop.

Step-based rather than epoch-based, because the synthetic and placeholder
corpora are tiny and "epoch" is not a meaningful unit at 20 clips per class.

Importable: `train()` takes a config and returns a `TrainResult`, so the smoke
test drives it directly instead of shelling out to the CLI.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from msl.config import ExperimentConfig
from msl.data.dataset import build_synthetic_loaders, infinite
from msl.data.vocab import Vocab, placeholder_vocab
from msl.models.build import SignClassifier, build_model, save_checkpoint
from msl.runtime import (
    describe_device,
    peak_rss_mb,
    resolve_device,
    seed_everything,
    system_memory_line,
)


@dataclass
class TrainResult:
    steps_run: int
    losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    device: str = "cpu"
    peak_rss_mb: float = 0.0


def cosine_warmup_lambda(step: int, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup then cosine decay to zero, as a multiplier on the base LR."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if max_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate_top1(
    model: SignClassifier, loader: DataLoader, device: torch.device
) -> float:
    """Top-1 accuracy. Full evaluation (top-5, per-class F1, confusion matrix)
    lands in train/evaluate.py at Milestone 4; early stopping only needs this."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1)
        correct += int((preds == y).sum())
        total += int(y.numel())
    model.train()
    return correct / total if total else 0.0


def train(
    cfg: ExperimentConfig,
    device: torch.device | None = None,
    vocab: Vocab | None = None,
    verbose: bool = True,
    loaders: tuple[DataLoader, DataLoader] | None = None,
) -> TrainResult:
    """Train an isolated-sign classifier.

    `loaders` lets a caller supply real (train, val) DataLoaders; without it the
    synthetic source is used, which is all Milestone 0 needs. The loop itself is
    source-agnostic -- it only ever sees `(B, T, F)` tensors and integer labels,
    and the labels must be indices into `vocab`, not arbitrary ids.
    """
    device = device or resolve_device("auto")
    seed_everything(cfg.train.seed)

    vocab = vocab or placeholder_vocab(cfg.synthetic.num_classes)
    train_loader, val_loader = loaders if loaders is not None else build_synthetic_loaders(cfg)

    model = build_model(cfg, num_classes=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step, cfg.train.warmup_steps, cfg.train.max_steps
        ),
    )

    writer = None
    if cfg.train.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=cfg.train.log_dir)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"device      : {describe_device(device)}")
        print(f"             {system_memory_line()}")
        print(f"arch        : {cfg.model.arch}  ({n_params:,} params)")
        print(f"input       : (B, {cfg.feature.seq_len}, {cfg.feature.n_features})")
        print(f"classes     : {len(vocab)}")

    result = TrainResult(steps_run=0, device=str(device))
    batches = infinite(train_loader)
    evals_without_improvement = 0
    model.train()

    for step in range(cfg.train.max_steps):
        x, y = next(batches)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        loss = criterion(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach())
        result.losses.append(loss_value)
        result.steps_run = step + 1
        if writer is not None:
            writer.add_scalar("train/loss", loss_value, step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        is_last = step == cfg.train.max_steps - 1
        if cfg.train.eval_every > 0 and ((step + 1) % cfg.train.eval_every == 0 or is_last):
            val_acc = evaluate_top1(model, val_loader, device)
            result.val_accuracies.append(val_acc)
            if writer is not None:
                writer.add_scalar("val/top1", val_acc, step)
            if verbose:
                print(
                    f"step {step + 1:>5}/{cfg.train.max_steps}  "
                    f"loss {loss_value:.4f}  val_top1 {val_acc:.3f}"
                )

            if val_acc > result.best_val_acc:
                result.best_val_acc = val_acc
                evals_without_improvement = 0
                save_checkpoint(
                    best_path, model, cfg, vocab.glosses, step + 1, val_acc, optimizer
                )
                result.best_checkpoint = best_path
            else:
                evals_without_improvement += 1
                if (cfg.train.early_stopping
                        and evals_without_improvement >= cfg.train.patience):
                    if verbose:
                        print(f"early stopping at step {step + 1} (val top-1 plateaued)")
                    break

    save_checkpoint(
        last_path, model, cfg, vocab.glosses, result.steps_run, result.best_val_acc, optimizer
    )
    result.last_checkpoint = last_path
    if result.best_checkpoint is None:  # never evaluated, e.g. a 5-step smoke run
        result.best_checkpoint = last_path

    if writer is not None:
        writer.close()

    result.peak_rss_mb = peak_rss_mb()
    if verbose:
        print(f"peak RSS    : {result.peak_rss_mb:.0f} MiB")
        print(f"checkpoints : {result.best_checkpoint}, {last_path}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train isolated sign recognition")
    parser.add_argument("--config", type=Path, default=Path("msl/configs/synthetic.yaml"))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="train on random tensors, no video files (spec S3a.1). "
        "Milestone 0 supports nothing else.",
    )
    parser.add_argument(
        "--arch",
        choices=["transformer", "bilstm", "tcn", "stgcn"],
        default=None,
        help="override the architecture in the config",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (cuda if available), cuda, or cpu",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    from dataclasses import replace

    args = build_arg_parser().parse_args(argv)
    cfg = ExperimentConfig.from_yaml(args.config)

    if not args.synthetic:
        raise SystemExit(
            "Milestone 0 is synthetic-only: pass --synthetic. The real-data path "
            "arrives with dataset.py in Milestone 2."
        )

    if args.arch:
        cfg = replace(cfg, model=replace(cfg.model, arch=args.arch))
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.out_dir is not None:
        overrides["out_dir"] = str(args.out_dir)
    if args.no_tensorboard:
        overrides["tensorboard"] = False
    if overrides:
        cfg = replace(cfg, train=replace(cfg.train, **overrides))

    train(cfg, device=resolve_device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
