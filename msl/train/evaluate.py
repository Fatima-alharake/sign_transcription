"""Evaluation for isolated sign recognition (spec S7).

Top-1, top-5, per-class F1, and a confusion matrix as a first-class output --
the point of the matrix is that signs differing only in handshape or only in
movement collapse into each other, and only the matrix shows which pairs.

Plot conventions, applied deliberately rather than by matplotlib default:

* **No dual-axis plot.** Loss and accuracy have unrelated scales; overlaying
  them on twin y-axes invents a relationship the data does not contain. They
  are stacked panels sharing an x-axis instead, one measure per axis.
* **Sequential = one hue, light to dark** for the confusion matrix, with a
  scale legend. A rainbow ramp makes equal magnitude steps look unequal.
* Hairline, solid, recessive grid and axes; no value printed on every cell
  (10,000 of them would be unreadable) -- the ranked pair list carries that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from msl.models.build import SignClassifier

# One-hue sequential ramp, light -> dark, for magnitude.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"
SERIES_1 = "#2a78d6"


@dataclass
class EvalResult:
    top1: float
    top5: float
    macro_f1: float
    n_clips: int
    n_classes: int
    confusion: np.ndarray = field(repr=False)
    per_class_f1: np.ndarray = field(repr=False)
    support: np.ndarray = field(repr=False)

    def worst_pairs(self, glosses: list[str], k: int = 15) -> list[tuple]:
        """Most-confused (true, predicted) pairs, off-diagonal only."""
        cm = self.confusion.copy()
        np.fill_diagonal(cm, 0)
        order = np.argsort(cm, axis=None)[::-1][:k]
        out = []
        for flat in order:
            i, j = np.unravel_index(flat, cm.shape)
            if cm[i, j] == 0:
                break
            out.append((glosses[i], glosses[j], int(cm[i, j]), int(self.support[i])))
        return out


@torch.no_grad()
def evaluate(
    model: SignClassifier,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> EvalResult:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    top1 = top5 = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        # top-5 is min(5, num_classes) so this does not explode on a tiny vocab
        k = min(5, logits.shape[1])
        ranked = logits.topk(k, dim=1).indices
        pred = ranked[:, 0]
        top1 += int((pred == y).sum())
        top5 += int((ranked == y[:, None]).any(dim=1).sum())
        total += int(y.numel())
        for t, p in zip(y.tolist(), pred.tolist()):
            confusion[t, p] += 1
    model.train()

    tp = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    # Macro-F1 over classes that actually appear in this split. Averaging in
    # absent classes as F1=0 would understate the model for a reason that has
    # nothing to do with the model.
    present = support > 0
    macro = float(f1[present].mean()) if present.any() else 0.0

    return EvalResult(
        top1=top1 / total if total else 0.0,
        top5=top5 / total if total else 0.0,
        macro_f1=macro,
        n_clips=total,
        n_classes=int(present.sum()),
        confusion=confusion,
        per_class_f1=f1,
        support=support.astype(np.int64),
    )


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def plot_curves(
    losses: list[float],
    val_accuracies: list[float],
    eval_every: int,
    out_path: str | Path,
    title: str,
    smooth: int = 25,
) -> Path:
    """Training loss and val top-1 as two stacked panels, never twin axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.18},
    )
    fig.patch.set_facecolor(SURFACE)

    steps = np.arange(1, len(losses) + 1)
    ax_loss.plot(steps, losses, color=SEQUENTIAL_BLUE[2], linewidth=0.8,
                 alpha=0.9, label="per step")
    if len(losses) >= smooth > 1:
        kernel = np.ones(smooth) / smooth
        rolling = np.convolve(losses, kernel, mode="valid")
        ax_loss.plot(steps[smooth - 1:], rolling, color=SERIES_1, linewidth=2.0,
                     label=f"{smooth}-step mean")
        legend = ax_loss.legend(loc="upper right", frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(INK_MUTED)  # text wears ink, never the series colour
    _style(ax_loss)
    ax_loss.set_ylabel("training loss", color=INK_MUTED, fontsize=9)

    if val_accuracies:
        val_steps = np.arange(1, len(val_accuracies) + 1) * eval_every
        ax_acc.plot(val_steps, val_accuracies, color=SERIES_1, linewidth=2.0,
                    solid_capstyle="round")
        best = int(np.argmax(val_accuracies))
        # Mark and label only the extreme, never every point.
        ax_acc.plot([val_steps[best]], [val_accuracies[best]], marker="o",
                    markersize=8, markerfacecolor=SERIES_1,
                    markeredgecolor=SURFACE, markeredgewidth=2.0)
        ax_acc.annotate(
            f"best {val_accuracies[best]:.3f}",
            xy=(val_steps[best], val_accuracies[best]),
            xytext=(8, 6), textcoords="offset points",
            fontsize=8, color=INK,
        )
    _style(ax_acc)
    ax_acc.set_ylabel("val top-1", color=INK_MUTED, fontsize=9)
    ax_acc.set_xlabel("step", color=INK_MUTED, fontsize=9)
    ax_acc.set_ylim(bottom=0)

    ax_loss.set_title(title, color=INK, fontsize=10, loc="left", pad=10)
    # Val is a stopping criterion, not a result. Say so on the figure itself so
    # the panel can never be screenshotted into a report as an accuracy claim.
    fig.text(0.5, 0.005,
             "val is signer-dependent (clip-level split inside the training "
             "signers) and is a stopping criterion only, never a result",
             ha="center", fontsize=7, color=INK_MUTED, style="italic")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


# Categorical slots 1-3. These three validate all-pairs CVD separation, which
# is the gate that matters when three lines overlap rather than sit adjacent.
SEED_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def plot_loss_seeds(
    losses_by_seed: dict[int, list[float]],
    out_path: str | Path,
    title: str,
    smooth: int = 25,
) -> Path:
    """Training loss for each seed of one variant, on a single axis.

    Loss only -- no val panel. Val is a stopping criterion, and a sweep figure
    that carries it invites someone to read the spread of a number that is not
    a result.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(SURFACE)
    for i, (seed, losses) in enumerate(sorted(losses_by_seed.items())):
        colour = SEED_SERIES[i % len(SEED_SERIES)]
        steps = np.arange(1, len(losses) + 1)
        if len(losses) >= smooth > 1:
            kernel = np.ones(smooth) / smooth
            rolling = np.convolve(losses, kernel, mode="valid")
            ax.plot(steps[smooth - 1:], rolling, color=colour, linewidth=2.0,
                    label=f"seed {seed}")
        else:
            ax.plot(steps, losses, color=colour, linewidth=2.0,
                    label=f"seed {seed}")
    _style(ax)
    ax.set_xlabel("step", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(f"training loss ({smooth}-step mean)", color=INK_MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=10)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK_MUTED)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_confusion(
    confusion: np.ndarray,
    glosses: list[str],
    out_path: str | Path,
    title: str,
    normalise: bool = True,
) -> Path:
    """Confusion matrix as a one-hue sequential heatmap with a scale legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = confusion.astype(np.float64)
    if normalise:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    cmap.set_under(SURFACE)  # an exactly-zero cell recedes into the surface

    n = len(glosses)
    fig, ax = plt.subplots(figsize=(11, 9.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(cm, cmap=cmap, vmin=1e-9, vmax=cm.max() or 1.0,
                      interpolation="nearest", aspect="equal")

    # 100 tick labels would be unreadable; label every 5th and keep the rest as
    # the axis range. The ranked pair list is what names specific confusions.
    ticks = np.arange(0, n, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([glosses[i] for i in ticks], rotation=90, fontsize=6,
                       color=INK_MUTED)
    ax.set_yticklabels([glosses[i] for i in ticks], fontsize=6, color=INK_MUTED)
    ax.set_xlabel("predicted", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("true", color=INK_MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=12)
    for side in ax.spines.values():
        side.set_linewidth(0.6)
        side.set_color(GRID)
    ax.tick_params(length=2, width=0.6, colors=INK_MUTED)

    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("fraction of true class" if normalise else "clips",
                  color=INK_MUTED, fontsize=8)
    bar.ax.tick_params(labelsize=7, colors=INK_MUTED, length=2, width=0.6)
    bar.outline.set_linewidth(0.6)
    bar.outline.set_edgecolor(GRID)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path
