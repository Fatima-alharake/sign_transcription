"""Device selection and memory reporting.

Kept separate from the training loop so tests can assert on device policy
without constructing a trainer.
"""

from __future__ import annotations

import os
import random

import numpy as np
import psutil
import torch


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA was explicitly requested but is not usable.

    Deliberately fatal rather than falling back to CPU: a silent CPU fallback
    on a machine with a GPU turns a 2-minute run into a 40-minute one and the
    only symptom is that everything feels slow.
    """


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve a --device argument.

    None or "auto" -> cuda if available, else cpu.
    "cuda"         -> cuda, or raise if unavailable.
    "cpu"          -> cpu.
    """
    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise CudaUnavailableError(
                "--device cuda was requested but torch.cuda.is_available() is False. "
                f"torch {torch.__version__} was built for CUDA "
                f"{torch.version.cuda or 'none (CPU-only wheel)'}. "
                "Reinstall from the CUDA wheel index: uv sync."
            )
        return torch.device(requested)
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unrecognised device {requested!r}; use auto, cuda or cpu")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return f"{device} (CPU)"
    index = device.index or 0
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / 1024**3
    return (
        f"{device} -- {props.name}, {total_gb:.1f} GiB VRAM, "
        f"compute {props.major}.{props.minor}, "
        f"torch {torch.__version__} / CUDA {torch.version.cuda}"
    )


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB.

    Uses the Windows peak working set where available; elsewhere falls back to
    current RSS, which understates the peak but is better than nothing.
    """
    info = psutil.Process(os.getpid()).memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows only
    return (peak if peak is not None else info.rss) / 1024**2


def system_memory_line() -> str:
    vm = psutil.virtual_memory()
    return (
        f"system RAM {vm.total / 1024**3:.1f} GiB total, "
        f"{vm.available / 1024**3:.1f} GiB available ({vm.percent:.0f}% used)"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
