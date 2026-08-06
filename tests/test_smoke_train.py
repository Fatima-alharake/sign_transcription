"""End-to-end smoke test: 5 steps on synthetic data, then ONNX export.

This is the Milestone 0 exit criterion (spec S3a.1, S10.0). It proves shapes
flow through the model, the loss computes, the training loop runs, checkpoints
are written and reloadable, and ONNX export succeeds -- without a single video
file, MediaPipe, or a camera.

It reports nothing about accuracy. The labels are random by construction.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from msl.config import ExperimentConfig
from msl.data.dataset import build_synthetic_loaders
from msl.infer.export_onnx import benchmark_cpu, export_onnx
from msl.models.build import build_model, load_checkpoint
from msl.runtime import (
    CudaUnavailableError,
    describe_device,
    peak_rss_mb,
    resolve_device,
    seed_everything,
    system_memory_line,
)
from msl.train.train_isolated import cosine_warmup_lambda, train

SMOKE_STEPS = 5


@pytest.fixture
def cfg(tmp_path) -> ExperimentConfig:
    """The shipped synthetic config, shrunk to 5 steps and pointed at tmp_path."""
    base = ExperimentConfig.from_yaml("msl/configs/synthetic.yaml")
    return replace(
        base,
        train=replace(
            base.train,
            max_steps=SMOKE_STEPS,
            warmup_steps=2,
            eval_every=SMOKE_STEPS,  # one evaluation, at the end
            out_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "runs"),
            tensorboard=False,  # keep the smoke test free of disk side effects
        ),
    )


def test_cuda_is_available() -> None:
    """Fail loudly rather than silently training on CPU.

    This machine has an RTX 3060 and the CUDA wheel is pinned in pyproject.toml.
    If this fails, the CPU-only wheel got installed -- re-run `uv sync`.
    """
    assert torch.cuda.is_available(), (
        f"CUDA unavailable. torch {torch.__version__} reports CUDA "
        f"{torch.version.cuda or 'none -- this is a CPU-only wheel'}. "
        "Expected the cu130 build from the pytorch-cu130 index."
    )
    print(f"\ndevice      : {describe_device(resolve_device('auto'))}")
    print(f"             {system_memory_line()}")


def test_requesting_cuda_when_absent_raises_rather_than_falling_back(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CudaUnavailableError, match="CUDA"):
        resolve_device("cuda")
    # ...but auto is allowed to degrade, that is what "auto" means.
    assert resolve_device("auto").type == "cpu"
    assert resolve_device("cpu").type == "cpu"


def test_lr_schedule_warms_up_then_decays_to_zero() -> None:
    warmup, total = 10, 100
    values = [cosine_warmup_lambda(s, warmup, total) for s in range(total)]
    assert values[0] == pytest.approx(0.1)
    assert values[warmup - 1] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(values[:warmup], values[1:warmup]))
    assert all(a >= b for a, b in zip(values[warmup:], values[warmup + 1 :]))
    assert values[-1] < 0.01


def test_trains_five_steps_with_finite_loss(cfg: ExperimentConfig) -> None:
    """The core assertion of Milestone 0."""
    device = resolve_device("auto")
    result = train(cfg, device=device, verbose=False)

    assert result.steps_run == SMOKE_STEPS
    assert len(result.losses) == SMOKE_STEPS
    for step, loss in enumerate(result.losses):
        assert math.isfinite(loss), f"loss became {loss} at step {step}"
        assert loss > 0.0

    assert result.device == str(device)
    print(f"\nlosses      : {[round(v, 4) for v in result.losses]}")
    print(f"peak RSS    : {result.peak_rss_mb:.0f} MiB")


@pytest.mark.parametrize("arch", ["transformer", "bilstm", "tcn"])
def test_every_architecture_trains(cfg: ExperimentConfig, arch: str) -> None:
    result = train(replace(cfg, model=replace(cfg.model, arch=arch)), verbose=False)
    assert result.steps_run == SMOKE_STEPS
    assert all(math.isfinite(v) for v in result.losses)


def test_training_updates_the_weights(cfg: ExperimentConfig) -> None:
    """A finite loss is not enough: a detached graph or a zeroed LR would also
    produce five finite numbers and no learning at all.

    To recover the *initial* weights we replay train()'s construction order
    under the same seed -- seed, build loaders, build model -- so `before` is
    genuinely the init the run started from and not an unrelated random draw.
    """
    seed_everything(cfg.train.seed)
    build_synthetic_loaders(cfg)
    before = {
        k: v.clone()
        for k, v in build_model(cfg, cfg.synthetic.num_classes).state_dict().items()
    }

    result = train(cfg, verbose=False)
    after = load_checkpoint(result.last_checkpoint)["model"]

    assert set(before) == set(after)
    changed = [k for k in before if not torch.equal(before[k].cpu(), after[k].cpu())]
    unchanged_weights = [
        k
        for k in before
        if k.endswith(".weight") and torch.equal(before[k].cpu(), after[k].cpu())
    ]
    assert changed, "no weight changed over 5 steps"
    assert not unchanged_weights, f"these weights never moved: {unchanged_weights}"


def test_checkpoint_is_written_and_reloadable(cfg: ExperimentConfig) -> None:
    result = train(cfg, verbose=False)
    assert result.last_checkpoint is not None and result.last_checkpoint.exists()
    assert result.best_checkpoint is not None and result.best_checkpoint.exists()

    ckpt = load_checkpoint(result.last_checkpoint)
    assert ckpt["step"] == SMOKE_STEPS
    assert len(ckpt["glosses"]) == cfg.synthetic.num_classes
    assert "optimizer" in ckpt

    model = build_model(cfg, num_classes=len(ckpt["glosses"]))
    model.load_state_dict(ckpt["model"])  # strict: every key must match


def test_run_is_reproducible(cfg: ExperimentConfig) -> None:
    """Same seed, same losses -- otherwise no result from this project is
    comparable to any other."""
    first = train(cfg, verbose=False).losses
    second = train(cfg, verbose=False).losses
    assert first == pytest.approx(second, abs=1e-5)


def test_exports_to_onnx_and_matches_pytorch(cfg: ExperimentConfig, tmp_path) -> None:
    """Export runs on CPU: that is the deployment target (spec S2)."""
    result = train(cfg, verbose=False)
    onnx_path = export_onnx(result.best_checkpoint, tmp_path / "model.onnx", verify=True)

    assert onnx_path.exists()
    size_mb = onnx_path.stat().st_size / 1024**2
    assert size_mb > 0

    latency_ms = benchmark_cpu(
        onnx_path, cfg.feature.seq_len, cfg.feature.n_features, runs=20
    )
    print(f"\nonnx        : {size_mb:.1f} MiB")
    print(f"cpu latency : {latency_ms:.2f} ms/clip ({1000 / latency_ms:.0f} clips/s)")
    # Spec S2 wants >=15 FPS on CPU. A 64-frame clip must therefore cost under
    # ~4.3 s; this is a sanity floor, not the real realtime test (Milestone 5).
    assert latency_ms < 4266


def test_peak_rss_stays_modest(cfg: ExperimentConfig) -> None:
    """16GB machine, typically ~90% committed. Synthetic training is O(1) in
    the corpus size, so anything large here means something is materialising
    data it should be streaming."""
    result = train(cfg, verbose=False)
    assert result.peak_rss_mb > 0
    assert result.peak_rss_mb < 6000, (
        f"peak RSS {result.peak_rss_mb:.0f} MiB is higher than a synthetic "
        "5-step run should need"
    )
    print(f"\npeak RSS    : {peak_rss_mb():.0f} MiB")
