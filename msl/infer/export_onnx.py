"""Export a trained checkpoint to ONNX and verify it against PyTorch.

Export always runs on CPU. The deployment target is CPU inference at >=15 FPS
(spec S2), training on the GPU is an implementation detail, and tracing on CUDA
can bake device-specific kernels into the graph.

The batch axis is dynamic; the time axis is not. Clips are resampled to a fixed
`seq_len` before batching, and the transformer's learned positional embedding is
sized to exactly that length, so a variable time axis would export a graph that
cannot legally be run at any other length.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from msl.config import ExperimentConfig
from msl.models.build import build_model, load_checkpoint


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset: int = 17,
    verify: bool = True,
    tolerance: float = 1e-4,
) -> Path:
    """Export `checkpoint_path` to `output_path`, returning the written path."""
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = ExperimentConfig.from_dict(_strip_derived(ckpt["config"]))
    glosses = ckpt["glosses"]

    model = build_model(cfg, num_classes=len(glosses))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, cfg.feature.seq_len, cfg.feature.n_features)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    if verify:
        _verify(model, output_path, cfg, tolerance)
    return output_path


def _strip_derived(raw: dict) -> dict:
    """Drop keys that are computed properties rather than stored fields.

    `asdict` on a frozen dataclass only serialises fields, but be defensive:
    a checkpoint written by an older revision may carry extra keys and
    `ExperimentConfig.from_dict` rejects unknown ones.
    """
    allowed = {"feature", "model", "train", "data", "synthetic"}
    return {k: v for k, v in raw.items() if k in allowed}


def _verify(
    model: torch.nn.Module,
    onnx_path: Path,
    cfg: ExperimentConfig,
    tolerance: float,
) -> None:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    # Batch of 4, not 1, so the dynamic batch axis is actually exercised.
    sample = rng.standard_normal(
        (4, cfg.feature.seq_len, cfg.feature.n_features), dtype=np.float32
    )

    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).numpy()
    onnx_out = session.run(["logits"], {"landmarks": sample})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > tolerance:
        raise AssertionError(
            f"ONNX output diverges from PyTorch by {max_diff:.2e} (> {tolerance:.0e})"
        )
    print(f"verified    : max |torch - onnx| = {max_diff:.2e} over a batch of 4")


def benchmark_cpu(onnx_path: str | Path, seq_len: int, n_features: int, runs: int = 50) -> float:
    """Single-clip CPU latency in ms. Spec S2 requires >=15 FPS at inference."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = np.zeros((1, seq_len, n_features), dtype=np.float32)

    for _ in range(5):  # warm up
        session.run(["logits"], {"landmarks": sample})

    start = time.perf_counter()
    for _ in range(runs):
        session.run(["logits"], {"landmarks": sample})
    return (time.perf_counter() - start) / runs * 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/model.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args(argv)

    path = export_onnx(args.checkpoint, args.output, args.opset, verify=not args.no_verify)
    size_mb = path.stat().st_size / 1024**2
    print(f"exported    : {path} ({size_mb:.1f} MiB, opset {args.opset})")

    if args.benchmark:
        ckpt = load_checkpoint(args.checkpoint)
        cfg = ExperimentConfig.from_dict(_strip_derived(ckpt["config"]))
        ms = benchmark_cpu(path, cfg.feature.seq_len, cfg.feature.n_features)
        print(f"cpu latency : {ms:.2f} ms/clip ({1000 / ms:.0f} clips/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
