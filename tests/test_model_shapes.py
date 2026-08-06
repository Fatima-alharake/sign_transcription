"""Model wiring: shapes flow, architectures are interchangeable, and the
backbone/head split actually survives a checkpoint round-trip."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from msl.config import ExperimentConfig, ModelConfig
from msl.models.build import build_model, load_backbone_only, save_checkpoint
from msl.models.encoder import build_backbone
from msl.models.heads import CTCHead

ARCHS = ["transformer", "bilstm", "tcn"]


@pytest.fixture
def cfg() -> ExperimentConfig:
    """Small config: these tests are about shapes, not capacity."""
    base = ExperimentConfig.from_yaml("msl/configs/synthetic.yaml")
    return replace(base, model=replace(base.model, d_model=32, n_layers=2, ffn_dim=64))


@pytest.mark.parametrize("arch", ARCHS)
def test_backbone_pools_to_d_model(cfg: ExperimentConfig, arch: str) -> None:
    model_cfg = replace(cfg.model, arch=arch)
    backbone = build_backbone(cfg.feature.n_features, model_cfg, cfg.feature.seq_len)
    x = torch.randn(3, cfg.feature.seq_len, cfg.feature.n_features)
    out = backbone(x)
    assert out.shape == (3, model_cfg.d_model)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ARCHS)
def test_classifier_emits_one_logit_per_class(cfg: ExperimentConfig, arch: str) -> None:
    model = build_model(replace(cfg, model=replace(cfg.model, arch=arch)), num_classes=7)
    out = model(torch.randn(2, cfg.feature.seq_len, cfg.feature.n_features))
    assert out.shape == (2, 7)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ARCHS)
def test_gradients_reach_every_parameter(cfg: ExperimentConfig, arch: str) -> None:
    """A parameter with no gradient is a wiring bug that training would hide."""
    model = build_model(replace(cfg, model=replace(cfg.model, arch=arch)), num_classes=4)
    model(torch.randn(4, cfg.feature.seq_len, cfg.feature.n_features)).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


@pytest.mark.parametrize("arch", ARCHS)
def test_batch_size_does_not_change_per_sample_output(cfg: ExperimentConfig, arch: str) -> None:
    """Guards against leakage across the batch dimension -- BatchNorm in the TCN
    makes this a real risk, so it must be in eval mode at inference."""
    model = build_model(replace(cfg, model=replace(cfg.model, arch=arch)), num_classes=4)
    model.eval()
    batch = torch.randn(5, cfg.feature.seq_len, cfg.feature.n_features)
    with torch.no_grad():
        together = model(batch)
        alone = model(batch[2:3])
    assert torch.allclose(together[2], alone[0], atol=1e-5)


def test_dropout_is_active_in_train_mode_only(cfg: ExperimentConfig) -> None:
    model = build_model(cfg, num_classes=4)
    x = torch.randn(2, cfg.feature.seq_len, cfg.feature.n_features)
    model.train()
    assert not torch.equal(model(x), model(x))
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_transformer_rejects_wrong_sequence_length(cfg: ExperimentConfig) -> None:
    """The learned positional embedding is sized to seq_len; a mismatch must
    fail loudly rather than broadcast into nonsense."""
    model = build_model(cfg, num_classes=4)
    with pytest.raises(ValueError, match="frames"):
        model(torch.randn(2, cfg.feature.seq_len + 1, cfg.feature.n_features))


def test_state_dict_separates_backbone_from_head(cfg: ExperimentConfig) -> None:
    keys = build_model(cfg, num_classes=4).state_dict().keys()
    assert any(k.startswith("backbone.") for k in keys)
    assert any(k.startswith("head.") for k in keys)
    assert all(k.startswith(("backbone.", "head.")) for k in keys)


def test_backbone_transfers_to_a_different_vocabulary_size(
    cfg: ExperimentConfig, tmp_path
) -> None:
    """The AUTSL -> Mongolian fine-tuning path (spec S3a.3): reload the
    backbone under a new head without touching the pretrained weights."""
    pretrained = build_model(cfg, num_classes=226)  # proxy corpus
    ckpt = tmp_path / "pretrained.pt"
    save_checkpoint(ckpt, pretrained, cfg, [f"SYNTH_{i:02d}" for i in range(226)], 1, 0.0)

    adapted = build_model(cfg, num_classes=12)  # smaller MSL vocabulary
    missing = load_backbone_only(adapted, ckpt)
    assert missing == []

    for (name, a), (_, b) in zip(
        adapted.backbone.named_parameters(), pretrained.backbone.named_parameters()
    ):
        assert torch.equal(a, b), f"backbone weight {name} did not transfer"

    assert adapted.head.fc.out_features == 12
    out = adapted(torch.randn(2, cfg.feature.seq_len, cfg.feature.n_features))
    assert out.shape == (2, 12)


def test_checkpoint_carries_the_vocabulary(cfg: ExperimentConfig, tmp_path) -> None:
    """A checkpoint without its gloss order is uninterpretable."""
    model = build_model(cfg, num_classes=3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, cfg, ["A", "B", "C"], step=5, best_val_acc=0.5)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["glosses"] == ["A", "B", "C"]
    assert loaded["step"] == 5
    assert loaded["config"]["model"]["arch"] == cfg.model.arch


def test_checkpoint_write_leaves_no_temp_file(cfg: ExperimentConfig, tmp_path) -> None:
    """Atomic write: temp file renamed into place, never left behind."""
    save_checkpoint(tmp_path / "c.pt", build_model(cfg, 3), cfg, ["A", "B", "C"], 1, 0.0)
    assert [p.name for p in tmp_path.iterdir()] == ["c.pt"]


def test_unknown_arch_names_the_valid_options(cfg: ExperimentConfig) -> None:
    with pytest.raises(ValueError, match="bilstm"):
        build_backbone(10, replace(cfg.model, arch="resnet"), cfg.feature.seq_len)


def test_deferred_components_fail_with_a_milestone_pointer(cfg: ExperimentConfig) -> None:
    with pytest.raises(NotImplementedError, match="Milestone 4"):
        build_backbone(10, replace(cfg.model, arch="stgcn"), cfg.feature.seq_len)
    with pytest.raises(NotImplementedError, match="Milestone 8"):
        CTCHead(d_model=32, num_classes=4)


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        ExperimentConfig.from_dict({"model": {"d_model": 64, "nlayers": 3}})
    with pytest.raises(ValueError, match="unknown config section"):
        ExperimentConfig.from_dict({"modle": {}})


def test_config_feature_dim_tracks_the_z_flag() -> None:
    with_z = ExperimentConfig.from_dict({"feature": {"use_z": True}})
    without_z = ExperimentConfig.from_dict({"feature": {"use_z": False}})
    assert with_z.feature.n_features > without_z.feature.n_features
