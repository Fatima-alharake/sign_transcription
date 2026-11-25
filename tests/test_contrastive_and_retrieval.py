import numpy as np
import torch

from msl.infer.retrieval import SignRetrievalGallery
from msl.models.heads import ProjectionHead
from msl.train.contrastive import SupConLoss, augment_landmarks_torch


def test_supcon_loss_identical_and_distinct():
    loss_fn = SupConLoss(temperature=0.07)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    loss_aligned = loss_fn(features, labels=labels)
    assert loss_aligned >= 0.0

    features_misaligned = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    loss_misaligned = loss_fn(features_misaligned, labels=labels)
    assert loss_misaligned > loss_aligned


def test_supcon_loss_multiview():
    loss_fn = SupConLoss(temperature=0.07)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    features = torch.randn(4, 2, 32)
    loss = loss_fn(features, labels=labels)
    assert not torch.isnan(loss)
    assert loss.ndim == 0
    assert float(loss) > 0.0


def test_supcon_loss_no_positives_returns_zero():
    loss_fn = SupConLoss(temperature=0.07)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    features = torch.randn(4, 16)
    loss = loss_fn(features, labels=labels)
    assert float(loss.detach()) == 0.0


def test_augment_landmarks_torch_shape_and_values():
    x = torch.ones(4, 64, 182)
    aug = augment_landmarks_torch(x, scale_range=(0.95, 1.05), noise_std=0.01)
    assert aug.shape == x.shape
    assert torch.all(aug > 0.8)
    assert torch.all(aug < 1.2)


def test_projection_head():
    head = ProjectionHead(d_model=64, proj_dim=32, dropout=0.0)
    emb = torch.randn(8, 64)
    proj = head(emb)
    assert proj.shape == (8, 32)


def test_sign_retrieval_gallery_query_and_predict():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.95, 0.05],
        ],
        dtype=np.float32,
    )
    gallery = SignRetrievalGallery(
        embeddings=embeddings,
        video_ids=["v0", "v1", "v2", "v3"],
        glosses=["book", "book", "drink", "drink"],
        gloss_ids=[0, 0, 1, 1],
        signer_ids=["s1", "s2", "s1", "s2"],
    )
    assert len(gallery) == 4

    q_book = np.array([0.99, 0.01, 0.0], dtype=np.float32)
    results = gallery.query(q_book, top_k=2)
    assert len(results) == 2
    assert results[0].gloss == "book"
    assert results[0].video_id == "v0"
    assert results[0].score > 0.9

    results_excl = gallery.query(q_book, top_k=2, exclude_video_id="v0")
    assert results_excl[0].video_id == "v1"

    pred_gloss, conf = gallery.predict_sign(q_book, k=3)
    assert pred_gloss == "book"
    assert conf > 0.5


def test_sign_retrieval_gallery_save_and_load(tmp_path):
    embeddings = np.random.randn(5, 16).astype(np.float32)
    gallery = SignRetrievalGallery(
        embeddings=embeddings,
        video_ids=[f"v{i}" for i in range(5)],
        glosses=[f"g{i}" for i in range(5)],
        gloss_ids=list(range(5)),
        signer_ids=[f"s{i}" for i in range(5)],
    )
    save_path = tmp_path / "gallery.npz"
    gallery.save(save_path)
    assert save_path.is_file()

    loaded = SignRetrievalGallery.load(save_path)
    assert len(loaded) == 5
    assert loaded.video_ids == gallery.video_ids
    assert loaded.glosses == gallery.glosses
    np.testing.assert_allclose(loaded.embeddings, gallery.embeddings, atol=1e-6)


def test_sign_retrieval_gallery_add_item_and_merge():
    g1 = SignRetrievalGallery(
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        video_ids=["v0"],
        glosses=["book"],
        gloss_ids=[0],
        signer_ids=["s0"],
    )
    g1.add_item(np.array([0.0, 1.0], dtype=np.float32), "v1", "drink", 1, "user")
    assert len(g1) == 2
    assert g1.video_ids == ["v0", "v1"]
    assert g1.glosses == ["book", "drink"]

    g2 = SignRetrievalGallery(
        embeddings=np.array([[0.5, 0.5]], dtype=np.float32),
        video_ids=["v2"],
        glosses=["chair"],
        gloss_ids=[2],
        signer_ids=["s2"],
    )
    g1.merge(g2)
    assert len(g1) == 3
    assert "v2" in g1.video_ids


def test_sign_retrieval_gallery_from_dataset_with_mirror():
    class DummyDataset:
        def __init__(self):
            self.rows = [
                {"video_id": "v0", "gloss": "book", "gloss_id": 0, "signer_id": "s0"},
                {"video_id": "v1", "gloss": "drink", "gloss_id": 1, "signer_id": "s1"},
            ]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return torch.randn(64, 30), self.rows[idx]["gloss_id"]

        def get_item(self, idx, mirror=False):
            return torch.randn(64, 30), self.rows[idx]["gloss_id"]

    class DummyModel(torch.nn.Module):
        def embed(self, x, normalize=True):
            return torch.randn(x.size(0), 16)

    gallery = SignRetrievalGallery.from_dataset(
        DummyModel(), DummyDataset(), device=torch.device("cpu"), mirror_exemplars=True
    )
    assert len(gallery) == 4
    assert "v0" in gallery.video_ids
    assert "v0_mirrored" in gallery.video_ids
    assert "v1" in gallery.video_ids
    assert "v1_mirrored" in gallery.video_ids
