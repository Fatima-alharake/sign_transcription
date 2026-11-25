from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from msl.models.build import SignClassifier


@dataclass(frozen=True)
class SearchResult:
    rank: int
    video_id: str
    gloss: str
    gloss_id: int
    signer_id: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignRetrievalGallery:
    def __init__(
        self,
        embeddings: np.ndarray,
        video_ids: list[str],
        glosses: list[str],
        gloss_ids: list[int],
        signer_ids: list[str],
    ) -> None:
        if len(embeddings) != len(video_ids):
            raise ValueError(
                f"embeddings count ({len(embeddings)}) != video_ids ({len(video_ids)})"
            )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.embeddings = embeddings / norms

        self.video_ids = list(video_ids)
        self.glosses = list(glosses)
        self.gloss_ids = [int(g) for g in gloss_ids]
        self.signer_ids = [str(s) for s in signer_ids]

    def __len__(self) -> int:
        return len(self.video_ids)

    def add_item(
        self,
        embedding: np.ndarray,
        video_id: str,
        gloss: str,
        gloss_id: int,
        signer_id: str = "user",
    ) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb = emb / norm
        self.embeddings = np.vstack([self.embeddings, emb]) if len(self.embeddings) > 0 else emb
        self.video_ids.append(str(video_id))
        self.glosses.append(str(gloss))
        self.gloss_ids.append(int(gloss_id))
        self.signer_ids.append(str(signer_id))

    def merge(self, other: SignRetrievalGallery) -> None:
        existing_vids = set(self.video_ids)
        for i, vid in enumerate(other.video_ids):
            if vid not in existing_vids:
                self.add_item(
                    other.embeddings[i],
                    vid,
                    other.glosses[i],
                    other.gloss_ids[i],
                    other.signer_ids[i],
                )
                existing_vids.add(vid)

    def remove_last_item(self) -> tuple[str, str] | None:
        if len(self.video_ids) == 0:
            return None
        vid = self.video_ids.pop()
        gloss = self.glosses.pop()
        self.gloss_ids.pop()
        self.signer_ids.pop()
        dim = self.embeddings.shape[1] if self.embeddings.ndim > 1 else 128
        self.embeddings = (
            self.embeddings[:-1]
            if len(self.embeddings) > 1
            else np.empty((0, dim), dtype=np.float32)
        )
        return vid, gloss

    @classmethod
    def from_dataset(
        cls,
        model: SignClassifier,
        dataset: Any,
        device: torch.device,
        batch_size: int = 32,
        mirror_exemplars: bool = True,
    ) -> SignRetrievalGallery:
        model.eval()
        embeddings_list: list[np.ndarray] = []
        video_ids: list[str] = []
        glosses: list[str] = []
        gloss_ids: list[int] = []
        signer_ids: list[str] = []

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        saved_aug = getattr(dataset, "_aug", None)
        if hasattr(dataset, "_aug"):
            dataset._aug = None
        try:
            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(device)
                    emb = model.embed(x, normalize=True).cpu().numpy()
                    embeddings_list.append(emb)

            for row in dataset.rows:
                video_ids.append(row["video_id"])
                glosses.append(row["gloss"])
                gloss_ids.append(int(row["gloss_id"]))
                signer_ids.append(str(row["signer_id"]))

            if mirror_exemplars and hasattr(dataset, "get_item"):
                mirrored_x_list: list[Tensor] = []
                for i in range(len(dataset)):
                    x_mir, _ = dataset.get_item(i, mirror=True)
                    mirrored_x_list.append(x_mir)

                with torch.no_grad():
                    for start_idx in range(0, len(mirrored_x_list), batch_size):
                        batch_x = torch.stack(
                            mirrored_x_list[start_idx : start_idx + batch_size]
                        ).to(device)
                        emb_mir = model.embed(batch_x, normalize=True).cpu().numpy()
                        embeddings_list.append(emb_mir)

                for row in dataset.rows:
                    video_ids.append(f"{row['video_id']}_mirrored")
                    glosses.append(row["gloss"])
                    gloss_ids.append(int(row["gloss_id"]))
                    signer_ids.append(f"{row['signer_id']}_mirrored")
        finally:
            if hasattr(dataset, "_aug"):
                dataset._aug = saved_aug

        all_embeddings = (
            np.concatenate(embeddings_list, axis=0) if embeddings_list else np.empty((0, 128))
        )
        return cls(
            embeddings=all_embeddings,
            video_ids=video_ids,
            glosses=glosses,
            gloss_ids=gloss_ids,
            signer_ids=signer_ids,
        )

    def query(
        self,
        query_emb: np.ndarray | Tensor,
        top_k: int = 5,
        exclude_video_id: str | None = None,
        exclude_signer_id: str | None = None,
    ) -> list[SearchResult]:
        if len(self) == 0:
            return []

        if isinstance(query_emb, Tensor):
            query_emb = query_emb.detach().cpu().numpy()

        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            q = q / norm

        sims = (q @ self.embeddings.T).flatten()
        ranked_indices = np.argsort(sims)[::-1]

        results: list[SearchResult] = []
        rank = 1
        for idx in ranked_indices:
            vid = self.video_ids[idx]
            sid = self.signer_ids[idx]

            if exclude_video_id is not None and (
                vid == exclude_video_id or vid.startswith(f"{exclude_video_id}_")
            ):
                continue
            if exclude_signer_id is not None and (
                sid == exclude_signer_id or sid.startswith(f"{exclude_signer_id}_")
            ):
                continue

            results.append(
                SearchResult(
                    rank=rank,
                    video_id=vid,
                    gloss=self.glosses[idx],
                    gloss_id=self.gloss_ids[idx],
                    signer_id=sid,
                    score=float(sims[idx]),
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def predict_sign(
        self,
        query_emb: np.ndarray | Tensor,
        k: int = 5,
    ) -> tuple[str, float]:
        matches = self.query(query_emb, top_k=k)
        if not matches:
            return ("unknown", 0.0)

        vote_scores: dict[str, float] = {}
        for m in matches:
            weight = max(0.0, (m.score + 1.0) / 2.0)
            vote_scores[m.gloss] = vote_scores.get(m.gloss, 0.0) + weight

        best_gloss = max(vote_scores, key=vote_scores.get)
        total_weight = sum(vote_scores.values()) + 1e-12
        confidence = vote_scores[best_gloss] / total_weight
        return (best_gloss, float(confidence))

    def class_probabilities(
        self,
        query_emb: np.ndarray | Tensor,
        num_classes: int,
        temperature: float = 0.08,
        user_boost: float = 0.05,
    ) -> np.ndarray | None:
        if len(self) == 0 or num_classes <= 0:
            return None

        if isinstance(query_emb, Tensor):
            query_emb = query_emb.detach().cpu().numpy()

        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            q = q / norm

        sims = (q @ self.embeddings.T).flatten()
        if user_boost > 0.0:
            user_mask = np.array([s == "user" for s in self.signer_ids], dtype=bool)
            if user_mask.any():
                sims = sims.copy()
                sims[user_mask] = np.minimum(1.0, sims[user_mask] + user_boost)

        g_ids = np.asarray(self.gloss_ids, dtype=np.int64)
        baseline_sim = float(np.min(sims)) if sims.size > 0 else -1.0
        class_sims = np.full(num_classes, baseline_sim, dtype=np.float32)
        for idx, sim in zip(g_ids, sims):
            if 0 <= idx < num_classes and sim > class_sims[idx]:
                class_sims[idx] = sim

        scaled = class_sims / max(temperature, 1e-4)
        scaled -= np.max(scaled)
        exp_scores = np.exp(scaled)
        return (exp_scores / (np.sum(exp_scores) + 1e-12)).astype(np.float32)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            video_ids=np.array(self.video_ids, dtype=object),
            glosses=np.array(self.glosses, dtype=object),
            gloss_ids=np.array(self.gloss_ids, dtype=np.int64),
            signer_ids=np.array(self.signer_ids, dtype=object),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> SignRetrievalGallery:
        data = np.load(path, allow_pickle=True)
        return cls(
            embeddings=data["embeddings"],
            video_ids=list(data["video_ids"]),
            glosses=list(data["glosses"]),
            gloss_ids=list(data["gloss_ids"]),
            signer_ids=list(data["signer_ids"]),
        )


@torch.no_grad()
def evaluate_retrieval(
    model: SignClassifier,
    gallery: SignRetrievalGallery,
    query_dataset: Any,
    device: torch.device,
    top_k: int = 5,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(
        query_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    r1_correct = 0
    r5_correct = 0
    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    queries_evaluated = 0

    gallery_signers = set(gallery.signer_ids)
    query_signers = {r["signer_id"] for r in query_dataset.rows}
    signer_disjoint = len(gallery_signers.intersection(query_signers)) == 0

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        query_label = int(y.item())
        query_vid = query_dataset.rows[idx]["video_id"]

        q_emb = model.embed(x, normalize=True).cpu().numpy()[0]
        matches = gallery.query(
            q_emb,
            top_k=len(gallery),
            exclude_video_id=query_vid,
        )
        if not matches:
            continue

        queries_evaluated += 1
        if matches[0].gloss_id == query_label:
            r1_correct += 1

        top5_gloss_ids = [m.gloss_id for m in matches[: min(5, len(matches))]]
        if query_label in top5_gloss_ids:
            r5_correct += 1

        first_rank: int | None = None
        n_rel = 0
        precisions: list[float] = []
        for rank_idx, m in enumerate(matches, start=1):
            if m.gloss_id == query_label:
                n_rel += 1
                precisions.append(n_rel / rank_idx)
                if first_rank is None:
                    first_rank = rank_idx

        reciprocal_ranks.append(1.0 / first_rank if first_rank is not None else 0.0)
        average_precisions.append(float(np.mean(precisions)) if precisions else 0.0)

    n_q = max(1, queries_evaluated)
    return {
        "recall_at_1": round(r1_correct / n_q, 4),
        "recall_at_5": round(r5_correct / n_q, 4),
        "mAP": round(float(np.mean(average_precisions)), 4) if average_precisions else 0.0,
        "mrr": round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0,
        "n_queries": queries_evaluated,
        "gallery_size": len(gallery),
        "signer_disjoint": signer_disjoint,
    }
