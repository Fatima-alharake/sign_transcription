"""Gloss vocabulary: the mapping between gloss labels and integer class ids.

NON-NEGOTIABLE (spec S4a): this module never generates, invents or suggests sign
vocabulary. It only reads a gloss list supplied by the operator, sourced from
native signers or a published MSL dictionary. `placeholder_vocab` below is the
sole exception and it emits deliberately non-lexical labels that cannot be
mistaken for signs.
"""

from __future__ import annotations

import json
from pathlib import Path


class Vocab:
    """Ordered gloss list with a stable gloss <-> id mapping.

    Order is significant: it fixes the column order of the classifier head, so
    a checkpoint is only interpretable alongside the vocab it was trained with.
    """

    def __init__(self, glosses: list[str]) -> None:
        if not glosses:
            raise ValueError("vocabulary is empty")
        duplicates = {g for g in glosses if glosses.count(g) > 1}
        if duplicates:
            raise ValueError(f"duplicate glosses: {sorted(duplicates)}")
        self._glosses: list[str] = list(glosses)
        self._to_id: dict[str, int] = {g: i for i, g in enumerate(self._glosses)}

    def __len__(self) -> int:
        return len(self._glosses)

    def __iter__(self):
        return iter(self._glosses)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vocab) and self._glosses == other._glosses

    def __repr__(self) -> str:
        return f"Vocab({len(self)} glosses)"

    @property
    def glosses(self) -> list[str]:
        return list(self._glosses)

    def to_id(self, gloss: str) -> int:
        try:
            return self._to_id[gloss]
        except KeyError:
            raise KeyError(f"gloss {gloss!r} not in vocabulary") from None

    def to_gloss(self, index: int) -> str:
        if not 0 <= index < len(self._glosses):
            raise IndexError(f"class id {index} out of range for {len(self)} glosses")
        return self._glosses[index]

    # --- persistence -------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocab":
        """Read a newline-delimited gloss list. Blank lines and # comments skipped."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        glosses = [s.strip() for s in lines]
        return cls([g for g in glosses if g and not g.startswith("#")])

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"glosses": self._glosses}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "Vocab":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(raw["glosses"]))


def placeholder_vocab(n: int) -> Vocab:
    """Non-lexical labels for synthetic-tensor runs (spec S3a.1).

    Deliberately named SYNTH_00.. so that nothing in a log, checkpoint or
    confusion matrix produced in synthetic mode can be mistaken for an MSL
    gloss. Real vocabularies are loaded from an operator-supplied file.
    """
    return Vocab([f"SYNTH_{i:02d}" for i in range(n)])
