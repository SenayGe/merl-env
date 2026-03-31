"""Deterministic subject-level split assignment."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Hashable, Mapping, Sequence


SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


@dataclass(slots=True, kw_only=True)
class SplitConfig:
    """Configuration for deterministic subject-level split assignment."""

    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2
    seed: int = 7

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Split fractions must sum to 1.0")
        for name, value in (
            ("train_frac", self.train_frac),
            ("val_frac", self.val_frac),
            ("test_frac", self.test_frac),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")


def _subject_sort_key(subject_id: Hashable) -> tuple[str, str]:
    return (type(subject_id).__name__, str(subject_id))


def assign_subject_splits(
    subject_ids: Sequence[Hashable],
    *,
    config: SplitConfig | None = None,
) -> dict[Hashable, str]:
    """Assign one stable split per unique subject."""

    split_config = config or SplitConfig()
    unique_subjects = sorted(set(subject_ids), key=_subject_sort_key)
    rng = random.Random(split_config.seed)
    rng.shuffle(unique_subjects)

    n_subjects = len(unique_subjects)
    n_train = int(round(split_config.train_frac * n_subjects))
    n_val = int(round(split_config.val_frac * n_subjects))

    if n_train > n_subjects:
        n_train = n_subjects
    if n_train + n_val > n_subjects:
        n_val = max(0, n_subjects - n_train)

    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train : n_train + n_val])

    assignments: dict[Hashable, str] = {}
    for subject_id in unique_subjects:
        if subject_id in train_subjects:
            assignments[subject_id] = "train"
        elif subject_id in val_subjects:
            assignments[subject_id] = "val"
        else:
            assignments[subject_id] = "test"
    return assignments


def assign_splits_to_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    subject_id_key: str = "subject_id",
    split_key: str = "split",
    config: SplitConfig | None = None,
) -> list[dict[str, Any]]:
    """Copy rows and attach a subject-level split column."""

    subject_ids = [row[subject_id_key] for row in rows]
    assignments = assign_subject_splits(subject_ids, config=config)

    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied[split_key] = assignments[row[subject_id_key]]
        out.append(copied)
    return out

