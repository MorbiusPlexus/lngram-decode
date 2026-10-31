"""Shared drafting contracts: sequence records and the drafter interface.

Offline acceptance (M3.T4) works on ``SeqRecord``s: a prompt plus the target's
greedy continuation, with the target's memory codes at every position.  A drafter
returns, for every start position of the continuation, gamma drafted tokens:

    row j  (j = 0 .. n_cont-1):  verified prefix = tokens[: prompt_len + j]
                                 drafts predict tokens[prompt_len + j : prompt_len + j + gamma]

Live-information rule (plan M3.T4.2): a drafter may use, for verified positions
(< prompt_len + j), the target's exact codes (unmodified models) or its own
recomputed codes (SD models); for drafted positions only codes it computes itself;
sketch online updates only from verified tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeqRecord:
    tokens: np.ndarray                 # int64 [prompt_len + n_cont]
    prompt_len: int
    text_type: str = "open"            # open | long | code
    codes: dict = field(default_factory=dict)  # memory block pos -> uint8 [len(tokens), R] (target's codes)
    doc_id: int = -1

    @property
    def n_cont(self) -> int:
        return len(self.tokens) - self.prompt_len


class Drafter(ABC):
    name: str = "drafter"

    @abstractmethod
    def drafts(self, rec: SeqRecord, gamma: int) -> np.ndarray:
        """int64 [rec.n_cont, gamma]: row j drafts the gamma tokens after tokens[:prompt_len + j]."""


def accepted_lengths(rec: SeqRecord, drafts: np.ndarray) -> np.ndarray:
    """Leading matches of each draft row against the target's greedy continuation.

    Rows whose reference is shorter than gamma (end of the continuation) are compared
    on the available tokens only; ``n_ref`` tells how many were available."""
    n, gamma = drafts.shape
    out = np.zeros(n, dtype=np.int64)
    for j in range(n):
        ref = rec.tokens[rec.prompt_len + j: rec.prompt_len + j + gamma]
        eq = drafts[j, : len(ref)] == ref
        out[j] = len(ref) if eq.all() else int(np.argmin(eq))
    return out


def full_rows(rec: SeqRecord, gamma: int) -> np.ndarray:
    """Mask of rows with a full gamma-token reference (use these for unbiased means)."""
    return np.arange(rec.n_cont) + gamma <= rec.n_cont
