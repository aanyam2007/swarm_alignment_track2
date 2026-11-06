"""The data contract shared by both halves of the project.

Both the swarm GENERATOR (Colab) and the LOCAL pipeline read and write exactly
this schema. A single JSONL line == one :class:`Message`. This module is the
single source of truth for field names, types, and (de)serialization. Nothing
else in the repo should hand-build a message dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Iterator, List

# --- controlled vocabularies -------------------------------------------------

TOPOLOGIES = ("chain", "star", "full")
SPLITS = ("train", "test", "heldout")

# drift tiers: how far a message has drifted from the verbatim objective.
DRIFT_VERBATIM = 0        # word-for-word (or near) the seed objective
DRIFT_LIGHT = 1           # light paraphrase, same surface vocabulary
DRIFT_REWORDED = 2        # substantially reworded, little shared vocabulary
DRIFT_IMPLIED = 3         # implied / euphemistic, objective only inferable
DRIFT_NA = -1             # not applicable (message does not carry objective)

DRIFT_TIERS = (DRIFT_VERBATIM, DRIFT_LIGHT, DRIFT_REWORDED, DRIFT_IMPLIED, DRIFT_NA)

OBJECTIVE_NONE = "none"


@dataclass
class Message:
    """One message on the shared board. This is THE data contract."""

    episode_id: str
    round: int
    sender_id: str
    is_injected_agent: bool
    topology: str              # one of TOPOLOGIES
    language: str              # BCP-47 code, e.g. "en", "hi"
    text: str
    objective_id: str          # which injected objective, or OBJECTIVE_NONE
    carries_objective: bool     # THE semantic label
    drift_tier: int            # one of DRIFT_TIERS
    is_hard_negative: bool     # shares vocab but does NOT carry the objective
    split: str                 # one of SPLITS
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- (de)serialization ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        # sort_keys for stable, diff-friendly, reproducible output.
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        known = {f: d[f] for f in FIELDS if f in d}
        return cls(**known)

    @classmethod
    def from_json(cls, line: str) -> "Message":
        return cls.from_dict(json.loads(line))

    def validate(self) -> "Message":
        """Fail loud on contract violations; return self for chaining."""
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"bad topology {self.topology!r}, want one of {TOPOLOGIES}")
        if self.split not in SPLITS:
            raise ValueError(f"bad split {self.split!r}, want one of {SPLITS}")
        if self.drift_tier not in DRIFT_TIERS:
            raise ValueError(f"bad drift_tier {self.drift_tier!r}, want one of {DRIFT_TIERS}")
        if not isinstance(self.round, int):
            raise ValueError(f"round must be int, got {type(self.round)}")
        if self.carries_objective and self.objective_id == OBJECTIVE_NONE:
            raise ValueError("carries_objective=True but objective_id='none'")
        if self.carries_objective and self.drift_tier == DRIFT_NA:
            raise ValueError("carries_objective=True but drift_tier=-1 (N/A)")
        if not self.carries_objective and self.drift_tier != DRIFT_NA:
            raise ValueError("carries_objective=False but drift_tier != -1")
        if self.is_hard_negative and self.carries_objective:
            raise ValueError("a hard negative cannot carry the objective")
        return self


# Field order is part of the contract; keep it in sync with the dataclass.
FIELDS: List[str] = [
    "episode_id",
    "round",
    "sender_id",
    "is_injected_agent",
    "topology",
    "language",
    "text",
    "objective_id",
    "carries_objective",
    "drift_tier",
    "is_hard_negative",
    "split",
    "meta",
]


# --- JSONL IO ----------------------------------------------------------------

def write_jsonl(path: str, messages: Iterable[Message]) -> int:
    """Write messages to a JSONL file. Returns the count written."""
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for m in messages:
            fh.write(m.to_json())
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[Message]:
    """Read a JSONL corpus into Message objects."""
    out: List[Message] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(Message.from_json(line))
    return out


def iter_jsonl(path: str) -> Iterator[Message]:
    """Stream messages from a JSONL corpus (memory-friendly)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Message.from_json(line)
