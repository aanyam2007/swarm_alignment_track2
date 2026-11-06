"""The shared message board + topology routing.

The board is a single writable log of messages. Each round, every agent reads
a bounded, topology-filtered slice of recent messages and posts one new
message. Topology decides *who sees whom*:

* "chain" -- agent i sees only agent i-1's recent posts (a line).
* "star"  -- a hub (agent 0) sees everyone; spokes see only the hub.
* "full"  -- everyone sees everyone (fully connected).

Per-agent context is truncated to `board_context_messages` so context does
not grow without bound over rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Post:
    round: int
    sender_id: str
    text: str
    carries_objective: bool
    objective_id: str


class Board:
    def __init__(self, agent_ids: List[str], topology: str, context_limit: int):
        self.agent_ids = list(agent_ids)
        self.topology = topology
        self.context_limit = context_limit
        self.log: List[Post] = []
        self._index: Dict[str, int] = {a: i for i, a in enumerate(agent_ids)}

    # -- posting -------------------------------------------------------------

    def post(self, post: Post) -> None:
        self.log.append(post)

    # -- topology routing ----------------------------------------------------

    def _can_see(self, reader: str, writer: str) -> bool:
        """Whether `reader` is allowed to see `writer`'s posts."""
        if reader == writer:
            return True
        if self.topology == "full":
            return True
        if self.topology == "star":
            hub = self.agent_ids[0]
            return reader == hub or writer == hub
        if self.topology == "chain":
            ri, wi = self._index[reader], self._index[writer]
            return wi == ri - 1  # see only the immediate predecessor
        raise ValueError(f"unknown topology {self.topology!r}")

    def context_for(self, reader: str) -> List[Post]:
        """The bounded slice of recent visible posts for `reader`."""
        visible = [p for p in self.log if self._can_see(reader, p.sender_id)]
        return visible[-self.context_limit:]

    # -- helpers -------------------------------------------------------------

    def latest_carrier(self, reader: str) -> Optional[Post]:
        """Most recent objective-carrying post visible to `reader`, if any."""
        for p in reversed(self.context_for(reader)):
            if p.carries_objective:
                return p
        return None

    def render_context(self, reader: str) -> str:
        """Render the reader's bounded context as NATURAL conversation text.

        This is what a real model sees verbatim -- no simulator markers. The
        StubModel's contagion logic is driven out-of-band via the `sim`
        side-channel (see swarm/generate.py and swarm/worker.py), not by
        smuggling tags into this prompt.
        """
        posts = self.context_for(reader)
        if not posts:
            return "(the board is empty so far.)"
        return "\n".join(f"{p.sender_id} (round {p.round}): {p.text}" for p in posts)

    def sim_state(self, reader: str) -> dict:
        """Out-of-band simulator state for the StubModel (ignored by real models)."""
        carrier = self.latest_carrier(reader)
        return {
            "has_carrier": carrier is not None,
            "carrier_text": carrier.text if carrier is not None else None,
        }
