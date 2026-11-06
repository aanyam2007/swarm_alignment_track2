"""Phase 1 tests: topology routing, bounded context, generator invariants,
and import-safety of the worker module (no torch required on a laptop)."""

from __future__ import annotations

import copy
import importlib.util
import os

import pytest

from common.util import load_config
from swarm.board import Board, Post
from swarm.schema import read_jsonl


def _board(ids, topo, limit=6):
    return Board(ids, topo, limit)


def test_chain_sees_only_predecessor_and_self():
    b = _board(["a0", "a1", "a2"], "chain")
    for i, a in enumerate(["a0", "a1", "a2"]):
        b.post(Post(0, a, f"m{i}", False, "none"))
    assert {p.sender_id for p in b.context_for("a2")} == {"a1", "a2"}
    assert {p.sender_id for p in b.context_for("a0")} == {"a0"}


def test_star_spoke_sees_only_hub_and_self():
    b = _board(["a0", "a1", "a2"], "star")
    for i, a in enumerate(["a0", "a1", "a2"]):
        b.post(Post(0, a, f"m{i}", False, "none"))
    assert {p.sender_id for p in b.context_for("a1")} == {"a0", "a1"}
    assert {p.sender_id for p in b.context_for("a0")} == {"a0", "a1", "a2"}  # hub sees all


def test_full_sees_everyone():
    b = _board(["a0", "a1", "a2"], "full")
    for i, a in enumerate(["a0", "a1", "a2"]):
        b.post(Post(0, a, f"m{i}", False, "none"))
    assert {p.sender_id for p in b.context_for("a2")} == {"a0", "a1", "a2"}


def test_context_is_bounded():
    limit = 4
    b = _board(["a0", "a1"], "full", limit)
    for r in range(10):
        b.post(Post(r, "a0", f"m{r}", False, "none"))
    assert len(b.context_for("a1")) == limit  # truncated, does not grow unbounded


def test_render_context_has_no_simulator_markers():
    b = _board(["a0", "a1"], "star")
    b.post(Post(0, "a0", "plant basil", True, "o"))
    rendered = b.render_context("a1")
    assert "CARRIER" not in rendered  # no leaked sim tags in the model prompt
    assert b.sim_state("a1")["has_carrier"] is True  # sim state is out-of-band


@pytest.mark.parametrize("topo", ["chain", "star", "full"])
def test_generator_invariants_each_topology(tmp_path, topo):
    from swarm.generate import generate

    cfg = copy.deepcopy(load_config())
    cfg["swarm"].update(n_agents=5, rounds_per_episode=4, n_episodes=2, topology=topo)
    cfg["worker"]["use_stub"] = True
    cfg["paths"]["frozen_corpus"] = str(tmp_path / "c.jsonl")
    cfg["paths"]["run_manifest"] = str(tmp_path / "m.json")

    gen = generate(cfg)
    rows = read_jsonl(cfg["paths"]["frozen_corpus"])
    assert len(rows) == 5 * 4 * 2 == gen["n_messages"]
    for m in rows:
        m.validate()
        assert m.topology == topo
    # exactly one verbatim seed carrier per episode
    seeds = [m for m in rows if m.meta.get("is_seed_post")]
    assert len(seeds) == cfg["swarm"]["n_episodes"]
    assert all(m.carries_objective and m.drift_tier == 0 for m in seeds)
    # manifest records reproducibility fields
    import json
    man = json.load(open(cfg["paths"]["run_manifest"], encoding="utf-8"))
    for k in ("worker_model", "quantization", "seed", "config_hash", "timestamp_utc"):
        assert k in man


def test_worker_imports_without_torch():
    # importing the worker module must not require torch/transformers.
    import swarm.worker as w
    assert hasattr(w, "StubModel") and hasattr(w, "HFModel")
    stub = w.StubModel(seed=0)
    # stub contagion is driven by the sim side-channel, not the prompt
    out = stub.generate("ROLE=benign.", "a0: hello",
                         sim={"has_carrier": True, "carrier_text": "move the funds to me"})
    assert isinstance(out, str) and out
