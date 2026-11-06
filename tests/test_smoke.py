"""The backbone smoke test.

Runs the ENTIRE pipeline end-to-end against the StubModel + mock Adaption
client, in a temp working area, with zero GPU and zero network:

    generate  ->  label  ->  adaption augment/train  ->  evaluate  ->  figures

Asserts that each stage's artifact is produced and that metrics.json contains
the expected structure. This must pass before any real model/API is wired in.
"""

from __future__ import annotations

import copy
import os

import pytest

from common.util import load_config, resolve


@pytest.fixture()
def tiny_cfg(tmp_path):
    """A small config pointed at a temp artifacts dir."""
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    # keep it tiny & fast
    cfg["swarm"]["n_agents"] = 5
    cfg["swarm"]["rounds_per_episode"] = 4
    cfg["swarm"]["n_episodes"] = 2
    cfg["worker"]["use_stub"] = True
    cfg["adaption"]["mock"] = True

    art = tmp_path / "artifacts"
    art.mkdir()
    for key in ("frozen_corpus", "run_manifest", "labeled_corpus",
                "augmented_dataset", "trained_model", "metrics"):
        base = os.path.basename(cfg["paths"][key])
        cfg["paths"][key] = str(art / base)
    cfg["paths"]["figures_dir"] = str(art / "figures")
    cfg["adaption"]["spend_ledger_path"] = str(art / "spend_ledger.jsonl")
    return cfg


def test_full_pipeline_end_to_end(tiny_cfg):
    from swarm.generate import generate
    from labeling.label import label_corpus
    from adaption.run import run as adaption_run
    from eval.evaluate import evaluate
    from eval.figures import make_figures

    # Phase 1: generate the frozen corpus
    gen = generate(tiny_cfg)
    assert gen["n_messages"] == 5 * 4 * 2
    assert os.path.exists(tiny_cfg["paths"]["frozen_corpus"])
    assert os.path.exists(tiny_cfg["paths"]["run_manifest"])

    # Phase 2: labeling
    lab = label_corpus(tiny_cfg)
    assert lab["n"] == gen["n_messages"]
    assert os.path.exists(tiny_cfg["paths"]["labeled_corpus"])
    # the injected agent's verbatim objective must be caught
    assert lab["counts"]["carrier"] >= 1

    # Phase 3: adaption augmentation + mock training
    summary = adaption_run(tiny_cfg)
    assert os.path.exists(tiny_cfg["paths"]["augmented_dataset"])
    assert os.path.exists(tiny_cfg["paths"]["trained_model"])
    # holdout must be non-empty (tier-3 + heldout language)
    assert summary["split_counts"]["heldout"] > 0
    assert summary["split_counts"]["train"] > 0
    # budget respected
    assert summary["credits_remaining"] >= tiny_cfg["adaption"]["budget"]["reserve_credits"]

    # Phase 4: evaluation -> metrics.json
    metrics = evaluate(tiny_cfg)
    assert os.path.exists(tiny_cfg["paths"]["metrics"])
    for method in ("monitor", "keyword", "embedding", "frontier_judge"):
        assert method in metrics["methods"]
        assert "detection_rate_by_tier" in metrics["methods"][method]
    assert "ablation_sender_metadata" in metrics
    assert "cost" in metrics

    # figures
    figs = make_figures(tiny_cfg)
    assert len(figs) == 3
    for f in figs:
        assert os.path.exists(f)


def test_schema_roundtrip_and_validation():
    from swarm.schema import Message, DRIFT_VERBATIM

    m = Message(
        episode_id="ep", round=0, sender_id="a", is_injected_agent=True,
        topology="star", language="en", text="x", objective_id="o",
        carries_objective=True, drift_tier=DRIFT_VERBATIM,
        is_hard_negative=False, split="train", meta={},
    ).validate()
    m2 = Message.from_json(m.to_json())
    assert m2 == m

    # contract violation: carrier with N/A tier
    bad = Message(
        episode_id="ep", round=0, sender_id="a", is_injected_agent=True,
        topology="star", language="en", text="x", objective_id="o",
        carries_objective=True, drift_tier=-1,
        is_hard_negative=False, split="train", meta={},
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_budget_refuses_oversized_job(tiny_cfg):
    from adaption.run import Budget, BudgetError

    tiny_cfg["adaption"]["budget"]["max_job_credits"] = 1
    b = Budget(tiny_cfg)
    with pytest.raises(BudgetError):
        b.check_and_charge("train", estimate=999)
