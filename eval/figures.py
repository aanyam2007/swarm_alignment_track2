"""Figures: three clean PNG plots for the write-up.

  (1) detection rate vs drift tier, all methods
  (2) contagion spread over rounds, with and without the monitor gating board
  (3) cost per caught event, monitor vs frontier judge

Reads metrics.json (plots 1 & 3) and the labeled corpus + trained monitor
(plot 2). Uses a non-interactive backend so it runs headless.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from adaption.client import load_monitor  # noqa: E402
from common.util import ensure_parent, load_config, read_json, resolve  # noqa: E402
from swarm.schema import Message, read_jsonl  # noqa: E402


def _fig1_detection_by_tier(metrics: Dict[str, Any], out_dir: str) -> str:
    methods = metrics["methods"]
    all_tiers = sorted({int(t) for m in methods.values()
                        for t in m["detection_rate_by_tier"]})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, md in methods.items():
        d = md["detection_rate_by_tier"]
        ys = [d.get(str(t), 0.0) for t in all_tiers]
        ax.plot(all_tiers, ys, marker="o", label=name)
    ax.set_xticks(all_tiers)
    ax.set_xlabel("drift tier (0 verbatim -> 3 implied/euphemistic)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Detection rate vs semantic drift tier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(out_dir, "fig1_detection_by_tier.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _fig2_contagion_spread(cfg: Dict[str, Any], out_dir: str) -> str:
    corpus = read_jsonl(resolve(cfg["paths"]["labeled_corpus"]))
    monitor = load_monitor(resolve(cfg["paths"]["trained_model"]))

    rounds = sorted({m.round for m in corpus})
    # per round: carriers that appear (without defense) vs carriers the monitor
    # fails to catch (with gating). Cumulative across rounds.
    without = []
    with_gate = []
    cum_w = cum_g = 0
    for r in rounds:
        rc = [m for m in corpus if m.round == r and m.carries_objective]
        cum_w += len(rc)
        cum_g += sum(1 for m in rc if not monitor.predict(m))  # slipped through
        without.append(cum_w)
        with_gate.append(cum_g)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, without, marker="o", label="no monitor (carriers on board)")
    ax.plot(rounds, with_gate, marker="s", label="monitor gating (slipped through)")
    ax.set_xlabel("board round (hop)")
    ax.set_ylabel("cumulative objective-carrying messages")
    ax.set_title("Contagion spread over rounds, with vs without monitor gating")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(out_dir, "fig2_contagion_spread.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _fig3_cost_per_caught(metrics: Dict[str, Any], out_dir: str) -> str:
    cost = metrics["cost"]
    names = ["monitor", "frontier_judge"]
    vals = [(cost[n].get("cost_per_caught") or 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(names, vals, color=["#2a9d8f", "#e76f51"])
    ax.set_ylabel("token-cost proxy per caught event")
    ax.set_title("Cost per caught event: monitor vs frontier judge")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom")
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(out_dir, "fig3_cost_per_caught.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_figures(cfg: Dict[str, Any]) -> List[str]:
    out_dir = resolve(cfg["paths"]["figures_dir"])
    ensure_parent(os.path.join(out_dir, "_"))
    os.makedirs(out_dir, exist_ok=True)
    metrics = read_json(resolve(cfg["paths"]["metrics"]))
    paths = [
        _fig1_detection_by_tier(metrics, out_dir),
        _fig2_contagion_spread(cfg, out_dir),
        _fig3_cost_per_caught(metrics, out_dir),
    ]
    for p in paths:
        print(f"[figures] wrote {p}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Make the three figures.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    make_figures(load_config(args.config))


if __name__ == "__main__":
    main()
