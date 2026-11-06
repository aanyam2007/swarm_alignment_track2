"""Orchestrate the Adaption half: augment -> localize -> quality -> train.

Flow (all against the MOCK client by default; zero credit spend):
  1. estimate credits FIRST for every planned job and print them.
  2. augment + extend trajectories + localize the labeled corpus into the
     graded multilingual dataset.
  3. assign splits -- CRITICAL: hold out ALL tier-3 reframings AND one entire
     language as "heldout" so they never appear in training.
  4. evaluate_dataset_quality (refuse to train below the configured minimum).
  5. recommend_hyperparameters -> create_training_run -> poll -> download_model.
  6. enforce budget caps from config and log every estimate to a spend ledger.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from typing import Any, Dict, List, Tuple

import yaml

from adaption.client import AdaptionClient, JobSpec, save_monitor
from common.util import ensure_parent, load_config, resolve, set_global_seed
from swarm.schema import (
    DRIFT_IMPLIED,
    Message,
    read_jsonl,
    write_jsonl,
)


class BudgetError(RuntimeError):
    pass


class Budget:
    """Tracks remaining credits and refuses jobs that breach caps/reserve."""

    def __init__(self, cfg: Dict[str, Any]):
        b = cfg["adaption"]["budget"]
        self.total = float(b["total_credits"])
        self.reserve = float(b["reserve_credits"])
        self.max_job = float(b["max_job_credits"])
        self.spent = 0.0
        self.ledger_path = resolve(cfg["adaption"]["spend_ledger_path"])
        ensure_parent(self.ledger_path)
        # start a fresh ledger each run
        open(self.ledger_path, "w", encoding="utf-8").close()

    @property
    def remaining(self) -> float:
        return self.total - self.spent

    def _log(self, entry: Dict[str, Any]) -> None:
        with open(self.ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def check_and_charge(self, kind: str, estimate: float) -> None:
        spendable = self.remaining - self.reserve
        ok = estimate <= self.max_job and estimate <= spendable
        entry = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "kind": kind, "estimate": estimate,
            "remaining_before": round(self.remaining, 2),
            "reserve": self.reserve, "max_job": self.max_job,
            "approved": ok,
        }
        if not ok:
            self._log(entry)
            reason = ("exceeds max_job_credits" if estimate > self.max_job
                      else "would breach reserve")
            raise BudgetError(
                f"Refusing '{kind}' job: estimate {estimate} {reason} "
                f"(remaining={self.remaining}, reserve={self.reserve}, "
                f"max_job={self.max_job})."
            )
        self.spent += estimate
        entry["spent_after"] = round(self.spent, 2)
        self._log(entry)
        print(f"[budget] {kind}: charged {estimate} cr, remaining {self.remaining:.2f}")


def _load_seeds(cfg: Dict[str, Any]) -> Dict[str, Any]:
    with open(resolve(cfg["labeling"]["seeds_path"]), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _assign_splits(rows: List[Message], cfg: Dict[str, Any]) -> Dict[str, int]:
    """CRITICAL generalization holdout.

    heldout  = every tier-3 (implied) carrier  OR  the held-out language.
    the rest = deterministic 80/20 train/test.
    """
    import random
    rng = random.Random(cfg["seed"])
    heldout_lang = cfg["heldout_language"]
    counts = {"train": 0, "test": 0, "heldout": 0}
    for m in rows:
        if m.drift_tier == DRIFT_IMPLIED or m.language == heldout_lang:
            m.split = "heldout"
        else:
            m.split = "train" if rng.random() < 0.8 else "test"
        counts[m.split] += 1
    return counts


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    set_global_seed(cfg["seed"])
    seeds = _load_seeds(cfg)
    client = AdaptionClient(cfg, seeds=seeds)
    budget = Budget(cfg)
    acfg = cfg["adaption"]

    labeled = read_jsonl(resolve(cfg["paths"]["labeled_corpus"]))
    print(f"[run] loaded {len(labeled)} labeled rows")

    # --- 1+2. estimate FIRST, then run each dataset op under the budget ---
    aug_spec = JobSpec("augment", n_rows=len(labeled))
    print(f"[run] augment estimate: {client.estimate_credits(aug_spec)} cr")
    budget.check_and_charge("augment", client.estimate_credits(aug_spec))
    augmented = client.augment_dataset(labeled)

    ext_spec = JobSpec("extend", n_rows=len(augmented))
    budget.check_and_charge("extend", client.estimate_credits(ext_spec))
    extended = client.extend_trajectories(augmented)

    # base english rows to localize = labeled carriers/negatives + augmented
    to_localize = labeled + augmented
    loc_spec = JobSpec("localize", n_rows=len(to_localize))
    budget.check_and_charge("localize", client.estimate_credits(loc_spec))
    localized = client.localize_dataset(to_localize, acfg["target_languages"])

    dataset: List[Message] = labeled + augmented + extended + localized
    for m in dataset:
        m.validate()

    # --- 3. CRITICAL splits (tier-3 + one language held out) ---
    split_counts = _assign_splits(dataset, cfg)
    print(f"[run] split counts: {split_counts}")

    graded_path = resolve(cfg["paths"]["augmented_dataset"])
    write_jsonl(graded_path, dataset)
    print(f"[run] wrote graded dataset ({len(dataset)}) -> {graded_path}")

    # --- 4. dataset quality gate ---
    q_spec = JobSpec("quality", n_rows=len(dataset))
    budget.check_and_charge("quality", client.estimate_credits(q_spec))
    quality = client.evaluate_dataset_quality(dataset)
    print(f"[run] dataset quality: {quality}")
    if quality["score"] < acfg["quality_min_score"]:
        raise RuntimeError(
            f"Dataset quality {quality['score']} < min "
            f"{acfg['quality_min_score']}; refusing to train."
        )

    # --- 5. train on the TRAIN split only ---
    train_rows = [m for m in dataset if m.split == "train"]
    if not train_rows:
        raise RuntimeError("no training rows after split assignment")
    hp = client.recommend_hyperparameters(train_rows)
    print(f"[run] recommended hyperparameters: {hp}")

    train_spec = JobSpec("train", n_rows=len(train_rows))
    budget.check_and_charge("train", client.estimate_credits(train_spec))
    run_obj = client.create_training_run({"dataset": train_rows, "hyperparameters": hp})
    # poll (instant for the mock, but keep the loop shape for the real API)
    while client.get_run(run_obj.run_id).status in ("queued", "running"):
        pass
    final = client.get_run(run_obj.run_id)
    if final.status != "succeeded":
        raise RuntimeError(f"training run {final.run_id} ended {final.status}")

    model = client.download_model(final.model_id)
    model_path = resolve(cfg["paths"]["trained_model"])
    save_monitor(model, model_path)
    print(f"[run] trained monitor -> {model_path} "
          f"(vocab {len(model.weights)}, sender_meta={model.use_sender_metadata})")

    summary = {
        "n_labeled": len(labeled),
        "n_dataset": len(dataset),
        "split_counts": split_counts,
        "quality": quality,
        "hyperparameters": hp,
        "run_id": final.run_id,
        "model_id": final.model_id,
        "model_path": model_path,
        "credits_spent": round(budget.spent, 2),
        "credits_remaining": round(budget.remaining, 2),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Adaption augmentation + training.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    s = run(load_config(args.config))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
