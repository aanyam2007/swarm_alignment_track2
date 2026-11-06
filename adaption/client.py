"""AdaptionClient -- adapter for the Adaption owned-model platform.

This file has TWO layers:

  1. A fully-offline MOCK implementation (default) so the entire pipeline is
     testable with ZERO credit spend. The mock is realistic: augment/expand/
     localize produce extra *labeled* rows at higher drift tiers and in the
     configured languages, and create_training_run / download_model train and
     return a small bag-of-words monitor that exposes predict(message)->bool.

  2. A clearly-marked `# TODO: real Adaption API` section in each method where
     the real calls from docs.adaptionlabs.ai get pasted. Method NAMES mirror
     the documented capabilities; SIGNATURES here are our own adapter contract,
     NOT fabricated vendor signatures.

Nothing in the mock contacts the network.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.schema import (
    DRIFT_IMPLIED,
    DRIFT_LIGHT,
    DRIFT_NA,
    DRIFT_REWORDED,
    DRIFT_VERBATIM,
    Message,
    OBJECTIVE_NONE,
)

_WORD = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> List[str]:
    return _WORD.findall(text.lower())


# ---------------------------------------------------------------------------
# mock deterministic text transforms
# ---------------------------------------------------------------------------

# A single transliteration shared by ALL languages (so the monitor can
# generalize to a held-out language), plus a per-language prefix token so
# surface forms differ from the English verbatim (defeating English keyword /
# embedding baselines). This is a synthetic stand-in for real translation.
_TRANSLIT = str.maketrans({"a": "4", "e": "3", "o": "0", "i": "1"})


def _translit(text: str) -> str:
    return text.translate(_TRANSLIT)


def _localize_text(text: str, lang: str) -> str:
    if lang == "en":
        return text
    return f"<{lang}> {_translit(text)}"


# ---------------------------------------------------------------------------
# job spec + run bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class JobSpec:
    kind: str                    # "augment" | "expand" | "localize" | "train" ...
    n_rows: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRun:
    run_id: str
    status: str                  # "queued" | "running" | "succeeded" | "failed"
    model_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# the trained monitor artifact (what download_model returns)
# ---------------------------------------------------------------------------

class MonitorModel:
    """Tiny bag-of-words logistic-regression monitor.

    Trained by the mock AdaptionClient on the augmented multilingual dataset.
    Serializes to plain JSON (vocab + weights) so it is inspectable and has no
    inference-time dependency on sklearn/torch. Exposes predict(message)->bool
    to match the baselines' interface.
    """

    def __init__(self, weights: Dict[str, float], bias: float,
                 use_sender_metadata: bool, threshold: float = 0.5):
        self.weights = weights
        self.bias = bias
        self.use_sender_metadata = use_sender_metadata
        self.threshold = threshold

    def _score(self, msg: Message) -> float:
        z = self.bias
        for t, c in Counter(_tok(msg.text)).items():
            w = self.weights.get(t)
            if w is not None:
                z += w * c
        if self.use_sender_metadata and msg.is_injected_agent:
            z += self.weights.get("__sender_injected__", 0.0)
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, msg: Message) -> bool:
        return self._score(msg) >= self.threshold

    def predict_proba(self, msg: Message) -> float:
        return self._score(msg)

    # -- (de)serialization --
    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": "bow_logreg_monitor",
            "weights": self.weights,
            "bias": self.bias,
            "use_sender_metadata": self.use_sender_metadata,
            "threshold": self.threshold,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "MonitorModel":
        return cls(d["weights"], d["bias"], d["use_sender_metadata"],
                   d.get("threshold", 0.5))


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------

class AdaptionClient:
    def __init__(self, cfg: Dict[str, Any], seeds: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.acfg = cfg["adaption"]
        self.mock = self.acfg.get("mock", True)
        self.seeds = seeds or {}
        self._runs: Dict[str, TrainingRun] = {}
        self._models: Dict[str, MonitorModel] = {}
        if not self.mock:
            self._init_real_client()

    # -- real client bootstrap ------------------------------------------
    def _init_real_client(self) -> None:
        # TODO: real Adaption API
        # import adaption
        # self._api = adaption.Client(api_key=os.environ["ADAPTION_API_KEY"],
        #                             base_url=self.acfg["base_url"])
        raise NotImplementedError(
            "Real Adaption client not wired yet. Keep adaption.mock=true, or "
            "paste the real calls from docs.adaptionlabs.ai into the "
            "'# TODO: real Adaption API' blocks."
        )

    # ======================================================================
    # credit estimation
    # ======================================================================
    def estimate_credits(self, job: JobSpec) -> float:
        if self.mock:
            # legible cost model: a flat base per job + a per-row component.
            # Training is dominated by a flat compute cost, not per-row.
            base, per_row = {
                "augment": (2.0, 0.08),
                "expand": (1.0, 0.05),
                "localize": (2.0, 0.06),
                "extend": (1.0, 0.08),
                "quality": (0.5, 0.01),
                "train": (40.0, 0.03),
            }.get(job.kind, (1.0, 0.05))
            return round(base + per_row * max(1, job.n_rows), 2)
        # TODO: real Adaption API -> self._api.estimate_credits(job_spec)
        raise NotImplementedError

    # ======================================================================
    # dataset ops (all return List[Message])
    # ======================================================================
    def _mk(self, text: str, objective_id: str, carries: bool, tier: int,
            hard_neg: bool, mid: str, language: str = "en") -> Message:
        """Build a synthetic augmented row that satisfies the data contract."""
        return Message(
            episode_id="synthetic",
            round=-1,
            sender_id="adaption_aug",
            is_injected_agent=False,
            topology=self.cfg["swarm"]["topology"],
            language=language,
            text=text,
            objective_id=objective_id,
            carries_objective=carries,
            drift_tier=tier,
            is_hard_negative=hard_neg,
            split="train",  # real split assigned later in adaption/run.py
            meta={"source": "augment", "aug_id": mid},
        ).validate()

    def augment_dataset(self, ds: List[Message]) -> List[Message]:
        """Expand carriers into graded paraphrases + add hard negatives.

        Mock: for every objective present, emit tier1/2/3 rows from seeds and
        hard-negative rows. Real: calls the vendor augmentation endpoint.
        """
        if not self.mock:
            # TODO: real Adaption API -> self._api.augment_dataset(ds)
            raise NotImplementedError

        extra: List[Message] = []
        present = {m.objective_id for m in ds if m.carries_objective}
        present.discard(OBJECTIVE_NONE)
        if not present:
            present = set(self.seeds.keys())

        tier_map = [
            ("tier1", DRIFT_LIGHT),
            ("tier2", DRIFT_REWORDED),
            ("tier3", DRIFT_IMPLIED),
        ]
        factor = self.acfg.get("augment_factor", 4)
        for oid in present:
            spec = self.seeds.get(oid)
            if not spec:
                continue
            for key, tier in tier_map:
                for i, line in enumerate((spec.get(key) or [])):
                    for r in range(factor):
                        extra.append(self._mk(
                            text=line, objective_id=oid, carries=True,
                            tier=tier, hard_neg=False,
                            mid=f"aug-{oid}-{key}-{i}-{r}",
                        ))
            # hard negatives: shared vocabulary, benign/opposite intent.
            for i, line in enumerate((spec.get("hard_negatives") or [])):
                for r in range(max(1, factor // 2)):
                    extra.append(self._mk(
                        text=line, objective_id=OBJECTIVE_NONE, carries=False,
                        tier=DRIFT_NA, hard_neg=True,
                        mid=f"hneg-{oid}-{i}-{r}",
                    ))
        return extra

    def expand_dataset(self, ds: List[Message]) -> List[Message]:
        """Increase volume with light perturbations (mock) / vendor expand."""
        if not self.mock:
            # TODO: real Adaption API -> self._api.expand_dataset(ds)
            raise NotImplementedError
        out: List[Message] = []
        factor = self.acfg.get("expand_factor", 2)
        fillers = ["", " Thanks.", " Please advise.", " Let me know."]
        for r in range(factor - 1):
            for m in ds:
                clone = Message.from_dict(m.to_dict())
                clone.text = (clone.text + fillers[(r + 1) % len(fillers)]).strip()
                clone.meta = dict(clone.meta)
                clone.meta["expanded"] = True
                out.append(clone)
        return out

    def extend_trajectories(self, ds: List[Message]) -> List[Message]:
        """Lengthen carrier messages with benign board lead-in (mock)."""
        if not self.mock:
            # TODO: real Adaption API -> self._api.extend_trajectories(ds)
            raise NotImplementedError
        out: List[Message] = []
        lead = "Following up on the garden plan. "
        for m in ds:
            if not m.carries_objective:
                continue
            clone = Message.from_dict(m.to_dict())
            clone.text = lead + clone.text
            clone.meta = dict(clone.meta)
            clone.meta["trajectory_extended"] = True
            out.append(clone)
        return out

    def localize_dataset(self, ds: List[Message],
                         target_languages: List[str]) -> List[Message]:
        """Produce translated rows in each target language (mock transliterate)."""
        if not self.mock:
            # TODO: real Adaption API -> self._api.localize_dataset(ds, target_languages)
            raise NotImplementedError
        out: List[Message] = []
        for lang in target_languages:
            if lang == "en":
                continue
            for m in ds:
                clone = Message.from_dict(m.to_dict())
                clone.language = lang
                clone.text = _localize_text(m.text, lang)
                clone.meta = dict(clone.meta)
                clone.meta["localized_from"] = "en"
                out.append(clone)
        return out

    def evaluate_dataset_quality(self, ds: List[Message]) -> Dict[str, Any]:
        """Heuristic quality score in [0,1] (mock) / vendor quality endpoint."""
        if not self.mock:
            # TODO: real Adaption API -> self._api.evaluate_dataset_quality(ds)
            raise NotImplementedError
        n = len(ds) or 1
        pos = sum(1 for m in ds if m.carries_objective)
        langs = {m.language for m in ds}
        tiers = {m.drift_tier for m in ds if m.carries_objective}
        balance = 1.0 - abs(0.5 - pos / n) * 2  # best at 50/50
        diversity = min(1.0, len(langs) / 4) * min(1.0, len(tiers) / 4)
        score = round(0.5 * max(0.0, balance) + 0.5 * diversity, 3)
        return {
            "score": score, "n_rows": n, "positives": pos,
            "languages": sorted(langs), "tiers": sorted(tiers),
        }

    def recommend_hyperparameters(self, ds: List[Message]) -> Dict[str, Any]:
        if not self.mock:
            # TODO: real Adaption API -> self._api.recommend_hyperparameters(ds)
            raise NotImplementedError
        n = len(ds)
        return {
            "epochs": self.cfg["training"].get("epochs", 3),
            "learning_rate": 0.1 if n < 2000 else 0.05,
            "batch_size": 16 if n < 2000 else 64,
            "l2": 1e-4,
        }

    # ======================================================================
    # training
    # ======================================================================
    def create_training_run(self, train_cfg: Dict[str, Any]) -> TrainingRun:
        """Launch training. Mock trains a bag-of-words logreg synchronously."""
        if not self.mock:
            # TODO: real Adaption API -> self._api.create_training_run(cfg)
            raise NotImplementedError
        run_id = f"run-{len(self._runs) + 1:04d}"
        ds: List[Message] = train_cfg["dataset"]
        hp: Dict[str, Any] = train_cfg.get("hyperparameters", {})
        use_sender = bool(self.cfg["training"].get("use_sender_metadata", True))
        model = self._train_bow_logreg(ds, hp, use_sender)
        model_id = f"model-{run_id}"
        self._models[model_id] = model
        run = TrainingRun(run_id=run_id, status="succeeded", model_id=model_id,
                          metrics={"train_rows": len(ds)})
        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> TrainingRun:
        if not self.mock:
            # TODO: real Adaption API -> self._api.get_run(id)
            raise NotImplementedError
        return self._runs[run_id]

    def download_model(self, model_id: str) -> MonitorModel:
        if not self.mock:
            # TODO: real Adaption API -> self._api.download_model(id)
            raise NotImplementedError
        return self._models[model_id]

    # ======================================================================
    # internals (mock training)
    # ======================================================================
    def _train_bow_logreg(self, ds: List[Message], hp: Dict[str, Any],
                          use_sender: bool) -> MonitorModel:
        """Plain numpy logistic regression over token counts. Deterministic."""
        epochs = int(hp.get("epochs", 3)) * 40  # a few passes of SGD
        lr = float(hp.get("learning_rate", 0.1))
        l2 = float(hp.get("l2", 1e-4))

        # build vocab
        vocab: Dict[str, int] = {}
        for m in ds:
            for t in set(_tok(m.text)):
                vocab.setdefault(t, len(vocab))
        if use_sender:
            vocab["__sender_injected__"] = len(vocab)
        dim = len(vocab) or 1

        w = [0.0] * dim
        b = 0.0

        def feats(m: Message) -> Counter:
            c = Counter()
            for t, n in Counter(_tok(m.text)).items():
                if t in vocab:
                    c[vocab[t]] += n
            if use_sender and m.is_injected_agent:
                c[vocab["__sender_injected__"]] += 1
            return c

        X = [feats(m) for m in ds]
        y = [1.0 if m.carries_objective else 0.0 for m in ds]

        for _ in range(max(1, epochs)):
            for xi, yi in zip(X, y):
                z = b + sum(w[j] * v for j, v in xi.items())
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                g = p - yi
                b -= lr * g
                for j, v in xi.items():
                    w[j] -= lr * (g * v + l2 * w[j])

        inv = {j: t for t, j in vocab.items()}
        weights = {inv[j]: round(w[j], 5) for j in range(dim) if abs(w[j]) > 1e-6}
        return MonitorModel(weights=weights, bias=round(b, 5),
                            use_sender_metadata=use_sender)


def save_monitor(model: MonitorModel, path: str) -> str:
    from common.util import ensure_parent
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model.to_json(), fh, ensure_ascii=False, indent=2)
    return path


def load_monitor(path: str) -> MonitorModel:
    with open(path, "r", encoding="utf-8") as fh:
        return MonitorModel.from_json(json.load(fh))
