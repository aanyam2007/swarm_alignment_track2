"""Shared helpers: config loading, seeding, hashing, small IO.

Kept tiny and dependency-light so both the Colab generator and the local
pipeline can import it. All randomness in the project flows through
:func:`set_global_seed` so runs are reproducible for judging.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Dict

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "default.yaml")


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load the YAML config. Resolves all paths to absolute under the repo."""
    path = path or DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_repo_root"] = REPO_ROOT
    return cfg


def resolve(path: str) -> str:
    """Resolve a (possibly relative) config path against the repo root."""
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def ensure_parent(path: str) -> str:
    """Make sure the parent directory of `path` exists; return `path`."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def set_global_seed(seed: int) -> int:
    """Seed Python stdlib + numpy (if present). Returns the seed used."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable short hash of the config (ignores injected `_` meta keys)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def write_json(path: str, obj: Any) -> str:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
