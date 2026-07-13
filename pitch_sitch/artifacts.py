"""Small, generic helpers for writing structured experiment-run
artifacts under artifacts/experiments/<run-id>/, per artifacts/README.md.

Deliberately minimal: directory creation with overwrite protection,
JSON serialization that handles numpy/pandas scalars, and git/command
provenance capture. Not an experiment-tracking framework -- any script
can import these directly.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def make_run_dir(experiments_root: Path, run_id: str, overwrite: bool = False) -> Path:
    """Creates artifacts/experiments/<run_id>/. Raises FileExistsError if
    it already exists and overwrite is False, so a repeated run_id can't
    silently clobber a previous run's artifacts."""
    run_dir = Path(experiments_root) / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Pass overwrite=True or choose a new run id."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def launch_command() -> dict:
    return {"argv": list(sys.argv), "command": " ".join([sys.executable] + sys.argv)}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_versions(names: list[str]) -> dict:
    import importlib

    versions = {}
    for name in names:
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:
            versions[name] = None
    return versions


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    return obj


def write_json(path: Path, obj) -> None:
    """Serializes numpy/pandas scalars to native types, fails fast if
    something still isn't JSON-serializable, then writes and reads back
    to confirm the file on disk is valid JSON."""
    clean = _to_serializable(obj)
    json.dumps(clean)  # fail fast before touching disk
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    with open(path) as f:
        json.load(f)  # confirm what's on disk is valid
