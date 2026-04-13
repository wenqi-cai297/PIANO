"""File I/O helpers: npz, json, yaml, and incremental JSONL.

All path arguments accept ``str | Path`` and are normalized internally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------

def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    """Save multiple numpy arrays to a compressed ``.npz`` file.

    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a ``.npz`` file and return its contents as a plain dict."""
    data = np.load(path, allow_pickle=False)
    return dict(data)


# ---------------------------------------------------------------------------
# JSON / JSONL helpers
# ---------------------------------------------------------------------------

def save_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    """Write *obj* as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    """Read a JSON file and return the parsed object."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file (one object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read all records from a JSONL file."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Returns the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
