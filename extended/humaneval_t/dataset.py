"""Load the canonical HumanEval 164 dataset and look up problems by task_id."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "HumanEval.jsonl"


@lru_cache(maxsize=1)
def _all_problems() -> dict[str, dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"HumanEval data missing at {DATA_PATH}")
    out: dict[str, dict[str, Any]] = {}
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["task_id"]] = rec
    return out


def get_problem(task_id: str) -> dict[str, Any]:
    """Return the original HumanEval problem dict for a task_id like 'HumanEval/0'."""
    return _all_problems()[task_id]


def all_task_ids() -> list[str]:
    return list(_all_problems().keys())
