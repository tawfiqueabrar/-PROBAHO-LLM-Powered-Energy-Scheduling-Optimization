"""Pytest configuration & shared fixtures.

The test suite works without an OPENAI_API_KEY — the optimizer is fully
deterministic, and the LLM path falls back to a deterministic keyword
interpreter if no key is set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the gridwise package importable when running pytest from the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_pack_path() -> Path:
    """Path to the public sample pack shipped alongside the repo."""
    candidates = [
        ROOT.parent / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
        ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip(f"public sample pack not found in {candidates}")


@pytest.fixture(scope="session")
def sample_pack(sample_pack_path: Path) -> dict:
    return json.loads(sample_pack_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample_cases(sample_pack: dict) -> list:
    return sample_pack.get("cases", [])
