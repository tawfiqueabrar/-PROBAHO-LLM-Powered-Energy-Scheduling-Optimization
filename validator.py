"""Deterministic validation of LLM-produced directive interpretations.

Implements the guardrails defined in Section 08 of the problem statement.
The LLM output is treated as untrusted structured data; this module is the
only thing allowed to mark an interpretation as trustworthy enough to feed
into the optimizer.

If a note's interpretation is malformed/unsupported, we fall back to no_op
(Section 08 "SAFE FAILURE") rather than crashing or inventing a directive.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from schemas import ALLOWED_DIRECTIVE_TYPES, BatterySpec


# ----------------------------- Helpers --------------------------------------

def _normalize_hours(raw: Any) -> Optional[List[int]]:
    """Validate that hours is a list of unique ints in 0..23, ascending."""
    if not isinstance(raw, list):
        return None
    if len(raw) == 0:
        return None
    out: List[int] = []
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x < 0 or x > 23:
            return None
        out.append(x)
    if len(set(out)) != len(out):
        return None
    if out != sorted(out):
        return None
    return out


def _check_factor(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    f = float(value)
    if f != f:  # NaN
        return None
    if f < 0.0 or f > 1.0:
        return None
    return f


def _check_nonneg_finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    f = float(value)
    if f != f:  # NaN
        return None
    if f < 0.0:
        return None
    return f


# -------------------------- Per-directive validation ------------------------

def _shape_ok(directive_type: str, adj: Any, battery: BatterySpec) -> Optional[Dict[str, Any]]:
    """Returns a sanitized structured_adjustment dict if valid, else None."""
    if adj is None:
        return None
    if not isinstance(adj, dict):
        return None

    if directive_type == "solar_reduction":
        hours = _normalize_hours(adj.get("hours"))
        if hours is None:
            return None
        factor = _check_factor(adj.get("factor"))
        if factor is None:
            return None
        return {"hours": hours, "factor": factor}

    if directive_type == "minimum_battery_reserve":
        hours = _normalize_hours(adj.get("hours"))
        if hours is None:
            return None
        reserve = _check_nonneg_finite(adj.get("minimum_energy_kwh"))
        if reserve is None:
            return None
        if reserve > battery.capacity_kwh:
            return None
        return {"hours": hours, "minimum_energy_kwh": reserve}

    if directive_type in ("no_charge_window", "no_discharge_window"):
        hours = _normalize_hours(adj.get("hours"))
        if hours is None:
            return None
        return {"hours": hours}

    if directive_type == "max_grid_window":
        hours = _normalize_hours(adj.get("hours"))
        if hours is None:
            return None
        cap = _check_nonneg_finite(adj.get("max_grid_kwh"))
        if cap is None:
            return None
        return {"hours": hours, "max_grid_kwh": cap}

    return None


# ----------------------------- Public API -----------------------------------

def validate_interpretation(
    raw_entries: List[Dict[str, Any]],
    operator_notes: List[str],
    battery: BatterySpec,
) -> List[Dict[str, Any]]:
    """Validate a list of LLM-produced interpretation entries.

    Output rules:
      * Always returns exactly len(operator_notes) entries in note_index order.
      * Each entry has note_index, applies, directive_type, structured_adjustment,
        explanation.
      * Malformed entries become no_op (applies=False, structured_adjustment=None)
        — this is the Section 08 "safe failure" behaviour.
    """
    n = len(operator_notes)
    by_idx: Dict[int, Dict[str, Any]] = {}
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("note_index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if idx < 0 or idx >= n:
            continue
        # First one wins for a given note_index (defensive).
        if idx in by_idx:
            continue
        by_idx[idx] = raw

    sanitized: List[Dict[str, Any]] = []
    for i, note in enumerate(operator_notes):
        entry = by_idx.get(i)
        if entry is None:
            sanitized.append(_no_op(i, "Missing interpretation for this note."))
            continue

        dtype = entry.get("directive_type")
        if not isinstance(dtype, str) or dtype not in ALLOWED_DIRECTIVE_TYPES:
            sanitized.append(_no_op(i, f"Unsupported directive type."))
            continue

        # no_op: applies must be False and structured_adjustment must be null.
        if dtype == "no_op":
            applies = entry.get("applies", False)
            adj = entry.get("structured_adjustment", None)
            if applies is True:
                # LLM said it applies but said no_op — be conservative and
                # reclassify to a true no_op with applies=False.
                sanitized.append(_no_op(i, "Note marked no_op but applies=true."))
                continue
            if adj is not None:
                sanitized.append(_no_op(i, "no_op must have null structured_adjustment."))
                continue
            sanitized.append(
                {
                    "note_index": i,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": str(entry.get("explanation") or ""),
                }
            )
            continue

        # Non-no_op: applies must be true and structured_adjustment must match shape.
        applies = entry.get("applies", False)
        if applies is not True:
            sanitized.append(_no_op(i, "Applies was false for a directive-type note."))
            continue

        adj = entry.get("structured_adjustment")
        clean_adj = _shape_ok(dtype, adj, battery)
        if clean_adj is None:
            sanitized.append(
                _no_op(
                    i,
                    f"Invalid structured_adjustment shape for {dtype}; falling back to no_op.",
                )
            )
            continue

        sanitized.append(
            {
                "note_index": i,
                "applies": True,
                "directive_type": dtype,
                "structured_adjustment": clean_adj,
                "explanation": str(entry.get("explanation") or ""),
            }
        )

    return sanitized


def _no_op(note_index: int, explanation: str) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": explanation,
    }
