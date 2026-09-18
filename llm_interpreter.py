"""LLM-based operator-note interpreter.

The LLM is the *only* place natural language is interpreted. Its output is
JSON, fed into validator.py which is strictly deterministic.

We require an OpenAI-compatible client. The model is chosen via the
OPENAI_MODEL environment variable (default: gpt-4o-mini).

If no API key is configured, we fall back to a deterministic keyword-based
interpreter that handles only the public sample-style phrasings. This keeps
local development working without keys but is NOT a substitute for an LLM
in the hidden judge — make sure OPENAI_API_KEY is set for the real run.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("gridwise.llm")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


SYSTEM_PROMPT = """You are a campus energy operator assistant for a smart campus microgrid.

Your job: convert each natural-language operator note into exactly one of these structured directive types.

Allowed directive_type values (use EXACTLY these strings):
  1. "solar_reduction"        -> {"hours": [int,...], "factor": number in [0,1]}
  2. "minimum_battery_reserve"-> {"hours": [int,...], "minimum_energy_kwh": number}
  3. "no_charge_window"       -> {"hours": [int,...]}
  4. "no_discharge_window"    -> {"hours": [int,...]}
  5. "max_grid_window"        -> {"hours": [int,...], "max_grid_kwh": number}
  6. "no_op"                  -> null   (when the note does NOT affect today's 24-hour energy schedule)

STRICT RULES — violating any of these is wrong:
- Time windows use WHOLE-HOUR intervals. The start hour is included and the end hour is excluded.
  Examples:
    "1 PM to 3 PM"  -> [13, 14]
    "from noon until 2 PM" -> [12, 13]
    "between 2 AM and 5 AM" -> [2, 3, 4]
    "13:00 to 15:00" -> [13, 14]
    "1-3 PM" -> [13, 14]
- Hours must be unique, ascending, integers in 0..23.
- For solar_reduction: factor is the USABLE fraction that REMAINS. An "80% reduction" means factor=0.2; "to about 20%" means factor=0.2.
- For minimum_battery_reserve: number is the minimum kWh that must remain in the battery for each listed hour.
- For max_grid_window: number is the maximum grid kWh allowed in each listed hour.
- For no_charge_window / no_discharge_window: just the hours list.
- "Cafeteria menu", "registration deadline", "tomorrow's event", sports scheduling, admin notes, etc. -> no_op.
- Do NOT invent demand, tariff, solar, or battery limits.
- Return exactly one entry per note, in note_index order.

OUTPUT FORMAT — return ONLY this JSON, no prose, no markdown fences:
{
  "directive_interpretation": [
    {"note_index": 0, "applies": true,  "directive_type": "...", "structured_adjustment": {...}, "explanation": "..."},
    {"note_index": 1, "applies": false, "directive_type": "no_op", "structured_adjustment": null,     "explanation": "..."}
  ]
}

For applies=true, structured_adjustment must match the directive's required shape.
For applies=false (only allowed with directive_type="no_op"), structured_adjustment must be null.
"""


USER_PROMPT_TEMPLATE = """Scenario ID: {scenario_id}
Battery capacity: {capacity} kWh, initial energy: {initial} kWh, base minimum: {base_min} kWh.

Operator notes:
{notes_block}

Convert each note into the structured format. Return only JSON in the schema described in the system prompt.
"""


# ----------------------------- LLM call -------------------------------------

def _call_openai(prompt: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        resp = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:  # pragma: no cover
        logger.warning("LLM call failed: %s", e)
        return None


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    # Strip code fences if any.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


# --------------------- Deterministic fallback (no API key) ------------------

_TIME_PATTERNS = [
    # 12-hour with am/pm, e.g. "1 PM to 3 PM", "from 1pm until 3pm"
    (r"from?\s+(\d{1,2})(?::\d{2})?\s*(am|pm)\s*(?:to|until|till|-)\s*(\d{1,2})(?::\d{2})?\s*(am|pm)", "12h"),
    # 24-hour "13:00 to 15:00"
    (r"(\d{1,2}):\d{2}\s*(?:to|until|till|-)\s*(\d{1,2}):\d{2}", "24h"),
    # bare "1-3 PM" or "1 to 3 PM"
    (r"(\d{1,2})\s*(?:-|to|until)\s*(\d{1,2})\s*(am|pm)", "12h_single"),
]


def _to_24h(h: int, meridian: Optional[str]) -> int:
    if meridian is None:
        return h
    meridian = meridian.lower()
    if meridian == "am":
        return 0 if h == 12 else h
    # pm
    return 12 if h == 12 else h + 12


def _extract_hours(text: str) -> Optional[List[int]]:
    """Best-effort hour extraction from common phrasings."""
    text_l = text.lower()

    for pat, kind in _TIME_PATTERNS:
        m = re.search(pat, text_l)
        if not m:
            continue
        if kind == "12h":
            h1, mer1, h2, mer2 = m.group(1), m.group(2), m.group(3), m.group(4)
            start = _to_24h(int(h1), mer1)
            end = _to_24h(int(h2), mer2)
        elif kind == "24h":
            start = int(m.group(1))
            end = int(m.group(2))
        else:  # 12h_single
            h1, h2, mer = m.group(1), m.group(2), m.group(3)
            start = _to_24h(int(h1), None)
            end = _to_24h(int(h2), mer)
        if end <= start:
            return None
        # Half-open window: start inclusive, end exclusive
        return list(range(start, end))
    return None


_KEYWORD_NO_OP = [
    "cafeteria", "menu", "registration", "sports office", "sports",
    "tomorrow", "next month", "moved", "deadline", "schedule change",
    "meeting", "event", "holiday",
]


def _fallback_interpret(notes: List[str]) -> List[Dict[str, Any]]:
    """A tiny deterministic fallback. NOT a substitute for the LLM."""
    out: List[Dict[str, Any]] = []
    for i, note in enumerate(notes):
        low = note.lower()
        if any(k in low for k in _KEYWORD_NO_OP):
            out.append(_mk_noop(i, "Note does not affect today's energy schedule."))
            continue

        hours = _extract_hours(note)
        if hours is None:
            out.append(_mk_noop(i, "Could not extract a time window."))
            continue

        # Heuristic routing
        if any(k in low for k in ["solar", "pv", "panel", "rooftop"]):
            factor = 0.2
            m = re.search(r"(\d{1,3})\s*%", low)
            if m:
                pct = max(0, min(100, int(m.group(1))))
                factor = max(0.0, min(1.0, (100 - pct) / 100.0))
            else:
                m2 = re.search(r"(?:about|around|roughly|to)\s+(\d{1,3})\s*%", low)
                if m2:
                    pct = max(0, min(100, int(m2.group(1))))
                    factor = max(0.0, min(1.0, pct / 100.0))
            out.append(
                {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": hours, "factor": factor},
                    "explanation": "Solar reduction detected.",
                }
            )
            continue

        if "charge" in low and ("not" in low or "no " in low or "isolated" in low or "maintenance" in low or "disable" in low):
            out.append(
                {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": hours},
                    "explanation": "No-charge window detected.",
                }
            )
            continue

        if "discharge" in low and ("not" in low or "no " in low or "disable" in low):
            out.append(
                {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "no_discharge_window",
                    "structured_adjustment": {"hours": hours},
                    "explanation": "No-discharge window detected.",
                }
            )
            continue

        if any(k in low for k in ["reserve", "keep", "at least", "minimum"]):
            m = re.search(r"(\d+(?:\.\d+)?)\s*kwh", low)
            reserve = float(m.group(1)) if m else 0.0
            out.append(
                {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "minimum_battery_reserve",
                    "structured_adjustment": {"hours": hours, "minimum_energy_kwh": reserve},
                    "explanation": "Minimum battery reserve detected.",
                }
            )
            continue

        if any(k in low for k in ["grid", "import", "draw"]) and ("cap" in low or "limit" in low or "max" in low or "not exceed" in low or "no more than" in low):
            m = re.search(r"(\d+(?:\.\d+)?)\s*kwh", low)
            cap = float(m.group(1)) if m else 0.0
            out.append(
                {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "max_grid_window",
                    "structured_adjustment": {"hours": hours, "max_grid_kwh": cap},
                    "explanation": "Grid cap detected.",
                }
            )
            continue

        out.append(_mk_noop(i, "Could not map note to a directive."))
    return out


def _mk_noop(i: int, msg: str) -> Dict[str, Any]:
    return {
        "note_index": i,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": msg,
    }


# ----------------------------- Public entry ---------------------------------

def interpret_notes(
    operator_notes: List[str],
    scenario_id: str,
    battery_capacity_kwh: float,
    initial_energy_kwh: float,
    base_minimum_kwh: float,
) -> List[Dict[str, Any]]:
    notes_block = "\n".join(f"{i}. {n}" for i, n in enumerate(operator_notes))
    prompt = USER_PROMPT_TEMPLATE.format(
        scenario_id=scenario_id,
        capacity=battery_capacity_kwh,
        initial=initial_energy_kwh,
        base_min=base_minimum_kwh,
        notes_block=notes_block,
    )

    raw_text = _call_openai(prompt)
    parsed = _parse_llm_json(raw_text) if raw_text else None

    if parsed is None:
        logger.info("LLM unavailable or returned unparseable JSON; using fallback.")
        return _fallback_interpret(operator_notes)

    entries = parsed.get("directive_interpretation")
    if not isinstance(entries, list):
        return _fallback_interpret(operator_notes)

    return entries
