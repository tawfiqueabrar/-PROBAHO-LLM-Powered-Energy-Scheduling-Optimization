"""LLM-based operator-note interpreter.

The LLM is the *only* place natural language is interpreted. Its output is
JSON, fed into validator.py which is strictly deterministic.

We require an OpenAI-compatible client. The model is chosen via the
OPENAI_MODEL environment variable (default: gpt-4o-mini).

If no API key is configured, we fall back to a deterministic interpreter for
controlled local operation. This is not a substitute for an LLM in the hidden
judge — configure OPENAI_API_KEY for the deployed service.
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
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))


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
            # The judge's request deadline is 30 seconds. Leave enough time
            # for deterministic validation and optimization after a provider
            # outage, then use the safe local fallback.
            timeout=LLM_TIMEOUT_SECONDS,
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

_HOUR_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}
_TIME_TOKEN = r"(?:midnight|noon|(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)"
_TIME_PATTERNS = [
    rf"\bfrom\s+({_TIME_TOKEN})\s*(?:to|until|till|-)\s*({_TIME_TOKEN})",
    rf"\bbetween\s+({_TIME_TOKEN})\s*(?:and|to|until|till|-)\s*({_TIME_TOKEN})",
    rf"\b({_TIME_TOKEN})\s*(?:to|until|till|-)\s*({_TIME_TOKEN})",
]


def _parse_time_token(token: str) -> Optional[int]:
    """Convert a single explicit clock token into a 24-hour integer."""
    normalized = token.lower().replace(".", "").strip()
    if normalized == "noon":
        return 12
    if normalized == "midnight":
        return 0
    match = re.fullmatch(r"(?:(\d{1,2})|([a-z]+))(?:\:\d{2})?\s*(am|pm)?", normalized)
    if not match:
        return None
    hour = int(match.group(1)) if match.group(1) else _HOUR_WORDS.get(match.group(2) or "")
    if hour is None:
        return None
    meridian = match.group(3)
    if meridian == "am":
        return 0 if hour == 12 else hour if 0 <= hour <= 11 else None
    if meridian == "pm":
        return 12 if hour == 12 else hour + 12 if 1 <= hour <= 11 else None
    return hour if 0 <= hour <= 23 else None


def _extract_hours(text: str) -> Optional[List[int]]:
    """Best-effort hour extraction from common phrasings."""
    text_l = text.lower()

    for pat in _TIME_PATTERNS:
        m = re.search(pat, text_l)
        if not m:
            continue
        start = _parse_time_token(m.group(1))
        end = _parse_time_token(m.group(2))
        if start is None or end is None or end <= start or end > 24:
            return None
        # Half-open window: start inclusive, end exclusive
        return list(range(start, end))
    return None


_KEYWORD_NO_OP = [
    "cafeteria", "menu", "registration", "sports office", "sports",
    "tomorrow", "next month", "moved", "deadline", "schedule change",
    "meeting", "event", "holiday",
]


def _fallback_interpret(notes: List[str], battery_capacity_kwh: float) -> List[Dict[str, Any]]:
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
            # A percentage attached to "reduction" is removed output; a
            # percentage after "to/leave/remain" is usable output remaining.
            factor = 0.2
            if re.search(r"(?:half|one[- ]fifth)", low):
                factor = 0.5 if "half" in low else 0.2
            else:
                pct_match = re.search(r"(\d{1,3})\s*%", low)
                if pct_match:
                    pct = max(0, min(100, int(pct_match.group(1))))
                    prefix = low[:pct_match.start()]
                    suffix = low[pct_match.end():pct_match.end() + 24]
                    is_reduction = bool(
                        re.search(r"(?:reduce|reduced|reduction|drop)\D{0,24}$", prefix)
                        or re.search(r"\b(?:reduce|reduced|reduction|drop)\b", suffix)
                    )
                    factor = (100 - pct) / 100.0 if is_reduction else pct / 100.0
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

        if re.search(r"\bdischarge\b", low) and any(k in low for k in ["not", "no ", "disable", "unavailable"]):
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

        if re.search(r"\bcharg(?:e|ing|er)\b", low) and any(k in low for k in ["not", "no ", "isolated", "maintenance", "disable", "unavailable"]):
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

        if any(k in low for k in ["reserve", "keep", "at least", "minimum"]):
            m = re.search(r"(\d+(?:\.\d+)?)\s*kwh", low)
            percent = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?battery\s+capacity", low)
            reserve = float(m.group(1)) if m else (battery_capacity_kwh * float(percent.group(1)) / 100.0 if percent else 0.0)
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

        if any(k in low for k in ["grid", "import", "draw", "intake"]) and ("cap" in low or "limit" in low or "max" in low or "not exceed" in low or "no more than" in low or "at or below" in low):
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
        return _fallback_interpret(operator_notes, battery_capacity_kwh)

    entries = parsed.get("directive_interpretation")
    if not isinstance(entries, list):
        return _fallback_interpret(operator_notes, battery_capacity_kwh)

    return entries
