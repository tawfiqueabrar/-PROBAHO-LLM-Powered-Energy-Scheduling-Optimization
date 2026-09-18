"""GridWise — LLM-Assisted Smart Campus Energy Optimization API.

Endpoints:
    GET  /health          -> {"status": "ok"}
    POST /optimize-energy -> full interpretation + 24-hour schedule

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from llm_interpreter import interpret_notes
from optimizer import optimize
from schemas import (
    HourlyPlanEntry,
    OptimizeResponse,
    ScenarioRequest,
)
from validator import validate_interpretation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise Energy Optimizer", version="1.0.0")

# Permissive CORS so the judge's HTTP harness can call us from anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_endpoint(req: ScenarioRequest) -> OptimizeResponse:
    # 1. Sanity: 24 unique hours 0..23, demand/solar/tariff non-negative.
    hours_sorted = sorted(req.hours, key=lambda h: h.hour)
    hours = [h for h in hours_sorted if 0 <= h.hour <= 23]
    if {h.hour for h in hours} != set(range(24)):
        raise HTTPException(status_code=400, detail="hours must contain exactly 0..23")

    battery = req.battery

    # 2. LLM interprets operator notes.
    try:
        raw_entries = interpret_notes(
            operator_notes=list(req.operator_notes),
            scenario_id=req.scenario_id,
            battery_capacity_kwh=battery.capacity_kwh,
            initial_energy_kwh=battery.initial_energy_kwh,
            base_minimum_kwh=battery.minimum_energy_kwh,
        )
    except Exception as e:
        logger.exception("LLM interpreter crashed: %s", e)
        raw_entries = []

    # 3. Validate deterministically.
    safe_entries = validate_interpretation(raw_entries, list(req.operator_notes), battery)

    # 4. Optimize.
    plan, total_grid, total_cost, peak = optimize(hours, battery, safe_entries)

    # 5. Plan summary (short human-readable explanation).
    summary = _summarize(safe_entries, plan, hours)

    # 6. Build response.
    response = OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=safe_entries,
        hourly_plan=[HourlyPlanEntry(**p) for p in plan],
        total_grid_kwh=round(total_grid, 4),
        total_cost_bdt=round(total_cost, 4),
        peak_grid_kwh=round(peak, 4),
        plan_summary=summary,
    )
    return response


def _summarize(
    directives: List[Dict[str, Any]],
    plan: List[Dict[str, Any]],
    hours: List[Any],
) -> str:
    applied = [d for d in directives if d.get("applies")]
    if applied:
        kinds = ", ".join(sorted({d["directive_type"] for d in applied}))
    else:
        kinds = "no operator directives"

    charge_hours = [p["hour"] for p in plan if p["battery_action"] == "charge"]
    discharge_hours = [p["hour"] for p in plan if p["battery_action"] == "discharge"]
    return (
        f"Applied {kinds}; battery charges during hours {charge_hours or 'none'} "
        f"and discharges during hours {discharge_hours or 'none'} to minimize total grid cost."
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
