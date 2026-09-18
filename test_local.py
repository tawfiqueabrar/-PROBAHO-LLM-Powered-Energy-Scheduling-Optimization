"""Local validation harness for GridWise.

Loops through every case in the public sample pack, POSTs the .input to a
running /optimize-energy service, and verifies that:

  1. The interpretation has one entry per note in note_index order.
  2. Each entry's directive_type matches the expected reference.
  3. Each applies flag matches the expected reference.
  4. structured_adjustment matches the expected reference (with tolerance).
  5. The hourly_plan has exactly 24 entries for hours 0..23.
  6. The aggregates (total_grid_kwh, total_cost_bdt, peak_grid_kwh) match
     the values recalculated from hourly_plan within the judge's 0.01 tolerance.
  7. End-of-day battery neutrality holds: battery_energy_after_kwh[23] == initial.
  8. Solar usage never exceeds effective solar (after any solar_reduction).
  9. Battery bounds and rate limits respected.

Usage:
    # Against a running server
    python test_local.py --url http://localhost:8000

    # Directly in-process (skips the HTTP layer)
    python test_local.py --inproc
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES = ROOT.parent / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"

TOL = 0.01


# ----------------------------- HTTP client ----------------------------------

def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_inproc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call the FastAPI app directly without going through HTTP."""
    from main import optimize_endpoint  # type: ignore
    from schemas import ScenarioRequest

    req = ScenarioRequest(**payload)
    resp = optimize_endpoint(req)
    return json.loads(resp.model_dump_json())


# ----------------------------- Validators -----------------------------------

class Checker:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def fail(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self) -> bool:
        return not self.errors


def check_interpretation(
    chk: Checker,
    response: Dict[str, Any],
    expected: Dict[str, Any],
    n_notes: int,
) -> None:
    interp = response.get("directive_interpretation")
    if not isinstance(interp, list):
        chk.fail("directive_interpretation missing or not a list")
        return

    if len(interp) != n_notes:
        chk.fail(f"directive_interpretation length {len(interp)} != {n_notes}")

    # note_index order and uniqueness
    seen_idx: List[int] = []
    for entry in interp:
        idx = entry.get("note_index")
        if not isinstance(idx, int):
            chk.fail(f"non-int note_index: {idx!r}")
            continue
        seen_idx.append(idx)

    if sorted(seen_idx) != list(range(n_notes)):
        chk.fail(f"note_index set/order wrong: {seen_idx}")

    # Per-entry shape
    exp_interp = expected.get("directive_interpretation", [])
    exp_by_idx = {e["note_index"]: e for e in exp_interp}

    for entry in interp:
        idx = entry.get("note_index")
        exp = exp_by_idx.get(idx)
        if exp is None:
            continue  # No reference; just validate shape.

        dtype = entry.get("directive_type")
        if dtype != exp.get("directive_type"):
            chk.fail(
                f"note {idx}: directive_type {dtype!r} != expected {exp.get('directive_type')!r}"
            )

        if entry.get("applies") != exp.get("applies"):
            chk.fail(
                f"note {idx}: applies {entry.get('applies')} != expected {exp.get('applies')}"
            )

        if dtype == "no_op":
            if entry.get("structured_adjustment") is not None:
                chk.fail(f"note {idx}: no_op must have null structured_adjustment")
            continue

        adj = entry.get("structured_adjustment")
        exp_adj = exp.get("structured_adjustment") or {}
        if not isinstance(adj, dict):
            chk.fail(f"note {idx}: structured_adjustment missing or not a dict")
            continue

        # Hours must match
        adj_hours = adj.get("hours", [])
        exp_hours = exp_adj.get("hours", [])
        if list(adj_hours) != list(exp_hours):
            chk.fail(f"note {idx}: hours {adj_hours} != expected {exp_hours}")

        # Numeric fields within tolerance
        for k in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if k in exp_adj:
                if k not in adj:
                    chk.fail(f"note {idx}: missing numeric field {k!r}")
                else:
                    try:
                        if abs(float(adj[k]) - float(exp_adj[k])) > TOL:
                            chk.fail(
                                f"note {idx}: {k} {adj[k]} differs from expected {exp_adj[k]} by > {TOL}"
                            )
                    except (TypeError, ValueError):
                        chk.fail(f"note {idx}: {k} not numeric")


def check_hourly_plan(chk: Checker, response: Dict[str, Any], input_payload: Dict[str, Any]) -> None:
    plan = response.get("hourly_plan")
    if not isinstance(plan, list) or len(plan) != 24:
        chk.fail(f"hourly_plan length {len(plan) if isinstance(plan, list) else 'N/A'} != 24")
        return

    hours_seen = set()
    battery_initial = float(input_payload["battery"]["initial_energy_kwh"])
    battery_capacity = float(input_payload["battery"]["capacity_kwh"])
    battery_min = float(input_payload["battery"]["minimum_energy_kwh"])
    max_chg = float(input_payload["battery"]["max_charge_kwh_per_hour"])
    max_dis = float(input_payload["battery"]["max_discharge_kwh_per_hour"])
    demand_map = {int(h["hour"]): float(h["demand_kwh"]) for h in input_payload["hours"]}
    solar_map = {int(h["hour"]): float(h["solar_kwh"]) for h in input_payload["hours"]}
    tariff_map = {int(h["hour"]): float(h["tariff_bdt_per_kwh"]) for h in input_payload["hours"]}

    # Effective solar after directive reductions
    eff_solar = dict(solar_map)
    for entry in response.get("directive_interpretation", []):
        if not entry.get("applies"):
            continue
        if entry.get("directive_type") != "solar_reduction":
            continue
        adj = entry.get("structured_adjustment") or {}
        factor = float(adj.get("factor", 1.0))
        for h in adj.get("hours", []):
            eff_solar[int(h)] = eff_solar.get(int(h), 0.0) * factor

    # Per-hour reserve floor (raises base minimum for listed hours)
    reserve_floor = {h: battery_min for h in range(24)}
    # Per-hour grid cap
    grid_cap = {h: float("inf") for h in range(24)}
    # Per-hour charge/discharge window blocks
    no_charge_h = set()
    no_discharge_h = set()

    for entry in response.get("directive_interpretation", []):
        if not entry.get("applies"):
            continue
        dt = entry.get("directive_type")
        adj = entry.get("structured_adjustment") or {}
        for h in adj.get("hours", []):
            h = int(h)
            if dt == "minimum_battery_reserve":
                reserve_floor[h] = max(reserve_floor[h], float(adj.get("minimum_energy_kwh", 0.0)))
            elif dt == "no_charge_window":
                no_charge_h.add(h)
            elif dt == "no_discharge_window":
                no_discharge_h.add(h)
            elif dt == "max_grid_window":
                grid_cap[h] = min(grid_cap[h], float(adj.get("max_grid_kwh", float("inf"))))

    prev_e = battery_initial
    sum_grid = 0.0
    sum_cost = 0.0
    peak = 0.0

    for entry in plan:
        h = int(entry.get("hour"))
        hours_seen.add(h)
        g = float(entry.get("grid_kwh", 0.0))
        s_used = float(entry.get("solar_used_kwh", 0.0))
        action = entry.get("battery_action")
        mag = float(entry.get("battery_kwh", 0.0))
        e_after = float(entry.get("battery_energy_after_kwh", 0.0))

        # Non-negativity
        for k, v in (("grid_kwh", g), ("solar_used_kwh", s_used), ("battery_kwh", mag)):
            if v < -TOL:
                chk.fail(f"hour {h}: {k} negative: {v}")

        # Battery action consistency
        if action == "idle" and abs(mag) > TOL:
            chk.fail(f"hour {h}: idle but battery_kwh={mag}")
        if action not in ("charge", "discharge", "idle"):
            chk.fail(f"hour {h}: invalid battery_action {action!r}")

        # Rate limits
        if action == "charge" and mag > max_chg + TOL:
            chk.fail(f"hour {h}: charge {mag} exceeds max_charge {max_chg}")
        if action == "discharge" and mag > max_dis + TOL:
            chk.fail(f"hour {h}: discharge {mag} exceeds max_discharge {max_dis}")
        if h in no_charge_h and action == "charge":
            chk.fail(f"hour {h}: charging inside no_charge_window")
        if h in no_discharge_h and action == "discharge":
            chk.fail(f"hour {h}: discharging inside no_discharge_window")

        # Solar cap
        if s_used > eff_solar.get(h, 0.0) + TOL:
            chk.fail(
                f"hour {h}: solar_used {s_used} exceeds effective_solar {eff_solar.get(h, 0.0)}"
            )

        # Grid cap
        if g > grid_cap[h] + TOL:
            chk.fail(f"hour {h}: grid_kwh {g} exceeds max_grid_window cap {grid_cap[h]}")

        # Battery bounds
        if e_after < reserve_floor[h] - TOL:
            chk.fail(
                f"hour {h}: battery_energy_after_kwh {e_after} below reserve_floor {reserve_floor[h]}"
            )
        if e_after > battery_capacity + TOL:
            chk.fail(f"hour {h}: battery_energy_after_kwh {e_after} above capacity {battery_capacity}")

        # Dynamics + balance
        if action == "charge":
            e_check = prev_e + mag
            charge_kwh = mag
            discharge_kwh = 0.0
        elif action == "discharge":
            e_check = prev_e - mag
            charge_kwh = 0.0
            discharge_kwh = mag
        else:
            e_check = prev_e
            charge_kwh = 0.0
            discharge_kwh = 0.0

        if abs(e_check - e_after) > TOL:
            chk.fail(
                f"hour {h}: battery dynamics mismatch: prev+act={e_check} vs reported {e_after}"
            )

        demand = demand_map[h]
        lhs = g + s_used + discharge_kwh
        rhs = demand + charge_kwh
        if abs(lhs - rhs) > TOL:
            chk.fail(f"hour {h}: energy balance LHS={lhs} != RHS={rhs}")

        prev_e = e_after
        sum_grid += g
        sum_cost += g * tariff_map[h]
        peak = max(peak, g)

    if hours_seen != set(range(24)):
        chk.fail(f"hourly_plan missing hours: {set(range(24)) - hours_seen}")

    # End-of-day neutrality
    if abs(prev_e - battery_initial) > TOL:
        chk.fail(
            f"end-of-day battery neutrality: final {prev_e} != initial {battery_initial}"
        )

    # Aggregate consistency
    if abs(sum_grid - float(response.get("total_grid_kwh", 0.0))) > TOL:
        chk.fail(f"total_grid_kwh mismatch: recomputed {sum_grid} vs reported {response.get('total_grid_kwh')}")
    if abs(sum_cost - float(response.get("total_cost_bdt", 0.0))) > TOL:
        chk.fail(f"total_cost_bdt mismatch: recomputed {sum_cost} vs reported {response.get('total_cost_bdt')}")
    if abs(peak - float(response.get("peak_grid_kwh", 0.0))) > TOL:
        chk.fail(f"peak_grid_kwh mismatch: recomputed {peak} vs reported {response.get('peak_grid_kwh')}")


def check_one_case(case: Dict[str, Any], response: Dict[str, Any]) -> Tuple[bool, List[str]]:
    chk = Checker(case["id"])
    inp = case["input"]
    expected = case.get("expected_output", {})
    check_interpretation(chk, response, expected, n_notes=len(inp["operator_notes"]))
    check_hourly_plan(chk, response, inp)
    return chk.ok(), chk.errors


# ----------------------------- Main loop ------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/optimize-energy")
    parser.add_argument("--inproc", action="store_true", help="call the FastAPI app directly")
    parser.add_argument(
        "--samples",
        default=str(DEFAULT_SAMPLES),
        help="path to the public sample pack JSON",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"Sample pack not found: {samples_path}")
        return 2

    data = json.loads(samples_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    print(f"Loaded {len(cases)} cases from {samples_path.name}")

    n_pass = 0
    n_fail = 0
    total_errors: List[str] = []
    for case in cases:
        inp = case["input"]
        try:
            if args.inproc:
                response = call_inproc(inp)
            else:
                response = post_json(args.url, inp)
        except Exception as e:
            print(f"[{case['id']}] HTTP ERROR: {e}")
            n_fail += 1
            continue

        ok, errs = check_one_case(case, response)
        if ok:
            print(f"[{case['id']}] OK")
            n_pass += 1
        else:
            print(f"[{case['id']}] FAIL ({len(errs)} issues)")
            for e in errs:
                print(f"    - {e}")
                total_errors.append(f"[{case['id']}] {e}")
            n_fail += 1

    print()
    print(f"Passed: {n_pass}/{len(cases)}    Failed: {n_fail}/{len(cases)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
