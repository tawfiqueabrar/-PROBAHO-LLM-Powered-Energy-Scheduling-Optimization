"""Tests for the LP optimizer (Section 09 invariants).

The optimizer must always produce a 24-hour plan that:
  - balances energy every hour
  - respects battery bounds and rate limits
  - enforces directive windows (no_charge, no_discharge, reserve, grid cap)
  - ends the day at the same battery level it started
"""

from __future__ import annotations

import pytest

from optimizer import optimize
from schemas import BatterySpec, HourEntry


def _mk_hours(demand_seq, solar_seq, tariff_seq):
    return [
        HourEntry(
            hour=h,
            demand_kwh=float(demand_seq[h]),
            solar_kwh=float(solar_seq[h]),
            tariff_bdt_per_kwh=float(tariff_seq[h]),
        )
        for h in range(24)
    ]


def _mk_battery(capacity=220, initial=110, minimum=40,
                max_chg=50, max_dis=50):
    return BatterySpec(
        capacity_kwh=capacity,
        initial_energy_kwh=initial,
        minimum_energy_kwh=minimum,
        max_charge_kwh_per_hour=max_chg,
        max_discharge_kwh_per_hour=max_dis,
    )


def _demands():
    return [90, 85, 80, 80, 85, 95, 110, 130, 150, 165, 175, 180,
            185, 180, 170, 165, 170, 185, 205, 215, 205, 175, 135, 105]


def _solars():
    return [0, 0, 0, 0, 0, 0, 5, 20, 50, 90, 130, 160,
            180, 170, 140, 90, 45, 10, 0, 0, 0, 0, 0, 0]


def _tariffs():
    return [6, 6, 5, 5, 5, 6, 8, 10, 12, 14, 16, 16,
            15, 14, 13, 14, 18, 22, 28, 30, 26, 18, 10, 7]


def _plan():
    hours = _mk_hours(_demands(), _solars(), _tariffs())
    bat = _mk_battery()
    plan, _, _, _ = optimize(hours, bat, [])
    return plan, hours, bat


# --------------------- Structural invariants ---------------------

def test_plan_has_24_entries_in_order():
    plan, _, _ = _plan()
    assert [p["hour"] for p in plan] == list(range(24))


def test_battery_action_is_valid_enum():
    plan, _, _ = _plan()
    for p in plan:
        assert p["battery_action"] in ("charge", "discharge", "idle")


def test_idle_must_have_zero_battery_kwh():
    plan, _, _ = _plan()
    for p in plan:
        if p["battery_action"] == "idle":
            assert p["battery_kwh"] == 0


def test_end_of_day_neutrality():
    plan, _, bat = _plan()
    assert abs(plan[-1]["battery_energy_after_kwh"] - bat.initial_energy_kwh) < 0.01


# --------------------- Per-hour invariants ------------------------

def test_energy_balance_every_hour():
    plan, hours, _ = _plan()
    for p in plan:
        h = p["hour"]
        demand = hours[h].demand_kwh
        if p["battery_action"] == "charge":
            charge, discharge = p["battery_kwh"], 0.0
        elif p["battery_action"] == "discharge":
            charge, discharge = 0.0, p["battery_kwh"]
        else:
            charge, discharge = 0.0, 0.0
        lhs = p["grid_kwh"] + p["solar_used_kwh"] + discharge
        rhs = demand + charge
        assert abs(lhs - rhs) < 0.01, f"hour {h}: balance LHS={lhs} RHS={rhs}"


def test_solar_used_never_exceeds_available():
    plan, hours, _ = _plan()
    for p in plan:
        h = p["hour"]
        assert p["solar_used_kwh"] <= hours[h].solar_kwh + 0.01


def test_battery_within_bounds():
    plan, _, bat = _plan()
    for p in plan:
        assert p["battery_energy_after_kwh"] >= bat.minimum_energy_kwh - 0.01
        assert p["battery_energy_after_kwh"] <= bat.capacity_kwh + 0.01


def test_rate_limits_respected():
    plan, _, bat = _plan()
    for p in plan:
        if p["battery_action"] == "charge":
            assert p["battery_kwh"] <= bat.max_charge_kwh_per_hour + 0.01
        if p["battery_action"] == "discharge":
            assert p["battery_kwh"] <= bat.max_discharge_kwh_per_hour + 0.01


# --------------------- Directive application ----------------------

def test_solar_reduction_reduces_usable_solar():
    hours = _mk_hours(_demands(), _solars(), _tariffs())
    bat = _mk_battery()
    plan, _, _, _ = optimize(
        hours, bat,
        [{"note_index": 0, "applies": True, "directive_type": "solar_reduction",
          "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
          "explanation": ""}],
    )
    # Effective solar in hours 12 & 13 should be 25% of original; usable <= that.
    for p in plan:
        if p["hour"] in (12, 13):
            original = _solars()[p["hour"]]
            assert p["solar_used_kwh"] <= original * 0.25 + 0.01


def test_no_charge_window_forces_charge_to_zero():
    hours = _mk_hours(_demands(), _solars(), _tariffs())
    bat = _mk_battery()
    plan, _, _, _ = optimize(
        hours, bat,
        [{"note_index": 0, "applies": True, "directive_type": "no_charge_window",
          "structured_adjustment": {"hours": [2, 3, 4]}, "explanation": ""}],
    )
    for p in plan:
        if p["hour"] in (2, 3, 4):
            assert p["battery_action"] != "charge"


def test_no_discharge_window_forces_discharge_to_zero():
    hours = _mk_hours(_demands(), _solars(), _tariffs())
    bat = _mk_battery()
    plan, _, _, _ = optimize(
        hours, bat,
        [{"note_index": 0, "applies": True, "directive_type": "no_discharge_window",
          "structured_adjustment": {"hours": [18, 19, 20]}, "explanation": ""}],
    )
    for p in plan:
        if p["hour"] in (18, 19, 20):
            assert p["battery_action"] != "discharge"


def test_max_grid_window_caps_grid_imports():
    hours = _mk_hours(_demands(), _solars(), _tariffs())
    bat = _mk_battery()
    plan, _, _, _ = optimize(
        hours, bat,
        [{"note_index": 0, "applies": True, "directive_type": "max_grid_window",
          "structured_adjustment": {"hours": [18, 19, 20], "max_grid_kwh": 100},
          "explanation": ""}],
    )
    for p in plan:
        if p["hour"] in (18, 19, 20):
            assert p["grid_kwh"] <= 100 + 0.01


def test_minimum_reserve_raises_battery_floor():
    hours = _mk_hours(_demands(), _solars(), _tariffs())
    bat = _mk_battery()
    plan, _, _, _ = optimize(
        hours, bat,
        [{"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
          "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 120},
          "explanation": ""}],
    )
    for p in plan:
        if p["hour"] in (18, 19, 20):
            assert p["battery_energy_after_kwh"] >= 120 - 0.01


# --------------------- Aggregate consistency ---------------------

def test_aggregates_match_recomputed_values():
    plan, hours, _ = _plan()
    total_grid = sum(p["grid_kwh"] for p in plan)
    total_cost = sum(p["grid_kwh"] * hours[p["hour"]].tariff_bdt_per_kwh for p in plan)
    peak = max(p["grid_kwh"] for p in plan)
    # Recompute via optimize (it returns these aggregates) and compare.
    plan2, tg, tc, pk = optimize(hours, _mk_battery(), [])
    assert abs(total_grid - tg) < 0.5
    assert abs(total_cost - tc) < 0.5
    assert abs(peak - pk) < 0.5
