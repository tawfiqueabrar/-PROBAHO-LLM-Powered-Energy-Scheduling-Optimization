"""Tests for the deterministic validator (Section 08 guardrails)."""

from __future__ import annotations

import pytest

from schemas import BatterySpec
from validator import validate_interpretation


@pytest.fixture
def battery() -> BatterySpec:
    return BatterySpec(
        capacity_kwh=500,
        initial_energy_kwh=200,
        minimum_energy_kwh=50,
        max_charge_kwh_per_hour=100,
        max_discharge_kwh_per_hour=100,
    )


def _note(s: str = "ignore me") -> str:
    return s


def test_returns_one_entry_per_note(battery):
    notes = [_note("Solar drops to 20% from 1 PM to 3 PM."),
             _note("Cafeteria menu changes tomorrow.")]
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
         "explanation": "..."},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "..."},
    ]
    out = validate_interpretation(raw, notes, battery)
    assert len(out) == 2
    assert [e["note_index"] for e in out] == [0, 1]


def test_unknown_directive_type_falls_back_to_no_op(battery):
    notes = [_note("something")]
    raw = [{"note_index": 0, "applies": True,
            "directive_type": "demand_reduction",   # not in the allowed enum
            "structured_adjustment": {"hours": [10]}, "explanation": ""}]
    out = validate_interpretation(raw, notes, battery)
    assert out[0]["applies"] is False
    assert out[0]["directive_type"] == "no_op"
    assert out[0]["structured_adjustment"] is None


def test_no_op_must_have_applies_false_and_null_adjustment(battery):
    notes = [_note("something")]
    raw = [{"note_index": 0, "applies": True, "directive_type": "no_op",
            "structured_adjustment": {"hours": [10]}, "explanation": ""}]
    out = validate_interpretation(raw, notes, battery)
    assert out[0]["applies"] is False


def test_solar_reduction_factor_must_be_in_0_1(battery):
    notes = [_note("Solar 150% boost from noon to 1 PM.")]
    raw = [{"note_index": 0, "applies": True, "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [12], "factor": 1.5}, "explanation": ""}]
    out = validate_interpretation(raw, notes, battery)
    assert out[0]["directive_type"] == "no_op"


def test_hours_must_be_sorted_unique_and_in_range(battery):
    notes = [_note("Solar reduced")]
    raw = [{"note_index": 0, "applies": True, "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [14, 13, 13]}, "explanation": ""}]
    out = validate_interpretation(raw, notes, battery)
    assert out[0]["directive_type"] == "no_op"


def test_minimum_reserve_cannot_exceed_capacity():
    bat = BatterySpec(
        capacity_kwh=100, initial_energy_kwh=50, minimum_energy_kwh=10,
        max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50,
    )
    notes = [_note("Keep 999 kWh in reserve")]
    raw = [{"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [10], "minimum_energy_kwh": 999}, "explanation": ""}]
    out = validate_interpretation(raw, notes, bat)
    assert out[0]["directive_type"] == "no_op"


def test_missing_note_becomes_no_op(battery):
    notes = [_note("a"), _note("b"), _note("c")]
    raw = [{"note_index": 0, "applies": True, "directive_type": "no_op",
            "structured_adjustment": None, "explanation": ""}]  # only 1 entry
    out = validate_interpretation(raw, notes, battery)
    assert len(out) == 3
    assert all(e["directive_type"] == "no_op" for e in out)


def test_applies_true_required_for_non_no_op(battery):
    notes = [_note("Do not charge from 2 to 4 PM.")]
    raw = [{"note_index": 0, "applies": False, "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [14, 15]}, "explanation": ""}]
    out = validate_interpretation(raw, notes, battery)
    # applies=False + non-no_op -> validator reclassifies to no_op.
    assert out[0]["directive_type"] == "no_op"


def test_max_grid_window_requires_finite_cap(battery):
    notes = [_note("Cap grid from 6 to 9 PM.")]
    raw = [{"note_index": 0, "applies": True, "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [18, 19, 20], "max_grid_kwh": -10}, "explanation": ""}]
    out = validate_interpretation(raw, notes, battery)
    assert out[0]["directive_type"] == "no_op"


def test_valid_minimum_reserve_is_preserved(battery):
    notes = [_note("Keep 120 kWh in reserve from 6 PM to 9 PM.")]
    raw = [{"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 120},
            "explanation": ""}]
    out = validate_interpretation(raw, notes, battery)
    assert out[0]["applies"] is True
    assert out[0]["directive_type"] == "minimum_battery_reserve"
    assert out[0]["structured_adjustment"]["hours"] == [18, 19, 20]
    assert out[0]["structured_adjustment"]["minimum_energy_kwh"] == 120
