"""Pydantic schemas for the GridWise optimization API.

These mirror exactly the request/response contracts defined in Sections 07 and 10
of the BUP CSE Fest 2026 Preliminary Problem Statement.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


# ---------- Allowed enums ----------

ALLOWED_DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

ALLOWED_BATTERY_ACTIONS = {"charge", "discharge", "idle"}


# ---------- Request schemas ----------

class HourEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class BatterySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str = Field(min_length=1)
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourEntry] = Field(min_length=24, max_length=24)
    battery: BatterySpec

    @property
    def operator_notes_list(self) -> List[str]:
        return self.operator_notes


# ---------- Directive interpretation schemas ----------

class SolarReductionAdj(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int]
    factor: float


class MinimumBatteryReserveAdj(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int]
    minimum_energy_kwh: float


class HoursOnlyAdj(BaseModel):
    """Used by no_charge_window and no_discharge_window."""
    model_config = ConfigDict(extra="forbid")
    hours: List[int]


class MaxGridWindowAdj(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int]
    max_grid_kwh: float


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note_index: int
    applies: bool
    directive_type: str
    structured_adjustment: Optional[dict] = None
    explanation: str = ""


# ---------- Hourly plan schemas ----------

class HourlyPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hour: int = Field(ge=0, le=23)
    grid_kwh: float = Field(ge=0)
    solar_used_kwh: float = Field(ge=0)
    battery_action: str
    battery_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float


# ---------- Response schema ----------

class OptimizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
