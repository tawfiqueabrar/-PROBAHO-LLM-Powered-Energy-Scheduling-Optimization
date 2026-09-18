"""Pydantic schemas for the GridWise optimization API.

These mirror exactly the request/response contracts defined in Sections 07 and 10
of the BUP CSE Fest 2026 Preliminary Problem Statement.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class BatterySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    @model_validator(mode="after")
    def energy_bounds_are_consistent(self) -> "BatterySpec":
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh must not exceed capacity_kwh")
        if not self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh:
            raise ValueError("initial_energy_kwh must be between minimum_energy_kwh and capacity_kwh")
        return self


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scenario_id: str = Field(min_length=1)
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourEntry] = Field(min_length=24, max_length=24)
    battery: BatterySpec

    @field_validator("operator_notes")
    @classmethod
    def notes_must_be_nonempty(cls, notes: List[str]) -> List[str]:
        cleaned = [note.strip() for note in notes]
        if any(not note for note in cleaned):
            raise ValueError("operator_notes must contain only non-empty strings")
        return cleaned

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
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    note_index: int
    applies: bool
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    ]
    structured_adjustment: Optional[dict] = None
    explanation: str = ""


# ---------- Hourly plan schemas ----------

class HourlyPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    hour: int = Field(ge=0, le=23)
    grid_kwh: float = Field(ge=0)
    solar_used_kwh: float = Field(ge=0)
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float


# ---------- Response schema ----------

class OptimizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
