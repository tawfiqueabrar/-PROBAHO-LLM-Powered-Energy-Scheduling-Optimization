"""End-to-end API tests against supplied scenario fixtures.

We invoke the FastAPI app in-process via TestClient so these tests don't need
a running server, network access, or an OpenAI key.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def _run_case(client: TestClient, payload: dict) -> dict:
    r = client.post("/optimize-energy", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_all_fixture_cases_pass_smoke_checks(client: TestClient, sample_cases):
    for case in sample_cases:
        resp = _run_case(client, case["input"])

        # 1. one entry per note
        n_notes = len(case["input"]["operator_notes"])
        assert len(resp["directive_interpretation"]) == n_notes, case["id"]

        # 2. 24 hourly entries
        assert len(resp["hourly_plan"]) == 24, case["id"]

        # 3. end-of-day neutrality
        initial = case["input"]["battery"]["initial_energy_kwh"]
        assert abs(resp["hourly_plan"][-1]["battery_energy_after_kwh"] - initial) < 0.01, case["id"]

        # 4. energy balance every hour
        demand_map = {h["hour"]: h["demand_kwh"] for h in case["input"]["hours"]}
        for p in resp["hourly_plan"]:
            h = p["hour"]
            if p["battery_action"] == "charge":
                charge, dis = p["battery_kwh"], 0.0
            elif p["battery_action"] == "discharge":
                charge, dis = 0.0, p["battery_kwh"]
            else:
                charge, dis = 0.0, 0.0
            lhs = p["grid_kwh"] + p["solar_used_kwh"] + dis
            rhs = demand_map[h] + charge
            assert abs(lhs - rhs) < 0.01, f"{case['id']} hour {h} balance {lhs}!={rhs}"

        # 5. aggregates match recomputed values
        tg = sum(p["grid_kwh"] for p in resp["hourly_plan"])
        tc = sum(p["grid_kwh"] * h["tariff_bdt_per_kwh"]
                 for p, h in zip(resp["hourly_plan"], case["input"]["hours"]))
        pk = max(p["grid_kwh"] for p in resp["hourly_plan"])
        assert abs(tg - resp["total_grid_kwh"]) < 0.01
        assert abs(tc - resp["total_cost_bdt"]) < 0.01
        assert abs(pk - resp["peak_grid_kwh"]) < 0.01


def test_invalid_request_returns_400(client: TestClient):
    bad = {"scenario_id": "X", "operator_notes": [], "hours": [], "battery": {}}
    r = client.post("/optimize-energy", json=bad)
    assert r.status_code == 400


def test_blank_operator_note_returns_400(client: TestClient):
    payload = {
        "scenario_id": "blank-note",
        "operator_notes": ["   "],
        "hours": [
            {"hour": h, "demand_kwh": 1, "solar_kwh": 0, "tariff_bdt_per_kwh": 1}
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": 10,
            "initial_energy_kwh": 5,
            "minimum_energy_kwh": 0,
            "max_charge_kwh_per_hour": 1,
            "max_discharge_kwh_per_hour": 1,
        },
    }
    assert client.post("/optimize-energy", json=payload).status_code == 400


def test_openapi_documents_actual_error_statuses(client: TestClient):
    responses = client.get("/openapi.json").json()["paths"]["/optimize-energy"]["post"]["responses"]
    assert responses["400"]["description"] == "Malformed or structurally invalid request."
    assert responses["422"]["description"] == "No feasible schedule satisfies the supplied constraints."
