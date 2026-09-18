"""Smoke tests that don't need the sample pack."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_optimize_minimal_request():
    payload = {
        "scenario_id": "MIN-1",
        "operator_notes": ["Cafeteria menu changes tomorrow."],
        "hours": [
            {"hour": h, "demand_kwh": 100.0, "solar_kwh": 0.0,
             "tariff_bdt_per_kwh": 6.0 + (h % 5)}
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": 200.0,
            "initial_energy_kwh": 100.0,
            "minimum_energy_kwh": 20.0,
            "max_charge_kwh_per_hour": 50.0,
            "max_discharge_kwh_per_hour": 50.0,
        },
    }
    r = client.post("/optimize-energy", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["scenario_id"] == "MIN-1"
    assert len(body["hourly_plan"]) == 24
    assert len(body["directive_interpretation"]) == 1
    # Irrelevant note -> no_op
    assert body["directive_interpretation"][0]["directive_type"] == "no_op"
