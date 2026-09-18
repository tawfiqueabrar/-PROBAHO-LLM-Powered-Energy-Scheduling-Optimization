# GridWise — Smart Campus Energy Optimization

LLM-assisted 24-hour energy scheduler for BUP CSE Fest 2026 Hackathon (Preliminary).

## What it does

* `GET /health` → `{"status": "ok"}`
* `POST /optimize-energy` → takes a 24-hour scenario + 1-3 natural-language operator notes, returns:
  * `directive_interpretation[]` — machine-checkable interpretation of each note
  * `hourly_plan[24]` — battery / solar / grid dispatch
  * `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh`, `plan_summary`

The LLM is on the **interpretation path** as required by the problem statement.
LLM output is treated as untrusted; `validator.py` applies deterministic guardrails
before any directive touches the optimizer. The optimizer is a linear program (PuLP/CBC).

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env       # then add your OPENAI_API_KEY
python main.py             # or: uvicorn main:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl http://localhost:8000/health
```

Send a scenario (see the public sample pack for full payloads):

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, `/health` + `/optimize-energy` |
| `schemas.py` | Pydantic request/response models (Section 07 & 10) |
| `llm_interpreter.py` | OpenAI-based note interpreter + deterministic fallback |
| `validator.py` | Guardrails from Section 08 — only safe data reaches the optimizer |
| `optimizer.py` | LP (PuLP/CBC) that minimizes total grid cost |

## Notes for the judge harness

* Endpoint names match exactly (`/health`, `/optimize-energy`).
* 24-hour horizon, end-of-day battery neutrality enforced in the LP.
* Time windows are half-open: "1 PM to 3 PM" → `[13, 14]`.
* `no_op` is the only directive with `applies=false`.
* If the LLM fails or returns malformed JSON, the service falls back to a deterministic keyword interpreter rather than crashing.

## Public test pack

`test_local.py` loops through all 10 public cases and verifies interpretation,
plan, aggregates, dynamics, balance, neutrality, and bounds:

```bash
# Against a running server on :8000
python test_local.py --url http://localhost:8000/optimize-energy

# Or skip the HTTP layer entirely
python test_local.py --inproc
```

## Pytest suite

```bash
pip install -r requirements-dev.txt
pytest                  # full suite (validator, optimizer, API, sample cases)
pytest -m smoke         # only smoke tests
pytest tests/test_optimizer.py -v
```

The suite runs without an OpenAI key — the optimizer is fully deterministic
and the interpreter falls back to a deterministic keyword matcher when no key
is set.

## Deploy

### Option 1 — Render (recommended)

```bash
# 1. Push to GitHub
# 2. Go to https://render.com → New + → Blueprint → select your repo
# 3. Set OPENAI_API_KEY in the Render dashboard (sync: false)
# 4. Render builds the Dockerfile and gives you a public HTTPS URL
```

The `render.yaml` blueprint is pre-configured with the right healthcheck path.

### Option 2 — Railway

```bash
npm i -g @railway/cli
railway login
railway up
railway variables set OPENAI_API_KEY=sk-...
```

### Option 3 — Fly.io

```bash
brew install flyctl    # or scoop install flyctl on Windows
fly auth signup
fly launch --copy-config --name gridwise-energy
fly secrets set OPENAI_API_KEY=sk-...
fly deploy
```

The included `fly.toml` is pre-wired with the right healthcheck (`/health`)
and a primary region of Singapore (`sin`) which is closest to Bangladesh.

### Option 4 — Local Docker

```bash
docker build -t gridwise .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... gridwise
curl http://localhost:8000/health
```

### Option 5 — Direct (no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python main.py         # http://localhost:8000
```
