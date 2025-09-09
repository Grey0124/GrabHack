# Project Synapse — Local Agent Demo

This repo contains a functional proof‑of‑concept autonomous logistics agent with:
- A CLI that accepts a disruption scenario and prints a transparent chain‑of‑thought (Thought/Action/Observation/Reflection)
- A FastAPI server exposing `/solve`, `/warmup`, and `/scenarios`
- A minimal web UI under `/ui` to run scenarios and view traces
- Optional long‑term memory using Ollama embeddings + ChromaDB


## 1) Prerequisites

- Python 3.10+ (recommended 3.11+)
- Ollama installed and running locally (http://localhost:11434)
  - Install: see https://ollama.com
  - Pull models:
    - `ollama pull llama3.1:8b`
    - `ollama pull nomic-embed-text` (for memory)


## 2) Setup (new device)

Clone and create a virtual environment, then install dependencies.

macOS/Linux (bash/zsh)
- `git clone <this-repo-url> && cd <repo>`
- `python -m venv .venv && source .venv/bin/activate`
- `python -m pip install -U pip`
- `pip install fastapi uvicorn httpx typer pydantic chromadb langgraph pytest`

Windows (PowerShell)
- `git clone <this-repo-url>; cd <repo>`
- `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
- `python -m pip install -U pip`
- `pip install fastapi uvicorn httpx typer pydantic chromadb langgraph pytest`

Optional: create a `.env` file (same directory as README.md) to override defaults:
- Example contents:
  - `MODEL_NAME=llama3.1:8b`
  - `OLLAMA_API_BASE=http://localhost:11434/v1`
  - `AGENT_MAX_STEPS=4`
  - `VECTOR_DIR=.vectorstore`


## 3) Warm up the model (faster first request)

Using native Ollama API (recommended):
- PowerShell: 
  - `$body = @'{ "model":"llama3.1:8b","messages":[{"role":"user","content":"warmup"}],"stream":false,"keep_alive":"30m","options":{"temperature":0,"num_predict":1} }'@`
  - `curl.exe http://localhost:11434/api/chat -H "Content-Type: application/json" -d $body`
- macOS/Linux:
  - `curl http://localhost:11434/api/chat -H 'Content-Type: application/json' -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"warmup"}],"stream":false,"keep_alive":"30m","options":{"temperature":0,"num_predict":1}}'`

Or via our API server after you start it (see section 6):
- `curl -X POST http://localhost:8000/warmup -H 'Content-Type: application/json' -d '{"keep_alive":"30m"}'`


## 4) Run the CLI (chain‑of‑thought in JSON)

Basic usage:
- `python -m src.cli_react --help`

Examples (macOS/Linux):
- `python -m src.cli_react --no-progress "Kitchen prep 40+ minutes; keep driver utilized; offer alternatives and notify."`
- `python -m src.cli_react --no-progress "Recipient unavailable at 123 Main St; valuable parcel"`

Examples (Windows PowerShell):
- `& python -m src.cli_react --no-progress 'Kitchen prep 40+ minutes; keep driver utilized; offer alternatives and notify.'`
- `& python -m src.cli_react --no-progress 'Recipient unavailable at 123 Main St; valuable parcel'`

Offline smoke (no Ollama calls):
- macOS/Linux: `AGENT_OFFLINE=1 python -m src.cli_react --no-progress "Recipient unavailable at 123 Main St; valuable parcel"`
- PowerShell: `$env:AGENT_OFFLINE='1'; & python -m src.cli_react --no-progress 'Recipient unavailable at 123 Main St; valuable parcel'`

Notes
- `--no-progress` keeps stdout pure JSON (easy to parse). Omit it to see live plan/act/reflect logs.
- Successful runs are remembered into `.vectorstore` (can be changed via `VECTOR_DIR`).


## 5) Seeded scenarios

You can copy/paste prompts from `src/scenarios/examples.json` or run them via the UI (next section).


## 6) Run the local API and web UI

Start the server:
- `uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload`

Open the UI:
- http://localhost:8000/ (redirects to `/ui/`)

UI features
- Warm Model button calls `/warmup` with keep‑alive.
- Scenario picker loads examples from `/scenarios`.
- “Offline mode” toggles `AGENT_OFFLINE` to run without model calls.
- Displays chain‑of‑thought trace (scratchpad) and raw JSON output.


## 7) Tests

Run the smoke test:
- macOS/Linux: `AGENT_OFFLINE=1 pytest -q`
- PowerShell: `$env:AGENT_OFFLINE='1'; .\.venv\Scripts\pytest.exe -q`


## 8) Configuration reference

Environment variables
- `MODEL_NAME` (default `llama3.1:8b`)
- `OLLAMA_API_BASE` (default `http://localhost:11434/v1`)
- `OLLAMA_TIMEOUT` (seconds; default `300`)
- `AGENT_MAX_STEPS` (default `3`)
- `AGENT_PROGRESS` (`1`/`0`; default `1`)
- `AGENT_OFFLINE` (`1`/`0`; default `0`)
- `VECTOR_DIR` (default `.vectorstore`)
- `EMBED_MODEL` (default `nomic-embed-text`)

Key files
- `src/cli_react.py` — CLI entrypoint
- `src/agent/graph.py` — ReAct plan/act/reflect + self‑critique repair loop
- `src/tools/logistics.py` — simulated logistics tools
- `src/mem/ltm.py` — embeddings + Chroma memory
- `src/api/server.py` — FastAPI server and UI hook
- `ui/` — static web UI


## 9) Troubleshooting

- Windows quoting errors (PowerShell): prefer single quotes around prompts or escape semicolons using backtick (`` `; ``).
- “invalid character '\'' looking for beginning of value”: Use `curl.exe` with PowerShell here‑string for JSON bodies.
- Timeouts via `/v1/chat/completions`: This project uses native Ollama `/api/chat` for reliability; ensure Ollama is running.
- Verify model is loaded: `ollama ps`. Unload: `ollama stop llama3.1:8b`.
