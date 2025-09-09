import json
import os
from typing import Optional, List, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent.state import AgentState


def load_dotenv(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass


def _native_base_from_env() -> str:
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1").rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


app = FastAPI(title="Synapse Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolveReq(BaseModel):
    disruption: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    progress: Optional[bool] = None
    offline: Optional[bool] = None


class WarmupReq(BaseModel):
    model: Optional[str] = None
    keep_alive: str = "30m"
    temperature: float = 0.0


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/scenarios")
def scenarios() -> List[dict]:
    path = os.path.join("src", "scenarios", "examples.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


@app.post("/warmup")
def warmup(req: WarmupReq) -> Any:
    load_dotenv()
    model = req.model or os.getenv("MODEL_NAME", "llama3.1:8b")
    base = _native_base_from_env()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "warmup"}],
        "stream": False,
        "keep_alive": req.keep_alive,
        "options": {"temperature": req.temperature, "num_predict": 1},
    }
    try:
        with httpx.Client(base_url=base, timeout=httpx.Timeout(60.0)) as client:
            r = client.post("/api/chat", json=payload)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/solve")
def solve(req: SolveReq) -> Any:
    load_dotenv()
    if req.model:
        os.environ["MODEL_NAME"] = req.model
    if req.base_url:
        os.environ["OLLAMA_API_BASE"] = req.base_url
    if req.timeout:
        os.environ["OLLAMA_TIMEOUT"] = str(req.timeout)
    if req.progress is not None:
        os.environ["AGENT_PROGRESS"] = "1" if req.progress else "0"
    if req.offline is not None:
        os.environ["AGENT_OFFLINE"] = "1" if req.offline else "0"

    try:
        from src.agent.graph import build_graph  # import after env overrides
        graph = build_graph()
        result = graph.invoke(AgentState(goal=req.disruption))
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        try:
            solved = bool(getattr(result, "solved", False)) if hasattr(result, "solved") else bool((payload or {}).get("solved"))
            if solved and isinstance(payload, dict):
                from src.mem.ltm import remember
                remember(payload)
        except Exception:
            pass
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- UI Hook: serve static files and redirect root to /ui/ ---
static_dir = os.path.join(os.getcwd(), "ui")
if os.path.isdir(static_dir):
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")


@app.get("/")
def _root_redirect():
    return RedirectResponse(url="/ui/")
