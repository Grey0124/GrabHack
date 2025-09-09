import json
import os
from typing import Optional

import typer

from src.agent.state import AgentState


app = typer.Typer(help="Project Synapse ReAct CLI")


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
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except Exception:
        # Fail-soft .env loader
        pass


@app.command()
def solve(
    disruption: str = typer.Argument(..., help="Disruption scenario description"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override model name (e.g., llama3.1:8b)"
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Override Ollama OpenAI-compatible base URL (default: http://localhost:11434/v1)",
    ),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", min=1.0, help="LLM request timeout in seconds (default: 300)"
    ),
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Print plan/act/reflect progress"
    ),
):
    """Solve a disruption with a multi-step ReAct loop."""
    load_dotenv()
    if model:
        os.environ["MODEL_NAME"] = model
    if base_url:
        os.environ["OLLAMA_API_BASE"] = base_url
    if timeout:
        os.environ["OLLAMA_TIMEOUT"] = str(timeout)
    os.environ["AGENT_PROGRESS"] = "1" if progress else "0"

    try:
        # Import after environment is prepared so graph picks up runtime flags
        from src.agent.graph import build_graph  # noqa: WPS433
        graph = build_graph()
        result: AgentState = graph.invoke(AgentState(goal=disruption))
        # LangGraph may return a plain dict; support both Pydantic and dict outputs
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        # Remember successful runs (fail-soft)
        try:
            solved = bool(getattr(result, "solved", False)) if hasattr(result, "solved") else bool((payload or {}).get("solved"))
            if solved and isinstance(payload, dict):
                from src.mem.ltm import remember  # runtime import to avoid unnecessary deps at startup
                remember(payload)
        except Exception:
            # Do not block CLI output on memory issues
            pass
        typer.echo(json.dumps(payload, indent=2, default=str))
    except Exception as exc:
        typer.echo(
            json.dumps({"error": str(exc), "type": exc.__class__.__name__}, indent=2),
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
