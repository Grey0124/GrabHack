import json
import os
from typing import Optional

import typer


app = typer.Typer(help="Project Synapse PoC CLI")


def load_dotenv(path: str = ".env") -> None:
    """Lightweight .env loader without external deps.

    Populates os.environ for KEY=VALUE lines. Does not override existing vars.
    """
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
        # Fail-soft: CLI should still run even if .env parsing fails
        pass


@app.command()
def run(
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
        None,
        "--timeout",
        min=1.0,
        help="LLM request timeout in seconds (default: 300)",
    ),
):
    """Run the minimal agent against a disruption scenario."""
    # Load .env before importing the agent (agent reads env at import)
    load_dotenv()

    if model:
        os.environ["MODEL_NAME"] = model
    if base_url:
        os.environ["OLLAMA_API_BASE"] = base_url
    if timeout:
        os.environ["OLLAMA_TIMEOUT"] = str(timeout)

    # Import after environment is prepared so defaults are picked up
    from src.agent.mva import mva_run  # noqa: WPS433 (runtime import by design)

    try:
        result = mva_run(disruption)
        typer.echo(json.dumps(result, indent=2))
    except Exception as exc:
        # Provide a structured error for easier debugging
        typer.echo(
            json.dumps({"error": str(exc), "type": exc.__class__.__name__}, indent=2),
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
