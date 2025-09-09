import json
import os
import re
from typing import Dict, Any

import httpx

from src.tools import logistics
import inspect


# Base URL and model can be overridden via environment variables
BASE_URL = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")
TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT", "300"))  # allow long cold-starts


TOOLS = {
    "check_traffic": logistics.check_traffic,
    "get_merchant_status": logistics.get_merchant_status,
    "contact_recipient_via_chat": logistics.contact_recipient_via_chat,
    "suggest_safe_drop_off": logistics.suggest_safe_drop_off,
    "find_nearby_locker": logistics.find_nearby_locker,
}


SYSTEM = (
    "You are a helpful last-mile logistics agent. "
    "Given a disruption, choose ONE best tool and valid JSON args. "
    "Only choose from: " + ", ".join(TOOLS.keys())
)


def _extract_json(text: str) -> Dict:
    """Attempt to parse JSON; if that fails, extract the first JSON object.

    Helps recover when the model adds prose around a JSON block.
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Model did not return valid JSON: {text!r}")


def _ollama_native_chat_json(messages, temperature: float) -> Dict:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
        "stream": True,
    }
    base = BASE_URL.replace("/v1", "")
    for attempt in (1, 2):
        try:
            with httpx.Client(base_url=base, timeout=httpx.Timeout(TIMEOUT_S)) as client:
                with client.stream("POST", "/api/chat", json=payload) as r:
                    r.raise_for_status()
                    content = ""
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue
                        msg = data.get("message", {})
                        if isinstance(msg, dict) and msg.get("content"):
                            content += msg["content"]
                        if data.get("done") is True:
                            break
                return _extract_json(content)
        except httpx.ReadTimeout:
            if attempt == 1:
                continue
            raise


def llm_json(prompt: str) -> Dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        # Ollama's OpenAI-compatible endpoint supports standard fields.
        # Keep response_format for JSON coercion where available.
        "response_format": {"type": "json_object"},
    }
    try:
        with httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(TIMEOUT_S)) as client:
            r = client.post("/chat/completions", json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return _extract_json(content)
    except httpx.ReadTimeout:
        # fallback to native API
        return _ollama_native_chat_json(payload["messages"], temperature=0.2)


def _filter_args_for_tool(fn, args: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    allowed = {name for name, p in sig.parameters.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    return {k: v for k, v in (args or {}).items() if k in allowed}


def decide_tool(disruption: str) -> Dict:
    ask = (
        f"Disruption: {disruption}\n"
        "Return JSON: {\"tool_name\": <one_of_tools>, \"arguments\": { ... }}\n"
        f"Tools: {list(TOOLS.keys())}"
    )
    plan = llm_json(ask)

    # Minimal validation
    if "tool_name" not in plan:
        raise KeyError("Model response missing 'tool_name'")
    if plan["tool_name"] not in TOOLS:
        raise ValueError(f"Unknown tool selected: {plan['tool_name']}")
    if "arguments" in plan and not isinstance(plan["arguments"], dict):
        raise TypeError("'arguments' must be an object if provided")
    return plan


def mva_run(disruption: str) -> Dict:
    plan = decide_tool(disruption)
    tool = TOOLS[plan["tool_name"]]
    raw_args = plan.get("arguments", {})
    obs = tool(**_filter_args_for_tool(tool, raw_args))
    return {"decision": plan, "observation": obs}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.agent.mva \"<disruption text>\"")
        sys.exit(2)
    result = mva_run(sys.argv[1])
    print(json.dumps(result, indent=2))
