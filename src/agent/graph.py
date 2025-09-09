import json
import os
import re
import inspect
from typing import Any, Dict, List

import httpx
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.tools import logistics
from src.mem.ltm import recall, remember


# Config via environment with sensible defaults
BASE_URL = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")
TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT", "300"))
MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "256"))
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "3"))
PROGRESS = os.getenv("AGENT_PROGRESS", "1") not in ("0", "false", "False", "no", "")
OFFLINE = os.getenv("AGENT_OFFLINE", "0") not in ("0", "false", "False", "no", "")


def _p(msg: str) -> None:
    if PROGRESS:
        print(msg, flush=True)


TOOLS = {
    "check_traffic": logistics.check_traffic,
    "get_merchant_status": logistics.get_merchant_status,
    "contact_recipient_via_chat": logistics.contact_recipient_via_chat,
    "suggest_safe_drop_off": logistics.suggest_safe_drop_off,
    "find_nearby_locker": logistics.find_nearby_locker,
}


def _tool_signature_str(name: str, fn) -> str:
    sig = inspect.signature(fn)
    params = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        annot = "str" if p.annotation is inspect._empty else getattr(p.annotation, "__name__", str(p.annotation))
        if p.default is inspect._empty:
            params.append(f"{p.name}: {annot}")
        else:
            params.append(f"{p.name}: {annot} = {p.default!r}")
    return f"{name}({', '.join(params)})"


def _filter_args_for_tool(fn, args: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    allowed = {name for name, p in sig.parameters.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    return {k: v for k, v in (args or {}).items() if k in allowed}


def _missing_required_args(fn, args: Dict[str, Any]) -> List[str]:
    sig = inspect.signature(fn)
    provided = set((args or {}).keys())
    missing = []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect._empty and name not in provided:
            missing.append(name)
    return missing


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Model did not return valid JSON: {text!r}")


def _ollama_native_chat_json(messages: List[Dict[str, str]], temperature: float) -> Dict[str, Any]:
    """Fallback to Ollama native /api/chat with JSON formatting.

    Uses `format: "json"` to coerce a valid JSON object response.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": MAX_TOKENS,
        },
        "stream": True,
    }
    base = BASE_URL.replace("/v1", "")
    # One retry on read timeout
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
                _p("[llm] native /api/chat stream timed out, retrying once...")
                continue
            raise


def _chat_json(
    messages: List[Dict[str, str]],
    response_format: Dict[str, Any] | None = None,
    temperature: float = 0.2,
):
    """Chat helper that directly uses Ollama's native /api/chat with JSON formatting.

    We avoid the OpenAI-compatible shim to prevent timeouts and ensure
    deterministic JSON decoding via `format: "json"`.
    """
    # Note: response_format is accepted for API compatibility, but we always
    # request JSON via native API and return a parsed object.
    # Offline fast-path for tests/smoke: avoid any network calls.
    if OFFLINE:
        try:
            user_content = (messages or [])[-1].get("content", "")
        except Exception:
            user_content = ""
        # If it's a reflection-style prompt, return a stop decision.
        if "\"stop\"" in user_content or "repair_action" in user_content:
            return {"stop": True, "why": "offline"}
        # Otherwise, produce a simple, valid plan using a known tool.
        return {
            "thought": "offline plan",
            "tool_name": "suggest_safe_drop_off" if "suggest_safe_drop_off" in TOOLS else list(TOOLS.keys())[0],
            "arguments": {"address": "test"},
        }
    return _ollama_native_chat_json(messages, temperature)


def plan_node(state: AgentState) -> AgentState:
    step_idx = int(state.collected_data.get("steps", 0)) + 1
    # Retrieve relevant memories to guide planning (fail-soft)
    try:
        tips = recall(state.goal)
    except Exception:
        tips = []

    tool_sigs = ", ".join(_tool_signature_str(n, f) for n, f in TOOLS.items())
    msg = [
        {
            "role": "system",
            "content": (
                "You are a last-mile logistics planner.\n"
                "PRIORITIES (in order): (1) food integrity (hot items), (2) minimize cascading driver delays, (3) customer communication & consent, (4) safety/compliance.\n"
                f"TOOLS: {', '.join(list(TOOLS.keys()))}.\n"
                f"TOOL SIGNATURES: {tool_sigs}.\n"
                "ALGORITHM: First confirm merchant delay and route status; then try contacting recipient for instructions; if unreachable, propose safe-drop if policy allows; else propose nearby locker; prefer actions that reduce downstream delays when driver has stacked deliveries.\n"
                "Output strictly JSON: {\"thought\":\"...\", \"tool_name\":\"...\", \"arguments\":{...}}.\n"
                "Only use argument keys exactly as in the signatures. Do not invent keys."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Goal: {state.goal}\n"
                f"Relevant past learnings: {json.dumps(tips)}\n"
                f"Prior steps: {json.dumps(state.scratchpad)}\n"
                'Return JSON only.'
            ),
        },
    ]
    plan = _chat_json(msg, response_format={"type": "json_object"})

    # Optional validator: ensure core factors are considered across plan + history
    try:
        must_consider = ["traffic", "merchant", "recipient"]
        history_txt = json.dumps(state.scratchpad).lower()
        plan_txt = json.dumps(plan).lower()
        if any(k not in (history_txt + plan_txt) for k in must_consider):
            nudge = [
                {
                    "role": "system",
                    "content": (
                        "You skipped one of: traffic, merchant, recipient. Revise the next single action based on the stated PRIORITIES and ALGORITHM."
                    ),
                },
                msg[1],
            ]
            plan = _chat_json(nudge, response_format={"type": "json_object"})
    except Exception:
        # Fail-soft if validation or nudging fails
        pass

    # Minimal validation and state update
    tool_name = plan.get("tool_name")
    if tool_name not in TOOLS:
        raise ValueError(f"Planner selected unknown tool: {tool_name}")
    arguments = plan.get("arguments") or {}
    thought = str(plan.get("thought", "")).strip()
    state.add_thought(thought)
    # Support both keys so act_node can pick either
    state.scratchpad.append({
        "action": {
            "tool": tool_name,
            "arguments": arguments,
            "args": arguments,
        }
    })
    _p(f"[plan #{step_idx}] tool={tool_name} args={json.dumps(arguments)} thought={thought}")
    return state


def act_node(state: AgentState) -> AgentState:
    last = state.scratchpad[-1]
    action = last.get("action") or {}
    tool_name = (action.get("tool") or "").strip()
    tool = TOOLS[tool_name]
    raw_args = action.get("arguments") or action.get("args") or {}
    filtered_args = _filter_args_for_tool(tool, raw_args)
    ignored = sorted(set(raw_args.keys()) - set(filtered_args.keys()))
    missing = _missing_required_args(tool, filtered_args)
    step_idx = int(state.collected_data.get("steps", 0)) + 1
    if missing:
        obs = {
            "error": "missing_required_args",
            "tool": tool_name,
            "missing": missing,
            "provided_keys": list((raw_args or {}).keys()),
            "hint": f"Required: {', '.join(missing)}",
        }
    else:
        obs = tool(**filtered_args)
        if ignored:
            # annotate note about ignored arguments to guide the planner
            note = {"ignored_args": ignored}
            if isinstance(obs, dict):
                obs = {**obs, **note}
            else:
                obs = {"result": obs, **note}
    state.add_observation(obs)
    if isinstance(obs, dict):
        state.collected_data.update(obs)
    else:
        state.collected_data["last_observation"] = obs
    # Increment step counter
    state.collected_data["steps"] = int(state.collected_data.get("steps", 0)) + 1
    try:
        preview = json.dumps(obs)
    except Exception:
        preview = str(obs)
    if len(preview) > 200:
        preview = preview[:200] + "..."
    _p(f"[act  #{step_idx}] tool={tool_name} observation={preview}")
    # Clear any pending repair flag after executing an action
    state.collected_data.pop("pending_repair", None)
    return state


def reflect_node(state: AgentState) -> AgentState:
    """Critique the last step and optionally inject a repair action.

    If the model proposes a repair_action, queue it and route back to `act`.
    Otherwise, decide to stop or continue planning.
    """
    steps = int(state.collected_data.get("steps", 0))
    msg = [
        {"role": "system", "content": "Critique last step; repair if needed."},
        {
            "role": "user",
            "content": (
                f"Goal: {state.goal}\n"
                f"History: {json.dumps(state.scratchpad)}\n"
                'Return JSON: {"stop": true|false, "why":"...", '
                '"repair_action": {"tool_name":"...","arguments":{}} | null}'
            ),
        },
    ]

    if steps >= MAX_STEPS:
        decision = {"stop": True, "why": f"Reached max steps ({MAX_STEPS})", "repair_action": None}
    else:
        decision = _chat_json(msg, response_format={"type": "json_object"})

    # Attach reflection to the last executed step for traceability
    try:
        state.scratchpad[-1]["reflection"] = decision
    except Exception:
        pass

    repair = decision.get("repair_action")
    if repair:
        tool_name = (repair.get("tool_name") or "").strip()
        arguments = repair.get("arguments") or {}
        # Queue a repair action and signal routing to `act`
        state.solved = False
        state.collected_data["pending_repair"] = True
        state.scratchpad.append({
            "thought": "repair",
            "action": {"tool": tool_name, "args": arguments, "arguments": arguments},
        })
        _p(f"[reflect] repair -> tool={tool_name} args={json.dumps(arguments)} why={decision.get('why','')}")
        return state

    state.solved = bool(decision.get("stop", False))
    _p(f"[reflect] stop={state.solved} why={decision.get('why','')}")
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("reflect", reflect_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "reflect")
    def _route_after_reflect(s: AgentState):
        if s.solved:
            return "END"
        if s.collected_data.get("pending_repair"):
            return "act"
        return "plan"

    g.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"END": END, "plan": "plan", "act": "act"},
    )
    return g.compile()
