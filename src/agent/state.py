"""Agent state for multi-step ReAct planning.

Provides a minimal but safe container to track goal, scratchpad steps
(Thought/Action/Observation), incremental data, and solved status.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Holds the evolving state across a multi-step interaction."""

    goal: str
    scratchpad: List[Dict[str, Any]] = Field(default_factory=list)
    collected_data: Dict[str, Any] = Field(default_factory=dict)
    solved: bool = False

    # Convenience helpers for ReAct-style traces
    def add_thought(self, text: str) -> None:
        self.scratchpad.append({"thought": text})

    def add_action(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> None:
        self.scratchpad.append({
            "action": {
                "tool": name,
                "arguments": arguments or {},
            }
        })

    def add_observation(self, observation: Any) -> None:
        self.scratchpad.append({"observation": observation})

    def mark_solved(self) -> None:
        self.solved = True

    def remember(self, key: str, value: Any) -> None:
        """Accumulate structured data useful for later steps or summary."""
        self.collected_data[key] = value

