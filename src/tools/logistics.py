"""Simulated logistics tools for Phase 1.1.

These deterministic stubs let the agent exercise planning and tool-use logic
without relying on real external services. You can swap them with real API
integrations later while preserving the function contracts.
"""

from __future__ import annotations

from typing import Dict


def check_traffic(route_id: str) -> Dict:
    """Simulate checking live traffic for a route.

    Args:
        route_id: Identifier of the route/segment being evaluated.

    Returns:
        Dict with overall status, the original route_id, and an observation.
        Observation values are simple deterministics for now (e.g., "clear").
    """
    return {"status": "ok", "route_id": route_id, "observation": "clear"}


def get_merchant_status(merchant_id: str) -> Dict:
    """Simulate querying merchant prep time and status.

    Args:
        merchant_id: Merchant identifier.

    Returns:
        Dict with merchant_id, prep_minutes and coarse availability signals.
    """
    return {
        "merchant_id": merchant_id,
        "prep_minutes": 40,
        "status": "open",
        "capacity": "high_wait",
    }


def contact_recipient_via_chat(order_id: str) -> Dict:
    """Simulate contacting the recipient to request delivery instructions.

    Args:
        order_id: Order identifier.

    Returns:
        Dict indicating whether instructions were obtained.
    """
    return {
        "order_id": order_id,
        "delivered_instructions": True,
        "message": "Please leave at concierge; I'll pick up in 10 min.",
    }


def suggest_safe_drop_off(address: str) -> Dict:
    """Suggest a safe drop-off option for an address.

    Args:
        address: The delivery address string.

    Returns:
        Dict with a suggested safe drop-off location.
    """
    return {
        "address": address,
        "suggestion": "Leave with building concierge",
        "risk": "low",
    }


def find_nearby_locker(address: str) -> Dict:
    """Suggest a nearby locker for alternative delivery.

    Args:
        address: The reference address from which to search.

    Returns:
        Dict with locker identifier and a short descriptor.
    """
    return {
        "address": address,
        "locker": "Locker-19 @ 2nd St",
        "distance_m": 180,
    }


__all__ = [
    "check_traffic",
    "get_merchant_status",
    "contact_recipient_via_chat",
    "suggest_safe_drop_off",
    "find_nearby_locker",
]

