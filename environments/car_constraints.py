"""
Car invariance lock.

Guarantees that the LLM lighting/reflection step never touches the car's
own geometry, proportions, paint, or wheels. The 3D car asset is treated
as an immutable, opaque object: the pipeline only ever mutates the
*environment* (background, lighting, reflections cast ONTO the car's
existing, unchanged surfaces).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import config


@dataclass(frozen=True)
class CarLockProfile:
    """Immutable descriptor of the car asset. Values here are metadata
    used for prompt context and audit logging only — they are never sent
    to the LLM as something it is allowed to modify.
    """

    model_id: str
    geometry_hash: str
    proportions: Dict[str, float] = field(default_factory=dict)
    paint_code: str = "unspecified"
    wheel_spec: str = "unspecified"

    def as_locked_context(self) -> Dict[str, Any]:
        """A read-only summary safe to reference in outputs, tagged LOCKED."""
        return {
            "car_model_id": self.model_id,
            "geometry_hash": self.geometry_hash,
            "paint_code": self.paint_code,
            "wheel_spec": self.wheel_spec,
            "status": "LOCKED - DO NOT MODIFY",
        }


def build_invariance_clause() -> str:
    """Returns the hard-constraint text injected into every Groq system
    prompt, instructing the model it may only reason about environment
    and reflection parameters.
    """
    locked = ", ".join(config.CAR_LOCK_FIELDS)
    return (
        "HARD CONSTRAINT: The 3D car model's geometry, proportions, paint, "
        f"and wheels ({locked}) are LOCKED and must remain completely "
        "unchanged. You are NOT permitted to propose, imply, or output any "
        "modification to these fields. Only reason about environment "
        "lighting direction, intensity, color temperature, and reflection "
        "parameters that will be cast onto the car's existing, fixed surfaces."
    )


def validate_no_car_mutation(llm_output: Dict[str, Any]) -> None:
    """Defensive check: raise if the LLM ever slips a locked field into its
    JSON output despite the system-prompt constraint.
    """
    violations = [f for f in config.CAR_LOCK_FIELDS if f in llm_output]
    if violations:
        raise ValueError(
            f"Car invariance violated — LLM attempted to mutate locked "
            f"fields: {violations}. Rejecting output."
        )