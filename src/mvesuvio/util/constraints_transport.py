from __future__ import annotations

import ast
import uuid

import dill

_REGISTRY_PREFIX = "registry:"
_constraint_registry: dict[str, object] = {}


def serialize_constraints(constraints: object) -> str:
    """Store constraints in-process and return a token for Mantid algorithm properties."""
    token = uuid.uuid4().hex
    _constraint_registry[token] = constraints
    return f"{_REGISTRY_PREFIX}{token}"


def deserialize_constraints(payload: str) -> object:
    """Resolve constraints token, with backward compatibility for legacy dill payloads."""
    if payload.startswith(_REGISTRY_PREFIX):
        token = payload[len(_REGISTRY_PREFIX) :]
        if token not in _constraint_registry:
            raise RuntimeError(f"Constraints token not found in registry: {token}")
        return _constraint_registry[token]

    # Backward compatibility for legacy payloads produced via str(dill.dumps(...)).
    return dill.loads(ast.literal_eval(payload))
