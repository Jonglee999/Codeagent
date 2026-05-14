"""Helper utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_string(value: str) -> str:
    """Create a SHA-256 hash of a string."""
    return hashlib.sha256(value.encode()).hexdigest()


def to_json(data: Any) -> str:
    """Convert data to JSON string."""
    return json.dumps(data, indent=2, default=str)
