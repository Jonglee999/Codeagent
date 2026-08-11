"""Safe metadata extraction for auditable memory recall events."""

from __future__ import annotations

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException


def parse_memory_hits(memory_xml: str, token_budget: int) -> list[dict[str, object]]:
    if not memory_xml:
        return []
    try:
        root = ElementTree.fromstring(memory_xml)
    except (ElementTree.ParseError, DefusedXmlException):
        return []
    estimated_tokens = min(token_budget, max(1, len(memory_xml) // 4))
    hits = []
    for element in root.findall("memory"):
        hits.append({
            "name": element.attrib.get("name", "unknown"),
            "type": element.attrib.get("type", "unknown"),
            "scope": element.attrib.get("scope", "unknown"),
            "score": float(element.attrib.get("score", "0") or 0),
            "token_budget": token_budget,
            "estimated_tokens": estimated_tokens,
        })
    return hits
