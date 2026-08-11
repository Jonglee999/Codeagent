from codeagent.memory.audit import parse_memory_hits


def test_parse_memory_hits_returns_safe_metadata() -> None:
    hits = parse_memory_hits(
        '<relevant_memories><memory name="verified" type="session" '
        'scope="project" score="0.83">private body</memory></relevant_memories>',
        token_budget=800,
    )

    assert hits == [{
        "name": "verified",
        "type": "session",
        "scope": "project",
        "score": 0.83,
        "token_budget": 800,
        "estimated_tokens": 32,
    }]
    assert "private body" not in str(hits)


def test_parse_memory_hits_rejects_xml_entity_expansion() -> None:
    payload = (
        '<!DOCTYPE memories [<!ENTITY secret SYSTEM "file:///etc/passwd">]>'
        '<relevant_memories><memory name="&secret;" /></relevant_memories>'
    )

    assert parse_memory_hits(payload, token_budget=800) == []
