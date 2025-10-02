from digest.compute_digest import _parse_authors


def test_parse_authors_various_shapes():
    assert _parse_authors(None) == []
    assert _parse_authors({}) == []
    assert _parse_authors({"authors": "A, B, C"}) == ["A", "B", "C"]
    assert _parse_authors({"authors": ["A", "B", "C"]}) == ["A", "B", "C"]
    assert _parse_authors({"authors": [{"name": "A"}, {"name": "B"}]}) == ["A", "B"]
