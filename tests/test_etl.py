from processing.metadata import extract_metadata


def test_metadata_rules():
    metadata = extract_metadata(
        "/Blok_Noord/Project-ABC/Building-B1/Phase-2/spec.md",
        {"project": [{"pattern": r"Project-([A-Z]+)"}],
         "building": [{"pattern": r"Building-([A-Z0-9]+)"}],
         "phase": [{"pattern": r"Phase-([A-Z0-9]+)"}]},
    )
    assert (metadata.project, metadata.building, metadata.phase) == ("ABC", "B1", "2")
