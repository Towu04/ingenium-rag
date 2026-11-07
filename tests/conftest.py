import pytest


@pytest.fixture
def sample_markdown() -> str:
    return "# Project\nThe building uses reinforced concrete." * 20
