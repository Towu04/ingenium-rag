"""Retrieval and citation evaluation metrics."""


def citation_precision(expected: set[str], actual: set[str]) -> float:
    """Fraction of cited sources that appear in the expected ground-truth source set."""
    if not actual:
        return 0.0
    return len(expected & actual) / len(actual)


def source_recall(expected: set[str], actual: set[str]) -> float:
    """Fraction of expected ground-truth sources surfaced in the retrieved citations."""
    if not expected:
        return 0.0
    return len(expected & actual) / len(expected)
