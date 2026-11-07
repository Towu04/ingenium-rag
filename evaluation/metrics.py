def citation_precision(expected: set[str], actual: set[str]) -> float:
    if not actual:
        return 0.0
    return len(expected & actual) / len(actual)
