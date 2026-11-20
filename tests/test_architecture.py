"""Automated AST architectural guard enforcing strict Pipes-and-Filters stage boundaries."""

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

ISOLATED_STAGES = frozenset(
    {"etl", "chunking", "embedding", "storage", "retrieval", "generation", "evaluation"}
)
INTERNAL_ROOTS = ISOLATED_STAGES | {"domain", "config", "pipeline", "api"}


def _imported_internal_roots(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in INTERNAL_ROOTS:
                roots.add(top)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in INTERNAL_ROOTS:
                    roots.add(top)
    return roots


def test_domain_and_config_have_zero_internal_dependencies() -> None:
    """`domain.py` and `config.py` are root modules and must never import any stage."""
    assert _imported_internal_roots(SRC_ROOT / "domain.py") == set()
    assert _imported_internal_roots(SRC_ROOT / "config.py") == set()


def test_stages_have_zero_horizontal_cross_stage_imports() -> None:
    """No stage (`etl`, `chunking`, `embedding`, `storage`, `retrieval`, `generation`, `evaluation`)
    may import another stage, `pipeline`, or `api`."""
    violations: list[str] = []
    for stage in sorted(ISOLATED_STAGES):
        stage_dir = SRC_ROOT / stage
        assert stage_dir.is_dir(), f"Expected stage directory missing: {stage_dir}"
        allowed = {stage, "domain", "config"}
        for py_file in sorted(stage_dir.rglob("*.py")):
            imported = _imported_internal_roots(py_file)
            forbidden = imported - allowed
            if forbidden:
                violations.append(
                    f"{py_file.relative_to(SRC_ROOT)} imports forbidden module(s): {sorted(forbidden)}"
                )
    assert not violations, "Cross-stage boundary violations:\n" + "\n".join(violations)


def test_no_print_calls_inside_src_library_code() -> None:
    """Library modules under `src/` must use `logging`, never ad-hoc `print()`."""
    offenders: list[str] = []
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{py_file.relative_to(SRC_ROOT)}:{node.lineno}")
    assert not offenders, "Found print() in src/ library code:\n" + "\n".join(offenders)
