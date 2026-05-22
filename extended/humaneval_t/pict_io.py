"""Thin wrapper around the PICT 3.3 binary.

Two responsibilities:
1. Render a Template into PICT's text model format (template variables + constraints).
2. Invoke pict.exe and parse its tab-separated output into a list of dict rows,
   one per concrete combination chosen by PICT's pairwise (or higher-order) algo.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from humaneval_t.schema import Template


# Resolution order for pict.exe (first hit wins):
#   1. HUMANEVAL_T_PICT environment variable
#   2. ./tools/pict/pict.exe bundled with the repo (works after a fresh clone)
#   3. `pict.exe` or `pict` on the system PATH
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # extended/
_BUNDLED_PICT = _PACKAGE_ROOT / "tools" / "pict" / "pict.exe"


def pict_exe() -> Path:
    """Return the path to pict.exe. Raises FileNotFoundError if none reachable."""
    override = os.environ.get("HUMANEVAL_T_PICT")
    if override:
        p = Path(override)
        if p.exists():
            return p
    if _BUNDLED_PICT.exists():
        return _BUNDLED_PICT
    on_path = shutil.which("pict.exe") or shutil.which("pict")
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        "pict.exe not found. Looked in: HUMANEVAL_T_PICT env var, "
        f"{_BUNDLED_PICT}, and system PATH."
    )


def template_to_pict_model(template: Template) -> str:
    """Render a Template into the text PICT consumes (one line per template variable)."""
    lines: list[str] = []
    for variable in template.variables:
        lines.append(f"{variable.name}: {', '.join(variable.values)}")
    if template.constraints:
        lines.append("")
        lines.extend(c.raw for c in template.constraints)
    return "\n".join(lines) + "\n"


def parse_pict_output(text: str) -> list[dict[str, str]]:
    """Parse PICT's TSV stdout. First line is the header with template variable names."""
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    return [dict(row) for row in reader]


def run_pict(template: Template, *, order: int = 2) -> list[dict[str, str]]:
    """Materialize a temp model file, run pict.exe, return parsed rows."""
    exe = pict_exe()

    model_text = template_to_pict_model(template)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pict", delete=False, encoding="utf-8"
    ) as f:
        f.write(model_text)
        model_path = f.name
    try:
        proc = subprocess.run(
            [str(exe), model_path, f"/o:{order}"],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        Path(model_path).unlink(missing_ok=True)
    return parse_pict_output(proc.stdout)
