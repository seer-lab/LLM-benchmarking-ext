"""Materialize a Template + PICT rows into ConcreteTask objects.

Substitution is pure string replacement of `<variable_name>` -> chosen value.
There is no LLM or other generation step — every byte of every concrete task
is either inherited from the original HumanEval problem or copied from a
value the template author wrote down.

We always prepend the *baseline* concrete task (every template variable at its
first/original value) as concrete_task #000, so delta metrics have a fixed
reference and so that the original HumanEval problem is itself a member of the
benchmark.
"""

from __future__ import annotations

import re

from humaneval_t.dataset import get_problem
from humaneval_t.schema import ConcreteTask, Template


_PLACEHOLDER_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")


def _substitute(text: str, bindings: dict[str, str]) -> str:
    """Replace every `<name>` with bindings[name].

    Unknown placeholders (e.g. literal `<numerator>` text that appears in the
    original HumanEval docstring as part of a format spec) are left untouched.
    The schema validator already enforces that every *declared* template
    variable is referenced somewhere; the reverse direction (every `<...>`
    matches a variable) would forbid problems whose docstrings legitimately
    contain `<word>` text such as HumanEval/144 ("<numerator>/<denominator>").
    """

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in bindings:
            return bindings[name]
        return m.group(0)  # leave the original `<name>` text alone

    return _PLACEHOLDER_RE.sub(repl, text)


def _build_templated_test(template: Template, row: dict[str, str]) -> str:
    """Build a `check(candidate)` source string by substituting each assertion line."""
    lines = ["def check(candidate):"]
    for raw_line in template.test_strategy.assertions or []:
        substituted = _substitute(raw_line.strip(), row)
        lines.append(f"    {substituted}")
    return "\n".join(lines) + "\n"


def materialize(template: Template, row: dict[str, str], *, row_index: int) -> ConcreteTask:
    """Apply one PICT row to a Template and produce a ConcreteTask."""
    missing = {v.name for v in template.variables} - set(row.keys())
    if missing:
        raise ValueError(f"PICT row missing template variables: {missing}")

    prompt = _substitute(template.templated_prompt, row)
    solution_body = _substitute(template.templated_solution, row)
    full_solution = prompt + solution_body

    original = get_problem(template.task_id)
    new_entry_point = row["fn"] if "fn" in row else original["entry_point"]

    if template.test_strategy.mode == "templated":
        test_src = _build_templated_test(template, row)
    else:
        # keep_original: reuse the HumanEval test verbatim. The harness imports
        # the renamed function via the concrete task's entry_point.
        test_src = original["test"]

    return ConcreteTask(
        task_id=template.task_id,
        concrete_task_id=f"{template.task_id}#{row_index:03d}",
        entry_point=new_entry_point,
        prompt=prompt,
        canonical_solution=full_solution,
        test=test_src,
        variable_values=dict(row),
    )


def materialize_all(template: Template, rows: list[dict[str, str]]) -> list[ConcreteTask]:
    """Force-include the baseline row at index 0, then any PICT rows that follow.

    If a PICT row happens to equal the baseline (every variable at its first
    value), we don't duplicate it — we drop the dupe and keep the baseline-first
    ordering.
    """
    baseline_row = template.original_values()
    baseline = materialize(template, baseline_row, row_index=0)

    concrete_tasks: list[ConcreteTask] = [baseline]
    seen_signatures: set[tuple[tuple[str, str], ...]] = {
        tuple(sorted(baseline_row.items()))
    }

    for r in rows:
        sig = tuple(sorted(r.items()))
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        concrete_tasks.append(materialize(template, r, row_index=len(concrete_tasks)))

    return concrete_tasks


def baseline_task(template: Template) -> ConcreteTask:
    """Convenience: just the baseline (all-original-values) concrete task."""
    return materialize(template, template.original_values(), row_index=0)
