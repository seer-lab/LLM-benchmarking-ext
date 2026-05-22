"""Pydantic models for HumanEval_T templates and concrete tasks.

Vocabulary (matches the paper, arXiv 2412.01526):

- **template task / Template** — the recipe for one HumanEval problem, listing
  the parameterizable variables and their allowed values.
- **template variable** — one parameter slot in a template, with a list of
  alternative values. The first value is the original/baseline.
- **concrete task** — a fully materialized instance of a template, produced by
  filling each template variable with one chosen value (a PICT row).
- **benchmark variant (V1, V2, ...)** — one assembled benchmark; a set of
  concrete tasks, one per template, with strict exclusivity across variants.

No LLM ever touches anything in this file or its consumers. All variation is
declarative and deterministic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


VariableKind = Literal["identifier", "lexical"]
# identifier: substitution renames a Python symbol (function name, arg name).
#             Every value must be a valid Python identifier.
# lexical:    substitution puts any string into the surrounding text. Values
#             are free-form (single chars like "(" or whole phrases like
#             "data points" are both fine). May appear in any of the three
#             template surfaces: templated_prompt, templated_solution, or
#             test_strategy.assertions.


REQUIRED_TEST_COUNT = 5
# When test_strategy.mode == "templated", a template MUST carry exactly this
# many assertions. Enforced at schema validation so half-written templates
# can't slip into the build. ("Each problem must have 5 test cases at all
# costs" — locked 2026-05-20.)


class TemplateVariable(BaseModel):
    """One parameterizable placeholder with its allowed values.

    The first entry of `values` is the *original* HumanEval value, so delta
    metrics can treat it as the canonical baseline.
    """

    name: str
    kind: VariableKind
    values: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_identifier_values(self) -> "TemplateVariable":
        if self.kind == "identifier":
            for v in self.values:
                if not v.isidentifier():
                    raise ValueError(
                        f"template variable {self.name!r}: value {v!r} is not a valid Python identifier"
                    )
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"template variable {self.name!r}: values contain duplicates")
        return self


class Constraint(BaseModel):
    """A PICT constraint string, passed through verbatim.

    Example: `IF [input_type] = "measurements" THEN [value_descriptor] = "data points";`
    """

    raw: str


class TestStrategy(BaseModel):
    """How per-concrete-task assertions are produced.

    Two modes:

    * `keep_original` — the materializer reuses HumanEval's original test()
      verbatim. The harness imports the renamed function via `entry_point`,
      so the same assertions still apply. Use this when only the surface
      wording / identifiers change and the underlying test inputs don't need
      to vary. No `assertions` field required.

    * `templated` — the template author writes exactly `REQUIRED_TEST_COUNT`
      `assert candidate(...) == ...` lines, optionally with `<variable>`
      placeholders. At materialization the placeholders are substituted from
      the PICT row, and the resulting lines are wrapped in a fresh
      `def check(candidate):`. Use this when the test inputs/outputs must
      vary with the chosen variable values (e.g. a bracket-matching problem
      where `<open>` / `<close>` differ across concrete tasks).
    """

    mode: Literal["keep_original", "templated"] = "keep_original"
    assertions: list[str] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "TestStrategy":
        if self.mode == "templated":
            if not self.assertions:
                raise ValueError(
                    f"test_strategy: 'templated' mode requires an 'assertions' list of "
                    f"exactly {REQUIRED_TEST_COUNT} entries."
                )
            if len(self.assertions) != REQUIRED_TEST_COUNT:
                raise ValueError(
                    f"test_strategy: 'templated' mode requires exactly {REQUIRED_TEST_COUNT} "
                    f"assertions, got {len(self.assertions)}."
                )
            for i, line in enumerate(self.assertions):
                stripped = line.strip()
                if not stripped.startswith("assert "):
                    raise ValueError(
                        f"test_strategy.assertions[{i}]: each line must start with "
                        f"'assert ' (got: {stripped[:40]!r})."
                    )
        elif self.assertions:
            raise ValueError(
                "test_strategy: 'assertions' is only valid when mode='templated'. "
                "Remove the assertions or switch mode to 'templated'."
            )
        return self


class Template(BaseModel):
    """A HumanEval_T template task for one HumanEval problem."""

    task_id: str
    entry_point: str  # the original HumanEval function name (for traceability)
    notes: str = ""

    # Whether this problem was part of the original workshop paper's 10-problem
    # subset (arXiv 2412.01526). Pure documentation — has no effect on
    # generation, just lets the UI badge it for context.
    paper_subset: bool = False

    # The original prompt with <variable_name> placeholders inserted.
    templated_prompt: str

    # The original canonical solution with <variable_name> placeholders. Both
    # identifier and lexical variables may appear here — the materializer is
    # a pure string substitution.
    templated_solution: str

    variables: list[TemplateVariable]
    constraints: list[Constraint] = []
    test_strategy: TestStrategy = Field(default_factory=TestStrategy)

    @model_validator(mode="after")
    def _validate(self) -> "Template":
        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("duplicate template variable names")

        assertions_text = "\n".join(self.test_strategy.assertions or [])
        for v in self.variables:
            ph = f"<{v.name}>"
            referenced = (
                ph in self.templated_prompt
                or ph in self.templated_solution
                or ph in assertions_text
            )
            if not referenced:
                raise ValueError(
                    f"template variable {v.name!r} declared but never referenced as {ph}"
                )
        return self

    def original_values(self) -> dict[str, str]:
        """Return the canonical (first) value for each template variable — the HumanEval baseline."""
        return {v.name: v.values[0] for v in self.variables}


class ConcreteTask(BaseModel):
    """A single materialized concrete task produced by filling a Template from a PICT row."""

    task_id: str
    concrete_task_id: str  # f"{task_id}#{row_index:03d}"
    entry_point: str  # post-substitution; the harness imports this name
    prompt: str
    canonical_solution: str
    test: str
    variable_values: dict[str, str]  # the PICT row that produced this concrete task
