"""Assemble N benchmark variants from per-template concrete tasks + deltas.

Hard constraints (strict mode):
  * Every variant Vk contains exactly one ConcreteTask per template.
  * Every non-baseline ConcreteTask is used in at most one variant (strict
    exclusivity — no concrete task shared across variants).
  * If any template has fewer than N non-baseline concrete tasks, the build
    fails loudly with the offending task_id(s).

Soft objective:
  * Minimize the *range* (max - min) of the per-variant total lexical delta.
    Range minimization is linear, well-behaved for CBC, and gives the same
    "balanced variant" effect the user asked for.

The baseline concrete task (index 0 of a template) is never put into any
variant — it represents the original HumanEval problem and is the comparison
point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pulp

from humaneval_t.delta import DeltaVector
from humaneval_t.schema import ConcreteTask


@dataclass(frozen=True)
class TemplateGroup:
    """One template's full set of concrete tasks plus matching delta vectors.

    `concrete_tasks[0]` is the baseline (delta vector all zeros).
    """

    task_id: str
    concrete_tasks: list[ConcreteTask]
    deltas: list[DeltaVector]

    def non_baseline_count(self) -> int:
        return max(0, len(self.concrete_tasks) - 1)


@dataclass
class AssembledVariant:
    variant_id: str
    concrete_tasks: list[ConcreteTask]
    total_lexical: float
    total_semantic: float
    total_difficulty: float

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "totals": {
                "lexical": self.total_lexical,
                "semantic": self.total_semantic,
                "difficulty": self.total_difficulty,
            },
            "concrete_tasks": [
                {
                    "task_id": ct.task_id,
                    "concrete_task_id": ct.concrete_task_id,
                    "entry_point": ct.entry_point,
                    "prompt": ct.prompt,
                    "canonical_solution": ct.canonical_solution,
                    "test": ct.test,
                    "variable_values": ct.variable_values,
                }
                for ct in self.concrete_tasks
            ],
        }


def _check_capacity(groups: Iterable[TemplateGroup], n: int) -> None:
    short: list[tuple[str, int]] = []
    for g in groups:
        if g.non_baseline_count() < n:
            short.append((g.task_id, g.non_baseline_count()))
    if short:
        details = "; ".join(f"{tid} has {k} non-baseline (< {n})" for tid, k in short)
        raise ValueError(f"insufficient PICT capacity for N={n}: {details}")


def assemble_variants(
    groups: list[TemplateGroup],
    *,
    n: int = 5,
    time_limit_s: int = 60,
    seed: int = 42,
) -> list[AssembledVariant]:
    """Solve the balanced-delta assignment ILP and return N variants.

    `seed` is passed to CBC's `randomCbcSeed` option so tie-breaking between
    multiple optimal assignments is reproducible across runs and CBC versions.
    """
    _check_capacity(groups, n)

    prob = pulp.LpProblem("humaneval_t_assemble", pulp.LpMinimize)

    # x[t][i][v] = 1 iff non-baseline concrete task i of template t goes to variant v.
    x: dict[tuple[int, int, int], pulp.LpVariable] = {}
    for t_idx, group in enumerate(groups):
        for i in range(1, len(group.concrete_tasks)):  # skip baseline
            for v in range(n):
                x[(t_idx, i, v)] = pulp.LpVariable(
                    f"x_{t_idx}_{i}_{v}", lowBound=0, upBound=1, cat=pulp.LpBinary
                )

    # Each (template, variant) gets exactly one concrete task.
    for t_idx, group in enumerate(groups):
        for v in range(n):
            prob += (
                pulp.lpSum(x[(t_idx, i, v)] for i in range(1, len(group.concrete_tasks))) == 1,
                f"one_per_variant_t{t_idx}_v{v}",
            )

    # Strict exclusivity: each concrete task in at most one variant.
    for t_idx, group in enumerate(groups):
        for i in range(1, len(group.concrete_tasks)):
            prob += (
                pulp.lpSum(x[(t_idx, i, v)] for v in range(n)) <= 1,
                f"exclusive_t{t_idx}_i{i}",
            )

    # Per-variant total lexical delta.
    s_v = {
        v: pulp.lpSum(
            x[(t_idx, i, v)] * group.deltas[i].lexical
            for t_idx, group in enumerate(groups)
            for i in range(1, len(group.concrete_tasks))
        )
        for v in range(n)
    }

    z_max = pulp.LpVariable("z_max", lowBound=0)
    z_min = pulp.LpVariable("z_min", lowBound=0)
    for v in range(n):
        prob += z_max >= s_v[v]
        prob += z_min <= s_v[v]
    prob += z_max - z_min  # objective: minimize range

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=time_limit_s,
        options=["randomCbcSeed", str(seed)],
    )
    status = prob.solve(solver)
    if pulp.LpStatus[status] not in {"Optimal", "Not Solved"}:
        raise RuntimeError(f"assembler ILP did not solve cleanly: {pulp.LpStatus[status]}")

    # Extract assignment.
    assembled: list[AssembledVariant] = []
    for v in range(n):
        chosen: list[ConcreteTask] = []
        totals = {"lexical": 0.0, "semantic": 0.0, "difficulty": 0.0}
        for t_idx, group in enumerate(groups):
            for i in range(1, len(group.concrete_tasks)):
                if pulp.value(x[(t_idx, i, v)]) > 0.5:
                    chosen.append(group.concrete_tasks[i])
                    totals["lexical"] += group.deltas[i].lexical
                    totals["semantic"] += group.deltas[i].semantic
                    totals["difficulty"] += group.deltas[i].difficulty
                    break
        assembled.append(
            AssembledVariant(
                variant_id=f"V{v + 1}",
                concrete_tasks=chosen,
                total_lexical=totals["lexical"],
                total_semantic=totals["semantic"],
                total_difficulty=totals["difficulty"],
            )
        )
    return assembled
