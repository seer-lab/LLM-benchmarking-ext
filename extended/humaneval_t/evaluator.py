"""Evaluate a ConcreteTask by running its tests against a candidate solution.

Two solver modes:

* `oracle`: use the concrete task's `canonical_solution` directly. Proves the
  test harness wires together correctly. No external dependencies. Use this in
  smoke checks and CI.

* `llm`: send the concrete task's prompt to an LLM provider and run the
  returned code. Wires to openai / anthropic / ollama in the existing v1
  layout. (Implementation deferred to a `solvers.py` module and only imported
  lazily so the core evaluator stays pure for tests.)

The harness mirrors HumanEval's classic pattern: exec the candidate solution
into a fresh namespace, exec the `test` source so a `check(candidate)` function
is defined, then call `check(candidate)` with the function looked up by the
concrete task's `entry_point`.
"""

from __future__ import annotations

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass

from humaneval_t.schema import ConcreteTask


@dataclass(frozen=True)
class EvalResult:
    concrete_task_id: str
    entry_point: str
    passed: bool
    error: str | None = None
    stdout: str = ""


def _run_check(candidate_source: str, concrete_task: ConcreteTask) -> EvalResult:
    namespace: dict = {}
    out_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(out_buf):
            exec(candidate_source, namespace)
            if concrete_task.entry_point not in namespace:
                return EvalResult(
                    concrete_task_id=concrete_task.concrete_task_id,
                    entry_point=concrete_task.entry_point,
                    passed=False,
                    error=f"entry_point {concrete_task.entry_point!r} not defined by candidate",
                    stdout=out_buf.getvalue(),
                )
            exec(concrete_task.test, namespace)
            check_fn = namespace.get("check")
            if check_fn is None:
                return EvalResult(
                    concrete_task_id=concrete_task.concrete_task_id,
                    entry_point=concrete_task.entry_point,
                    passed=False,
                    error="concrete_task.test did not define a check() function",
                    stdout=out_buf.getvalue(),
                )
            check_fn(namespace[concrete_task.entry_point])
    except AssertionError as e:
        return EvalResult(
            concrete_task_id=concrete_task.concrete_task_id,
            entry_point=concrete_task.entry_point,
            passed=False,
            error=f"AssertionError: {e}",
            stdout=out_buf.getvalue(),
        )
    except Exception:
        return EvalResult(
            concrete_task_id=concrete_task.concrete_task_id,
            entry_point=concrete_task.entry_point,
            passed=False,
            error=traceback.format_exc(),
            stdout=out_buf.getvalue(),
        )
    return EvalResult(
        concrete_task_id=concrete_task.concrete_task_id,
        entry_point=concrete_task.entry_point,
        passed=True,
        stdout=out_buf.getvalue(),
    )


def evaluate_oracle(concrete_task: ConcreteTask) -> EvalResult:
    """Treat the canonical solution as the candidate. Should always pass."""
    return _run_check(concrete_task.canonical_solution, concrete_task)


def evaluate_candidate(concrete_task: ConcreteTask, candidate_source: str) -> EvalResult:
    """Run an external candidate (e.g. LLM-generated code) against the tests."""
    return _run_check(candidate_source, concrete_task)
