"""Three independent delta metrics per ConcreteTask against its baseline.

All three are computed deterministically and rule-based:

* **lexical**: 1 - normalized Levenshtein similarity of the prompt text.
  Proxy for surface change (and hence memorization-evading potential).

* **semantic**: 1 - TF-IDF cosine similarity of the prompt text.
  Catches whether word-level meaning shifted (a concrete task that's lexically
  different but semantically equivalent should score low here).

* **difficulty**: L2 distance in standardized AST/structural features of the
  canonical solution. Because identifier-only renaming preserves the AST,
  this is near-zero by construction for `keep_original` test strategy — we
  track it primarily as a guardrail (a non-zero value flags drift).

The delta vector is computed per (concrete_task, baseline) pair. The baseline
is always concrete_task index 0 of a Template (the all-original-values row).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
from Levenshtein import ratio as levenshtein_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from humaneval_t.schema import ConcreteTask


@dataclass(frozen=True)
class DeltaVector:
    """Triple of independent delta scores against a baseline. All in [0, ~1]."""

    lexical: float
    semantic: float
    difficulty: float

    def as_dict(self) -> dict[str, float]:
        return {
            "lexical": self.lexical,
            "semantic": self.semantic,
            "difficulty": self.difficulty,
        }


def lexical_delta(baseline_text: str, candidate_text: str) -> float:
    return 1.0 - levenshtein_ratio(baseline_text, candidate_text)


def _ast_features(source: str) -> np.ndarray:
    """Return a small fixed-length feature vector summarizing AST shape."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    node_count = 0
    func_count = 0
    branch_count = 0  # if / for / while / try
    call_count = 0
    max_depth = 0

    def walk(node: ast.AST, depth: int) -> None:
        nonlocal node_count, func_count, branch_count, call_count, max_depth
        node_count += 1
        max_depth = max(max_depth, depth)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_count += 1
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try)):
            branch_count += 1
        if isinstance(node, ast.Call):
            call_count += 1
        for child in ast.iter_child_nodes(node):
            walk(child, depth + 1)

    walk(tree, 0)
    return np.array([node_count, func_count, branch_count, call_count, max_depth], dtype=float)


def difficulty_delta(baseline_solution: str, candidate_solution: str) -> float:
    bv = _ast_features(baseline_solution)
    cv = _ast_features(candidate_solution)
    denom = np.linalg.norm(bv) + 1e-9
    return float(np.linalg.norm(cv - bv) / denom)


def semantic_delta_batch(baseline_text: str, candidate_texts: list[str]) -> list[float]:
    """Compute TF-IDF cosine deltas for many candidates against one baseline."""
    corpus = [baseline_text, *candidate_texts]
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [float(1.0 - s) for s in sims]


def compute_deltas(concrete_tasks: list[ConcreteTask]) -> list[DeltaVector]:
    """Compute delta vectors for every concrete task against concrete_tasks[0] (baseline).

    The baseline's own delta is (0, 0, 0) by construction.
    """
    if not concrete_tasks:
        return []
    baseline = concrete_tasks[0]
    candidates = concrete_tasks[1:]
    candidate_prompts = [c.prompt for c in candidates]

    sem_deltas = (
        semantic_delta_batch(baseline.prompt, candidate_prompts) if candidates else []
    )

    out: list[DeltaVector] = [DeltaVector(0.0, 0.0, 0.0)]
    for c, sem in zip(candidates, sem_deltas):
        lex = lexical_delta(baseline.prompt, c.prompt)
        diff = difficulty_delta(baseline.canonical_solution, c.canonical_solution)
        out.append(DeltaVector(lexical=lex, semantic=sem, difficulty=diff))
    return out
