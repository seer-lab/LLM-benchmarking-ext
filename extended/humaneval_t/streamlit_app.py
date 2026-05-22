"""HumanEval_T extended — template authoring workbench.

Pick a HumanEval problem, write the template's template variables and values
by hand, preview the PICT output live, save to templates/. No automation in
the authoring step — every template is hand-crafted.

The only program-side help offered: when you start a new template for a
problem, the UI seeds an identifier template variable for the function name
and one per positional argument, with only the original value. You add
alternatives.

Vocabulary (matches arXiv 2412.01526):

- **template task** -> stored as a JSON in templates/HumanEval_*.json
- **template variable** -> one parameter slot in a template
- **concrete task** -> a fully materialized version of a template
- **benchmark variant** -> the assembled output, one per V*.json

Run:
    streamlit run humaneval_t/streamlit_app.py
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from humaneval_t.dataset import all_task_ids, get_problem  # noqa: E402
from humaneval_t.materializer import materialize_all  # noqa: E402
from humaneval_t.pict_io import run_pict, template_to_pict_model  # noqa: E402
from humaneval_t.schema import Template  # noqa: E402


# ── friendly error messages ────────────────────────────────────────────────


def _explain_validation_error(err: ValidationError) -> tuple[str, str]:
    """Translate a pydantic ValidationError into (headline, what_to_do)."""
    first = err.errors()[0] if err.errors() else None
    if first is None:
        return ("Template is not valid.", "Check the template variables, constraints, and templated text.")

    msg = first.get("msg", "") or ""
    msg = msg.removeprefix("Value error, ").removeprefix("Assertion failed, ")

    # Pattern 1: template variable declared but never referenced
    m = re.search(r"template variable '([^']+)' declared but never referenced as <([^>]+)>", msg)
    if m:
        name = m.group(1)
        return (
            f"Template variable **`{name}`** is listed in your template variables table but the placeholder `<{name}>` doesn't appear anywhere in the templated_prompt or templated_solution.",
            f"Two options:\n"
            f"1. **Insert** `<{name}>` into the templated_prompt where you want it filled — open the *templated_prompt* expander and find the word you want to vary.\n"
            f"2. Or **delete** the `{name}` template variable if you don't actually need it.",
        )

    # Pattern 2: test_strategy errors
    if "test_strategy" in msg and "templated" in msg and "exactly" in msg:
        return (
            "Wrong number of assertions for `templated` mode.",
            "The *Test strategy* section requires **exactly 5** `assert candidate(...) == ...` lines. "
            "Fill in any blanks or remove extras.",
        )
    if "test_strategy" in msg and "must start with 'assert '" in msg:
        return (
            "One of the test assertions doesn't start with `assert `.",
            "Each of the 5 assertion lines must begin with `assert ` — e.g. `assert candidate(\"...\") == [...]`.",
        )
    if "test_strategy" in msg and "'assertions' is only valid when mode='templated'" in msg:
        return (
            "You have assertions defined but the test mode is `keep_original`.",
            "Either switch the test mode to *templated* (to use those assertions) or clear the assertions list.",
        )

    # Pattern 3: identifier template variable has a non-identifier value
    m = re.search(r"template variable '([^']+)': value '([^']+)' is not a valid Python identifier", msg)
    if m:
        name, bad = m.group(1), m.group(2)
        suggestion = re.sub(r"[^A-Za-z0-9_]+", "_", bad).strip("_") or "var_name"
        return (
            f"Template variable **`{name}`** is type *identifier* (a Python name), so its values can't contain spaces or punctuation. The value `\"{bad}\"` isn't a valid Python identifier.",
            f"Two options:\n"
            f"1. Rewrite the value with underscores: e.g. `{suggestion}` instead of `{bad}`.\n"
            f"2. Or change the variable **kind** from *identifier* to *lexical* if this isn't meant to be a code symbol.",
        )

    # Pattern 4: duplicate values
    m = re.search(r"template variable '([^']+)': values contain duplicates", msg)
    if m:
        name = m.group(1)
        return (
            f"Template variable **`{name}`** has the same value listed more than once.",
            f"Remove the duplicate(s) from the values box for `{name}` (comma-separated list — each entry must be unique).",
        )

    # Pattern 5: duplicate template variable names
    if "duplicate template variable names" in msg:
        return (
            "Two template variables have the same **name**.",
            "Template variable names must be unique — rename one of the duplicates.",
        )

    return ("Template is not valid.", msg or str(err))


def _explain_pict_error(e: Exception) -> tuple[str, str]:
    stderr = ""
    if isinstance(e, subprocess.CalledProcessError):
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        if not stderr and stdout:
            stderr = stdout

    if "constraint" in stderr.lower() or "if [" in stderr.lower():
        return (
            "PICT couldn't parse your constraints.",
            f"Check the constraint syntax. Each line must look like:\n"
            f"```\nIF [variable_a] = \"value_x\" THEN [variable_b] = \"value_y\";\n```\n"
            f"Use **square brackets** around template variable names, **double quotes** around values, and end each rule with a **semicolon**.\n\n"
            f"PICT said:\n```\n{stderr}\n```",
        )
    if "unknown parameter" in stderr.lower() or "is not declared" in stderr.lower():
        return (
            "A constraint refers to a template variable that doesn't exist.",
            f"One of your constraints mentions a name you don't have in the template variables table. Check spelling.\n\n"
            f"PICT said:\n```\n{stderr}\n```",
        )
    if stderr:
        return ("PICT rejected the input.", f"PICT said:\n```\n{stderr}\n```")
    return ("PICT failed.", f"Raw error:\n```\n{e}\n```")


def _explain_materialization_error(e: Exception) -> tuple[str, str]:
    if isinstance(e, KeyError):
        name = e.args[0] if e.args else "?"
        return (
            f"The templated_prompt or templated_solution contains `<{name}>` but no template variable named **`{name}`** is defined.",
            f"Two options:\n"
            f"1. **Add** a template variable called `{name}` with at least one value.\n"
            f"2. Or **remove** `<{name}>` from the templated text.",
        )
    return ("Couldn't materialize a concrete task.", f"Raw error:\n```\n{e}\n```")


def _show_error(headline: str, what_to_do: str) -> None:
    st.error(headline)
    st.markdown(what_to_do)


# ── paths ──────────────────────────────────────────────────────────────────


FINAL_DIR = ROOT / "templates"
SKIP_FILE = ROOT / "templates" / "skipped.json"
OUTPUT_DIR = ROOT / "output"  # extended/output/  — gitignored, repo-local

FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────


def _safe_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", task_id)


def _final_path(task_id: str) -> Path:
    return FINAL_DIR / f"{_safe_name(task_id)}.json"


def _load_skipped() -> set[str]:
    if SKIP_FILE.exists():
        return set(json.loads(SKIP_FILE.read_text("utf-8")))
    return set()


def _save_skipped(skipped: set[str]) -> None:
    SKIP_FILE.write_text(json.dumps(sorted(skipped), indent=2) + "\n", encoding="utf-8")


def _status(task_id: str, skipped: set[str]) -> str:
    if task_id in skipped:
        return "skipped"
    if _final_path(task_id).exists():
        return "final"
    return "todo"


def _is_paper_subset(task_id: str) -> bool:
    """Return True if the saved template marks this task as part of the
    workshop paper's 10-problem subset."""
    p = _final_path(task_id)
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("paper_subset", False))
    except Exception:
        return False


def _seed_template(task_id: str) -> dict:
    """Create a starter template: identifier template variables from the AST."""
    p = get_problem(task_id)
    prompt = p["prompt"]
    entry_point = p["entry_point"]

    parseable = prompt.rstrip() + "\n    pass\n"
    tree = ast.parse(parseable)
    func = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
    if func is None:
        raise ValueError(f"no def found in {task_id}")
    def_line = next(line for line in prompt.splitlines() if line.lstrip().startswith("def "))

    variables: list[dict] = [{"name": "fn", "kind": "identifier", "values": [entry_point]}]
    arg_names: list[tuple[str, str]] = []  # (original, variable_name)
    for i, arg in enumerate(func.args.args, start=1):
        var_name = f"arg{i}"
        variables.append({"name": var_name, "kind": "identifier", "values": [arg.arg]})
        arg_names.append((arg.arg, var_name))

    new_def = def_line
    new_def = re.sub(r"\b" + re.escape(entry_point) + r"\b", "<fn>", new_def, count=1)
    for original, var_name in arg_names:
        new_def = re.sub(r"\b" + re.escape(original) + r"\b", f"<{var_name}>", new_def, count=1)

    templated_prompt = prompt.replace(def_line, new_def, 1)
    templated_prompt = re.sub(
        r"\b" + re.escape(entry_point) + r"\b", "<fn>", templated_prompt
    )

    templated_solution = p["canonical_solution"]
    for original, var_name in arg_names:
        templated_solution = re.sub(
            r"\b" + re.escape(original) + r"\b", f"<{var_name}>", templated_solution
        )
    templated_solution = re.sub(
        r"\b" + re.escape(entry_point) + r"\b", "<fn>", templated_solution
    )

    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "notes": "",
        "templated_prompt": templated_prompt,
        "templated_solution": templated_solution,
        "variables": variables,
        "constraints": [],
        "test_strategy": {"mode": "keep_original"},
    }


def _load_template(task_id: str) -> dict:
    if _final_path(task_id).exists():
        return json.loads(_final_path(task_id).read_text("utf-8"))
    return _seed_template(task_id)


def _get_draft(task_id: str) -> dict:
    """Return the live, in-session draft. Persists across reruns via session_state."""
    key = f"draft::{task_id}"
    if key not in st.session_state:
        st.session_state[key] = _load_template(task_id)
    return st.session_state[key]


# ── sidebar ───────────────────────────────────────────────────────────────


def render_view_picker() -> str:
    st.sidebar.title("HumanEval_T extended")
    st.sidebar.caption("Built by Riddhi More")
    return st.sidebar.radio(
        "view",
        ["Author templates", "Build manually", "Browse built variants", "Guide"],
        index=0,
        horizontal=False,
        label_visibility="collapsed",
    )


def render_sidebar() -> str:
    skipped = _load_skipped()
    task_ids = all_task_ids()

    counts = {"final": 0, "skipped": 0, "todo": 0}
    for tid in task_ids:
        counts[_status(tid, skipped)] += 1
    total = len(task_ids)
    st.sidebar.metric("Finalized", f"{counts['final']} / {total}")
    st.sidebar.progress(counts["final"] / total)
    st.sidebar.write(f"todo: **{counts['todo']}** · skipped: **{counts['skipped']}**")

    filter_mode = st.sidebar.radio(
        "filter",
        ["all", "todo", "final", "skipped"],
        index=1,
        horizontal=True,
    )

    visible: list[str] = []
    for tid in task_ids:
        s = _status(tid, skipped)
        if filter_mode == "all" or filter_mode == s:
            visible.append(tid)

    if not visible:
        st.sidebar.info("nothing matches that filter")
        return task_ids[0]

    badges = {"final": "[F]", "skipped": "[S]", "todo": "[-]"}
    options = []
    for tid in visible:
        base = f"{badges[_status(tid, skipped)]} {tid}"
        if _is_paper_subset(tid):
            # Documentation marker for paper-subset problems. Placed before
            # the task_id so the task_id remains the last whitespace token.
            base = base.replace(" ", " 📄  ", 1)  # "[F] 📄  HumanEval/0"
        options.append(base)
    pick = st.sidebar.selectbox("problem", options, index=0)

    st.sidebar.divider()
    with st.sidebar.expander("Build benchmark", expanded=False):
        st.caption(
            f"Assemble N variants from your finalized templates. "
            f"Writes to `{OUTPUT_DIR}`."
        )
        n_variants = st.number_input("variants (N)", min_value=2, max_value=20, value=5, step=1)
        seed = st.number_input(
            "seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            step=1,
            help="CBC random-seed for reproducible tie-breaking between multiple ILP optima. "
                 "Same seed + same templates → identical V*.json every build.",
        )
        time_limit = st.number_input(
            "time limit (s)",
            min_value=10,
            max_value=900,
            value=60,
            step=10,
            help="Max seconds CBC may spend searching for the optimum. Recommended 30–300. "
                 "Larger N or more templates ⇒ raise it. If a build reports a non-zero "
                 "lexical range, bump this up for a tighter optimum.",
        )
        st.caption(
            "Recommended: **N = 5**, **seed = 42**, **time limit = 60s** "
            "(↑ for tighter balance). See *Guide → How variants get balanced* for details."
        )
        if st.button("Build now", key="sidebar_build", type="primary"):
            _build_benchmark_and_report(int(n_variants), int(seed), int(time_limit))

    return pick.rsplit(" ", 1)[-1]


def _build_benchmark_and_report(n_variants: int, seed: int = 42, time_limit_s: int = 120) -> None:
    from humaneval_t.assembler import TemplateGroup, assemble_variants
    from humaneval_t.delta import compute_deltas

    paths = sorted(FINAL_DIR.glob("HumanEval_*.json"))
    if not paths:
        st.sidebar.error("No finalized templates yet — save at least one with **Save as final** before building.")
        return

    groups: list = []
    sparse: list[tuple[str, int]] = []
    for path in paths:
        try:
            template = Template.model_validate(json.loads(path.read_text("utf-8")))
            rows = run_pict(template, order=2)
            concrete_tasks = materialize_all(template, rows)
            deltas = compute_deltas(concrete_tasks)
            non_baseline = len(concrete_tasks) - 1
            if non_baseline < n_variants:
                sparse.append((template.task_id, non_baseline))
            groups.append(
                TemplateGroup(task_id=template.task_id, concrete_tasks=concrete_tasks, deltas=deltas)
            )
        except Exception as e:
            st.sidebar.error(f"`{path.name}` failed: {e}")
            return

    if sparse:
        msg = ", ".join(f"`{tid}` ({k})" for tid, k in sparse)
        st.sidebar.error(
            f"These template(s) have fewer than N={n_variants} non-baseline concrete tasks: {msg}. "
            "Either reduce N, or add more values / lexical variables to those templates."
        )
        return

    # CBC runs in a background thread so the sidebar can show a live countdown.
    # The solver may finish well before `time_limit_s` if it proves optimum early.
    result: dict = {}

    def _run_solver() -> None:
        try:
            result["variants"] = assemble_variants(
                groups, n=n_variants, time_limit_s=time_limit_s, seed=seed
            )
        except Exception as exc:  # noqa: BLE001  surfaced to UI below
            result["error"] = exc

    thread = threading.Thread(target=_run_solver, daemon=True)
    thread.start()

    timer_slot = st.sidebar.empty()
    bar_slot = st.sidebar.empty()
    start = time.monotonic()
    while thread.is_alive():
        elapsed = time.monotonic() - start
        remaining = max(0.0, time_limit_s - elapsed)
        timer_slot.metric(
            "solver time left",
            f"{remaining:5.1f}s",
            delta=f"elapsed {elapsed:4.1f}s",
            delta_color="off",
        )
        bar_slot.progress(min(1.0, elapsed / max(1, time_limit_s)))
        time.sleep(0.5)
    thread.join()
    timer_slot.empty()
    bar_slot.empty()

    if "error" in result:
        st.sidebar.error(f"assembler failed: {result['error']}")
        return
    variants = result["variants"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for v in variants:
        (OUTPUT_DIR / f"{v.variant_id}.json").write_text(
            json.dumps(v.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "n_variants": n_variants,
        "seed": seed,
        "time_limit_s": time_limit_s,
        "templates": [{"task_id": g.task_id, "concrete_task_count": len(g.concrete_tasks)} for g in groups],
        "variants": [
            {
                "variant_id": v.variant_id,
                "total_lexical": v.total_lexical,
                "total_semantic": v.total_semantic,
                "total_difficulty": v.total_difficulty,
                "concrete_task_ids": [ct.concrete_task_id for ct in v.concrete_tasks],
            }
            for v in variants
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    lex = [v.total_lexical for v in variants]
    st.sidebar.success(
        f"Wrote {n_variants} variant(s) + manifest to `{OUTPUT_DIR}`. "
        f"Lexical-delta range = {max(lex) - min(lex):.4f}."
    )


# ── main view ─────────────────────────────────────────────────────────────


def render_original_body(task_id: str) -> None:
    """Render the original HumanEval problem body. Caller controls the outer container."""
    p = get_problem(task_id)
    if _is_paper_subset(task_id):
        st.info(
            "📄 **Paper subset** — this problem is one of the 10 problems used in the workshop paper "
            "*Addressing Data Leakage in HumanEval Using Combinatorial Test Design* "
            "(arXiv 2412.01526). The template's lexical variable values mirror the paper's "
            "exact wording. Documentation marker only — has no effect on generation.",
            icon="📄",
        )
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Prompt**")
        st.code(p["prompt"], language="python")
    with right:
        st.markdown("**Canonical solution**")
        st.code(p["canonical_solution"], language="python")
    with st.expander("Original test (rarely needed while authoring)", expanded=False):
        st.code(p["test"], language="python")


def render_variable_editor(task_id: str, draft: dict) -> dict:
    """Edit the template variables list of `draft` in place."""
    st.markdown("**Template variables**")
    deleted = False
    for i, var in enumerate(draft["variables"]):
        with st.container(border=True):
            cols = st.columns([2, 1, 6, 1])
            with cols[0]:
                var["name"] = st.text_input(
                    f"name##{task_id}##{i}",
                    value=var["name"],
                    label_visibility="collapsed",
                )
            with cols[1]:
                var["kind"] = st.selectbox(
                    f"kind##{task_id}##{i}",
                    ["identifier", "lexical"],
                    index=0 if var["kind"] == "identifier" else 1,
                    label_visibility="collapsed",
                )
            with cols[2]:
                values_text = st.text_input(
                    f"values##{task_id}##{i}",
                    value=", ".join(var["values"]),
                    label_visibility="collapsed",
                    help="comma-separated; first value is the original/baseline",
                )
                var["values"] = [v.strip() for v in values_text.split(",") if v.strip()]
            with cols[3]:
                if st.button("del", key=f"del_var::{task_id}::{i}"):
                    draft["variables"].pop(i)
                    deleted = True
                    break
    if deleted:
        st.rerun()
    if st.button("+ add variable", key=f"add_var::{task_id}"):
        draft["variables"].append(
            {
                "name": f"var_{len(draft['variables']) + 1}",
                "kind": "lexical",
                "values": [""],
            }
        )
        st.rerun()
    return draft


def render_constraint_editor(task_id: str, draft: dict) -> dict:
    st.markdown("**PICT constraints** *— optional; skip unless PICT is producing combinations that don't make sense*")
    with st.expander("What's a PICT constraint?", expanded=False):
        st.markdown(
            """
By default PICT freely combines every value of every template variable.
Sometimes that produces a row whose word combination reads as nonsense even
though each value is fine on its own. A constraint tells PICT *"never emit a
row where these specific values co-occur."*

**Concrete example.** Say you have:

- `input_type`: `numbers`, `float values`, `measurements`
- `value_descriptor`: `values`, `elements`, `data points`

Without any constraint, PICT can emit `input_type=measurements` with
`value_descriptor=values`. The materialized docstring reads *"Check if in
given list of measurements, are any two values closer than the threshold."*
That mixes a scientific-register noun (`measurements`) with a generic one
(`values`) — sounds off. To force `measurements` to always pair with
`data points`, add:

```
IF [input_type] = "measurements" THEN [value_descriptor] = "data points";
```

PICT will then drop any candidate row that violates this rule.

**Syntax rules:**

- Variables in **`[square brackets]`**.
- Values in **`"double quotes"`**.
- End each rule with a **`;`** (semicolon).
- One rule per line.
- Operators inside `IF`/`THEN`: `=`, `<>`, `<`, `>`, `<=`, `>=`, plus `AND`, `OR`, `NOT`.

**More examples:**

```
# Force a paired choice
IF [input_type] = "measurements" THEN [value_descriptor] = "data points";

# Combine conditions
IF [arg_list] = "data_points" AND [fn] = "any_too_close" THEN [threshold_descriptor] = "tolerance";

# Forbid a specific combination outright
IF [fn] = "truncate_number" THEN [arg1] <> "text";
```

**When to use:** only when you've clicked *Run PICT* and noticed a generated
row reads as nonsense. Otherwise leave this empty.
"""
        )
    raw = "\n".join(c["raw"] for c in draft.get("constraints", []))
    new_raw = st.text_area(
        f"constraints##{task_id}",
        value=raw,
        label_visibility="collapsed",
        height=100,
        placeholder='IF [input_type] = "measurements" THEN [value_descriptor] = "data points";',
    )
    draft["constraints"] = [
        {"raw": line.strip()} for line in new_raw.splitlines() if line.strip()
    ]
    return draft


def render_template_text(task_id: str, draft: dict) -> dict:
    with st.expander("templated_prompt", expanded=True):
        draft["templated_prompt"] = st.text_area(
            f"templated_prompt##{task_id}",
            value=draft["templated_prompt"],
            label_visibility="collapsed",
            height=260,
        )
    with st.expander("templated_solution", expanded=False):
        draft["templated_solution"] = st.text_area(
            f"templated_solution##{task_id}",
            value=draft["templated_solution"],
            label_visibility="collapsed",
            height=200,
        )
    return draft


def render_test_strategy_editor(task_id: str, draft: dict) -> dict:
    """Edit how per-concrete-task tests are produced.

    Two modes:
      keep_original — use HumanEval's original test verbatim.
      templated     — write 5 assert lines (with `<variable>` placeholders) that get
                       substituted per PICT row.
    """
    from humaneval_t.schema import REQUIRED_TEST_COUNT  # imported lazily

    st.markdown("**Test strategy**")
    strategy = draft.get("test_strategy") or {"mode": "keep_original"}
    if not isinstance(strategy, dict):
        strategy = {"mode": "keep_original"}

    mode_options = ["keep_original", "templated"]
    current_mode = strategy.get("mode", "keep_original")
    if current_mode not in mode_options:
        current_mode = "keep_original"
    mode = st.radio(
        "test mode",
        mode_options,
        index=mode_options.index(current_mode),
        horizontal=True,
        key=f"test_mode::{task_id}",
        captions=[
            "Reuse HumanEval's original test — just rename the function. Use when test inputs don't need to vary.",
            f"Write exactly {REQUIRED_TEST_COUNT} assertions with `<variable>` placeholders. Use when test inputs themselves vary.",
        ],
        label_visibility="collapsed",
    )
    strategy["mode"] = mode

    if mode == "templated":
        with st.expander("How does `templated` mode work?", expanded=False):
            st.markdown(
                f"Write **exactly {REQUIRED_TEST_COUNT} lines**, each a Python "
                "`assert candidate(...) == ...` statement."
            )
            st.markdown(
                """
Use `<variable_name>` placeholders inside string literals or anywhere else.
At materialization time the placeholders are substituted using the PICT row's
values, and the lines are wrapped into a fresh `def check(candidate):`
function — replacing HumanEval's original test.

**Tiny example.** Imagine a `greet(name)` function whose docstring and
solution use a greeting word. You add one template variable:

- `greeting` (lexical) with values: `Hello`, `Hi`, `Hey`

Your templated_solution is:

```python
    return '<greeting>, ' + name + '!'
```

In *Test strategy* you pick `templated` and write five lines like:

```
assert candidate('Alice') == '<greeting>, Alice!'
assert candidate('Bob')   == '<greeting>, Bob!'
assert candidate('Eve')   == '<greeting>, Eve!'
assert candidate('')      == '<greeting>, !'
assert candidate('Zoe')   == '<greeting>, Zoe!'
```

When PICT picks the row `greeting=Hi`, the materialized check function is:

```python
def check(candidate):
    assert candidate('Alice') == 'Hi, Alice!'
    assert candidate('Bob')   == 'Hi, Bob!'
    assert candidate('Eve')   == 'Hi, Eve!'
    assert candidate('')      == 'Hi, !'
    assert candidate('Zoe')   == 'Hi, Zoe!'
```

The same five lines work for every value of `greeting` — you only write
them once. Use this mode whenever the *expected output* (or the *input*)
needs to track the variable, not just the surrounding wording.
"""
            )
        existing = strategy.get("assertions") or []
        if not isinstance(existing, list):
            existing = []
        # Pad or trim to REQUIRED_TEST_COUNT.
        existing = (existing + [""] * REQUIRED_TEST_COUNT)[:REQUIRED_TEST_COUNT]
        new_assertions: list[str] = []
        for i in range(REQUIRED_TEST_COUNT):
            new_assertions.append(
                st.text_input(
                    f"assertion {i + 1}",
                    value=existing[i],
                    key=f"test_assertion::{task_id}::{i}",
                    placeholder="assert candidate(...) == ...",
                )
            )
        strategy["assertions"] = new_assertions
    else:
        # Drop any stale assertions field so the schema validator doesn't refuse.
        strategy.pop("assertions", None)

    draft["test_strategy"] = strategy
    return draft


def render_preview(task_id: str, draft: dict) -> None:
    """Cache PICT output in session_state so the concrete-task slider survives reruns."""
    st.markdown("**PICT preview**")
    pict_key = f"pict_output::{task_id}"
    model_key = f"pict_model::{task_id}"
    ct_key = f"pict_concrete_tasks::{task_id}"

    if st.button("Run PICT", key=f"run_pict_btn::{task_id}"):
        for k in (pict_key, model_key, ct_key):
            st.session_state.pop(k, None)
        try:
            template = Template.model_validate(draft)
        except ValidationError as e:
            _show_error(*_explain_validation_error(e))
            return
        try:
            rows = run_pict(template, order=2)
        except Exception as e:
            _show_error(*_explain_pict_error(e))
            return
        try:
            concrete_tasks = materialize_all(template, rows)
        except Exception as e:
            _show_error(*_explain_materialization_error(e))
            return
        st.session_state[pict_key] = rows
        st.session_state[model_key] = template_to_pict_model(template)
        st.session_state[ct_key] = concrete_tasks

    rows = st.session_state.get(pict_key)
    if not rows:
        st.caption("Click *Run PICT* to preview the concrete tasks this template would produce.")
        return

    concrete_tasks = st.session_state.get(ct_key, [])
    st.success(f"PICT produced **{len(rows)}** pairwise-covering rows")
    with st.expander("PICT model file", expanded=False):
        st.code(st.session_state.get(model_key, ""))

    # Mark the row that corresponds to the original (all template variables at
    # their first/canonical value) with a ★ prepended to its first cell. PICT
    # always covers this combination as part of pairwise coverage, but the
    # materializer + assembler exclude it from any built variant.
    baseline_row = {
        v["name"]: v["values"][0]
        for v in draft.get("variables", [])
        if v.get("values")
    }
    first_col = next(iter(rows[0].keys()), None) if rows else None
    annotated_rows = []
    for r in rows:
        is_baseline = bool(baseline_row) and all(
            r.get(k) == baseline_row.get(k) for k in baseline_row
        )
        if is_baseline and first_col is not None:
            r2 = dict(r)
            r2[first_col] = f"★ {r2[first_col]}"
            annotated_rows.append(r2)
        else:
            annotated_rows.append(r)
    st.dataframe(annotated_rows, width="stretch", height=320)
    st.caption(
        "The row marked **★** is the all-canonical-values combination — "
        "i.e. the original HumanEval problem. It's kept as the comparison baseline "
        "but is **never** included in any built benchmark variant."
    )
    if not concrete_tasks:
        return
    st.caption(
        f"materialized {len(concrete_tasks)} concrete task(s) "
        f"(baseline + {len(concrete_tasks) - 1} from PICT)"
    )
    if len(concrete_tasks) == 1:
        st.info("Only the baseline exists. Add more values to your template variables (or add lexical variables) so PICT has combinations to enumerate.")
        idx = 0
    else:
        idx = st.slider(
            "concrete task to preview",
            0,
            len(concrete_tasks) - 1,
            0,
            key=f"preview_idx::{task_id}",
        )
    ct = concrete_tasks[idx]
    st.markdown(f"**{ct.concrete_task_id}**")
    st.code(ct.prompt, language="python")


def render_actions(task_id: str, draft: dict) -> None:
    skipped = _load_skipped()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Save as final", type="primary"):
            try:
                Template.model_validate(draft)
            except ValidationError as e:
                headline, what_to_do = _explain_validation_error(e)
                _show_error("Can't save: " + headline, what_to_do)
                return
            _final_path(task_id).write_text(
                json.dumps(draft, indent=2) + "\n", encoding="utf-8"
            )
            st.success(f"Saved **{task_id}** to `templates/`. The sidebar will mark it `[F]`.")
    with col2:
        if task_id in skipped:
            if st.button("Un-skip"):
                skipped.discard(task_id)
                _save_skipped(skipped)
                st.rerun()
        else:
            if st.button("Skip problem"):
                skipped.add(task_id)
                _save_skipped(skipped)
                st.warning(f"{task_id} marked skipped — it won't be required for the benchmark.")
    with col3:
        if st.button("Reset"):
            st.session_state[f"draft::{task_id}"] = _seed_template(task_id)
            st.info("Draft reset to a fresh signature-extracted seed. Click *Save as final* to persist.")
            st.rerun()


# ── views ─────────────────────────────────────────────────────────────────


def render_manual_builder_view() -> None:
    """Walk through finalized templates and hand-pick concrete tasks for each variant.

    Alternative to the ILP-optimized *Build now* button. You see every PICT-
    generated concrete task per template and decide which goes into V1, V2, …
    Strict exclusivity within a template is enforced by warnings; the totals
    are computed from the assignments and written to output/ when you click
    *Build variants from current assignments*.
    """
    from humaneval_t.delta import compute_deltas

    st.title("Manual variant builder")
    st.caption(
        "Pick the concrete task that goes into each variant, template by template. "
        "The auto builder (sidebar *Build now*) does this with an optimizer; this view lets you override its choices."
    )

    paths = sorted(FINAL_DIR.glob("HumanEval_*.json"))
    if not paths:
        st.info(
            "No finalized templates yet. Switch to *Author templates*, finalize at least one, then come back."
        )
        return

    # State keys
    key_n = "manual::n_variants"
    key_idx = "manual::current_idx"
    key_assign = "manual::assignments"  # {task_id: {V1: ct_id, V2: ct_id, ...}}

    if key_n not in st.session_state:
        st.session_state[key_n] = 5
    if key_idx not in st.session_state:
        st.session_state[key_idx] = 0
    if key_assign not in st.session_state:
        st.session_state[key_assign] = {}

    # ── Controls row: N picker + progress + nav ──
    top = st.columns([2, 5, 1, 1, 2])
    with top[0]:
        n = st.number_input(
            "Variants (N)",
            min_value=2,
            max_value=20,
            value=st.session_state[key_n],
            step=1,
        )
        st.session_state[key_n] = int(n)
    total = len(paths)
    current = min(st.session_state[key_idx], total - 1)
    with top[1]:
        st.progress((current + 1) / total, text=f"Template {current + 1} of {total}")
    with top[2]:
        if st.button("← Prev", disabled=current == 0, key="manual_prev"):
            st.session_state[key_idx] = max(0, current - 1)
            st.rerun()
    with top[3]:
        if st.button("Next →", disabled=current >= total - 1, key="manual_next"):
            st.session_state[key_idx] = min(total - 1, current + 1)
            st.rerun()
    with top[4]:
        if st.button("Reset all assignments", key="manual_reset"):
            st.session_state[key_assign] = {}
            st.session_state[key_idx] = 0
            st.rerun()

    st.divider()

    # ── Current template ──
    current_path = paths[current]
    try:
        template = Template.model_validate(json.loads(current_path.read_text("utf-8")))
    except Exception as e:
        st.error(f"Couldn't load {current_path.name}: {e}")
        return

    st.subheader(f"{template.task_id}  ·  entry_point: `{template.entry_point}`")

    # Compute (or fetch cached) concrete tasks + deltas for this template
    ct_cache_key = f"manual::cts::{template.task_id}"
    if ct_cache_key not in st.session_state:
        try:
            rows = run_pict(template, order=2)
            concrete_tasks = materialize_all(template, rows)
            deltas = compute_deltas(concrete_tasks)
            st.session_state[ct_cache_key] = (concrete_tasks, deltas)
        except Exception as e:
            st.error(f"PICT/materialize failed for `{template.task_id}`: {e}")
            return
    concrete_tasks, deltas = st.session_state[ct_cache_key]

    if len(concrete_tasks) < 2:
        st.warning(
            f"`{template.task_id}` only has a baseline — no concrete tasks to assign. "
            "Add more values or lexical variables to this template."
        )
        return

    # ── Concrete tasks table ──
    st.markdown("**Available concrete tasks** (baseline `#000` is the original problem and is not assignable):")
    table_rows = []
    for i, (ct, d) in enumerate(zip(concrete_tasks, deltas)):
        table_rows.append(
            {
                "id": ct.concrete_task_id,
                "entry_point": ct.entry_point,
                "lex_delta": round(d.lexical, 4),
                "sem_delta": round(d.semantic, 4),
                "variable_values": json.dumps(ct.variable_values, sort_keys=True),
            }
        )
    st.dataframe(table_rows, width="stretch", height=240)

    # Preview the prompt for any concrete task
    with st.expander("Preview the prompt of a concrete task", expanded=False):
        ids = [ct.concrete_task_id for ct in concrete_tasks]
        picked_id = st.selectbox(
            "concrete task",
            ids,
            index=0,
            key=f"manual_preview_pick::{template.task_id}",
        )
        ct = next(c for c in concrete_tasks if c.concrete_task_id == picked_id)
        st.code(ct.prompt, language="python")

    # ── Assignment selectboxes ──
    st.markdown(
        "**Assign concrete tasks to variants** (each variant gets at most one concrete task from this template; leave at *skip* to omit this template from a variant):"
    )
    template_assignments = st.session_state[key_assign].get(template.task_id, {})
    assignable_ids = [ct.concrete_task_id for ct in concrete_tasks[1:]]  # exclude baseline
    options = ["skip"] + assignable_ids

    n_variants = int(st.session_state[key_n])
    cols = st.columns(n_variants)
    for v_idx in range(n_variants):
        var_id = f"V{v_idx + 1}"
        current_choice = template_assignments.get(var_id, "skip")
        if current_choice not in options:
            current_choice = "skip"
        with cols[v_idx]:
            choice = st.selectbox(
                var_id,
                options,
                index=options.index(current_choice),
                key=f"manual_assign::{template.task_id}::{var_id}",
            )
            template_assignments[var_id] = choice
    st.session_state[key_assign][template.task_id] = template_assignments

    # Uniqueness check
    picked = [v for v in template_assignments.values() if v != "skip"]
    duplicates = [v for v in set(picked) if picked.count(v) > 1]
    if duplicates:
        st.warning(
            f"Same concrete task assigned to multiple variants: {', '.join(duplicates)}. "
            "Each concrete task can only appear in one variant — fix before building."
        )

    st.divider()

    # ── Build action ──
    st.markdown("### Build the benchmark")
    st.caption(
        "When you click below, every template's current assignments are used. Templates you haven't visited yet have all *skip* assignments — those variants will simply omit that template."
    )
    if st.button("Build variants from current assignments", type="primary", key="manual_build"):
        _build_manual_variants(paths, st.session_state[key_assign], n_variants)


def _build_manual_variants(
    template_paths: list[Path],
    assignments_by_task: dict[str, dict[str, str]],
    n_variants: int,
) -> None:
    """Materialize the user's manual assignments into V*.json + manifest.json."""
    from humaneval_t.delta import compute_deltas

    # Collect per-variant concrete tasks + delta sums.
    variants_data: dict[str, dict] = {
        f"V{i + 1}": {"concrete_tasks": [], "totals": {"lexical": 0.0, "semantic": 0.0, "difficulty": 0.0}}
        for i in range(n_variants)
    }

    # Per-template duplicate enforcement.
    duplicate_problems: list[str] = []
    template_summaries: list[dict] = []

    for path in template_paths:
        try:
            template = Template.model_validate(json.loads(path.read_text("utf-8")))
        except Exception as e:
            st.sidebar.error(f"`{path.name}` failed to load: {e}")
            return

        task_id = template.task_id
        ct_cache_key = f"manual::cts::{task_id}"
        if ct_cache_key not in st.session_state:
            # User hasn't visited this template — compute its concrete tasks now.
            try:
                rows = run_pict(template, order=2)
                concrete_tasks = materialize_all(template, rows)
                deltas = compute_deltas(concrete_tasks)
                st.session_state[ct_cache_key] = (concrete_tasks, deltas)
            except Exception as e:
                st.error(f"PICT/materialize failed for `{task_id}`: {e}")
                return
        concrete_tasks, deltas = st.session_state[ct_cache_key]
        ct_by_id = {ct.concrete_task_id: (ct, d) for ct, d in zip(concrete_tasks, deltas)}

        assignments = assignments_by_task.get(task_id, {})
        picked = [c for c in assignments.values() if c != "skip"]
        if len(picked) != len(set(picked)):
            duplicate_problems.append(task_id)

        for v_idx in range(n_variants):
            var_id = f"V{v_idx + 1}"
            chosen_id = assignments.get(var_id, "skip")
            if chosen_id == "skip" or chosen_id not in ct_by_id:
                continue
            ct, d = ct_by_id[chosen_id]
            variants_data[var_id]["concrete_tasks"].append(ct)
            variants_data[var_id]["totals"]["lexical"] += d.lexical
            variants_data[var_id]["totals"]["semantic"] += d.semantic
            variants_data[var_id]["totals"]["difficulty"] += d.difficulty

        template_summaries.append({"task_id": task_id, "concrete_task_count": len(concrete_tasks)})

    if duplicate_problems:
        st.error(
            "Aborting: some templates have the same concrete task assigned to multiple variants — "
            f"{', '.join(duplicate_problems)}. Fix the duplicates in those templates and try again."
        )
        return

    # Write each variant JSON.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for var_id, vd in variants_data.items():
        out = {
            "variant_id": var_id,
            "totals": vd["totals"],
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
                for ct in vd["concrete_tasks"]
            ],
        }
        (OUTPUT_DIR / f"{var_id}.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8"
        )

    # Manifest
    manifest = {
        "n_variants": n_variants,
        "build_mode": "manual",
        "templates": template_summaries,
        "variants": [
            {
                "variant_id": var_id,
                "total_lexical": vd["totals"]["lexical"],
                "total_semantic": vd["totals"]["semantic"],
                "total_difficulty": vd["totals"]["difficulty"],
                "concrete_task_ids": [ct.concrete_task_id for ct in vd["concrete_tasks"]],
            }
            for var_id, vd in variants_data.items()
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    counts = {var_id: len(vd["concrete_tasks"]) for var_id, vd in variants_data.items()}
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    st.success(
        f"Built {n_variants} variant(s) manually. Concrete tasks per variant: {summary}. "
        "Switch to *Browse built variants* to inspect."
    )


def _render_levenshtein_dp_table():
    """Render the Levenshtein DP table for 'kitten' -> 'sitting' as a matplotlib figure.

    Shows the dynamic-programming matrix of edit distances and highlights the
    optimal-path cells. The bottom-right cell (3) is the final Levenshtein
    distance used in the lexical-delta formula.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    a = "kitten"
    b = "sitting"
    m, n = len(a), len(b)

    # Classic Levenshtein DP.
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost,  # match / substitution
            )

    # Trace one optimal path from (m, n) back to (0, 0).
    path = [(m, n)]
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                i, j = i - 1, j - 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                i -= 1
            else:
                j -= 1
        elif i > 0:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    path_set = set(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(-0.5, n + 1)
    ax.set_ylim(m + 1, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Column headers: characters of b.
    headers_b = [""] + list(b)
    for j, ch in enumerate(headers_b):
        ax.text(j + 0.5, -0.3, ch, ha="center", va="center", fontsize=11, fontweight="bold")
    # Row headers: characters of a.
    headers_a = [""] + list(a)
    for i, ch in enumerate(headers_a):
        ax.text(-0.3, i + 0.5, ch, ha="center", va="center", fontsize=11, fontweight="bold")

    # Cells.
    for i in range(m + 1):
        for j in range(n + 1):
            on_path = (i, j) in path_set
            face = "#fde68a" if on_path else "#f9fafb"
            edge = "#9ca3af"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=face, edgecolor=edge, linewidth=0.8))
            ax.text(j + 0.5, i + 0.5, str(int(dp[i][j])), ha="center", va="center", fontsize=11)

    # Highlight the answer cell.
    ax.add_patch(
        plt.Rectangle((n, m), 1, 1, fill=False, edgecolor="#dc2626", linewidth=2.2)
    )
    ax.set_title(
        "Levenshtein DP table for 'kitten' → 'sitting' (distance = 3)",
        fontsize=11,
        pad=10,
    )
    fig.tight_layout()
    return fig


def _render_cosine_diagram():
    """Render two vectors with the angle between them for the semantic-delta explanation."""
    import math
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    # Two vectors. v_b is the baseline, v_c is a candidate at ~35° away.
    ax.set_xlim(-0.4, 5)
    ax.set_ylim(-0.4, 5)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="#d1d5db", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#d1d5db", linewidth=0.8, zorder=0)

    vb = (4.0, 0.6)
    vc = (3.0, 2.4)
    ax.annotate("", xy=vb, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.0))
    ax.annotate("", xy=vc, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.0))
    ax.text(vb[0] + 0.15, vb[1], r"$\mathbf{v}_b$ (baseline)", fontsize=11, color="#2563eb")
    ax.text(vc[0] + 0.15, vc[1] + 0.1, r"$\mathbf{v}_c$ (candidate)", fontsize=11, color="#dc2626")

    # Angle arc.
    from matplotlib.patches import Arc
    angle_b = math.degrees(math.atan2(vb[1], vb[0]))
    angle_c = math.degrees(math.atan2(vc[1], vc[0]))
    ax.add_patch(
        Arc((0, 0), 1.6, 1.6, angle=0, theta1=angle_b, theta2=angle_c, color="#16a34a", linewidth=1.8)
    )
    mid = math.radians((angle_b + angle_c) / 2)
    ax.text(0.95 * math.cos(mid), 0.95 * math.sin(mid), r"$\theta$", fontsize=13, color="#16a34a")

    ax.set_title(
        r"Cosine similarity $= \cos\theta$  ·  semantic delta $= 1 - \cos\theta$",
        fontsize=11,
        pad=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def _render_assembler_flow_diagram():
    """High-level pipeline flow boxes-and-arrows."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    boxes = [
        (2, "PICT rows\n(per template)", "#dbeafe"),
        (27, "Concrete tasks\n+ lexical Δ", "#dcfce7"),
        (52, "ILP solver\n(PuLP + CBC)", "#fef3c7"),
        (77, "V1, V2, …, VN\n(balanced sums)", "#fce7f3"),
    ]
    box_w, box_h, y = 21, 16, 7
    for x, label, color in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.4", facecolor=color, edgecolor="#374151", linewidth=1.0,
        ))
        ax.text(x + box_w / 2, y + box_h / 2, label, ha="center", va="center", fontsize=10)
    for i in range(len(boxes) - 1):
        ax.annotate(
            "",
            xy=(boxes[i + 1][0], y + box_h / 2),
            xytext=(boxes[i][0] + box_w, y + box_h / 2),
            arrowprops=dict(arrowstyle="->", color="#374151", lw=1.4),
        )
    return fig


def _render_assembler_example_chart():
    """Stacked bars: a tiny worked example showing the balanced ILP assignment.

    Three templates T1/T2/T3, three variants V1/V2/V3, each variant takes one
    concrete task per template. The picks shown here are ONE optimal assignment
    that minimizes the range of per-variant sums.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    variants = ["V1", "V2", "V3"]
    t1_picks = [0.6, 0.2, 0.1]
    t2_picks = [0.05, 0.4, 0.25]
    t3_picks = [0.15, 0.3, 0.45]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.55
    bot = np.zeros(3)
    ax.bar(variants, t1_picks, width, label="T1", color="#3b82f6", bottom=bot)
    bot += t1_picks
    ax.bar(variants, t2_picks, width, label="T2", color="#10b981", bottom=bot)
    bot += t2_picks
    ax.bar(variants, t3_picks, width, label="T3", color="#f59e0b", bottom=bot)
    totals = [t1_picks[i] + t2_picks[i] + t3_picks[i] for i in range(3)]
    for i, total in enumerate(totals):
        ax.text(i, total + 0.03, f"Σ = {total:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("total lexical delta")
    ax.set_title(f"Variant sums after ILP   ·   range = {max(totals) - min(totals):.2f}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(totals) + 0.18)
    fig.tight_layout()
    return fig


def render_variants_view() -> None:
    """Browse the assembled benchmark variants currently sitting in OUTPUT_DIR."""
    st.title("Built benchmark variants")
    st.caption(f"Reading from `{OUTPUT_DIR}`")

    variant_files = sorted(OUTPUT_DIR.glob("V*.json"))
    manifest_path = OUTPUT_DIR / "manifest.json"

    if not variant_files:
        st.info(
            f"No variants in `{OUTPUT_DIR}` yet.\n\n"
            "Switch back to *Author templates*, finalize a few templates, then open the "
            "*Build benchmark* expander in the sidebar and click **Build now**."
        )
        return

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except Exception:
            manifest = None
    else:
        manifest = None

    if manifest:
        cols = st.columns(4)
        cols[0].metric(
            "Variants",
            manifest.get("n_variants", len(variant_files)),
            help="Number of benchmark variants assembled (N).",
        )
        cols[1].metric(
            "Templates per variant",
            len(manifest.get("templates", [])),
            help="How many finalized templates contributed to the build. Each variant contains exactly one concrete task per template.",
        )
        lex = [v.get("total_lexical", 0.0) for v in manifest.get("variants", [])]
        if lex:
            cols[2].metric(
                "Lex range (max-min)",
                f"{max(lex) - min(lex):.4f}",
                help="max(total_lex across variants) − min(total_lex across variants). This is what the assembler MINIMIZES — closer to 0 means the variants are more equally reworded. A large range means one variant ended up much more (or less) reworded than the others.",
            )
            cols[3].metric(
                "Lex mean",
                f"{sum(lex) / len(lex):.3f}",
                help="Average total_lexical across variants. Higher = the benchmark as a whole carries more rewording from the originals (stronger leakage probe).",
            )

        with st.expander("How are these numbers calculated?", expanded=False):
            st.markdown("### The three delta metrics, in detail")

            st.markdown("#### 1. Lexical delta — surface change")
            st.markdown(
                "Measures how much the *characters* of the prompt text changed against the baseline. "
                "Implemented via the `python-Levenshtein` library's `ratio()` function:"
            )
            st.latex(r"\text{lexical}(b, c) \;=\; 1 \;-\; \frac{|b| + |c| - d_{\text{Lev}}(b, c)}{|b| + |c|}")
            st.markdown(
                "where:\n"
                "- $b$ = baseline prompt, $c$ = candidate concrete-task prompt\n"
                "- $|b|, |c|$ = character lengths\n"
                "- $d_{\\text{Lev}}(b, c)$ = **Levenshtein edit distance**: the minimum number of "
                "single-character **insertions**, **deletions**, or **substitutions** needed to turn "
                "$b$ into $c$. (`python-Levenshtein.ratio` internally weights substitutions as cost 2 — "
                "equivalent to an insert + delete pair — which is why the denominator above is "
                "$|b| + |c|$, not $\\max(|b|,|c|)$.)"
            )
            st.markdown("**Worked example.** Let $b$ = `kitten`, $c$ = `sitting`.")
            st.markdown(
                "| Step | Action | Result |\n"
                "|---|---|---|\n"
                "| 1 | substitute `k` → `s` | `sitten` |\n"
                "| 2 | substitute `e` → `i` | `sittin` |\n"
                "| 3 | insert `g` at end | `sitting` |"
            )
            st.markdown(
                "Classical Levenshtein distance $d = 3$. With substitution cost 2 the library treats this "
                "as $2{+}2{+}1 = 5$. With $|b|+|c| = 6+7 = 13$:"
            )
            st.latex(r"\text{lexical} = 1 - \frac{13 - 5}{13} = \frac{5}{13} \approx 0.385")
            st.markdown(
                "Higher values = more reworded. ~0.1 = a few synonym swaps. ~0.3+ = substantial rewording. "
                "0 = identical text."
            )
            st.pyplot(_render_levenshtein_dp_table())
            st.caption("Reference: https://en.wikipedia.org/wiki/Levenshtein_distance")

            st.markdown("---")
            st.markdown("#### 2. Semantic delta — meaning change")
            st.markdown(
                "Measures how much the *words* shifted in meaning, weighted by their importance in the "
                "vocabulary. Implemented as the cosine distance of TF-IDF vectors:"
            )
            st.latex(r"\text{semantic}(b, c) \;=\; 1 \;-\; \frac{\mathbf{v}_b \cdot \mathbf{v}_c}{\|\mathbf{v}_b\|\;\|\mathbf{v}_c\|}")
            st.markdown(
                "where $\\mathbf{v}_b, \\mathbf{v}_c$ are **TF-IDF** (term-frequency × "
                "inverse-document-frequency) vectors built from the baseline plus all candidate prompts "
                "of one template:"
            )
            st.latex(r"\text{tfidf}(t, d) \;=\; \underbrace{\frac{f_{t,d}}{\sum_{t'} f_{t',d}}}_{\text{TF: how often } t \text{ appears in } d} \;\cdot\; \underbrace{\log\frac{N}{|\{d' : t \in d'\}|}}_{\text{IDF: how rare } t \text{ is overall}}")
            st.markdown(
                "- $t$ = a token (word), $d$ = a document (one prompt), $N$ = number of documents\n"
                "- **TF** makes frequent words in a document weigh more\n"
                "- **IDF** down-weights common words (`the`, `a`) and up-weights rare ones (`tolerance`, "
                "`palindrome`)\n\n"
                "Cosine distance treats the two prompts as vectors in this weighted word-space and "
                "measures the angle between them. 0 = identical word usage, 1 = no shared meaningful "
                "vocabulary."
            )
            st.markdown(
                "**Why it can be high while lexical is low:** replacing `threshold` with `tolerance` is "
                "~9 character edits (a small lexical delta inside a 200-char prompt), but TF-IDF sees a "
                "complete swap of one *rare, important* word for another (a larger semantic delta)."
            )
            st.pyplot(_render_cosine_diagram())
            st.caption("References: https://en.wikipedia.org/wiki/Tf%E2%80%93idf · https://en.wikipedia.org/wiki/Cosine_similarity")

            st.markdown("---")
            st.markdown("#### 3. Difficulty delta — solution structural change")
            st.markdown(
                "Measures how much the **abstract syntax tree** (AST) of the canonical solution changed. "
                "We extract a 5-dimensional feature vector per solution:"
            )
            st.markdown(
                "- `node_count` — total AST nodes\n"
                "- `func_count` — number of `def` statements (incl. nested)\n"
                "- `branch_count` — `if` / `for` / `while` / `try` count\n"
                "- `call_count` — function calls\n"
                "- `max_depth` — deepest nesting level"
            )
            st.markdown("then compute the *normalized L2 distance* between baseline and candidate vectors:")
            st.latex(r"\text{difficulty}(b, c) \;=\; \frac{\bigl\|\,\mathbf{f}(c) - \mathbf{f}(b)\,\bigr\|_2}{\bigl\|\,\mathbf{f}(b)\,\bigr\|_2 + \varepsilon}")
            st.markdown(
                "With `keep_original` test strategy this is **always 0** because the solution's AST is "
                "byte-identical post-rename (renaming identifiers doesn't change node count, branch "
                "count, etc.). Tracked as a guardrail — non-zero means an authoring mistake (e.g. the "
                "templated_solution was edited beyond simple identifier substitution)."
            )

            st.markdown("---")
            st.markdown("### Per variant and across all variants")
            st.markdown(
                "**Per variant** (one row in the manifest summary table):\n\n"
                "- `total_lex`, `total_sem`, `total_diff` = sum of the corresponding per-concrete-task "
                "delta across all concrete tasks in that variant.\n"
                "- `concrete_tasks` = how many tasks the variant contains (= number of finalized templates).\n\n"
                "**Across all variants** (the four top metrics):\n\n"
                "- `Variants` = N (set when you clicked *Build now*).\n"
                "- `Templates per variant` = number of finalized templates that fed the build.\n"
                "- `Lex range (max - min)` = max(`total_lex`) − min(`total_lex`) across variants. "
                "**This is the assembler's optimization target** — the ILP picks one concrete task per "
                "template per variant such that this range is minimized, subject to strict exclusivity "
                "(no concrete task in two variants).\n"
                "- `Lex mean` = mean of `total_lex` across variants.\n\n"
                "**Note on semantic balance:** the assembler does NOT balance semantic or difficulty — "
                "only lexical. So `total_sem` will naturally vary across variants even when `total_lex` "
                "is locked. That variation is expected (typically ~10–20% spread). If a single variant's "
                "`total_sem` looks alarmingly high (say > 30% above the rest), inspect its individual "
                "concrete tasks for meaning drift; otherwise it's just a side-effect of the lexical-only "
                "optimization."
            )

            st.markdown("### How to read the numbers")
            st.markdown(
                "- `Lex range` < ~5% of `Lex mean` ⇒ variants are well balanced.\n"
                "- `Lex mean` small ⇒ templates aren't varying much; add more values per variable.\n"
                "- `total_sem` for one variant clearly higher than the others ⇒ inspect its concrete tasks "
                "for meaning drift.\n"
                "- `total_diff` should be 0 across the board for `keep_original` templates. Non-zero flags "
                "an authoring mistake."
            )

        with st.expander("Manifest summary table", expanded=False):
            rows = []
            for v in manifest.get("variants", []):
                rows.append(
                    {
                        "variant": v["variant_id"],
                        "total_lex": round(v.get("total_lexical", 0.0), 4),
                        "total_sem": round(v.get("total_semantic", 0.0), 4),
                        "total_diff": round(v.get("total_difficulty", 0.0), 4),
                        "concrete_tasks": len(v.get("concrete_task_ids", [])),
                    }
                )
            if rows:
                st.dataframe(rows, width="stretch")
    else:
        st.warning("No `manifest.json` found — showing variant files only.")

    st.divider()

    variant_names = [v.stem for v in variant_files]
    pick = st.selectbox("Inspect a variant", variant_names, index=0)
    selected = OUTPUT_DIR / f"{pick}.json"
    try:
        variant = json.loads(selected.read_text("utf-8"))
    except Exception as e:
        st.error(f"Couldn't read {selected.name}: {e}")
        return

    totals = variant.get("totals", {})
    cols = st.columns(3)
    cols[0].metric(
        "lexical",
        f"{totals.get('lexical', 0.0):.4f}",
        help="Sum of per-concrete-task lexical deltas in this variant. Each delta = 1 − Levenshtein_ratio(baseline_prompt, this_prompt). Higher = more surface rewording.",
    )
    cols[1].metric(
        "semantic",
        f"{totals.get('semantic', 0.0):.4f}",
        help="Sum of per-concrete-task semantic deltas in this variant. Each delta = 1 − cosine(TF-IDF(baseline), TF-IDF(this)). High values can indicate meaning drift — inspect individual concrete tasks.",
    )
    cols[2].metric(
        "difficulty",
        f"{totals.get('difficulty', 0.0):.4f}",
        help="Sum of per-concrete-task difficulty deltas. Each delta = L2 distance of AST features between this canonical solution and the baseline's. Always 0 for keep_original templates; non-zero means a templated_solution was edited beyond simple identifier renaming.",
    )

    concrete_tasks = variant.get("concrete_tasks", [])
    st.write(f"**{len(concrete_tasks)} concrete task(s)** in this variant")

    if not concrete_tasks:
        return
    ct_labels = [f"{c['concrete_task_id']}  (entry: {c['entry_point']})" for c in concrete_tasks]
    ct_pick = st.selectbox("Concrete task to inspect", ct_labels, index=0)
    ct = concrete_tasks[ct_labels.index(ct_pick)]

    st.markdown(f"**Variable values chosen:** `{ct.get('variable_values', {})}`")
    tab_prompt, tab_solution, tab_test = st.tabs(["Prompt", "Canonical solution", "Test"])
    with tab_prompt:
        st.code(ct.get("prompt", ""), language="python")
    with tab_solution:
        st.code(ct.get("canonical_solution", ""), language="python")
    with tab_test:
        st.code(ct.get("test", ""), language="python")

    st.divider()
    st.caption(f"File on disk: `{selected}`")


def render_author_view() -> None:
    task_id = render_sidebar()
    draft = _get_draft(task_id)

    # Original at the top in an expander so the editor gets the full width
    # below it (the 50/50 side-by-side layout was too squished for both the
    # docstring and the editor's wide value lists).
    with st.expander(f"Original — {task_id}  ·  entry_point: {get_problem(task_id)['entry_point']}", expanded=True):
        render_original_body(task_id)

    st.divider()

    st.subheader("Template")
    st.info(
        "**To add a template variable:** "
        "(1) in the *templated_prompt* below, wrap the word you want to vary in `<...>` "
        "— e.g. change `number` to `<number_word>`. "
        "(2) Then in the **template variables** table further down, click **+ add variable**, "
        "give it the same name (`number_word`), set its kind, and list comma-separated values "
        "(first one = the original word).",
        icon="💡",
    )
    draft["notes"] = st.text_input(
        f"notes##{task_id}", value=draft.get("notes", "")
    )
    draft = render_template_text(task_id, draft)
    draft = render_variable_editor(task_id, draft)
    draft = render_constraint_editor(task_id, draft)
    draft = render_test_strategy_editor(task_id, draft)
    render_actions(task_id, draft)

    st.divider()
    render_preview(task_id, draft)


def render_guide_view() -> None:
    """Explain what every concept and button means, in plain language."""
    st.title("Guide")
    st.caption("What this app is, what each piece does, and how to read the numbers.")

    with st.expander("What this app is for (read first)", expanded=True):
        st.markdown(
            """
The HumanEval benchmark has 164 fixed programming problems. LLMs have probably seen them during training, so high scores partly reflect **memorization** rather than real ability.

This app helps you build **reworded versions** of every HumanEval problem. The underlying task stays identical (so the same test cases still pass), but the wording in the docstring and the names of the function/arguments change. If an LLM scores well on the originals but worse on the reworded versions, that gap is evidence of memorization.

You do this for each of the 164 problems, then the app combines the rewordings into **benchmark variants** that you run against any LLM.

This is the extension of the paper *Addressing Data Leakage in HumanEval Using Combinatorial Test Design* (arXiv 2412.01526) from 10 sampled problems to all 164.
            """
        )

    with st.expander("The four views, in plain English"):
        st.markdown(
            """
- **Author templates** — the workbench. Pick a HumanEval problem, mark which words can be swapped for other equivalent words, list the alternatives, save. Repeat for the next problem.
- **Build manually** — walk through your finalized templates one at a time and hand-pick which concrete task goes into V1, V2, … per template. Alternative to the optimizer-driven *Build now* button in the sidebar.
- **Browse built variants** — after a build (manual or auto), this view lets you read what was produced.
- **Guide** — what you're reading.
            """
        )

    with st.expander("Glossary — the words that show up everywhere"):
        st.markdown(
            """
Vocabulary mirrors the paper (arXiv 2412.01526):

- **Template (template task)** — a recipe for one HumanEval problem. It describes which parts of the docstring/signature can vary and what the alternatives are. Stored on disk as `templates/HumanEval_*.json`.
- **Template variable** — one parameterizable part of the template. Example: in `"check if any two values are closer than the threshold"`, the words `values` and `threshold` are filled by template variables — they could resolve to `elements`, `data points`, `tolerance`, etc.
- **Variable kind** — `identifier` means the variable is a code symbol (function name or argument name; must be a valid Python identifier, no spaces). `lexical` means it's English wording in the docstring.
- **Value** — one option for a template variable. Each variable has a list of values; the *first* one is the original wording from HumanEval (the baseline).
- **Baseline / concrete task #000** — what you get when every template variable is at its first value. This equals the original HumanEval problem, byte-for-byte equivalent in behavior.
- **PICT** — the Microsoft tool that picks combinations of template variable values. Given a template, PICT outputs a tab-separated table where each row is one concrete combination. Default is *pairwise coverage*: every pair of values from any two variables appears together in at least one row.
- **Concrete task** — a fully reworded problem (prompt + solution + tests) produced from one PICT row.
- **Benchmark variant (V1, V2, …)** — a complete benchmark: one concrete task per finalized template. If you have 50 templates finalized and ask for N=5 variants, each variant has 50 reworded problems (one per template), and no concrete task appears in more than one variant.
- **Constraint** — an *optional* rule that tells PICT to skip nonsense value combinations. By default PICT freely combines every value of every variable; sometimes that produces a docstring like *"Given a list of measurements, are any two values closer..."* where the registers clash. A constraint like `IF [input_type] = "measurements" THEN [value_descriptor] = "data points";` forces a paired choice. Syntax: variables in `[brackets]`, values in `"double quotes"`, each rule ends with `;`. Operators: `=`, `<>`, `<`, `>`, `<=`, `>=`, plus `AND`, `OR`, `NOT`. Most templates need zero constraints — only add one if *Run PICT* shows you a combination that reads wrong.
            """
        )

    with st.expander("The three delta numbers (what they measure)"):
        st.markdown(
            """
A "delta" is a number that scores how *different* a concrete task is from the baseline. We track three independently:

- **lexical delta** — how much the surface text changed. Roughly: 0 means identical wording, ~0.3+ means substantial rewording. Computed as `1 - Levenshtein_ratio(baseline_prompt, concrete_task_prompt)`. This is the **primary** signal for "is this rewording strong enough to dodge memorization?"
- **semantic delta** — how much the *meaning* of the words changed. Computed via TF-IDF cosine distance. Low if the new wording uses synonyms; high if it uses unrelated words (which usually means you've broken the problem semantics — a warning sign).
- **difficulty delta** — how much the *underlying solution* changed. Computed from AST features (lines of code, branch count, etc.). With our default `keep_original` test strategy, this is **always 0** because we only rename identifiers, never restructure the algorithm. Tracked as a guardrail — non-zero means something unexpected happened.
            """
        )

    with st.expander("How variants get balanced — the assembler in detail"):
        st.markdown(
            "**Code:** [`humaneval_t/assembler.py`](../humaneval_t/assembler.py) → "
            "`assemble_variants(groups, n=5, time_limit_s=60)`.\n\n"
            "When you click *Build now*, this function turns the variant-assignment "
            "problem into an **Integer Linear Program (ILP)** and hands it to the "
            "**CBC** solver via [PuLP](https://coin-or.github.io/pulp/)."
        )

        st.markdown("### Pipeline at a glance")
        st.pyplot(_render_assembler_flow_diagram())
        st.caption(
            "PICT enumerates pairwise-covering rows per template → the materializer "
            "turns each row into a concrete task with a lexical-delta score → the ILP "
            "picks N concrete tasks per template and assigns them to variants such "
            "that the per-variant delta sums are as equal as possible."
        )

        st.markdown("### A worked example")
        st.markdown(
            "Suppose you have **3 templates** (T1, T2, T3) and you want to build "
            "**N = 3 variants**. The materializer + delta module gave you 3 concrete "
            "tasks per template, each with its own lexical delta:"
        )
        st.markdown(
            "| | task A | task B | task C |\n"
            "|---|---:|---:|---:|\n"
            "| **T1** | 0.10 | 0.20 | 0.60 |\n"
            "| **T2** | 0.05 | 0.25 | 0.40 |\n"
            "| **T3** | 0.15 | 0.30 | 0.45 |"
        )
        st.markdown(
            "Every variant has to take exactly one concrete task from each template, "
            "and no task can be reused across variants. That means the assembler "
            "essentially picks a *permutation* of {A, B, C} per template — there are "
            "$3! \\cdot 3! \\cdot 3! = 216$ possible combinations to consider.\n\n"
            "Two of them illustrate why the choice matters:"
        )
        st.markdown(
            "**Greedy assignment** (V1 takes the biggest, V3 the smallest):\n\n"
            "| | V1 | V2 | V3 |\n"
            "|---|---:|---:|---:|\n"
            "| **T1** | 0.60 | 0.20 | 0.10 |\n"
            "| **T2** | 0.40 | 0.25 | 0.05 |\n"
            "| **T3** | 0.45 | 0.30 | 0.15 |\n"
            "| **Σ** | **1.45** | **0.75** | **0.30** |\n\n"
            "→ range = 1.15. V1 is **way** more reworded than V3 — that variant is "
            "an 'easier' benchmark and the comparison across models is contaminated."
        )
        st.markdown(
            "**ILP-balanced assignment** (one of several optima):\n\n"
            "| | V1 | V2 | V3 |\n"
            "|---|---:|---:|---:|\n"
            "| **T1** | 0.60 | 0.20 | 0.10 |\n"
            "| **T2** | 0.05 | 0.40 | 0.25 |\n"
            "| **T3** | 0.15 | 0.30 | 0.45 |\n"
            "| **Σ** | **0.80** | **0.90** | **0.80** |\n\n"
            "→ range = 0.10. All three variants are within a hair of each other — "
            "fair to compare model scores across them. **That's exactly what the ILP "
            "computes.**"
        )
        st.pyplot(_render_assembler_example_chart())
        st.caption(
            "Stacked bars show which template (T1 blue, T2 green, T3 orange) "
            "contributed how much to each variant under the ILP-balanced assignment. "
            "Σ on top is the variant's total lexical delta. With 164 templates and "
            "more concrete tasks per template the same idea scales up — usually "
            "achieving range = 0.0000 because the search space is large enough."
        )

        st.markdown("#### Step 1: Capacity check")
        st.markdown(
            "Before any optimization, `_check_capacity(groups, n)` walks every "
            "template and asserts that the number of non-baseline concrete tasks "
            "(`len(group.concrete_tasks) - 1`) is at least N. If any template falls "
            "short, the function raises `ValueError` with the offending template IDs "
            "and the build refuses to proceed. The baseline (`#000`) is excluded by "
            "construction so it never lands in any variant."
        )

        st.markdown("#### Step 2: Decision variables")
        st.latex(r"x_{t,i,v} \in \{0, 1\}, \quad t = 1..T, \; i = 1..K_t, \; v = 1..N")
        st.markdown(
            "One binary variable per *(template t, non-baseline concrete-task index i, "
            "variant v)*. The semantics:\n"
            "- $x_{t,i,v} = 1$ means concrete task $i$ of template $t$ goes into variant $v$.\n"
            "- $K_t$ = number of non-baseline concrete tasks PICT + materializer produced for template $t$.\n"
            "- Index $i$ ranges over $\\{1, \\dots, K_t\\}$ — index 0 (the baseline) is never a decision variable."
        )

        st.markdown("#### Step 3: Constraints")
        st.markdown(
            "Two hard constraints, both encoded as linear equalities/inequalities so "
            "CBC can solve them in milliseconds."
        )
        st.markdown("**(a) Coverage** — every (template, variant) cell gets exactly one concrete task:")
        st.latex(r"\sum_{i=1}^{K_t} x_{t,i,v} \;=\; 1 \quad \forall\, t, v")
        st.markdown(
            "So if you have 164 templates and N = 5, every variant ends up with exactly "
            "164 concrete tasks (one per template)."
        )
        st.markdown("**(b) Strict exclusivity** — each concrete task appears in at most one variant:")
        st.latex(r"\sum_{v=1}^{N} x_{t,i,v} \;\le\; 1 \quad \forall\, t, i")
        st.markdown(
            "Combined with the capacity check, every PICT-generated rewording of a "
            "given template that *is* used appears in exactly one variant — no leakage "
            "between V1 and V2."
        )

        st.markdown("#### Step 4: Objective")
        st.markdown("First, define each variant's lexical-delta total:")
        st.latex(r"S_v \;=\; \sum_{t,i} x_{t,i,v} \cdot \delta_{\mathrm{lex}}(t, i)")
        st.markdown(
            "where $\\delta_{\\mathrm{lex}}(t, i)$ is the per-concrete-task lexical "
            "delta computed earlier by `humaneval_t/delta.py` (Levenshtein-based). "
            "We want all the $S_v$ to be as equal as possible. We do **range** "
            "minimization rather than variance because range is linear and CBC handles "
            "linear ILPs instantly; variance would be quadratic and need a different "
            "solver. Introduce two helper variables:"
        )
        st.latex(r"z_{\max} \;\ge\; S_v \quad \forall\, v")
        st.latex(r"z_{\min} \;\le\; S_v \quad \forall\, v")
        st.markdown("Then minimize their gap:")
        st.latex(r"\text{minimize } \quad z_{\max} - z_{\min}")
        st.markdown(
            "CBC squeezes $z_{\\max}$ down and $z_{\\min}$ up subject to the "
            "constraints, which forces the $S_v$ together. When CBC returns "
            "*Optimal*, the achievable range is what you see in the *Lex range "
            "(max - min)* metric on the variants page — frequently `0.0000` because "
            "the search space is small enough that perfect balance is reachable."
        )

        st.markdown("#### Step 5: Extracting the assignment")
        st.markdown(
            "After solve, the function reads back `pulp.value(x[t, i, v]) > 0.5` for "
            "each (t, i, v) and collects the chosen concrete tasks into "
            "`AssembledVariant` objects. It also recomputes the totals for all three "
            "deltas (lexical, semantic, difficulty) from the chosen instances so the "
            "manifest shows the *actual* sums, not just the objective-relevant ones. "
            "Semantic and difficulty are tracked but **not optimized** — that's why "
            "their per-variant totals can vary."
        )

        st.markdown("#### Tuning the build (N, seed, time limit)")
        st.markdown(
            "Three knobs in the *Build benchmark* sidebar expander (and the CLI):\n\n"
            "- **N — number of variants** (default 5, recommended range 3–10). Match the paper at "
            "5 unless you have a specific reason to go up or down. Larger N means each template "
            "needs more PICT-generated concrete tasks (the capacity check fails if any template "
            "has fewer than N non-baseline rows).\n"
            "- **seed** (default 42). Passed to CBC as `randomCbcSeed`. Same seed + same templates ⇒ "
            "byte-identical V\\*.json across runs. Change it only if you want to explore an "
            "alternative optimum (different seeds traverse the branch-and-bound tree in different "
            "orders and may converge on different but equally-optimal assignments).\n"
            "- **time limit (s)** (default 60, recommended range 30–300). Max wall-clock seconds "
            "CBC may spend searching. If a build reports a non-zero `Lex range`, raise it: CBC will "
            "keep searching for a tighter optimum. For 164 templates × ~15 concrete tasks × N=5 "
            "the global optimum is usually reachable in well under a minute, but a different seed "
            "may need more time to converge."
        )

        st.markdown("#### Why range and not variance")
        st.markdown(
            "- **Range** (`max − min`) is the difference of two linear expressions → "
            "still linear after the $z_{\\max}, z_{\\min}$ trick → solved by CBC.\n"
            "- **Variance** ($\\frac{1}{N}\\sum (S_v - \\bar{S})^2$) is quadratic → "
            "needs a QP solver and would take seconds-to-minutes instead of "
            "milliseconds for the same N.\n"
            "- For our use case, range and variance give effectively the same "
            "assignment when the achievable range is tiny (often 0), so range is the "
            "cheaper, equally good choice."
        )

        st.markdown("#### Why not balance semantic and difficulty too?")
        st.markdown(
            "It would mean a multi-objective ILP (Pareto frontier or weighted sum), "
            "which is doable but adds complexity. The paper's primary leakage signal "
            "is lexical, so the current single-objective formulation is what's "
            "implemented. Semantic and difficulty are computed and surfaced in the "
            "manifest as **guardrails** — non-zero deviations or anomalies are "
            "diagnostic, not optimization targets. Adding semantic to the objective "
            "is a future-work hook in `assembler.py`."
        )

    with st.expander("Buttons and fields in *Author templates*"):
        st.markdown(
            """
**Sidebar (top → bottom):**

- **View radio** — switch between this view, the variants browser, and the guide.
- **Finalized N/164** — how many templates you've saved.
- **filter** — show all problems, only `todo`, only `final`, or only `skipped`.
- **problem dropdown** — pick a HumanEval problem. Badges: `[F]` finalized, `[S]` skipped, `[-]` to-do.
- **Build benchmark expander** — when ready, set N and click *Build now*. Writes variants to `extended/output/` inside the repo.

**Main page, left side:** read-only view of the original HumanEval problem (prompt, canonical solution, test). For reference only.

**Main page, right side — the editor:**

- **notes** — free-form notes for your future self.
- **Template variables table** — one row per variable. Each has a name, a kind (`identifier` for code symbols, `lexical` for docstring wording), and a comma-separated list of values (first one is the original).
  - **del** — removes that template variable.
  - **+ add variable** — adds a new variable. After clicking this, rename it, set the kind, and list the values.
- **The workflow for adding a variable:** open the *templated_prompt* expander, find the word you want to vary, wrap it in `<...>` (e.g. `number` → `<number_word>`). Then click *+ add variable* in the table above, name it `number_word`, and list alternatives like `number, value, decimal, float`. PICT will then generate combinations.
- **PICT constraints** — one rule per line. Used to forbid nonsensical combinations.
- **templated_prompt** — the original prompt with `<variable_name>` placeholders inserted where you want substitution to happen. Edit this directly.
- **templated_solution** — same for the canonical solution. Only `identifier` template variables may appear here.

**Action buttons (right side, in a row):**

- **Save as final** — validates the template and writes it to `templates/HumanEval_*.json`. Sidebar updates to `[F]`.
- **Skip problem** — mark this problem as not-templatable; it won't be required for the benchmark. Sidebar updates to `[S]`.
- **Reset** — throws away your current edits in this session and starts over with just the auto-extracted `fn`/`arg` identifier variables.

**Bottom of page — PICT preview:**

- **Run PICT** — feeds the current template to PICT and shows how many concrete tasks come out. Also runs the materializer.
- **PICT model file** (expander) — the actual text passed to `pict.exe`. Useful for debugging constraints.
- **Combination table** — first 10 PICT rows.
- **concrete task to preview** slider — drag to see any of the materialized concrete tasks in full.
            """
        )

    with st.expander("Buttons and fields in *Build manually*"):
        st.markdown(
            """
This view is the manual alternative to the sidebar's *Build now* button. The auto build uses an optimizer that minimizes per-variant lexical-delta range under strict exclusivity. The manual builder lets you override that — useful if you want a specific concrete task in a specific variant.

- **Variants (N)** — how many variants you're assembling. Defaults to 5.
- **Progress bar** — which template you're currently looking at.
- **← Prev / Next →** — navigate between finalized templates.
- **Reset all assignments** — clears every assignment across every template and sends you back to template #1.
- **Available concrete tasks** table — every concrete task PICT can produce from the current template, with its lexical and semantic deltas vs the baseline. Baseline (`#000`) is shown but not assignable — it IS the original HumanEval problem.
- **Preview the prompt of a concrete task** (expander) — pick any concrete task ID to read its full materialized prompt.
- **V1, V2, …** dropdowns — for each variant, choose one of the assignable concrete tasks (or *skip* to leave that variant without this template). If you accidentally pick the same concrete task for two variants you get a warning and the build refuses to proceed.
- **Build variants from current assignments** — writes V1.json … VN.json + manifest.json to `output/`. The `build_mode` field in the manifest is set to `manual` so you can tell apart manual vs auto builds later.
            """
        )

    with st.expander("Buttons and fields in *Browse built variants*"):
        st.markdown(
            """
- **Top metrics** — count of variants, templates per variant, lexical-delta range across variants (lower is more balanced), mean lexical delta.
- **Manifest summary table** (expander) — per-variant totals of all three deltas + concrete-task count.
- **Inspect a variant** dropdown — pick V1, V2, …
- **Variant detail metrics** — totals for the chosen variant.
- **Concrete task to inspect** dropdown — pick one concrete task from this variant.
- **Tabs** — Prompt / Canonical solution / Test for the chosen concrete task.
- **File path** — where this variant lives on disk so you can open it in another tool.
            """
        )

    with st.expander("Where files live"):
        st.markdown(
            f"""
All paths below are relative to the repo's `extended/` directory:

- **HumanEval source data:** `data/HumanEval.jsonl`
- **Your finalized templates:** `templates/HumanEval_*.json`
- **Skipped task list:** `templates/skipped.json`
- **Built benchmark variants:** `output/` (V1.json, V2.json, …, manifest.json) — gitignored
- **PICT binary:** `tools/pict/pict.exe` (committed to the repo so it travels with a clone). Override location by setting the `HUMANEVAL_T_PICT` env var, or by putting `pict` on the system PATH.
- **Python virtualenv:** `.venv/` — gitignored, created during one-time setup
            """
        )

    with st.expander("Workflow — the loop"):
        st.markdown(
            """
1. Pick a HumanEval problem in the sidebar.
2. Look at the original on the left.
3. On the right, decide: which words can be reworded without changing what the problem is asking?
4. For each word, do two things: (a) open the *templated_prompt* expander and wrap the word in `<name>` (e.g. `number` → `<number_word>`); (b) click *+ add variable* in the table, name it `number_word`, set kind (`lexical` for docstring wording, `identifier` for code symbols), and list 2–4 alternative words comma-separated (first one = the original).
5. (Optional) Edit the existing identifier template variables (`fn`, `arg1`, …) in the table to add alternatives for the function/argument names — those are the strongest leakage probes.
6. (Optional) Add constraints if some value combinations would be incoherent.
7. Click **Run PICT** at the bottom. Aim for at least N rows (default N=5) so the template can supply enough rewordings for every variant.
8. Drag the *concrete task to preview* slider through a few rewordings to sanity-check they still make sense.
9. Click **Save as final**.
10. Move to the next problem.
11. Once you have at least N templates done, open the *Build benchmark* expander in the sidebar, set N, click **Build now**. Variants appear in `extended/output/`. Switch to the *Browse built variants* view to inspect them.
            """
        )

    with st.expander("Common errors and what to do"):
        st.markdown(
            """
- **"Template variable 'X' declared but never referenced as `<X>`"** — you added the variable but didn't put `<X>` anywhere in the templated_prompt/solution. Either insert the placeholder or delete the variable.
- **"template variable 'X': value 'foo bar' is not a valid Python identifier"** — identifier variables can't have spaces. Use underscores (`foo_bar`) or change the kind to *lexical*.
- **"lexical template variable 'X' must not appear in templated_solution"** — you put a lexical `<X>` inside the Python code. Move it to the docstring only, or change the kind to *identifier* if it really is a code symbol.
- **"insufficient PICT capacity for N=5: HumanEval/2 has 0 non-baseline"** — a template you finalized has only one value per variable, so PICT can't produce any combinations. Open it and add more values, or reduce N.
- **PICT constraint syntax error** — make sure each line ends with `;`, names are in `[brackets]`, and values are in `"double quotes"`.
            """
        )

    with st.expander("FAQ"):
        st.markdown(
            """
**Do the rewordings have to all mean exactly the same thing?**
Yes — that's the whole point. If a rewording changes the task, the original tests won't pass and the rewording is silently broken. The *semantic delta* number flags when a rewording drifts too far.

**Why does the function name change too?**
Renaming the function is one of the strongest leakage probes. A model that has memorized `has_close_elements` won't recognize `any_within_tolerance` even though they do the same thing.

**Can I edit a template after saving it?**
Yes. Pick it from the sidebar; the editor loads the saved version. Edit, click *Save as final* again to overwrite.

**What happens to skipped problems?**
They're ignored by the build step. If you skip 16 of 164, your benchmark has 148 problems per variant, and you should benchmark against the same 148-problem subset of the original HumanEval to keep the comparison fair.

**Can I run the benchmark against an LLM from inside this app?**
Not yet — the LLM solver wiring is a separate piece. The build step writes the variants in a format ready for evaluation; you run them with the evaluator module (or your own harness) once you're ready.
            """
        )


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="HumanEval_T extended", layout="wide")
    view = render_view_picker()
    st.sidebar.divider()
    if view == "Author templates":
        render_author_view()
    elif view == "Build manually":
        render_manual_builder_view()
    elif view == "Browse built variants":
        render_variants_view()
    else:
        render_guide_view()


if __name__ == "__main__":
    main()
