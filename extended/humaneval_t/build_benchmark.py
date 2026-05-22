"""Build N benchmark variants from every finalized template.

End-to-end pipeline runner:
  final templates -> PICT -> materialize -> delta -> assemble -> variants/V*.json

CLI:
    python -m humaneval_t.build_benchmark
    python -m humaneval_t.build_benchmark --variants 5
    python -m humaneval_t.build_benchmark --variants 7 --time-limit 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humaneval_t.assembler import TemplateGroup, assemble_variants
from humaneval_t.delta import compute_deltas
from humaneval_t.materializer import materialize_all
from humaneval_t.pict_io import run_pict
from humaneval_t.schema import Template

ROOT = Path(__file__).resolve().parents[1]  # extended/
FINAL_DIR = ROOT / "templates"
OUT_DIR = ROOT / "output"  # extended/output/  — gitignored, repo-local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", "-n", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=60, help="ILP solver time limit (s)")
    parser.add_argument("--order", type=int, default=2, help="PICT order (default: 2 = pairwise)")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="CBC random-seed for reproducible tie-breaking when multiple ILP optima exist (default: 42)",
    )
    args = parser.parse_args()

    paths = sorted(FINAL_DIR.glob("HumanEval_*.json"))
    if not paths:
        raise SystemExit(f"no finalized templates under {FINAL_DIR}")

    print(f"loading {len(paths)} finalized template(s) ...")
    groups: list[TemplateGroup] = []
    for path in paths:
        template = Template.model_validate(json.loads(path.read_text(encoding="utf-8")))
        rows = run_pict(template, order=args.order)
        concrete_tasks = materialize_all(template, rows)
        deltas = compute_deltas(concrete_tasks)
        groups.append(
            TemplateGroup(task_id=template.task_id, concrete_tasks=concrete_tasks, deltas=deltas)
        )
        print(
            f"  {template.task_id}: {len(concrete_tasks)} concrete task(s)  "
            f"({len(concrete_tasks) - 1} non-baseline)"
        )

    print(f"\nassembling N={args.variants} variants (seed={args.seed}) ...")
    variants = assemble_variants(
        groups, n=args.variants, time_limit_s=args.time_limit, seed=args.seed
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for v in variants:
        out = OUT_DIR / f"{v.variant_id}.json"
        out.write_text(json.dumps(v.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(
            f"  {out.relative_to(ROOT)}  -> {len(v.concrete_tasks)} task(s), "
            f"tot_lex={v.total_lexical:.3f}"
        )

    manifest = {
        "n_variants": args.variants,
        "seed": args.seed,
        "templates": [
            {"task_id": g.task_id, "concrete_task_count": len(g.concrete_tasks)} for g in groups
        ],
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
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    lex = [v.total_lexical for v in variants]
    print(f"\nlexical range (max-min) = {max(lex) - min(lex):.4f}")
    print(f"manifest -> {(OUT_DIR / 'manifest.json').relative_to(ROOT)}")
    print("done.")


if __name__ == "__main__":
    main()
