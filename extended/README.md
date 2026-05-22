# HumanEval_T extended

Hand-crafted templates + PICT-driven pairwise combinatorial expansion of every HumanEval problem, for leakage-resistant benchmarking.
Extension of the workshop paper *Addressing Data Leakage in HumanEval Using Combinatorial Test Design* ([arXiv 2412.01526](https://arxiv.org/abs/2412.01526)) from 10 problems to all 164.

## Setup

```powershell
cd HumanEval_T
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r extended/requirements.txt
```

## Run

```powershell
# from the repo root (HumanEval_T/)
streamlit run app.py
```

The Guide view inside the app explains every screen and every metric.

## CLI build

```powershell
cd HumanEval_T/extended
python -m humaneval_t.build_benchmark --variants 5
```

Variants land in `output/V*.json` + `manifest.json`.

## Layout

| Path | Purpose |
|---|---|
| `humaneval_t/` | Python package — schema, PICT wrapper, materializer, delta, assembler, evaluator, Streamlit app |
| `templates/` | Finalized templates (one JSON per HumanEval problem; paper subset has `paper_subset: true`) |
| `data/HumanEval.jsonl` | The 164 original problems |
| `tools/pict/pict.exe` | Bundled PICT 3.3 binary (Windows) |
| `output/` | Built benchmark variants (gitignored) |
| `../app.py` | Streamlit launcher (lives at the repo root, one level up) |

## Vocabulary 
| Term | Meaning |
|---|---|
| **template task** | Recipe for one HumanEval problem with placeholder variables |
| **template variable** | One placeholder + its allowed values; first value is the original |
| **concrete task** | One materialized rewording of a template (one PICT row applied) |
| **benchmark variant** | A complete benchmark, one concrete task per template, with strict exclusivity across variants |
