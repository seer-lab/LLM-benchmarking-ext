# HumanEval_T

Two versions of the work live here:

- **`old/`** — workshop paper artifacts ([arXiv 2412.01526](https://arxiv.org/abs/2412.01526)): the original 10-problem subset, meta-prompts, and v1 evaluator.
- **`extended/`** — **current work.** Hand-crafted templates + PICT-driven pairwise expansion across all 164 HumanEval problems, plus a Streamlit authoring/inspection UI.

## Run

```powershell
pip install -r extended/requirements.txt
streamlit run app.py
```

For full setup, CLI, and layout details, see [`extended/README.md`](extended/README.md).
