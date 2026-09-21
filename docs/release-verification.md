# Release verification

`tutorials/pix2struct_docvqa_colab.ipynb` is a standalone `E2E` Candidate until its exact committed source SHA and notebook blob execute top-to-bottom in a clean supported runtime. Static checks, unit tests, code-cell compilation, and generator parity are necessary but are not runtime evidence.

## Automatic source coverage

CI and local pre-flight must verify:

- all three carried modules match `src/pix2struct_docvqa_pipeline/` after only documented standalone rewrites;
- the inline model manifest equals the committed eight-file manifest for `google/pix2struct-docvqa-base` at `63f6b3de436e39f75c7a486881a9c2c14a7f4e89`;
- runtime dependencies are exact pins, the notebook has no repository dependency, every code cell compiles, and no outputs or execution counts are committed;
- corpus generation reproduces `SAMPLE_DIGEST`, train/validation/test documents are disjoint, and duplicate-id and leakage probes are refused;
- the notebook calls the public baseline, frozen evaluation, adaptation, held-out evaluation, artifact export, fresh reload, and answer-parity paths;
- `tools/validate_release_assets.py`, `python tools/build_notebook.py --check`, `ruff check src tests tools`, and `pytest -q -o addopts= tests` pass.

## Clean-runtime gate

The serial Kaggle GPU suite must:

1. resolve a pushed 40-character commit SHA and notebook blob SHA before submission;
2. use a fresh Kaggle kernel with Internet enabled and an explicit Tesla T4 accelerator;
3. execute the committed notebook verbatim, allowing only the executor shim required by the suite;
4. confirm the notebook records the same source revision and uses the embedded immutable model manifest;
5. confirm all ten code cells complete after any required dependency-install restart;
6. retain the input manifest, comparison metrics, training history, predictions CSV, adapter manifest and weights, and 8/8 reload parity;
7. report runtime identity, wall time, staged files/bytes, and any warnings or retries.

A run may pass structurally even if adapted held-out metrics regress; the negative delta must remain in the evidence. Promotion requires successful completion and faithful evidence, not a preselected metric gain.

## Recorded executions

| Notebook | Source SHA / blob | Date | Runtime | Result |
|---|---|---|---|---|
| `pix2struct_docvqa_colab.ipynb` (`E2E`) | pending committed carrier | pending | Kaggle Tesla T4 | **PENDING** — no local model/GPU run; scheduled for the serial GPU batch |

## Promotion rule

Keep **Candidate** until the row above records a passing clean run of the exact blob. A later change to any carried module, notebook template, dependency pin, manifest, or generated notebook creates a new blob and invalidates the prior runtime evidence. Promotion updates this file, `STATUS.md`, `README.md`, and `tutorials/README.md` in one evidence-only commit; merge remains an explicit maintainer action.
