# Release verification

`tutorials/pix2struct_docvqa_colab.ipynb` is a standalone `E2E` carrier that is **Release-grade** for the exact commit and notebook blob recorded below, and returns to **Candidate** whenever the blob changes, until that exact blob executes top-to-bottom in a clean supported runtime. Static checks, unit tests, code-cell compilation, and generator parity are necessary but are not runtime evidence.

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
| `pix2struct_docvqa_colab.ipynb` (`E2E`) | `2602e66` / `078435c42445` (notebook `NOTEBOOK_SOURCE.repository_revision` `8a43e98`, the source revision the notebook was bound to; `8a43e98..2602e66` changes only the notebook; embedded `module_sha256` `f41e4c163285…`) | 2026-09-26 (03:22–03:29 UTC) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pix2struct-docvqa` v3), serial suite, `USE_BYOD = False`, sample-path defaults; blob fetched at the 40-char SHA and Git-blob verified; HF cache clean at start | **PASSED** — 393.2 s wall (pass 1 181.1 s stopped at the install cell with a pip dependency-resolver `CellExecutionError` and the notebook's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3), kernel restarted after the install cell; pass 2 212.1 s), 10/10 post-restart code cells ok; image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` (image torch 2.10.0+cu128, transformers 5.0.0, pillow 11.3.0), Python 3.12.13, Tesla T4 15360 MiB, driver 580.159.04; after the inline pins: torch 2.14.0+cu130 (CUDA 13.0), transformers 4.57.6, pillow 11.3.0, device `cuda:0`, float32; staged 18 files / 1133 MB (8 manifest files at `63f6b3de436e39f75c7a486881a9c2c14a7f4e89`, 1,133,308,924 B, all size- and SHA-256-verified, incl. `model.safetensors` 1,129,177,976 B); synthetic corpus digest `b9e7b0b27ae120979c988da745c8d0324de4267925f8c53b7a2f7f66317a438a` equal to the pinned `SAMPLE_DIGEST` on Kaggle's Pillow 11.3.0, split 72 / 18 / 30 rows, document-disjoint; duplicate-id and document-leakage probes refused; trainable 18,878,976 of 282,285,696 parameters (final two decoder blocks), 2 epochs, lr 0.0002, batch 1, seed 0, best epoch 1 by validation ANLS (validation n = 18: frozen ANLS 0.9815 / EM 0.9444; epoch 1 ANLS 1.0 / EM 1.0, train loss 0.2374; epoch 2 ANLS 1.0 / EM 1.0, train loss 0.2676), adaptation 94.7 s; held-out test (30 rows, `max_new_tokens` 32): ANLS empty 0.0 / majority 0.2949 / frozen 1.0 / adapted 1.0; exact match 0.0 / 0.0 / 1.0 / 1.0; delta vs frozen ANLS 0.0, exact match 0.0 (the frozen model already answers every synthetic test question exactly, so this sample cannot show an adaptation gain); frozen evaluation 21.222 s, adapted 19.596 s; `adapter.safetensors` 75,519,448 B sha256 `b8b49eccff0e8b754e71de9f7fd07644b98f6791255cd418c517338efa09be5c`, fresh reload answer parity 8/8 identical; preserved output sha256: `pix2struct_docvqa_result.json` `97ef8f96d996…`, `pix2struct_docvqa_predictions.csv` `a17e155b79fa…`, `pix2struct_docvqa_input_manifest.json` `5afd17338c84…`, adapter `manifest.json` `cab0afb91132…`; no warnings in cell stderr. One seeded synthetic split, one runtime, no dispersion estimate |
| Superseded `E2E` attempt | `14f2cbd` / `285a26d12034` | 2026-09-26 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pix2struct-docvqa` v2) | **FAILED** — 227.9 s, 4/10 code cells ok after the install restart; model-staging cell 11 raised `TypeError: object supporting the buffer API required` because the standalone notebook's embedded `samples._sha256` shadowed `pipeline._sha256`; fixed by the rename in `8a43e98` and the regenerated notebook in `2602e66`. Not evidence for the current blob |

## Promotion rule

The current carrier holds **Release-grade** for commit `2602e66` / notebook blob `078435c42445` on the passing clean run recorded above; the measured values are one seeded synthetic split on one runtime, not a DocVQA benchmark. Keep **Candidate** for any carrier whose exact blob has no passing clean-run row. A later change to any carried module, notebook template, dependency pin, manifest, or generated notebook creates a new blob and invalidates the prior runtime evidence. Promotion updates this file, `STATUS.md`, `README.md`, and `tutorials/README.md` in one evidence-only commit; merge remains an explicit maintainer action.
