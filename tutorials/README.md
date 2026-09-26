# Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/tutorials/pix2struct_docvqa_colab.ipynb)

Notebook specification: **DIMER Notebook Specification 2.0**. The notebook is a **standalone** (§4), generated carrier; edit the package or `tools/notebook_template.py`, then run `python tools/build_notebook.py`. Do not edit the notebook by hand.

| Notebook | Profile | Mode | Carrier | Default path | Runtime | BYOD | Release status |
|---|---|---|---|---|---|---|---|
| `pix2struct_docvqa_colab.ipynb` | `E2E` | `GUIDED` | standalone; carries `pipeline.py`, `metrics.py`, and `samples.py` | deterministic 72/18/30 document-QA split; empty/majority/frozen baselines; final-two-decoder-block adaptation; held-out ANLS/exact match; safetensors export and fresh reload parity | CUDA GPU; clean Kaggle Tesla T4 run recorded 2026-09-26 | optional zip with `records.csv` and page images; gated off by default | **Release-grade** — commit `2602e66` / blob `078435c42445` passed on Kaggle Tesla T4 (10/10 code cells); record in [`docs/release-verification.md`](../docs/release-verification.md) |

The generated corpus contains fictional business documents and is created locally in the runtime. Documents, rather than question rows, define split membership. The notebook records point estimates from one seeded synthetic split and makes no DocVQA benchmark or real-document claim.

The default path needs no repository clone, credential, upload, or external dataset. It stages only the immutable eight-file model snapshot, verifies every recorded size and SHA-256, refuses CPU for adaptation, and exports `outputs/pix2struct_docvqa_input_manifest.json`, `outputs/pix2struct_docvqa_result.json`, `outputs/pix2struct_docvqa_predictions.csv`, and `outputs/pix2struct_docvqa_adapter/{adapter.safetensors,manifest.json}`.

`tools/validate_release_assets.py`, `tools/build_notebook.py --check`, unit tests, and code-cell compilation are source checks. They are not clean-runtime execution evidence; a changed notebook blob returns to Candidate until its own clean run is recorded in `docs/release-verification.md`.

## AI Assistance Disclosure

This repository’s code and documentation were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for review, validation, and release decisions.
