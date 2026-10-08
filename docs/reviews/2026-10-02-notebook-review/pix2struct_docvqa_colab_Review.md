# Notebook review: `tutorials/pix2struct_docvqa_colab.ipynb`

Notebook Review Framework v1 review against DIMER Notebook Specification 2.2. Review only: nothing in the
repository was changed. Finding prefix: **PSD**.

## 1. Scope and evidence

### Review contract

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/pix2struct-docvqa-pipeline` |
| Notebook | `tutorials/pix2struct_docvqa_colab.ipynb` |
| Reviewed revision | `origin/main` = `04c9d77c59bcc4f4765acedc7ef289417a7dfbb9` (notebook blob `078435c42445b13b1917a8577aa81c70b091ea10`). PR #6, the E2E carrier PR, is already merged into this revision |
| Generated from | `metadata.dimer.generated_from.revision` `8a43e9831650…`, `tools/build_notebook.py` / `tools/notebook_template.py`. `build_notebook.py --check` reports the notebook up to date |
| Requirements baseline | NOTEBOOK_SPEC **2.2** (ml-worker `origin/main`). The notebook declares **2.0** |
| Profile / mode | `E2E` / `GUIDED` (metadata and opening cell) |
| Intended audience | Stated as knowing Python, train/validation/test splits, autoregressive generation, and why held-out ANLS is not a confidence score (Prerequisites) |
| Supported runtime | Colab or a Kaggle-style Python 3.12 runtime **with a CUDA GPU**. The notebook refuses CPU for adaptation |
| Promised outcomes | Stage and digest-verify the pinned 8-file snapshot; generate and validate a deterministic synthetic document-QA corpus with document-disjoint splits; empty and majority baselines plus the frozen model; bounded adaptation of decoder blocks 10–11 with validation-ANLS epoch selection; held-out ANLS and exact match; safetensors adapter export, fresh reload and answer parity; optional BYOD through the same validation and split |

### Existing execution evidence

`docs/release-verification.md` records a **PASSED** Kaggle Tesla T4 run of commit `2602e66` / blob `078435c42445`.
That blob is the one under review, so the record covers this revision. The run was "10/10 **post-restart** code
cells". Pass 1 (181.1 s) stopped at the install cell with a pip resolver `CellExecutionError` and the notebook's
stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3). The kernel was restarted and pass 2
(212.1 s) completed. The record also shows **frozen held-out ANLS 1.0 / exact match 1.0, adapted 1.0 / 1.0**, and
frozen validation ANLS 0.9815. No Colab run of this blob is recorded. No BYOD run and no run of a "Next
experiment" are recorded.

### Journeys and evidence basis

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection of all 23 cells | Barriers found: PSD-M2, PSD-M4, PSD-m4 |
| Clean default | Documented execution evidence (Kaggle T4, exact blob). Not run here: there is no local GPU by workspace rule, and the notebook refuses CPU | Completes only after a manual restart (PSD-M1). The frozen model already scores 1.0 on the test split, so the comparison shows nothing (PSD-M2). **Colab not verified** |
| Active learning | Source inspection, plus direct execution at **reduced scale**: a random 12-decoder-block, 32-wide Pix2Struct built from the committed config, on CPU, calling `adapt` twice as a rerun of Section 6 does | No rerun instructions exist. Rerunning Section 6 before Section 8 stacks adaptation, and the artifact stops reproducing the in-memory model. Rerunning after Section 8 hits a `NameError` (PSD-M3). Full scale: not verified |
| Reuse and recovery | Direct execution of the BYOD branch's verbatim load/split/validate logic with a stubbed upload (no model). Artifact reload: documented evidence (8/8) | A valid zip with nested folders is accepted and split. Missing `records.csv` and a missing column give actionable messages. A cancelled upload, a non-image page and a too-small dataset do not (PSD-m2). The documented "set `USE_BYOD = True` after the default completes" path fails at Section 5 unless the learner reruns from Section 3 (PSD-M3). BYOD with the real model: **not verified** |

Limitations: no learner observation. No Colab execution. Full-scale numbers come from the repository record and
were not reproduced. Repository checks run here are source checks, not execution evidence (REL8):
`build_notebook.py --check` exited 0 and `validate_release_assets.py` reported PASS. `pytest -q -o addopts= tests`
gave 39 passed and 1 failed. The failure is `test_generated_sample_is_pinned_and_document_disjoint`, because the probe
environment has Pillow 12.3.0, not the pinned 11.3.0. The corpus probe confirms the digest is Pillow-version
dependent (`5c793fb2…` under 12.3.0 against the pinned `b9e7b0b2…`). This is an environment artifact, not a finding.

### Promise → evidence trace

| Claim | Implementation | Observable result | Learner interpretation |
|---|---|---|---|
| Run all completes in a fresh GPU runtime (opening cell, "Run all") | Cell 3 pinned install plus stale-module guard | Kaggle T4: restart required, then pass 2 OK | Contradicted by the documented run (PSD-M1) |
| Pinned, digest-verified snapshot | Cell 11 `stage_missing_files` + `verify_snapshot` | Documented: 8 files, 1,133,308,924 B verified | Delivered |
| Deterministic corpus, document-disjoint split, refusal probes | Cell 13 | Documented: 72/18/30, digest equal to `SAMPLE_DIGEST`, 2 refusals | Delivered |
| Baselines and frozen model | Cell 15 | Documented: empty 0.0, majority 0.2949, frozen **1.0** ANLS | Delivered, but the frozen model is already perfect (PSD-M2) |
| Bounded adaptation with validation selection | Cell 17 `pipe.adapt` | Documented: best epoch 1, decided by validation 0.9815 → 1.0 (one of 18 rows) | Runs. Cannot show an effect on this sample (PSD-M2). Breaks on rerun (PSD-M3) |
| Held-out comparison with deltas | Cell 19 | Documented: delta 0.0 / 0.0 | Nothing to interpret. The notebook does not say why (PSD-M2) |
| Export, fresh reload, answer parity | Cell 21 | Documented: 8/8 identical answers | Load is digest- and tensor-checked. The answer-parity check cannot tell adapted from base on this sample (PSD-m1) |
| BYOD after the default completes | Cell 13 `USE_BYOD` branch | Valid zip reaches the split (probe). Rerunning from Section 4 then Section 5 hits `NameError: pipe` | Partly delivered (PSD-M3, PSD-m2) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Verify model and corpus identities | Printed dicts in Sections 3–4 | Shown. No question asks the learner to read them (PSD-M4) |
| Inspect record and split contracts | Section 4 manifest and refusal probes | Shown. No activity (PSD-M4) |
| Compare baselines with the frozen model | Section 5 print | The frozen model is at ceiling, so the comparison teaches only "the model already knows this" (PSD-M2) |
| Run bounded decoder adaptation | Section 6 constants | Exercised on the default path. No guided change. Changing a value and rerunning gives invalid results or a `NameError` (PSD-M3) |
| Interpret ANLS/EM as one seeded synthetic estimate | Section 7 plus "Interpretation and limits" | Caveats are well written. The 1.0 / 1.0 result is never explained (PSD-M2) |
| Export, verify and reload the adapter | Section 8 | Exercised. The parity check is weak on this sample (PSD-m1) |

## 2. Separate judgments

- **Technical correctness:** the default path is carefully engineered: an immutable revision, per-file digests,
  `trust_remote_code=False`, a bundled CC0 header font instead of an unpinned Hub font download, transactional
  `adapt`, and manifest, digest and tensor-set checks before an adapter loads. The BYOD loader resolves files by
  basename, so zips with folders work. There are two defects. The documented install restart is a `MUST` failure.
  `adapt` mutates the shared `pipe` and starts from its current weights, which together with `del pipe` in
  Section 8 makes every rerun either invalid or a `NameError`.
- **Promise fulfilment:** identity, corpus, split, baseline, export and reload promises are met by documented
  evidence. "Run all completes" is not met. The adaptation and held-out comparison runs, but on this corpus it
  cannot show an adaptation effect, and the notebook does not tell the learner. BYOD is reachable only through an
  undocumented rerun route.
- **Learner experience:** the limits and the honest "not a benchmark" framing are clear. But the GUIDED learner
  meets about 52,000 characters of unlabelled carried module code before any model step. There is no How-to-use
  section, roadmap, glossary, troubleshooting, prediction, checkpoint, coded activity or conclusion scaffold. The
  learner's main numeric result (frozen 1.0, adapted 1.0, delta 0) arrives without the explanation that STATUS.md
  and README.md give ("the frozen model already saturates this sample").
- **Spec conformance:** fails the `MUST`s RUN1, RUN10 and ENV6 (restart). It also fails DAT19 for the cancelled-upload,
  non-image and too-small BYOD cases. It declares spec 2.0 rather than 2.2. GDL1–GDL14, EXE2 and EXE5 `SHOULD`
  deviations are not recorded as deviations. All other applicable `MUST`s checked by source inspection appear met on
  the default path: ST, MOD, DAT1–DAT9, VAL, SPL1/SPL5/SPL6/SPL7, FT1–FT8, EVAL1–EVAL7/EVAL14, ART1/ART4/ART5/ART8
  and VER1–VER3.

## 3. Findings

### Major

**PSD-M1 — Section 1 install cell: the default `Run all` requires a manual restart.**
Cell 3 `pip install`s exact pins (`torch==2.14.0`, `numpy==2.5.3`, `transformers==4.57.6`, …) into the running
kernel. If a pinned distribution was already imported, it raises "Restart the runtime, then rerun from the top". The
only recorded clean run of this blob hit exactly that. Kaggle T4 pass 1 stopped at the install cell (`cuda-bindings`,
`numpy`, plus a pip resolver error), and the kernel was restarted before pass 2 completed. The release gate itself
(`docs/release-verification.md`, clean-runtime step 5) accepts "after any required dependency-install restart".
STATUS.md, README.md and `tutorials/README.md` record Release-grade.
- *Consequence:* a learner choosing Run all on a fresh hosted runtime stops at cell 3. A Run all that needs a
  restart is not conformant (§5 closing rule, §25.7), so the Release-grade label overstates the evidence.
- *Evidence:* documented execution evidence (release-verification.md, 2026-09-26 row: "pass 1 181.1 s stopped at
  the install cell … kernel restarted after the install cell; pass 2 212.1 s), 10/10 post-restart code cells").
  Source inspection of cell 3 and `tools/build_notebook.py:40–70` (install body, emitted at ~`:415–426`). Colab: not
  verified.
- *Recommended correction:* adopt the fleet's uv isolated-environment pattern, which is how the capstone and newer
  workshop notebooks already run in one pass. The setup cell bootstraps uv, creates an isolated managed interpreter
  (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with
  `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that
  environment. The kernel's preloaded NumPy/torch are then never replaced, so no restart can be required. Reference
  implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb`
  and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not
  add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py`,
  regenerate, and re-qualify with a one-pass hosted Run all. Then correct the release record and clean-runtime step 5
  so a restart-dependent run is not reported as a `Run all` PASS.
- *Acceptance check:* a fresh Colab (or Kaggle) GPU runtime executes every code cell in order in one kernel, with
  no restart and no error output, recorded against the new blob. `docs/release-verification.md` no longer accepts or
  describes a restart as part of a passing run.
- *Spec:* RUN1, RUN10, ENV6 (MUST); REL2, REL11.

**PSD-M2 — Sections 5–7 and "Interpretation and limits": the synthetic corpus is at ceiling for the frozen model,
so the central frozen-vs-adapted demonstration cannot show anything, and the notebook never says so.**
The corpus is 40 clean, code-drawn invoice pages with three fixed question templates (invoice number, PO number,
total due; probe `corpus.distinct_question_strings`). On the only recorded run the frozen checkpoint already
scored held-out ANLS 1.0 / EM 1.0 and validation ANLS 0.9815. Adaptation produced 1.0 / 1.0 and a delta of 0.0, and
epoch selection was decided by one validation question out of 18. STATUS.md and README.md state that "the frozen
model already saturates this sample". The notebook does not: the probe finds no ceiling or saturation wording in
any markdown cell. Section 7 only says "Improvement is not an execution gate", and the limits section says "A
negative delta is valid evidence".
- *Consequence:* the learner runs bounded fine-tuning and sees no change in any number, with no explanation. The
  learning objectives "compare … baselines with the frozen model", "run bounded decoder adaptation" and "interpret
  ANLS and exact match" cannot be exercised on this result. A plausible wrong conclusion is "adaptation does
  nothing", or that the run is broken. This is framework dimension 3 (the experiment cannot discriminate) and
  dimension 5 (the main output is unexplained). Training loss is the only signal that anything changed, and FT7
  forbids presenting it as quality evidence.
- *Evidence:* documented execution evidence (release-verification.md 2026-09-26 row; STATUS.md). Source inspection
  of `samples.py` `_document` / `build_sample_dataset` and cells 14–22. Static probe `ceiling_warning_in_markdown:
  false`.
- *Recommended correction:* make the default sample leave headroom for the frozen model, and explain the result in
  either case. Options in `src/pix2struct_docvqa_pipeline/samples.py` (then regenerate): add harder question
  templates (date, contact, city, arithmetic or layout-dependent fields), visual noise or rotation, or answer
  formats the base model gets wrong. Re-pin `SAMPLE_DIGEST` and verify on a hosted run that frozen test ANLS is
  clearly below 1.0. Whatever the corpus, add a markdown cell after Section 7 (in `tools/notebook_template.py`
  ~`:163–170`) telling the learner how to read a zero or near-ceiling delta and pointing them to the predictions CSV.
- *Acceptance check:* on a recorded hosted run of the new blob, frozen held-out ANLS is below 0.95 or the notebook
  explicitly explains a ceiling result in its own markdown. The static probe `ceiling_warning_in_markdown` (or an
  equivalent check for the explanation) passes.
- *Spec:* no single ID. Related: EVAL3, EVAL15, FT7. Framework dimensions 1, 3 and 5.

**PSD-M3 — Sections 3–8: there are no rerun instructions; the natural BYOD rerun fails with `NameError`, and a
rerun before Section 8 stacks adaptation and breaks the artifact.**
`pipe` is built only in cell 11 (Section 3) and deleted in cell 21 (Section 8, `del pipe`). Cells 15, 17 and 19 use
it (static probe). `pipe.adapt` (carried `pipeline.py`, `adapt`) trains in place from the pipeline's *current*
weights and still labels epoch 0 `"frozen model"`.
- The BYOD instruction says "After the default sample completes, set `USE_BYOD = True` and upload one zip". It does
  not name the cells to rerun. Rerunning from Section 4 (the cell holding the toggle) reaches Section 5 and raises
  `NameError: name 'pipe' is not defined`, with no recovery hint. Only a rerun from Section 3, or a full Run all,
  works.
- "Next experiments: … compare one versus two trainable decoder blocks" has no code and no rerun route. A learner
  who edits `TRAINABLE_DECODER_LAYERS` and reruns Section 6 before Section 8 continues training the adapted model. Its
  "frozen" epoch 0 is the previous adapter. Block 10 stays modified in memory but is not exported, so the artifact no
  longer reproduces the in-memory model. The 8-answer parity check cannot be relied on to catch this (PSD-m1).
- *Consequence:* the promised BYOD path stops with an unexplained error, or, on the stacking route, the learner's
  "frozen vs adapted" and "1 vs 2 blocks" conclusions are wrong and the exported adapter misdescribes its lineage.
  This is framework dimensions 7 and 8 (an exercise leaves the notebook in an inconsistent state; a promised
  user-data path does not work as written). It is consistent in kind with the sibling `pix2struct-ai2d` review
  (PSA-M2). Here the BYOD route fails loudly rather than silently, which the severity still reflects as Major because
  the promised path is not completable from the instructions.
- *Evidence:* static probe (`del_pipe_cell: 21`, `cells_using_pipe_before_del: [15, 17, 19]`,
  `pipe_built_only_in_cell: [11]`, `byod_rerun_instruction_names_cells: false`). Direct execution at **reduced
  scale** (`stacked_adaptation`): after `adapt(layers=2)` then `adapt(layers=1)`, run 2 did not start from base
  (`run2_started_from_base: false`) and still reported epoch 0 as `"frozen model"`. All 14 `decoder.layer.10.*` tensors
  differ from base, and the artifact reloaded onto a fresh base mismatches the in-memory model on 14 tensors.
  Rebuilding the pipeline, which is what Section 3 does, restores base (`rebuild_pipeline_restores_base: true`).
- *Recommended correction:* make every experiment start from the verified base, and say exactly what to rerun.
  Either rebuild `pipe` with `Pix2StructDocVQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` at the top of
  Section 5 (no download: the snapshot is already verified), or capture the base state of the adaptable tensors after
  Section 3 and restore it before Sections 5 and 6. State in the BYOD text and the "Next experiments" text: "set the
  value, then run from Section 5 to the end" (or Run all). Change `tools/notebook_template.py` around lines 24–27
  (BYOD text), 127–135 (Section 5), 150–160 (Section 6) and 225 (next experiments).
- *Acceptance check:* after a completed default run, setting `USE_BYOD = True` and following the written
  instruction reaches Section 8 without `NameError`, and Section 5 reports `adapted: False`. Rerunning Section 6 with
  `TRAINABLE_DECODER_LAYERS = 1` per the written instruction gives an epoch-0 validation score equal to the frozen
  score, and every non-exported decoder tensor equals base. A reduced-scale probe like `stacked_adaptation` reports
  0 mismatched tensors.
- *Spec:* DAT13 (MUST, same semantics for BYOD), RUN9, UX7, GDL10, VER3–VER5, ART8.

**PSD-M4 — Whole notebook: the declared `GUIDED` layer is largely missing, and about 52,000 characters of carried
code are unlabelled.**
Static probe: no "How to use this notebook", roadmap, glossary, troubleshooting, prediction prompt, collapsible
answers, "what to notice" guidance or conclusion scaffold. There are 0 `cellView: form` cells and no "Infrastructure"
label. The three carried module cells hold 36,605, 2,397 and 13,125 characters and sit between Section 1 and the
first model step. No code cell is a learner activity. The only learner-directed content is one "Next experiments"
sentence with no code.
- *Consequence:* the stated learner can follow the default path but is never asked to apply the stated objectives,
  such as reading the split manifest, predicting the frozen score or explaining an error. The long infrastructure
  cells look like prerequisite knowledge (framework dimensions 4 and 6). Severity is set by learner consequence, and
  rated the same as the sibling PSA-M3. Spec conformance alone is `SHOULD`.
- *Evidence:* source inspection plus a static probe (`results.json` `static.guided_markers`,
  `cellView_form_cells: 0`, `learner_activity_code_cells: []`).
- *Recommended correction:* in `tools/build_notebook.py` / `tools/notebook_template.py`, add How-to-use, a roadmap
  and an Input → Model → Output line to the opening. Title the carried-module, manifest and install cells
  `# @title Infrastructure: …` with `cellView: form` (parity tests must account for the title line). Add a prediction
  before Sections 5 and 7, a collapsible "Check your reasoning" after them, and one coded Predict → Change one thing →
  Run → Observe → Explain activity built on the PSD-M3 fix. Add a troubleshooting section (no GPU, download or digest
  failure, OOM, BYOD rejections) and a conclusion template. Follow the 2.2 reference notebook named in spec §25.13.
- *Acceptance check:* the static probe reports all `guided_markers` true, `cellView_form_cells` ≥ 4, and every
  carried or install cell titled "Infrastructure". At least one coded experiment cell exists that does not run under
  default Run all.
- *Spec:* GDL1–GDL14, UX5, UX8 (SHOULD).

### Minor

**PSD-m1 — Section 8: the reload parity check compares answers the base model also produces, so it cannot detect
an adapter that failed to apply.**
Cell 21 asserts that 8 held-out answers from the reloaded pipeline equal the live adapted pipeline's. On this sample
the frozen and adapted models both score exact match 1.0 on all 30 test rows, so the base model answers these 8
questions the same way. The assert would pass even if no adapter tensor reached the model. `load_artifact` does
verify the digest and the exact tensor set, so the structural load is checked. The *reproduction* evidence that
VER5 asks to be distinguished from loading is not. *Evidence:* documented execution evidence (frozen EM 1.0 = adapted
EM 1.0) plus source inspection. That base and adapted raw strings match on the parity rows is inferred from the
normalized EM. *Correction:* before `del pipe`, record the adapted tensors (or decoder logits on the 8 parity rows),
and after reload assert tensor equality (or logits within an explicit tolerance). Alternatively choose parity rows
where frozen and adapted answers differ, once PSD-M2 gives such rows. Change `tools/notebook_template.py` ~`:193–203`.
*Acceptance:* deliberately loading the base without the adapter makes the Section 8 check fail. *Spec:* VER4, VER5.

**PSD-m2 — Section 4 BYOD branch: three expected failures give no actionable message, the minimum size is
undocumented, and there is no location field or size limit.**
A cancelled upload raises a bare `StopIteration` with an empty message. A non-image file named in `records.csv`
raises `UnidentifiedImageError: cannot identify image file <_io.BytesIO object …>`, which names neither the file
nor the row. A 10-document / 30-row dataset is rejected with "6 records; 8..5000 required", which names neither the
split nor the number of documents needed. Each split must hold ≥ 8 rows, and with the 0.15 / 0.2 fractions that
means roughly ≥ 14 documents at 3 questions each; the notebook states no minimum. The branch always opens
`files.upload()` and has no location field. The loader reads every zip member into memory with no member-count or
expanded-size ceiling. The positive cases work: nested folders are resolved by basename, and a missing `records.csv`
or column gets a clear message. *Evidence:* direct execution of the branch logic (`byod_replay`: 6 cases).
*Correction:* in `samples.load_byod_dataset` / the template's BYOD block (`tools/notebook_template.py:84–95`), check
for an empty upload, wrap image decoding with the file name and row id, report the split and the document count on
a too-small split, and state the minimum in the BYOD markdown. Add `BYOD_ZIP_PATH = ''  # @param` that bypasses the
upload when set, and an expanded-size cap. *Acceptance:* each of the three inputs raises a `ValueError` naming the
failed rule and the fix, and setting `BYOD_ZIP_PATH` runs without importing `google.colab`. *Spec:* DAT19 (MUST),
UX10, EXE1, EXE2, §20 (expanded-size SHOULD).

**PSD-m3 — Metadata, opening and README: the notebook declares spec 2.0 where the fleet baseline is 2.2.**
*Evidence:* `metadata.dimer.notebook_spec = "2.0"`. The opening cell, `NOTEBOOK_SOURCE`, `tutorials/README.md` and the
template docstring all say 2.0. *Correction:* migrate to 2.2 together with PSD-M4, and update the validator in the
same change (§32 item 6). *Acceptance:* metadata, opening cell, registry and validator agree on 2.2. *Spec:* §3.4, §32.

**PSD-m4 — Prerequisites and Section 5: no runtime estimate, and the CPU refusal arrives only after the 1.13 GB
download.**
The Colab badge opens a CPU runtime by default. The notebook says CUDA is required, but the refusal is in cell 15
(Section 5). Cells 3–13 run first on CPU, including the ~1.13 GB snapshot download in cell 11. No duration is given;
the T4 record has 212.1 s for the post-restart pass. *Evidence:* source inspection plus static probe
(`cuda_check_cell: 15`, `model_download_cell: 11`, `runtime_estimate_in_markdown: false`). *Correction:* move the CUDA
check into the Section 1 cell, before any download, with a "Runtime → Change runtime type → T4 GPU" instruction. State the
environment-labelled T4 duration in the Prerequisites. *Acceptance:* on a CPU runtime the first code cell to fail
is the Section 1 cell, before any model download. The Prerequisites state a T4 duration. *Spec:* RUN12, UX12, GDL1.

**PSD-m5 — Section 1: the environment variable `DIMER_NOTEBOOK_CI_PREINSTALLED` is read but never documented.**
Setting it skips the install. Under a non-pinned Pillow the Section 4 digest assert then fails with a bare tuple of
digests (the probe saw `5c793fb2…` under Pillow 12.3.0). *Evidence:* static probe (`env_var_documented_in_markdown:
false`). Corpus probe. *Correction:* one sentence in the Section 1 markdown, plus a message on the digest assert
naming the Pillow pin. *Acceptance:* the variable is named and explained in markdown, and the digest assert names
the pinned Pillow version. *Spec:* EXE5.

### Suggestions

- **PSD-S1 — Sections 5 and 7.** Print the `adapted` flag that `pipe.evaluate` already returns, so the learner can
  see which weights were scored.
- **PSD-S2 — Section 7.** Show the held-out rows with ANLS < 1 (or "no errors") inline from `adapted['rows']`, which
  turns the "inspect every held-out error" next experiment into a one-cell activity.
- **PSD-S3 — "Next experiments".** Provide the 1-vs-2-blocks comparison as an optional coded cell once PSD-M3 makes
  reruns start from base.
- **PSD-S4 — References.** Give DOIs/APA entries for Lee et al. (2023, Pix2Struct), Mathew et al. (2021, DocVQA) and
  Biten et al. (2019, ANLS), matching the 2.2 reference notebook's scholarly-grounding pattern (UX2, SRC10).

## 4. Readiness

**Needs revision.** Open Majors:
- PSD-M1: a documented manual restart, which fails `MUST`s RUN1, RUN10 and ENV6.
- PSD-M2: the corpus is at ceiling, so the central comparison is uninformative and unexplained.
- PSD-M3: rerun and BYOD routes either fail or give invalid results.
- PSD-M4: the guided layer is missing.

DAT19 (MUST) is also unmet for three BYOD failure modes (PSD-m2). The snapshot, corpus, split, artifact and reload
engineering is otherwise sound, and the exact-blob Kaggle T4 run evidences it. Remaining gates after the fixes:
- a no-restart clean run of the new blob on Colab or Kaggle, showing frozen headroom or an in-notebook ceiling
  explanation;
- a recorded BYOD positive and negative run with the real model (REL12);
- the 2.2 migration.

## 5. Verified vs inferred

- **Verified here (direct execution, CPU, reduced scale):** BYOD branch logic on 6 inputs. Stacked adaptation and
  artifact mismatch on a random 12-block Pix2Struct, and a rebuild restoring base. Corpus question templates and
  split counts. Generator `--check`, release-asset validator and unit tests (static/unit, not execution evidence).
- **From documented evidence:** the default-path numbers, the ceiling (frozen 1.0) and the restart (Kaggle T4,
  exact blob).
- **From source inspection:** the `NameError` on rerun after Section 8 (`del pipe` in cell 21, `pipe` built only in
  cell 11). It was not executed against the real notebook kernel.
- **Inferred:** that Colab also trips the restart guard (Kaggle did; Colab is not verified); that stacked
  adaptation changes real-model answers at full scale (shown on tensors at reduced scale); that the base model's raw
  strings equal the adapted ones on the 8 parity rows (PSD-m1, from normalized EM).
- **Most likely wrong:** the severity of PSD-M2. The release record, STATUS.md and README.md are candid about the
  saturation, and spec 2.2 has no explicit "headroom" requirement. A reviewer could treat it as a Minor explanation gap
  (add one markdown cell) rather than a Major validity problem with the sample.
