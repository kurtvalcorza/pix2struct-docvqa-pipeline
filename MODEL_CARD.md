---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: visual-question-answering
task: "Others - Document Question Answering"
base_model: google/pix2struct-docvqa-base
date_published: "2023-03-21"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2023-03-21T09:45:02Z, https://huggingface.co/api/models/google/pix2struct-docvqa-base — the Transformers-format conversion); the Pix2Struct paper and T5X checkpoints are from 2022-10 (arXiv:2210.03347), and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# Pix2Struct DocVQA-base — OCR-free Document Question Answering (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fpix2struct--docvqa--base-ffcc4d?style=flat)](https://huggingface.co/google/pix2struct-docvqa-base)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Fpix2struct-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/pix2struct)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2210.03347-b31b1b.svg)](https://arxiv.org/abs/2210.03347)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/tutorials/pix2struct_docvqa_colab.ipynb) [`pix2struct_docvqa_colab.ipynb`](https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/tutorials/pix2struct_docvqa_colab.ipynb)  
  *Five authored questions over an invoice-style form rendered in code with the pinned `google/pix2struct-docvqa-base` weights: one answer string per question under a caller-owned token budget, and `anls` / `exact_match` against the authored answers as sanity evidence only — no DocVQA benchmark.*

---

#### Description

`google/pix2struct-docvqa-base` is the Transformers-format release of the Pix2Struct *base* model fine-tuned on DocVQA, from "Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding" (Lee et al., arXiv:2210.03347), converted from the T5X checkpoints by the Hugging Face team and pinned here to revision `63f6b3de436e39f75c7a486881a9c2c14a7f4e89` (the Hub's `main` on 2026-09-14). The snapshot declares a 12-layer ViT-style image encoder and 12-layer T5-style text decoder, 282M parameters in the 1.13 GB float32 `model.safetensors`. The VQA processor renders the question above the page, scales the composite to at most 2,048 patches, and the decoder generates an answer. This repository adds digest-verified loading, a bundled header font, validated inference, ANLS/exact-match evaluation, deterministic synthetic and BYOD corpus helpers, empty and training-majority baselines, bounded adaptation of the final *k* decoder blocks (default two), validation-ANLS epoch selection, and safetensors adapter export/reload bound to the pinned base-weight digest. The vision encoder, embeddings, language-model head, and all non-selected decoder blocks remain frozen.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is extractive document question answering without OCR: input one page image (`PIL.Image.Image`, any mode, converted to RGB), one natural-language question (up to 256 characters) and a token budget; output one short answer string, the number of tokens generated and whether the budget was exhausted. Envisioned applications are field extraction from forms, invoices, receipts, letters and reports where the fields can be phrased as questions ("What is the invoice number?", "Who is the customer?"), triage of scanned business documents, and interactive lookup over a page the user is reading — with the answer checked against the page by a human or a downstream rule. The pipeline is an inference component and a zero-configuration baseline for document QA, not a certified extractor for any specific document family.

###### Primary Intended Users

Intended users are machine-learning engineers, document-processing developers, and data analysts integrating document question answering into research prototypes or in-house document tooling. A user is expected to understand that the output is *generated text* — it carries no probability, no correctness signal and no location on the page, and the model answers every question, including unanswerable ones, with equal fluency — that the answer is expected to be a short span copied from the page (DocVQA's answer style) rather than a computed or reasoned result, that the token budget is theirs to set (a long address can exceed a small budget and be reported `truncated`), that the question is rendered into the image with a font this repository supplies (not the one used at fine-tuning time), that the model was fine-tuned on scanned English business documents so photographs, handwriting, non-Latin scripts and unusual layouts are distribution shifts, that greedy decoding is reproducible on a fixed device but GPU and CPU outputs need not match, and that accuracy can only be measured on labelled question/answer pairs they supply. Users who need OCR text, answer localisation, multi-page documents or batch throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no OCR output, no answer localisation or evidence region, no confidence, no multi-page or PDF handling (one page image per call), no batching (each question is a full encoder pass), no sampling or beam search, no abstention (the model cannot say "not on the page"), and no arithmetic, comparison or multi-hop reasoning — DocVQA answers are spans read from the page.
2. **Input boundary:** `answer` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, empty or non-string questions, questions longer than `MAX_QUESTION_CHARS = 256`, and budgets outside `[1, MAX_NEW_TOKENS = 128]` (`ValueError`/`TypeError`). Every page is scaled to fill at most 2048 patches of 16×16 px, so text that is only a few pixels tall at that scale is unlikely to be read, and a long question header consumes part of that budget. The bundled header font covers printable ASCII; a question with other characters renders with missing glyphs.
3. **Input boundary:** the fine-tuning data is DocVQA — scanned typed and printed business documents in English (letters, forms, invoices, reports) with short-span answers. Photographs of documents, handwriting, screenshots of web pages or apps, charts, non-Latin scripts and questions whose answer is not a span on the page fall outside what the upstream authors evaluated and what this repository measured; results on them are undefined, not merely degraded. A page that is not a document at all still produces an answer (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on extracted values — automated invoice payment, claims adjudication, KYC or identity checks, clinical or legal record extraction — without a human comparing the answer with the page, and a locally measured ANLS on the deployment's own labelled question/answer pairs.

#### Factors

###### Groups

This pipeline is not human-centric by design: it answers a question about a page image and never classifies, identifies or scores people. The fine-tuning data (DocVQA: 12,767 scanned documents from the UCSF Industry Documents Library with 50,000 questions, per the dataset paper) contains no evaluation groups in the demographic sense, and neither the upstream authors nor this repository audited it for anything of the kind. What does vary is the document population: DocVQA is English-language, typed or printed, mid-twentieth-century-onward industry correspondence and forms, so pages in other languages and scripts, handwritten or historical typesetting, modern designer layouts, and non-Western document conventions are the groups whose answer accuracy is unknown, not known to be equal. Where pages carry personal data — medical records, HR files, identity documents, correspondence naming individuals — a question turns the pipeline into a targeted extractor; the operator who processes such documents is responsible for a fairness and privacy audit on their own page set, stratified by document family, before relying on the output.

###### Instrumentation

The upstream pretraining "instrument" is a web renderer (screenshots of pages paired with their HTML) and the fine-tuning instrument is the scanner that produced the DocVQA archive: greyscale or colour scans at archival resolution, with the question rendered above each page in the Arial font the Pix2Struct preprocessing uses. Inference images arrive from whatever produced them — a PDF renderer at some DPI, a scanner, a phone camera, a screenshot — and resolution, skew, JPEG artefacts, contrast and font all change the visual evidence; the 2048-patch budget fixes the encoder's input size regardless of the source, so a small-type page loses detail while a large blank margin costs patches. This repository renders the question with Pillow's bundled Aileron rather than Arial, a second instrument change whose effect was measured only on the synthetic form (5/5). The pipeline validates type, size and question length only; it cannot detect a low-DPI render, a skewed scan, or a question that the page cannot answer. The synthetic tutorial form (Pillow's bundled font, ruled table, generous margins, crisp black-on-white) is itself a rendering instrument cleaner than any DocVQA scan.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible (float32) — the CPU figures below are from the reference machine, and the CUDA path was exercised only by the Kaggle Tesla T4 E2E run under Runtime. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 8-file, 1.13 GB snapshot 0.59 s; load 5.26 s; one 850×1100 rendered invoice form, five questions at the default budget, 2.29–3.00 s per question (4–12 tokens each); a 4096×4096 blank page 4.19 s — cost is dominated by the encoder pass over the 2048-patch input, which repeats for every question, and only weakly by the answer length. Data environment: the model assumes the image is one scanned or rendered English business document whose answer is a span on the page; the synthetic tutorial form satisfies that assumption and is where the measured behaviour holds. Photographs, handwriting, unusual layouts, non-Latin scripts and unanswerable questions violate it to degrees this repository did not measure, and the pipeline reports no signal when they do — nor when the image is not a document at all.

#### Metrics

###### Performance Measures

Inference returns generated text with no per-answer score, probability, or correctness signal; `new_tokens` and `truncated` describe generation only. Corpus evaluation reports mean ANLS and exact-match rate plus per-row predictions. ANLS is normalized Levenshtein similarity, maximized over accepted answers and zeroed below 0.5; exact match uses the same lower-case, punctuation-stripped, whitespace-collapsed normalization. The E2E notebook also reports an empty-answer baseline and a one-answer training-majority baseline, then frozen and adapted model results on one fixed 30-row synthetic test split. These are point estimates without dispersion and are not DocVQA benchmark results. Measured on Kaggle Tesla T4 (2026-09-26, commit `2602e66`): held-out ANLS empty 0.0 / majority 0.2949 / frozen 1.0 / adapted 1.0, exact match 0.0 / 0.0 / 1.0 / 1.0 on 30 synthetic rows; the frozen model already saturates this sample, so it cannot show whether adaptation helps.

###### Decision thresholds

No score threshold exists in the model: it generates tokens until end-of-sequence or the budget and nothing is filtered or abstained. The decision parameter is the **token budget** `max_new_tokens`, default `DEFAULT_MAX_NEW_TOKENS = 32` (a field-sized budget chosen by this repository; the checkpoint's own `text_config.max_length` is 20, and the five smoke answers needed 4–12 tokens) with ceiling `MAX_NEW_TOKENS = 128`. A budget that is too small is reported, not hidden: `truncated` is true whenever `new_tokens` reaches it, and the caller should raise the budget and rerun. The evaluation helper carries one threshold of its own, `ANLS_THRESHOLD = 0.5`, which is DocVQA's published convention, not a value tuned here. Decoding is greedy (`do_sample=False`) with no temperature, beams or repetition penalty. A deployment owns choosing the budget per document family and deciding how an answer is verified against the page before it is used.

###### Approaches to uncertainty and variability

The generated corpus fixes seed 42, 40 documents, three questions per document, and document-disjoint 72/18/30 train/validation/test rows. Training fixes seed 0 and shuffles rows deterministically on one library/device stack, but GPU kernels and autoregressive decoding need not be bit-identical across hardware or builds. Validation chooses one of two epochs; the test set is not consulted during selection. One seeded split supplies no confidence interval or standard deviation, and the synthetic layouts are much cleaner than real scans. Repeat seeds and external document-family test sets are required before any generalization claim; no per-answer confidence exists.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The upstream paper describes pretraining on 80M masked web-page screenshots paired with simplified HTML (from the C4 corpus's URLs) and fine-tuning on DocVQA, whose pages come from the UCSF Industry Documents Library — public archives of tobacco-industry correspondence, forms and reports that name real people and organisations; personal data in the fine-tuning corpus is therefore present by construction, and web screenshots can contain more. Neither was audited here. This repository distributes code, tests, and documentation; it does not distribute the 1,129,177,976-byte `model.safetensors`, which is staged locally under `weights/pix2struct-docvqa-base/` and git-ignored, and it ships no sample documents — the tutorial form is rendered in code with fictitious names. The operator must audit the pages they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will answer "What is the patient's name?" as readily as "What is the invoice number?".

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — extracting amounts and payees for automated payment, reading identity documents for verification, pulling diagnoses or dosages from clinical records, screening applications by fields read from scans — would be admissible only with human comparison of every answer against the page (the model invents an answer when none exists and gives no signal), a locally measured ANLS on the deployment's own labelled pages, a documented budget policy with `truncated` handling, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 8 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, refuses a snapshot whose image processor is not the VQA variant, and the smoke run loaded and answered with `HF_HUB_OFFLINE=1`. The upstream `pytorch_model.bin` (pickle) is neither listed nor loaded. **The inference-time font download the upstream processor performs (`ybelkada/fonts/Arial.TTF`, unpinned, unlisted) is replaced** by `header_font_bytes()` — Pillow's bundled Aileron Regular (CC0) — and the loader calls the image processor directly because `Pix2StructProcessor.__call__` drops the `font_bytes` keyword. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported; another asserts the header font bytes are TrueType and stable.
- **Input integrity:** the public `validate_inputs(image, questions, *, max_new_tokens)` stage applies exactly the checks `answer` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input, the checked questions, the budget and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; empty, non-string or over-long questions and out-of-range or boolean budgets are rejected; `answer` raises on a malformed runner result; `evaluation_report` rejects mismatched or empty accepted-answer lists.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; greedy decoding with no sampling; a fixed, bundled header font; every result carries `model_id`, `model_revision`, the checked question, the budget, `new_tokens`, `truncated`, the device and the dtype.
- **Refusals:** no batching, no download without the explicit flag, no Hub access at inference time, no sampling, no pickle deserialisation, no attempt to guess whether the page can answer the question.
- **Training and artifact integrity:** `adapt` exposes bounded epochs, learning rate, batch size and final-block count; trains only the implied `decoder.layer.*` tensors; restores the initial state and frozen flags after any failure; and retains the highest validation-ANLS epoch. `save_artifact` writes safetensors plus a manifest carrying the base id, immutable revision, base-weight digest, exact tensor list, file size and SHA-256. `load_artifact` verifies those fields and shapes before applying tensors and refuses any extra, missing, foreign, or corrupted tensor.
- **Dataset integrity:** generated pages contain fictional data under CC0-1.0; `validate_dataset` enforces 8–5,000 records, unique ids, valid document ids, decodable bounded images, questions and accepted answers; split checks keep every document in one partition; BYOD zip/directory loading flattens member basenames and refuses duplicates rather than extracting archives.

###### Risks and harms

- **Invented answers:** the model has no abstention — a blank page yielded `51759 7951` in the smoke run — so an unanswerable question, a wrong page or a mis-phrased field name produces a fluent, plausible value with no signal; downstream consumers that trust the string (payment systems, databases, LLM pipelines) inherit the error silently.
- **Near-miss values:** a digit transposed in an amount or a date is still a short, confident span; ANLS on labelled pages is the only way to see the rate, and the tutorial's 5/5 says nothing about it.
- **Automation bias:** exact answers on a clean form invite trust that generated text has not earned; the header-font substitution adds an untested variable.
- **Truncation:** an answer longer than the budget is cut (reported via `truncated`); a caller who ignores the flag ships a partial value.
- **Targeted extraction and privacy exposure:** a question turns any page into a lookup for a named field — including personal data — with no content check.
- **Bias amplification:** any document family DocVQA under-represents (non-English, handwritten, modern designer layouts, non-Western conventions) is reproduced as uneven accuracy, undetected because no per-family evaluation exists.
- **Resource use:** a 1.13 GB model and ~2.3–3.0 s per question on the reference CPU with a full encoder pass per question (about 0.65–0.71 s per question on a Kaggle Tesla T4 in the E2E run); a question-heavy workload scales linearly.

###### Use cases

Prohibited even where the model would work: querying documents in order to extract personal data for surveillance, profiling, social scoring, or unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access; processing documents the operator has no right to process, or paywalled and licence-restricted material in breach of its terms; deceptive uses that present generated answers as verified document facts or as evidence; and any use that violates the upstream Apache-2.0 licence terms, the terms of the deployment that runs the pipeline, or the consent and data-protection obligations attached to the documents processed. Autonomous high-consequence actions triggered by unreviewed answers are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `google/pix2struct-docvqa-base`
- Revision: `63f6b3de436e39f75c7a486881a9c2c14a7f4e89`
- Snapshot manifest: `weights/pix2struct-docvqa-base/dimer-base-manifest.json`, 8 files, `totalBytes` 1133308924
- `model.safetensors` SHA-256: `067f7f314d87fa56daa5bcfaf36fa0b33ceebf7b7d4fae6a1e51ab7af64ee0b5` (1,129,177,976 bytes, float32)
- `config.json` SHA-256: `8d39973772a4218b555e30daecabdd5ea11aa1345dd711ff7f88fa90750b464f` (4,892 bytes; `Pix2StructForConditionalGeneration`)
- `preprocessor_config.json` SHA-256: `c84e4eebc84171d6069533d9f0147ec7b4afd02ab78697cb5c30f9419ef7dc45` (249 bytes; `is_vqa` true, `max_patches` 2048, 16×16 patches)
- Weight format: SafeTensors; loader `Pix2StructForConditionalGeneration.from_pretrained(<dir>, local_files_only=True, trust_remote_code=False, dtype=float32)` with `Pix2StructProcessor` from the same directory, the image processor called with `header_text=<question>` and `font_bytes=header_font_bytes()`. The upstream `pytorch_model.bin` is not part of the manifest and is never loaded.

## Input/output contract

- `Pix2StructDocVQAPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`; float32 on both.
- `answer(image, question, *, max_new_tokens=32) -> dict` with keys `answer` (stripped decoded text), `question` (whitespace-collapsed), `image_size`, `new_tokens`, `truncated`, `generation` (`max_new_tokens`, `do_sample` false, `decoding` greedy), `device`, `dtype`, `source`, `model_id`, `model_revision`.
- `normalize_answer(text) -> str`; `anls(prediction, golds, *, threshold=0.5) -> float`; `exact_match(prediction, golds) -> bool`; `header_font_bytes() -> bytes`.
- Ceilings and constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_QUESTION_CHARS = 256`, `MAX_NEW_TOKENS = 128`, `DEFAULT_MAX_NEW_TOKENS = 32`, `DECODING = "greedy"`, `MAX_PATCHES = 2048`, `ANLS_THRESHOLD = 0.5`, `INPUT_SCHEMA`.
- `validate_inputs(image, questions, *, max_new_tokens, names) -> dict`; `evaluation_report(results, golds=None, *, sample_kind) -> dict` where `golds` holds one sequence of accepted answers per result; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.
- `evaluate(records, *, max_new_tokens=32) -> dict` validates and scores 1–2,000 document-QA rows with ANLS and exact match; `adapt(train, val=None, *, epochs=3, lr=2e-4, batch_size=1, trainable_decoder_layers=2, seed=0, progress=None) -> dict` accepts 8–5,000 training rows, trains only the final 1–12 decoder blocks, clips gradients at 1.0, and keeps the highest-validation-ANLS epoch (or final epoch without validation).
- `save_artifact(directory, metadata=None)` writes `adapter.safetensors` and `manifest.json`; `load_artifact(directory)` verifies the supported format/version, pinned base id/revision/weight digest, exactly one contained artifact file, its byte count and SHA-256, the exact tensor names and shapes, then applies it; `from_artifact(...)` loads a verified base and adapter.
- `samples.py` provides `build_sample_dataset`, `validate_dataset`, `dataset_digest`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`, and CSV/manifest helpers. `metrics.py` provides `qa_metrics`, `empty_baseline`, and `majority_answer_baseline`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing renders the question as a header (Pillow's bundled Aileron, 36 pt, wrapped at 80 characters, black on white) above the page and scales the composite to at most 2048 16×16 patches (`Pix2StructImageProcessor`, snapshot defaults); greedy decoding.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.59 s (8 files, 1.13 GB); load 5.26 s; `answer` on a synthetic 850×1100 invoice-style form (header, supplier, five labelled fields, a four-row line-item table, totals and a payment-terms line, rendered with Pillow's bundled font) at `max_new_tokens=32` → "What is the invoice number?" `NW-2026-0417` (12 tokens, 3.00 s); "Who is the customer?" `Blue Yonder Airlines` (5 tokens, 2.29 s); "What is the total due?" `$1,099.20` (10 tokens, 2.84 s); "What is the due date?" `11 April 2026` (10 tokens, 2.91 s); "How many cargo pallets were invoiced?" `40` (4 tokens, 2.29 s); no truncation; `evaluation_report` against the authored answers: `anls` 1.0, `exact_match` 1.0, verdict `sample-sanity`; 4096×4096 blank page, "What is the invoice number?" → `51759 7951` (12 tokens, 4.19 s).
- E2E tutorial, measured 2026-09-26 on Kaggle Tesla T4 (commit `2602e66`, notebook blob `078435c42445`, torch 2.14.0+cu130, transformers 4.57.6, pillow 11.3.0, `cuda:0`, float32; 10/10 code cells, 393.2 s wall incl. one restart after the install cell): synthetic corpus digest matched the pinned `SAMPLE_DIGEST`; one seeded document-disjoint 72 / 18 / 30 split; final-two-decoder-block adaptation (18,878,976 of 282,285,696 parameters trainable), 2 epochs, 94.7 s, best epoch 1 by validation ANLS (frozen validation ANLS 0.9815 / EM 0.9444 → 1.0 / 1.0, n = 18); held-out test (30 rows): ANLS empty 0.0, majority 0.2949, frozen 1.0, adapted 1.0; exact match 0.0, 0.0, 1.0, 1.0; delta vs frozen 0.0 on both; evaluation 21.2 s frozen / 19.6 s adapted for 30 questions; `adapter.safetensors` 75,519,448 B; fresh reload answer parity 8/8. The frozen model already answers every synthetic test question exactly, so this sample demonstrates the adaptation mechanics, not a gain. One synthetic split, one runtime, no dispersion estimate; full record in `docs/release-verification.md`.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed for the current E2E carrier: repeat seeds, a held-out set on which the frozen model is not already saturated, the BYOD branch on real documents, questions with non-ASCII characters, scans or photographs, non-Latin scripts, or the upstream Arial header rendering for comparison.

## References

- Lee et al. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding. ICML 2023. https://arxiv.org/abs/2210.03347
- Mathew, Karatzas, Jawahar. DocVQA: A Dataset for VQA on Document Images. WACV 2021. https://arxiv.org/abs/2007.00398
- Biten et al. Scene Text Visual Question Answering (the ANLS metric). ICCV 2019. https://arxiv.org/abs/1905.13648
- Upstream code: https://github.com/google-research/pix2struct
- Upstream card: https://huggingface.co/google/pix2struct-docvqa-base
- Transformers `Pix2Struct` documentation: https://huggingface.co/docs/transformers/model_doc/pix2struct
- Aileron font (Pillow's bundled default, CC0): https://dotcolon.net/fonts/aileron
