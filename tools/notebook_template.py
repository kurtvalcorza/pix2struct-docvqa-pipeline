"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "pix2struct_docvqa_pipeline",
    "repo_name": "pix2struct-docvqa-pipeline",
    "stem": "pix2struct_docvqa",
    "notebook_name": "pix2struct_docvqa_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "Pix2StructDocVQAPipeline",
    "weights_key": "pix2struct-docvqa-base",
    "runtime_imports": ["torch", "transformers"],
    "title": "Pix2Struct DocVQA-base — DIMER OCR-free document question answering tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/tutorials/pix2struct_docvqa_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fpix2struct--docvqa--base-ffcc4d?style=flat",
            "https://huggingface.co/google/pix2struct-docvqa-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Fpix2struct-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/pix2struct",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2210.03347-b31b1b.svg", "https://arxiv.org/abs/2210.03347"),
    ],
    "capability": "OCR-free document question answering — one page image plus one natural-language question → one short answer string — using the pinned `google/pix2struct-docvqa-base` weights",
    "intro": (
        "At inference the Pix2Struct image-encoder/text-decoder (a ViT-style encoder over variable-resolution 16×16 patches "
        "and a 12-layer text decoder, 282M parameters, pretrained by parsing masked web screenshots into HTML and fine-tuned "
        "on DocVQA) reads the **question rendered as a text header above the page** — the Pix2Struct convention for "
        "visual question answering — scales the composite to fill at most 2048 patches, and generates the answer text "
        "token by token. Decoding is greedy (`do_sample=False`) under a caller-owned `max_new_tokens` budget. "
        "**No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens in this "
        "notebook — the upstream checkpoint supplies the weights, processor and tokenizer, and the carried module adds "
        "snapshot verification, the input contract (image side ceilings, a non-empty question up to 256 characters, the token "
        "budget), a fixed output contract, an offline header font (Pillow's bundled Aileron replaces the Hub font the upstream "
        "processor would otherwise download), and the `anls`, `exact_match`, `validate_inputs` and `evaluation_report` "
        "helpers. The default sample is an invoice-style form rendered in code with five authored questions and accepted "
        "answers, so ANLS and exact-match are demonstration (plumbing) evidence for one page, not a DocVQA benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, render a synthetic form with authored question/answer pairs (or upload your own "
        "page and write your own questions) and validate it into an input manifest, choose a token budget, run the supported "
        "task, read the answers correctly (generated text, no score, a `truncated` flag), exercise an optional BYOD path, "
        "produce an evaluation report that is `sample-sanity` with `anls` and `exact_match` only when accepted answers exist "
        "and `not-measurable` otherwise, and export the answers, the annotated page and provenance."
    ),
    "exclusions": (
        "PDF or multi-page documents (one page image per call), OCR output or answer localisation (the model returns text, "
        "not the region it read), questions that need arithmetic or reasoning across pages, batch throughput, sampling or "
        "beam search, evaluation on the DocVQA benchmark (registration-gated and not bundled; only authored questions on a "
        "rendered page are scored here), and any training. The model was fine-tuned on scanned business documents in "
        "English; photographs, handwriting, non-Latin scripts and long free-text answers are outside what this notebook "
        "measures, and a fluent wrong answer carries no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 5.3 s to load and 2.3–3.0 s per question on the 850×1100 rendered form in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 1.13 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; what normalised Levenshtein similarity (ANLS) measures; that a confident answer is not a correct one.",
        "- **Data:** the default sample is a deterministic 850×1100 invoice-style form rendered in code with Pillow's bundled font (header, supplier, five labelled fields, a four-row line-item table, totals, a payment-terms line) with five authored questions and their accepted answers, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar) of a **single document page**, any colour mode, sides between 16 and 4096 px, plus your own questions typed into the form field. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Render the synthetic form or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: an invoice-style form — an `INVOICE` "
                "header, a supplier name and address, five labelled fields (invoice number, dates, customer, purchase order), "
                "a four-row line-item table, subtotal/VAT/total lines and a payment-terms sentence — is rendered with Pillow's "
                "bundled font at 850×1100, the same page the repository's smoke run used. Five questions are authored against "
                "it, each with the accepted answer(s) as written on the page; they are the references for the `anls` and "
                "`exact_match` sanity checks later. They are not a labelled dataset, so nothing here is a DocVQA measurement. "
                "The image digest is printed for the record. BYOD is optional and disabled by default; when enabled, upload one "
                "page image and type your questions (one per line) — no accepted answers exist for them, so the evaluation "
                "report will be `not-measurable`.\n\n"
                "The token budget is a **caller-owned request parameter**: `max_new_tokens` bounds the answer "
                "(`DEFAULT_MAX_NEW_TOKENS = 32` fits any field on this form; `MAX_NEW_TOKENS = 128` is the ceiling). Nothing "
                "is validated in this cell — the next section hands the image and the questions to the pipeline's own "
                "validation stage, which is the only checker. Look for a dictionary naming the sample kind, the page size and "
                "digest, the budget and the number of questions."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "byod_questions = 'What is the invoice number?\\nWho is the customer?'  # @param {{type:\"string\"}}\n"
                "max_new_tokens = 32  # @param {{type:\"integer\"}}\n\n\n"
                "def synthetic_form(width=850, height=1100):\n"
                "    \"\"\"An invoice-style form rendered with Pillow's bundled font; returns page + [(question, accepted answers)].\"\"\"\n"
                "    page = Image.new('RGB', (width, height), 'white')\n"
                "    d = ImageDraw.Draw(page)\n"
                "    body, bold, head = ImageFont.load_default(size=18), ImageFont.load_default(size=20), ImageFont.load_default(size=30)\n"
                "    d.text((70, 60), 'INVOICE', fill='black', font=head)\n"
                "    d.text((70, 110), 'Northwind Traders Ltd.', fill='black', font=bold)\n"
                "    d.text((70, 136), '14 Harbour Road, Portsmouth PO1 3AX', fill=(40, 40, 40), font=body)\n"
                "    fields = [('Invoice number:', 'NW-2026-0417'), ('Invoice date:', '12 March 2026'), ('Due date:', '11 April 2026'), ('Customer:', 'Blue Yonder Airlines'), ('Purchase order:', 'PO-88213')]\n"
                "    y = 200\n"
                "    for label, value in fields:\n"
                "        d.text((70, y), label, fill='black', font=bold)\n"
                "        d.text((300, y), value, fill='black', font=body)\n"
                "        y += 32\n"
                "    d.rectangle([70, 400, 780, 640], outline='black', width=2)\n"
                "    cols = [70, 420, 540, 660, 780]\n"
                "    rows = [('Description', 'Qty', 'Unit price', 'Amount'), ('Cargo pallets (standard)', '40', '$18.50', '$740.00'), ('Shrink wrap rolls', '12', '$9.25', '$111.00'), ('Handling fee', '1', '$65.00', '$65.00')]\n"
                "    for r, row in enumerate(rows):\n"
                "        yy = 400 + r * 48\n"
                "        if r:\n"
                "            d.line([(70, yy), (780, yy)], fill=(120, 120, 120), width=1)\n"
                "        for c, cell in enumerate(row):\n"
                "            d.text((cols[c] + 10, yy + 14), cell, fill='black', font=bold if r == 0 else body)\n"
                "    for c in cols[1:-1]:\n"
                "        d.line([(c, 400), (c, 640)], fill=(120, 120, 120), width=1)\n"
                "    d.text((540, 670), 'Subtotal:', fill='black', font=bold)\n"
                "    d.text((680, 670), '$916.00', fill='black', font=body)\n"
                "    d.text((540, 700), 'VAT (20%):', fill='black', font=bold)\n"
                "    d.text((680, 700), '$183.20', fill='black', font=body)\n"
                "    d.text((540, 736), 'Total due:', fill='black', font=head)\n"
                "    d.text((680, 736), '$1,099.20', fill='black', font=head)\n"
                "    d.text((70, 900), 'Payment terms: 30 days from invoice date. Bank: Solent Mutual, sort code 40-11-22.', fill=(40, 40, 40), font=body)\n"
                "    qa = [\n"
                "        ('What is the invoice number?', ['NW-2026-0417']),\n"
                "        ('Who is the customer?', ['Blue Yonder Airlines']),\n"
                "        ('What is the total due?', ['$1,099.20', '1,099.20']),\n"
                "        ('What is the due date?', ['11 April 2026']),\n"
                "        ('How many cargo pallets were invoiced?', ['40']),\n"
                "    ]\n"
                "    return page, qa\n\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    questions = [line.strip() for line in byod_questions.splitlines() if line.strip()]\n"
                "    golds = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic form: no randomness, so no seed is needed and the digest is stable per Pillow build.\n"
                "    image, qa = synthetic_form()\n"
                "    questions, golds = [q for q, _ in qa], [g for _, g in qa]\n"
                "    image_name = 'synthetic_invoice_850x1100.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'max_new_tokens': max_new_tokens, 'n_questions': len(questions), 'has_golds': golds is not None}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `answer` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, each question a non-empty string of at most "
                "`MAX_QUESTION_CHARS` characters (whitespace collapsed), and `max_new_tokens` in `[1, MAX_NEW_TOKENS]` — and "
                "returns an **input manifest** naming the schema (including the header-rendering preprocessing and the "
                "decoding rule), the input's observed mode and size, the checked questions, the budget and the verdict. The "
                "manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also "
                "validates a blank question and records the pipeline's own error message as a finding. Inside the pipeline the "
                "image is converted to RGB, the question is rendered above it, and the composite is scaled to the patch budget; "
                "nothing else is dropped or altered. The pipeline cannot tell whether the image is a document or whether the "
                "question is answerable from it: that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PATCHES': MAX_PATCHES, 'MAX_QUESTION_CHARS': MAX_QUESTION_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING}}}})\n"
                "input_manifest = validate_inputs(image, questions, max_new_tokens=max_new_tokens, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, ['   '])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'blank-question-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Answer the questions and read the output correctly\n\n"
                "`answer` returns, per question, a dict with `answer` (the decoded text, stripped), the checked `question`, "
                "`image_size`, `new_tokens`, a `truncated` flag that is true when the budget was exhausted, the generation "
                "settings and the model identity. **No score exists**: the answer is generated text with no probability and no "
                "correctness signal, and a fluent answer is not evidence that it was read from the page. Greedy decoding is "
                "deterministic on a fixed device and dtype; CUDA kernel selection can change a token and therefore the rest of "
                "the answer, so GPU and CPU outputs need not match. Each call renders the question as a header and re-encodes "
                "the page, so cost is per question (about 2.3–3.0 s each on the reference CPU). As recorded in the model card, "
                "the repository's CPU smoke on this same form answered all five authored questions exactly — and answered "
                "`51759 7951` to \"What is the invoice number?\" on a blank 4096×4096 page: the model always produces an "
                "answer, whether or not one exists."
            ),
            "code": (
                "import time\n\n"
                "results, seconds = [], []\n"
                "for question in questions:\n"
                "    t0 = time.time()\n"
                "    results.append(pipe.answer(image, question, max_new_tokens=max_new_tokens))\n"
                "    seconds.append(round(time.time() - t0, 2))\n"
                "print({{'device': pipe.device, 'dtype': pipe.dtype, 'seconds_per_question': seconds, 'any_truncated': any(r['truncated'] for r in results)}})\n"
                "for result in results:\n"
                "    print(f\"Q: {{result['question']}}\\n   A: {{result['answer']!r}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\")\n"
                "if any(r['truncated'] for r in results):\n"
                "    print('A budget was exhausted: that answer is incomplete. Raise max_new_tokens (ceiling MAX_NEW_TOKENS) and rerun.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No accuracy is "
                "reported by default: DocVQA-style accuracy needs labelled question/answer pairs on pages from the deployment "
                "domain, and this repository ships none (the DocVQA benchmark itself is registration-gated). The repository's "
                "metric helpers are `anls` — Average Normalised Levenshtein Similarity, the benchmark's official metric: "
                "`1 − edits / max(len)` over normalised strings, maximised over the accepted answers, scored 0 below the 0.5 "
                "threshold — and `exact_match` after the same normalisation (lower-case, punctuation removed, whitespace "
                "collapsed). When accepted answers are supplied the report carries the mean `anls`, the `exact_match` rate and "
                "one entry per question, with the verdict `sample-sanity`. On the synthetic path those answers are values "
                "**you rendered yourself**, so a perfect score proves only that the input contract, header rendering, forward "
                "pass and decoding round-trip. On BYOD no accepted answers exist, the verdict is `not-measurable`, and the "
                "report states what would make the task measurable. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, golds, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'per_question')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:12}} {{metric['value']:.3f}}  ({{metric['estimation']}})\")\n"
                "for entry in report.get('per_question', []):\n"
                "    print(f\"  anls {{entry['anls']:.2f}}  exact {{str(entry['exact_match']):5}}  {{entry['question']}} -> {{entry['prediction']!r}} (accepted: {{entry['golds']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No accepted answers exist for these questions, so nothing is scored; read the answers against the page yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves every result (question, answer, `new_tokens`, `truncated`, the budget), the "
                "evaluation report, the input manifest, the sample identity, digest and accepted answers, the notebook's source "
                "(repository, revision, embedded module digest, generator), the model identifier, the immutable model revision, "
                "the model licence, and the runtime identity (Python, `torch`, `transformers`, device). The question/answer "
                "pairs are also written as CSV with explicit `image`, `question`, `answer`, `new_tokens`, `truncated` columns, "
                "and an annotated PNG shows the page with the questions and answers printed in a panel beneath it for visual "
                "inspection (the model returns no location, so nothing is drawn on the page itself) — a supplement to, not a "
                "replacement for, the machine-readable files. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "panel_height = 30 + 26 * len(results)\n"
                "annotated = Image.new('RGB', (image.width, image.height + panel_height), 'white')\n"
                "annotated.paste(image.convert('RGB'), (0, 0))\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "draw.line([(0, image.height + 1), (image.width, image.height + 1)], fill=(120, 120, 120), width=2)\n"
                "panel_font = ImageFont.load_default(size=16)\n"
                "for index, result in enumerate(results):\n"
                "    draw.text((20, image.height + 12 + 26 * index), f\"{{result['question']}}  ->  {{result['answer']}}\", fill=(40, 90, 220), font=panel_font)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'predictions': results,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'questions': questions, 'accepted_answers': golds}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'question', 'answer', 'new_tokens', 'truncated'])\n"
                "    for result in results:\n"
                "        writer.writerow([image_name, result['question'], result['answer'], result['new_tokens'], result['truncated']])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The answers are the text the model generates after reading a page with the question printed above it; nothing "
        "in the output scores that text, the model returns no location or evidence, and it answers every question — "
        "including one about a blank page — with equal fluency. On the synthetic form the `anls` and `exact_match` values in "
        "the evaluation report compare the answers with values you rendered yourself and the verdict is `sample-sanity`, which "
        "proves only that the input contract, header rendering, forward pass and decoding work (the repository's smoke run "
        "scored 5/5 exact on this page); they say nothing about scans, photographs, dense multi-column layouts, handwriting, "
        "non-Latin scripts, questions that need arithmetic or reasoning, or answers longer than a field, and a BYOD result is a "
        "single-page observation with the verdict `not-measurable`. **The model answers any question about any image** and "
        "stops only at end-of-sequence or the token budget: check `truncated`, and treat a plausible answer to an unanswerable "
        "question as the expected failure mode, not an exception. The pipeline provides no OCR, no answer localisation, no "
        "multi-page handling, no benchmark evaluation and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** ask a question the form cannot answer (`What is the delivery address?`) and see the model invent "
        "one; ask `What is the VAT amount?` and compare with `$183.20`; lower `max_new_tokens` to 2 and watch `truncated` turn "
        "true; enable `USE_BYOD` with a page you know, type your questions, then pass your own accepted answers to "
        "`evaluation_report` to see the verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/pix2struct\n"
        "- Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding (Lee et al., 2022): https://arxiv.org/abs/2210.03347\n"
        "- DocVQA: A Dataset for VQA on Document Images (Mathew, Karatzas, Jawahar, 2020): https://arxiv.org/abs/2007.00398\n"
        "- Scene Text Visual Question Answering — the ANLS metric (Biten et al., 2019): https://arxiv.org/abs/1905.13648"
    ),
}
