"""E2E notebook template for the standalone NOTEBOOK_SPEC 2.2 carrier."""

# ruff: noqa: E501

TEMPLATE = {
    "package": "pix2struct_docvqa_pipeline",
    "repo_name": "pix2struct-docvqa-pipeline",
    "stem": "pix2struct_docvqa",
    "notebook_name": "pix2struct_docvqa_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "require_cuda": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "Pix2StructDocVQAPipeline",
    "weights_key": "pix2struct-docvqa-base",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers"],
    "run_all": (
        "Selecting **Run all** in a fresh supported GPU runtime builds an isolated hash-locked environment from the pinned dependencies (nothing is installed into the notebook kernel, so no restart is needed), carries the three "
        "repository modules without cloning the repository, stages and digest-verifies the immutable model snapshot, "
        "builds and validates the deterministic synthetic corpus, records two non-neural baselines and the frozen model, "
        "fine-tunes only the final two decoder blocks with validation-ANLS epoch selection, evaluates held-out documents, "
        "exports a safetensors adapter bound to the base digest, reloads it in a fresh model instance, checks answer parity, "
        "and writes machine-readable evidence. CUDA is required (Section 1 stops before any download when it is absent); no credential or upload is needed. The recorded Kaggle Tesla T4 pass took 212.1 s after the environment was in place (94.7 s of it the adaptation); the first Run all also downloads the 1.13 GB snapshot and builds the environment."
    ),
    "byod": (
        "After the default sample completes, set `USE_BYOD = True` and `BYOD_PATH` (a zip already in the runtime; on Colab an empty path opens an upload dialog) in Section 4, then re-run Section 4 and every code cell of Sections 5–8 in order (Section 5 reloads the frozen pipeline from the verified snapshot, so the frozen numbers are the base model's on your data) for one zip containing `records.csv` plus its "
        "page images (folders inside the zip are fine; files are matched by name). CSV columns are `id,document_id,file,question,answers`; separate accepted answers with `|`. The same "
        "validation and document-grouped split apply: every split needs at least 8 rows, which with the 0.15 / 0.20 validation / test fractions means 17 documents at three questions each (10 documents are refused by name); at most 5,001 zip members and 2 GiB extracted. BYOD is optional and remains inside the hosted runtime."
    ),
    "title": "Pix2Struct DocVQA-base — bounded document-QA adaptation (standalone E2E)",
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
            "https://img.shields.io/badge/%F0%9F%A4%97-google%2Fpix2struct--docvqa--base-ffcc4d?style=flat",
            "https://huggingface.co/google/pix2struct-docvqa-base",
        ),
    ],
    "capability": "end-to-end adaptation of OCR-free document question answering: validated page/question/answer records → held-out ANLS comparison → reloadable safetensors adapter",
    "intro": (
        "Pix2Struct renders each question above its page, encodes the composite as at most 2,048 patches, and generates a "
        "short answer. This tutorial makes that path trainable without changing its inference contract: the vision encoder, "
        "embeddings, language head and first ten decoder blocks stay frozen; only decoder blocks 10 and 11 are updated. A "
        "deterministic, code-generated business-document corpus avoids registrations, remote dataset drift and private data. "
        "Documents—not question rows—define the train/validation/test boundary. Validation ANLS selects the epoch; the test "
        "set is used only for frozen/adapted comparison. This carrier remains Candidate until its exact committed blob passes "
        "a clean supported GPU run."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Kaggle with a GPU and has met train/validation/test splits, and wants to see how an OCR-free document question-answering model reads a page image, how its answers are scored (ANLS and exact match) against simple baselines, and how a small part of it is fine-tuned and exported without fooling themselves. The audience is students and practitioners preparing their own document-QA data; no prior experience with Pix2Struct or fine-tuning is assumed — each term is explained where it first matters and again in the **Glossary**. A CUDA GPU is required (a T4 is enough).\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | page images with a question and one or more accepted answers: 40 generated invoice-like documents, 120 questions (72 train, 18 validation, 30 test rows, split by document) or your own zip |\n| Model | Pix2Struct DocVQA-base: the question is rendered above the page, the composite is encoded as at most 2,048 patches, and a decoder generates the answer; only decoder blocks 10 and 11 are trained |\n| Output | a short answer per question, held-out ANLS and exact match beside an empty and a majority-answer baseline, and a safetensors adapter that reloads with identical answers |\n\n**How to use this notebook.** Choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the recorded hosted run of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried modules and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from the recorded Kaggle T4 run of 26 September 2026. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the corpus, validation and a document-level split *(evaluation practice: leakage)* → 5 two baselines and the frozen model *(core concept: ANLS)* → 6 bounded adaptation of two decoder blocks *(core concept: what is trained)* → 7 held-out comparison *(evaluation practice: a saturated sample)* → 8 export, free, reload and answer parity *(engineering)* → conclude."
    )]},
    "learning_objectives": (
        "verify immutable model and corpus identities; inspect record and split contracts; compare empty-answer and "
        "training-majority baselines with the frozen model; run bounded decoder adaptation; interpret ANLS and exact match as "
        "one seeded synthetic-domain estimate; and export, verify and reload the adapter without pickle."
    ),
    "exclusions": (
        "DocVQA benchmark claims, PDFs or multi-page reasoning, OCR boxes or answer localisation, confidence calibration, "
        "full-model fine-tuning, hyperparameter search, production throughput, and local execution evidence. The synthetic "
        "corpus tests a narrow invoice-like domain and cannot establish real-scan performance."
    ),
    "prerequisites": [
        '- **Learner:** basic Python and Colab or Kaggle familiarity; no prior experience with Pix2Struct or fine-tuning. ANLS, exact match, baselines, document-level splits and adapters are explained where they are first used and again in the Glossary.',
        "- **Runtime:** a fresh Google Colab or Kaggle **Linux x86_64** runtime with a CUDA GPU; Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so nothing is installed into the kernel and no restart is needed. The notebook refuses CPU in Section 1, before any download (choose **Runtime → Change runtime type → T4 GPU** first). The pinned checkpoint is about 1.13 GB and is downloaded at its immutable revision, then checked against the embedded manifest.",
        "- **Duration (T4):** the recorded Kaggle Tesla T4 pass of 26 September 2026 took **212.1 s** for the ten code cells once the environment was in place (adaptation 94.7 s, frozen and adapted evaluation about 20 s each); a first Run all adds the environment build and the 1.13 GB download. No CPU figure exists because the notebook does not run on CPU.",
        "- **Knowledge:** Python, supervised train/validation/test splits, autoregressive generation, and why held-out ANLS is not a confidence score.",
        "- **Data:** the default path generates 40 fictional documents and 120 question rows in code under CC0-1.0: 72 train, 18 validation and 30 held-out test rows. Optional BYOD must contain only documents you are authorized to process, at least 17 documents at three questions each (every split needs 8 rows), at most 5,001 zip members and 2 GiB extracted; refusals name the zip, the row and the rule. Do not upload confidential or restricted data to a hosted notebook.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Build or upload the corpus, then validate it\n\n"
                "The default corpus has three questions per document and fixed document-level splits. `validate_dataset` checks "
                "every id, decoded image, question and accepted answer; `check_split_disjoint` refuses leakage. The aggregate "
                "digest must equal `SAMPLE_DIGEST` (the page images depend on the pinned Pillow 11.3.0 build, which the isolated environment provides). Two refusal probes demonstrate duplicate-id and cross-split rejection. With `USE_BYOD`, the zip is checked for its member count and extracted size before it is read, every split must hold at least `MIN_RECORDS` (8) rows, and each refusal names the zip, the row and the rule."
                '\n\n**Predict before running:** why split by document instead of by question? How many train, validation and test rows will the 120 questions give?'
            ),
            "code": (
                "import json\n"
                "import os\n"
                "import zipfile\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42\n"
                "os.makedirs('outputs', exist_ok=True)\n\n\n"
                'def byod_file(path, kind, suffixes=()):\n'
                '    """BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog."""\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if not source.is_file():\n'
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise RuntimeError(f'BYOD is on but its path field is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set the path field.') from None\n"
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '        name, payload = next(iter(uploaded.items()))\n'
                "        source = Path('work') / Path(name).name\n"
                '        source.parent.mkdir(parents=True, exist_ok=True)\n'
                '        source.write_bytes(payload)\n'
                '    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n'
                '        raise ValueError(f\'{{source.name}}: expected a {{kind}} ending in {{" or ".join(suffixes)}}.\')\n'
                '    return source\n'
                '\n'
                '\n'
                "if USE_BYOD:\n"
                "    byod_zip = byod_file(BYOD_PATH, 'zip with records.csv and page images', ('.zip',))\n"
                "    if not zipfile.is_zipfile(byod_zip):\n"
                "        raise ValueError(f'{{byod_zip.name}}: not a zip archive; pack records.csv and the page images into one .zip.')\n"
                "    BYOD_MAX_MEMBERS, BYOD_MAX_EXPANDED_BYTES = MAX_RECORDS + 1, 2 * 1024 ** 3  # one page per record plus records.csv; 2 GiB extracted\n"
                "    with zipfile.ZipFile(byod_zip) as archive:\n"
                "        members = [m for m in archive.infolist() if not m.is_dir()]\n"
                "        expanded = sum(m.file_size for m in members)\n"
                "    if len(members) > BYOD_MAX_MEMBERS or expanded > BYOD_MAX_EXPANDED_BYTES:\n"
                "        raise ValueError(f'{{byod_zip.name}}: {{len(members)}} files and {{expanded:,}} bytes when extracted exceed the BYOD ceiling of {{BYOD_MAX_MEMBERS}} files / {{BYOD_MAX_EXPANDED_BYTES:,}} bytes (a dataset holds at most MAX_RECORDS = {{MAX_RECORDS}} rows); pack fewer or smaller pages.')\n"
                "    records = load_byod_dataset(str(byod_zip))\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    documents = {{name: len({{r['document_id'] for r in rows}}) for name, rows in splits.items()}}\n"
                "    small = {{name: len(rows) for name, rows in splits.items() if len(rows) < MIN_RECORDS}}\n"
                "    if small:\n"
                "        raise ValueError(f'{{byod_zip.name}}: split(s) {{small}} hold fewer than MIN_RECORDS = {{MIN_RECORDS}} rows (rows per split {{ {{name: len(rows) for name, rows in splits.items()}} }}, documents per split {{documents}}, {{len(records)}} rows on {{sum(documents.values())}} documents in total). Every split needs at least {{MIN_RECORDS}} rows; with the 0.15 / 0.20 validation / test fractions that is 17 documents at three questions each — add documents or questions.')\n"
                "    print({{'byod_zip': byod_zip.name, 'members': len(members), 'expanded_bytes': expanded, 'rows': len(records), 'documents_per_split': documents}})\n"
                "    data_source = 'BYOD'\n"
                "else:\n"
                "    splits = build_sample_dataset()\n"
                "    data_source = CORPUS_NAME\n\n"
                "validated = {{name: validate_dataset(rows) for name, rows in splits.items()}}\n"
                "splits = {{name: value['records'] for name, value in validated.items()}}\n"
                "split_counts = check_split_disjoint(splits)\n"
                "all_records = [record for rows in splits.values() for record in rows]\n"
                "corpus_digest = dataset_digest(all_records)\n"
                "if not USE_BYOD and corpus_digest != SAMPLE_DIGEST:\n"
                "    import PIL\n"
                "    raise AssertionError(f'corpus digest {{corpus_digest[:16]}}... differs from SAMPLE_DIGEST {{SAMPLE_DIGEST[:16]}}...: the page images depend on the Pillow build (pinned pillow==11.3.0; this environment runs Pillow {{PIL.__version__}}); run in the isolated environment Section 1 builds')\n"
                "findings = []\n"
                "try:\n"
                "    validate_dataset([*splits['train'][:8], {{**splits['train'][0]}}])\n"
                "except ValueError as exc:\n"
                "    findings.append({{'probe': 'duplicate-id', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "try:\n"
                "    check_split_disjoint({{'train': splits['train'], 'test': [splits['train'][0]]}})\n"
                "except ValueError as exc:\n"
                "    findings.append({{'probe': 'document-leakage', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "assert len(findings) == 2\n"
                "input_manifest = {{'source': data_source, 'license': CORPUS_LICENSE if not USE_BYOD else 'caller-owned', 'splits': split_counts, 'digest': corpus_digest, 'expected_digest': None if USE_BYOD else SAMPLE_DIGEST, 'document_disjoint': True, 'findings': findings}}\n"
                "Path('outputs/{stem}_input_manifest.json').write_text(json.dumps(input_manifest, indent=2), encoding='utf-8')\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Three questions about the same invoice share its layout and values, so a question-level split would let the model see a test document in training. The recorded run split 72 / 18 / 30 rows with no document shared; the corpus digest matched `SAMPLE_DIGEST`, and both refusal probes (duplicate id, document leakage) were rejected.</details>'
            ),
        },
        {
            "md": (
                "## 5. Record non-neural baselines and the frozen model\n\n"
                "The empty baseline proves the scorer's zero floor. The majority baseline ignores page and question and uses "
                "one training-only answer. The frozen checkpoint is evaluated before any update. ANLS gives partial credit only "
                "at normalized similarity 0.5 or above; exact match is stricter. These are one-split point estimates. The printed `adapted` flag must be `False`: it says which weights were scored, and the cell stops if a re-run somehow scored an adapted pipeline (it reloads the frozen one first)."
                '\n\n**Predict before running:** what ANLS will the empty answer and the majority answer score, and how close to 1.0 will the frozen model be on these clean synthetic invoices?'
            ),
            "code": (
                "import torch\n\n"
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('The E2E default path requires a CUDA GPU; use a supported GPU runtime.')\n"
                "gpu_name = torch.cuda.get_device_name(0)\n"
                '# SWP-F: adapt() trains decoder blocks 10-11 of `pipe` in place, and Section 8 frees `pipe`. If this pipeline was\n'
                '# already adapted or freed (a re-run), start again from the pinned, digest-verified snapshot, so the frozen scores and\n'
                '# every new adaptation begin from the frozen weights they are labelled with.\n'
                "if globals().get('pipe') is None or pipe.adapter is not None:\n"
                '    import gc\n'
                '\n'
                '    pipe = None\n'
                '    gc.collect()\n'
                '    if torch.cuda.is_available():\n'
                '        torch.cuda.empty_cache()\n'
                '    pipe = Pix2StructDocVQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n'
                "    print({{'reloaded_frozen_pipeline': True, 'reason': 'the previous pipeline had been adapted in place or freed'}})\n"
                '\n'
                "train_records = splits['train']\n"
                "val_records = splits['validation']\n"
                "test_records = splits['test']\n"
                "empty = empty_baseline(test_records)\n"
                "majority = majority_answer_baseline(test_records, train_records)\n"
                "frozen = pipe.evaluate(test_records)\n"
                "print({{'gpu': gpu_name, 'empty': {{k: empty[k] for k in ('anls', 'exact_match')}}, 'majority': {{k: majority[k] for k in ('anls', 'exact_match', 'answer')}}, 'frozen': {{k: frozen[k] for k in ('anls', 'exact_match', 'n', 'seconds', 'adapted')}}}})\n"
                "if frozen['adapted']:\n"
                "    raise RuntimeError('the pipeline scored here carries an adapter; these are not frozen-model numbers')"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>In the recorded run the empty baseline scored ANLS 0.0 (the scorer's floor), the majority answer 0.2949 (one training answer happens to be close to some test answers) with exact match 0.0, and the frozen model **1.0** ANLS and exact match: it already answers every synthetic test question exactly. Keep that in mind for Section 7.</details>"
            ),
        },
        {
            "md": (
                "## 6. Adapt the final two decoder blocks\n\n"
                "`adapt` freezes every parameter except decoder blocks 10 and 11. AdamW runs for two bounded epochs with "
                "gradient clipping; epoch 0 records the frozen validation score and the highest validation ANLS wins. Training "
                "is transactional, and the test split is not consulted during selection. Because `adapt` updates `pipe` in place, this cell "
                "(and Section 5) first reload the frozen pipeline from the verified snapshot when `pipe` was already adapted or freed by an "
                "earlier run, so a re-run with other values never stacks on a previous adaptation."
                '\n\n**Predict before running:** the validation split already scores near the top for the frozen model. Which epoch will validation selection keep?'
            ),
            "code": (
                "EPOCHS = 2\n"
                "LEARNING_RATE = 2e-4\n"
                "BATCH_SIZE = 1\n"
                "TRAINABLE_DECODER_LAYERS = 2\n\n"
                "def report_epoch(entry):\n"
                "    print(json.dumps(entry, ensure_ascii=False))\n\n"
                '# SWP-F: adapt() trains decoder blocks 10-11 of `pipe` in place, and Section 8 frees `pipe`. If this pipeline was\n'
                '# already adapted or freed (a re-run), start again from the pinned, digest-verified snapshot, so the frozen scores and\n'
                '# every new adaptation begin from the frozen weights they are labelled with.\n'
                "if globals().get('pipe') is None or pipe.adapter is not None:\n"
                '    import gc\n'
                '\n'
                '    pipe = None\n'
                '    gc.collect()\n'
                '    if torch.cuda.is_available():\n'
                '        torch.cuda.empty_cache()\n'
                '    pipe = Pix2StructDocVQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n'
                "    print({{'reloaded_frozen_pipeline': True, 'reason': 'the previous pipeline had been adapted in place or freed'}})\n"
                '\n'
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, seed=0, progress=report_epoch)\n"
                "assert adapt_result['n_train'] == len(train_records)\n"
                "assert adapt_result['n_val'] == len(val_records)\n"
                "assert all(name.startswith(('decoder.layer.10.', 'decoder.layer.11.')) for name in adapt_result['trainable_names'])\n"
                "print({{k: adapt_result[k] for k in ('n_trainable', 'n_total', 'best_epoch', 'selection', 'seconds')}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the recorded run the frozen validation score (epoch 0) was ANLS 0.9815 / exact match 0.9444, and epochs 1 and 2 both reached 1.0 / 1.0, so the earliest best epoch, **1**, was kept. Two decoder blocks (18,878,976 of 282,285,696 parameters) trained in about 95 s on a T4.</details>'
            ),
        },
        {
            "md": (
                "## 7. Evaluate the selected adapter on held-out documents\n\n"
                "The selected model is evaluated once on the held-out test rows. The comparison reports absolute ANLS and "
                "exact match plus deltas versus the frozen checkpoint and baselines. Improvement is not an execution gate: "
                "regressions remain evidence and must be recorded.\n\n"
                "**How to read a zero or near-ceiling delta.** On the recorded run the frozen model already answered every held-out "
                "synthetic question exactly (ANLS 1.0, exact match 1.0), so the adapted model could not score higher and the delta was "
                "0.0: this sample is **saturated** — it checks that adaptation, epoch selection, export and reload work, not that "
                "adaptation helps. The cell reports `sample_saturated` (frozen ANLS at or above 0.95), the ANLS headroom the frozen "
                "model left, which weights were scored, and the held-out rows the adapted model did not answer exactly (Section 8 "
                "writes all of them to `outputs/{stem}_predictions.csv`). A zero delta is therefore the expected outcome on the default "
                "corpus, not a broken run; a gain can only show on data the frozen model gets wrong — your own pages through BYOD, or a "
                "harder test set."
                '\n\n**Predict before running:** given Section 5, what is the largest test gain the adapted model *could* show here?'
            ),
            "code": (
                "adapted = pipe.evaluate(test_records)\n"
                "sample_saturated = frozen['anls'] >= 0.95\n"
                "print({{'scored_weights': {{'frozen_adapted_flag': frozen['adapted'], 'adapted_adapted_flag': adapted['adapted']}}, 'sample_saturated': sample_saturated, 'frozen_anls_headroom': round(1.0 - frozen['anls'], 4)}})\n"
                "if sample_saturated:\n"
                "    print('The frozen model already scores ANLS >= 0.95 on these held-out rows, so this sample cannot show an adaptation gain: a zero delta is the expected outcome, not a failure.')\n"
                "not_exact = [{{'id': row['id'], 'question': row['question'], 'prediction': row['prediction'], 'accepted': row['answers'], 'anls': row['anls']}} for row in adapted['rows'] if row['anls'] < 1.0]\n"
                "print({{'held_out_rows_not_answered_exactly': not_exact[:10] or 'none: every held-out answer matched an accepted answer exactly', 'count': len(not_exact)}})\n"
                "comparison = {{\n"
                "    'anls': {{'empty': empty['anls'], 'majority': majority['anls'], 'frozen': frozen['anls'], 'adapted': adapted['anls']}},\n"
                "    'exact_match': {{'empty': empty['exact_match'], 'majority': majority['exact_match'], 'frozen': frozen['exact_match'], 'adapted': adapted['exact_match']}},\n"
                "    'delta_vs_frozen': {{'anls': round(adapted['anls'] - frozen['anls'], 4), 'exact_match': round(adapted['exact_match'] - frozen['exact_match'], 4)}},\n"
                "    'n_test': len(test_records),\n"
                "    'estimation': 'one seeded synthetic document split; no dispersion estimate',\n"
                "    'sample_saturated': sample_saturated,\n"
                "}}\n"
                "print(json.dumps(comparison, indent=2))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>None: the frozen model already scored 1.0, so the recorded delta was 0.0 for both ANLS and exact match. This sample is **saturated** — it checks that adaptation, export and reload work, not that adaptation helps. A harder test set (real scans, unseen layouts) is needed to measure a gain.</details>'
            ),
        },
        {
            "md": (
                "## 8. Export, reload, and verify answer parity\n\n"
                "The artifact contains only trained decoder tensors in safetensors. Its manifest binds them to the exact base "
                "model id, revision and weight digest, plus the artifact size and digest. The live model is released before a "
                "fresh base loads the adapter; eight held-out answers must match before results are written. Because the frozen model answers these rows the same way on a saturated sample, answer parity alone cannot tell whether the adapter reached the model, so the cell also records every trained tensor before the live model is freed and requires the reloaded tensors to be identical — a base loaded without the adapter fails that check whenever training changed any tensor."
                '\n\n**Predict before running:** after freeing the model and reloading the adapter on a fresh base, will the eight held-out answers match exactly?'
            ),
            "code": (
                "import csv\n"
                "import gc\n"
                "import platform\n"
                "import transformers\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source, 'corpus_digest': corpus_digest}})\n"
                "parity_records = test_records[:8]\n"
                "expected_answers = [pipe.answer(record['image'], record['question'])['answer'] for record in parity_records]\n"
                "trained_names = sorted(adapt_result['trainable_names'])\n"
                "live_state = pipe._model.state_dict()\n"
                "expected_tensors = {{name: live_state[name].detach().cpu().clone() for name in trained_names}}\n"
                "del live_state\n"
                "del pipe\n"
                "gc.collect()\n"
                "torch.cuda.empty_cache()\n"
                "reloaded = Pix2StructDocVQAPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device='cuda:0')\n"
                "actual_answers = [reloaded.answer(record['image'], record['question'])['answer'] for record in parity_records]\n"
                "assert actual_answers == expected_answers\n"
                "reloaded_state = reloaded._model.state_dict()\n"
                "tensor_parity = {{'identical_tensors': sum(torch.equal(expected_tensors[name], reloaded_state[name].detach().cpu()) for name in trained_names), 'of': len(trained_names)}}\n"
                "del reloaded_state\n"
                "assert tensor_parity['identical_tensors'] == tensor_parity['of'], tensor_parity\n"
                "reload_parity = {{'identical_answers': len(actual_answers), 'of': len(expected_answers), **tensor_parity}}\n"
                "result = {{'comparison': comparison, 'frozen': frozen, 'adapted': adapted, 'adaptation': adapt_result, 'reload_parity': reload_parity, 'input_manifest': input_manifest, 'notebook_source': NOTEBOOK_SOURCE, 'repository_revision': NOTEBOOK_SOURCE['repository_revision'], 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'model_license': MODEL_LICENSE, 'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': reloaded.device, 'gpu': gpu_name}}}}\n"
                "Path('outputs/{stem}_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['id', 'document_id', 'question', 'prediction', 'accepted_answers', 'anls', 'exact_match'])\n"
                "    for row in adapted['rows']:\n"
                "        writer.writerow([row['id'], row['document_id'], row['question'], row['prediction'], '|'.join(row['answers']), row['anls'], row['exact_match']])\n"
                "print({{'artifact_files': sorted(path.name for path in artifact_dir.iterdir()), 'reload_parity': reload_parity, 'outputs': sorted(os.listdir('outputs'))}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>Yes: the recorded run reported 8 of 8 identical answers, with a 75,519,448-byte `adapter.safetensors` bound to the base model's digest. The tensor check (every trained tensor identical after reload) was added after that run and has no recorded number yet.</details>"
            ),
        },
        {
            "md": (
                "### Optional experiment: one trainable decoder block instead of two (off by default)\n\n"
                "**Predict → change one thing → run → observe → explain.** Set `RUN_BLOCK_COMPARISON = True` and run this cell: it loads a "
                "**fresh frozen pipeline** from the verified snapshot (so nothing stacks on the adapter you just exported), adapts only "
                "decoder block 11 with the Section 6 settings, scores the same held-out rows, saves the one-block adapter to its own "
                "folder and prints both runs side by side. Predict first: will one block reach the same held-out ANLS as two, how much "
                "smaller will its artifact be (count the tensors), and will validation selection keep the same epoch? On the saturated "
                "default corpus both runs can only tie at the ceiling, which is itself the observation to explain. The default artifact "
                "and `result.json` are untouched; the experiment costs one more adaptation (about 95 s on a T4 for two blocks)."
            ),
            "code": (
                "RUN_BLOCK_COMPARISON = False  # @param {{type:\"boolean\"}}\n\n"
                "if not RUN_BLOCK_COMPARISON:\n"
                "    print({{'block_comparison_experiment': 'skipped (set RUN_BLOCK_COMPARISON = True to run it); nothing was trained or written'}})\n"
                "else:\n"
                "    experiment_pipe = Pix2StructDocVQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)  # fresh frozen weights: the experiment never touches `reloaded` or the exported artifact\n"
                "    one_block = experiment_pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=1, seed=0, progress=report_epoch)\n"
                "    one_block_test = experiment_pipe.evaluate(test_records)\n"
                "    experiment_dir = Path('outputs/{stem}_adapter_experiment_1block')\n"
                "    experiment_pipe.save_artifact(experiment_dir, metadata={{'tutorial': '{stem}', 'experiment': 'one trainable decoder block', 'data_source': data_source, 'corpus_digest': corpus_digest}})\n"
                "    experiment_manifest = json.loads((experiment_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "    default_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "    summary = {{\n"
                "        'one_block': {{'best_epoch': one_block['best_epoch'], 'n_trainable': one_block['n_trainable'], 'test_anls': one_block_test['anls'], 'test_exact_match': one_block_test['exact_match'], 'artifact_tensors': len(experiment_manifest['tensors']), 'artifact_bytes': experiment_manifest['files'][0]['bytes']}},\n"
                "        'two_blocks': {{'best_epoch': adapt_result['best_epoch'], 'n_trainable': adapt_result['n_trainable'], 'test_anls': adapted['anls'], 'test_exact_match': adapted['exact_match'], 'artifact_tensors': len(default_manifest['tensors']), 'artifact_bytes': default_manifest['files'][0]['bytes']}},\n"
                "        'sample_saturated': sample_saturated, 'default_artifact_unchanged': True,\n"
                "    }}\n"
                "    print({{'block_comparison_experiment': summary}})\n"
                "    del experiment_pipe\n"
                "    gc.collect()\n"
                "    torch.cuda.empty_cache()"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>No recorded run of this experiment exists, so there is no reference number. What to look for: the one-block artifact holds about half the tensors and bytes of the two-block one (block 11 plus the final layer norm); on the saturated default corpus both reach the ceiling, so the held-out numbers tie and only the validation curve and the kept epoch can differ — a tie here is evidence about the sample, not about the number of blocks.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A successful run proves that this exact carrier can verify the pinned snapshot, create and validate its fixed corpus, "
        "measure frozen and adapted held-out behavior, serialize only the intended tensors, and reproduce selected answers "
        "after a fresh reload. It does not prove DocVQA benchmark quality, real-document generalization, calibration, robustness "
        "to handwriting or unseen layouts, or production fitness. ANLS and exact match are corpus averages, not answer "
        "confidence. A negative delta is valid evidence. The generated names and transactions are fictional; BYOD users own "
        "consent, licensing, retention and access control.\n\n"
        "Successful execution proves that the recorded repository revision, carried in this notebook without the repository being "
        "reachable, completes the stated E2E path. It does **not** establish benchmark superiority or production readiness.\n\n"
        "**Next experiments (none affects the default path or its exported artifact):** every re-run of the Section 5 or Section 6 cell first "
        "reloads the frozen pipeline from the verified snapshot when `pipe` was adapted or freed, so each experiment starts from the base "
        "weights and its epoch 0 equals the Section 5 frozen score. Change `EPOCHS`, `LEARNING_RATE` or `TRAINABLE_DECODER_LAYERS` in "
        "Section 6, then re-run the Section 6, 7 and 8 cells in order (Section 8 overwrites the exported adapter with the new run); run "
        "the one-block comparison cell above; add an external document-domain test set or your own pages through BYOD (Section 4 cell, "
        "then Sections 5–8) — the only way a gain can show, since the default corpus is saturated; repeat across seeds; stratify by "
        "field type and layout; and read the held-out rows Section 7 lists.\n\n"
        '## Troubleshooting\n\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"This notebook needs a CUDA GPU" (Section 1) or "The E2E default path requires a CUDA GPU" (Section 5)** — choose **Runtime → Change runtime type → T4 GPU** and Run all again; Section 1 stops before any download.\n'
        '- **The delta in Section 7 is 0.0** — expected on the default corpus: the frozen model already answers every synthetic question (`sample_saturated: True`); it is not a broken run. A gain can only show on pages the frozen model gets wrong.\n'
        '- **A re-run scores "frozen" numbers that differ from the first pass** — Sections 5 and 6 reload the frozen pipeline whenever `pipe` was adapted or freed (they print `reloaded_frozen_pipeline`); `frozen["adapted"]` must be `False`.\n'
        '- **"The isolated environment\'s Python process exited"** or **CUDA out of memory** — restart the session and choose **Run all**; if you re-ran Sections 5–7 after Section 8, restart instead (the reloaded adapter still holds GPU memory).\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again.\n'
        "- **The corpus digest differs from `SAMPLE_DIGEST`** — the page images depend on the Pillow build's bundled font rendering; use the pinned environment Section 1 builds.\n"
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one …"** — set `BYOD_PATH` to the zip in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **BYOD: "exceed the BYOD ceiling", "not a zip archive", "hold fewer than MIN_RECORDS", or a `load_byod_dataset` / `validate_dataset` refusal** — the message names the zip, the `records.csv` row and the rule (member count or extracted size, too few rows in a split, duplicate id, a file that is not a decodable image, empty question or answer); 17 documents at three questions each is the minimum.\n\n'
        '## Glossary\n\n'
        '- **OCR-free document QA** — answering a question about a page image without a separate text-recognition step.\n'
        '- **Patches** — the composite image (question header plus page) is cut into at most 2,048 patches the encoder reads.\n'
        '- **ANLS** — average normalised Levenshtein similarity: per question, 1 minus the normalised edit distance to the closest accepted answer, counted only at 0.5 or above, then averaged.\n'
        '- **Exact match** — the share of answers identical to an accepted answer after normalisation.\n'
        '- **Empty / majority-answer baseline** — answering nothing; answering the most common training answer to every question.\n'
        '- **Document-level split** — all questions about one document stay in one split, so the test documents are unseen.\n'
        '- **Saturated sample** — the frozen model already scores the maximum, so no adaptation gain can show.\n'
        '- **Epoch / validation selection** — one pass over the training rows; keeping the epoch with the best validation ANLS (epoch 0, the frozen model, included).\n'
        '- **Adapter / reload parity** — the trained decoder tensors only, bound to the base digest; the reloaded model gives identical answers.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: your pages and questions through the same cells.\n\n'
        '## Conclusion (your notes)\n\nWrite your conclusion from **your** run (optional); the recorded run is only a reference:\n\n'
        '- Empty ___, majority ___, frozen ___, adapted ___ (test ANLS); the delta was ___.\n'
        '- The kept epoch was ___ because ___.\n'
        '- What this sample can and cannot show about adaptation: ___.\n'
        '- One harder test set I would build next: ___.\n\n'
        "## References\n\n"
        "- Repository: https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline\n"
        "- Repository model card: https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream checkpoint: https://huggingface.co/{MODEL_ID}\n"
        "- Lee, K., Joshi, M., Turc, I., Hu, H., Liu, F., Eisenschlos, J., Khandelwal, U., Shaw, P., Chang, M.-W., & Toutanova, K. (2023). Pix2Struct: Screenshot parsing as pretraining for visual language understanding. *Proceedings of the 40th International Conference on Machine Learning* (PMLR 202). https://arxiv.org/abs/2210.03347 (https://doi.org/10.48550/arXiv.2210.03347)\n"
        "- Mathew, M., Karatzas, D., & Jawahar, C. V. (2021). DocVQA: A dataset for VQA on document images. *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*. https://arxiv.org/abs/2007.00398 (https://doi.org/10.48550/arXiv.2007.00398)\n"
        "- Biten, A. F., Tito, R., Mafla, A., Gomez, L., Rusiñol, M., Valveny, E., Jawahar, C. V., & Karatzas, D. (2019). Scene text visual question answering (the ANLS metric). *IEEE/CVF International Conference on Computer Vision (ICCV)*. https://arxiv.org/abs/1905.13648 (https://doi.org/10.48550/arXiv.1905.13648)\n"
        "- DIMER Notebook Specification 2.2 (fleet specs in the ml-worker repository)"
    ),
}
