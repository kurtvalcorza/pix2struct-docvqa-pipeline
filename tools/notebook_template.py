"""E2E notebook template for the standalone NOTEBOOK_SPEC 2.0 carrier."""

# ruff: noqa: E501

TEMPLATE = {
    "package": "pix2struct_docvqa_pipeline",
    "repo_name": "pix2struct-docvqa-pipeline",
    "stem": "pix2struct_docvqa",
    "notebook_name": "pix2struct_docvqa_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "Pix2StructDocVQAPipeline",
    "weights_key": "pix2struct-docvqa-base",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers"],
    "run_all": (
        "Selecting **Run all** in a fresh supported GPU runtime installs the pinned dependencies, carries the three "
        "repository modules without cloning the repository, stages and digest-verifies the immutable model snapshot, "
        "builds and validates the deterministic synthetic corpus, records two non-neural baselines and the frozen model, "
        "fine-tunes only the final two decoder blocks with validation-ANLS epoch selection, evaluates held-out documents, "
        "exports a safetensors adapter bound to the base digest, reloads it in a fresh model instance, checks answer parity, "
        "and writes machine-readable evidence. CUDA is required; no credential or upload is needed."
    ),
    "byod": (
        "After the default sample completes, set `USE_BYOD = True` and upload one zip containing `records.csv` plus its "
        "page images. CSV columns are `id,document_id,file,question,answers`; separate accepted answers with `|`. The same "
        "validation and document-grouped split apply. BYOD is optional and remains inside the hosted runtime."
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
        "- **Runtime:** a fresh Google Colab or Kaggle-style Python 3.12 runtime with a CUDA GPU. The notebook refuses CPU for adaptation. The pinned checkpoint is about 1.13 GB and is downloaded at its immutable revision, then checked against the embedded manifest.",
        "- **Knowledge:** Python, supervised train/validation/test splits, autoregressive generation, and why held-out ANLS is not a confidence score.",
        "- **Data:** the default path generates 40 fictional documents and 120 question rows in code under CC0-1.0: 72 train, 18 validation and 30 held-out test rows. Optional BYOD must contain only documents you are authorized to process. Do not upload confidential or restricted data to a hosted notebook.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Build or upload the corpus, then validate it\n\n"
                "The default corpus has three questions per document and fixed document-level splits. `validate_dataset` checks "
                "every id, decoded image, question and accepted answer; `check_split_disjoint` refuses leakage. The aggregate "
                "digest must equal `SAMPLE_DIGEST`. Two refusal probes demonstrate duplicate-id and cross-split rejection."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42\n"
                "os.makedirs('outputs', exist_ok=True)\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    upload_name = next(iter(uploaded))\n"
                "    Path(upload_name).write_bytes(uploaded[upload_name])\n"
                "    records = load_byod_dataset(upload_name)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD'\n"
                "else:\n"
                "    splits = build_sample_dataset()\n"
                "    data_source = CORPUS_NAME\n\n"
                "validated = {{name: validate_dataset(rows) for name, rows in splits.items()}}\n"
                "splits = {{name: value['records'] for name, value in validated.items()}}\n"
                "split_counts = check_split_disjoint(splits)\n"
                "all_records = [record for rows in splits.values() for record in rows]\n"
                "corpus_digest = dataset_digest(all_records)\n"
                "if not USE_BYOD:\n"
                "    assert corpus_digest == SAMPLE_DIGEST, (corpus_digest, SAMPLE_DIGEST)\n"
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
                "## 5. Record non-neural baselines and the frozen model\n\n"
                "The empty baseline proves the scorer's zero floor. The majority baseline ignores page and question and uses "
                "one training-only answer. The frozen checkpoint is evaluated before any update. ANLS gives partial credit only "
                "at normalized similarity 0.5 or above; exact match is stricter. These are one-split point estimates."
            ),
            "code": (
                "import torch\n\n"
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('The E2E default path requires a CUDA GPU; use a supported GPU runtime.')\n"
                "gpu_name = torch.cuda.get_device_name(0)\n"
                "train_records = splits['train']\n"
                "val_records = splits['validation']\n"
                "test_records = splits['test']\n"
                "empty = empty_baseline(test_records)\n"
                "majority = majority_answer_baseline(test_records, train_records)\n"
                "frozen = pipe.evaluate(test_records)\n"
                "print({{'gpu': gpu_name, 'empty': {{k: empty[k] for k in ('anls', 'exact_match')}}, 'majority': {{k: majority[k] for k in ('anls', 'exact_match', 'answer')}}, 'frozen': {{k: frozen[k] for k in ('anls', 'exact_match', 'n', 'seconds')}}}})"
            ),
        },
        {
            "md": (
                "## 6. Adapt the final two decoder blocks\n\n"
                "`adapt` freezes every parameter except decoder blocks 10 and 11. AdamW runs for two bounded epochs with "
                "gradient clipping; epoch 0 records the frozen validation score and the highest validation ANLS wins. Training "
                "is transactional, and the test split is not consulted during selection."
            ),
            "code": (
                "EPOCHS = 2\n"
                "LEARNING_RATE = 2e-4\n"
                "BATCH_SIZE = 1\n"
                "TRAINABLE_DECODER_LAYERS = 2\n\n"
                "def report_epoch(entry):\n"
                "    print(json.dumps(entry, ensure_ascii=False))\n\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, seed=0, progress=report_epoch)\n"
                "assert adapt_result['n_train'] == len(train_records)\n"
                "assert adapt_result['n_val'] == len(val_records)\n"
                "assert all(name.startswith(('decoder.layer.10.', 'decoder.layer.11.')) for name in adapt_result['trainable_names'])\n"
                "print({{k: adapt_result[k] for k in ('n_trainable', 'n_total', 'best_epoch', 'selection', 'seconds')}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate the selected adapter on held-out documents\n\n"
                "The selected model is evaluated once on the held-out test rows. The comparison reports absolute ANLS and "
                "exact match plus deltas versus the frozen checkpoint and baselines. Improvement is not an execution gate: "
                "regressions remain evidence and must be recorded."
            ),
            "code": (
                "adapted = pipe.evaluate(test_records)\n"
                "comparison = {{\n"
                "    'anls': {{'empty': empty['anls'], 'majority': majority['anls'], 'frozen': frozen['anls'], 'adapted': adapted['anls']}},\n"
                "    'exact_match': {{'empty': empty['exact_match'], 'majority': majority['exact_match'], 'frozen': frozen['exact_match'], 'adapted': adapted['exact_match']}},\n"
                "    'delta_vs_frozen': {{'anls': round(adapted['anls'] - frozen['anls'], 4), 'exact_match': round(adapted['exact_match'] - frozen['exact_match'], 4)}},\n"
                "    'n_test': len(test_records),\n"
                "    'estimation': 'one seeded synthetic document split; no dispersion estimate',\n"
                "}}\n"
                "print(json.dumps(comparison, indent=2))"
            ),
        },
        {
            "md": (
                "## 8. Export, reload, and verify answer parity\n\n"
                "The artifact contains only trained decoder tensors in safetensors. Its manifest binds them to the exact base "
                "model id, revision and weight digest, plus the artifact size and digest. The live model is released before a "
                "fresh base loads the adapter; eight held-out answers must match before results are written."
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
                "del pipe\n"
                "gc.collect()\n"
                "torch.cuda.empty_cache()\n"
                "reloaded = Pix2StructDocVQAPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device='cuda:0')\n"
                "actual_answers = [reloaded.answer(record['image'], record['question'])['answer'] for record in parity_records]\n"
                "assert actual_answers == expected_answers\n"
                "reload_parity = {{'identical_answers': len(actual_answers), 'of': len(expected_answers)}}\n"
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
        "**Next experiments:** repeat across seeds; add an external document-domain test set; compare one versus two trainable "
        "decoder blocks; stratify by field type and layout; and inspect every held-out error.\n\n"
        "## References\n\n"
        "- Repository: https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline\n"
        "- Repository model card: https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream checkpoint: https://huggingface.co/{MODEL_ID}\n"
        "- Pix2Struct paper: https://arxiv.org/abs/2210.03347\n"
        "- DocVQA paper: https://arxiv.org/abs/2007.00398\n"
        "- ANLS origin: https://arxiv.org/abs/1905.13648"
    ),
}
