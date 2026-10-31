"""E2E notebook template for the standalone NOTEBOOK_SPEC 2.5 carrier."""

# ruff: noqa: E501

TEMPLATE = {
    "package": "pix2struct_docvqa_pipeline",
    "repo_name": "pix2struct-docvqa-pipeline",
    "stem": "pix2struct_docvqa",
    "notebook_name": "pix2struct_docvqa_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "notebook_kind": "TUTORIAL",
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
        "and writes machine-readable evidence. CUDA is required (Section 1 stops before any download when it is absent); no credential or upload is needed. The first Run all also downloads the 1.13 GB snapshot and builds the environment. Recorded hosted runs, each with the notebook revision (blob) it ran, its wall time and its cell count, are listed in `docs/release-verification.md` in the repository."
    ),
    "byod": (
        "After the default sample completes, set `USE_BYOD = True` and `BYOD_PATH` (a zip already in the runtime; on Colab an empty path opens an upload dialog) in Section 4, then re-run Section 4 and every code cell of Sections 5–8 in order (Section 5 reloads the frozen pipeline from the verified snapshot, so the frozen numbers are the base model's on your data) for one zip containing `records.csv` plus its "
        "page images (folders inside the zip are fine; files are matched by name). CSV columns are `id,document_id,file,question,answers`; separate accepted answers with `|`. The same "
        "validation and document-grouped split apply: every split needs at least 8 rows, which with the 0.15 / 0.20 validation / test fractions means 17 documents at three questions each (10 documents are refused by name); at most 5,001 zip members and 2 GiB extracted. Document ids keep the split document-disjoint, but they cannot detect the same page uploaded twice under different ids, pages of one related document given different ids, or near-duplicates: give such pages one document id yourself. BYOD is optional and remains inside the hosted runtime."
    ),
    # DOC17-DOC21: canonical NOTEBOOK_SPEC 2.5 use notice (BYOD sentence kept: this notebook has a BYOD path).
    "use_notice": (
        "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n"
        "> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n"
        "> model weights and datasets under their own licenses (see the model card) and uses compute\n"
        "> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n"
        "> responsible for having the right to use it. Results here are tutorial evidence, not\n"
        "> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n"
        "> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n"
        "> its use."
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
        "Pix2Struct [1] renders each question above its page, encodes the composite as at most 2,048 patches, and generates a "
        "short answer. This tutorial makes that path trainable without changing its inference contract: the vision encoder, "
        "embeddings, language head and first ten decoder blocks stay frozen; only decoder blocks 10 and 11 are updated. A "
        "deterministic, code-generated business-document corpus (`samples.py` in the repository [7]) avoids registrations, remote dataset drift and private data. "
        "Documents—not question rows—define the train/validation/test boundary. Validation ANLS selects the epoch; the test "
        "set is used only for frozen/adapted comparison. This carrier remains Candidate until its exact committed blob passes "
        "a clean supported GPU run.\n\n"
        "**Data context and overlap.** The checkpoint `google/pix2struct-docvqa-base` was fine-tuned on DocVQA [2], [6]. The default "
        "corpus is generated in code (40 fictional CC0 documents, digest-pinned), so it cannot overlap that fine-tuning data; but "
        "DocVQA train or validation pages are **not** held-out data for this checkpoint, so do not use them as an external test set "
        "or as BYOD test documents.\n\n"
        "**What the default sample can show.** All three splits come from the same generator, so the test pages are **new document "
        "instances** with shared templates, vendors and layouts (two layouts, five vendors), not unseen templates; the results do "
        "not measure generalisation to unfamiliar layouts. The generator also has restrictions that are kept because "
        "`SAMPLE_DIGEST` pins its pages: only two of its six cities (Manila and Baguio) are reachable, and each vendor always "
        "has the same contact. The questions ask only for the invoice number, purchase-order number and total. Section 4 prints "
        "the vendor, city and layout frequencies per split."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Kaggle with a GPU and has met train/validation/test splits, and wants to see how an OCR-free document question-answering model reads a page image, how its answers are scored (ANLS [3] and exact match, following the ANLS rules of the MP-DocVQA-Framework evaluator [4], consistent with the RRC DocVQA task description [5]) against simple baselines, and how a small part of it is fine-tuned and exported without fooling themselves. The audience is students and practitioners preparing their own document-QA data; no prior experience with Pix2Struct or fine-tuning is assumed — each term is explained where it first matters and again in the **Glossary**. A CUDA GPU is required (a T4 is enough).\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | page images with a question and one or more accepted answers: 40 generated invoice-like documents, 120 questions (72 train, 18 validation, 30 test rows, split by document) or your own zip |\n| Model | Pix2Struct DocVQA-base: the question is rendered above the page, the composite is encoded as at most 2,048 patches, and a decoder generates the answer; only decoder blocks 10 and 11 are trained |\n| Output | a short answer per question, held-out ANLS and exact match beside an empty baseline and a raw first-answer frequency baseline, and a safetensors adapter that reloads with identical answers |\n\n**How to use this notebook.** Choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried modules and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer quoted from the recorded run of an earlier revision of this notebook, blob `c1ea0b6d` (commit `f197e08`, Colab T4, 8 October 2026), not re-run for this revision (see `docs/release-verification.md`); that run scored answers with the earlier punctuation-insensitive normalisation, and a hosted run of this revision is pending. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the corpus, validation and a document-level split *(evaluation practice: leakage)* → 5 two baselines and the frozen model *(core concept: ANLS)* → 6 bounded adaptation of two decoder blocks *(core concept: what is trained)* → 7 held-out comparison *(evaluation practice: headroom near the ceiling)* → 8 export, free, reload and answer parity *(engineering)* → conclude."
    )]},
    "learning_objectives": (
        "verify immutable model and corpus identities; inspect record and split contracts; compare empty-answer and "
        "raw first-answer frequency baselines with the frozen model; explain what training, validation selection (epoch 0 included) "
        "and a held-out test estimate each do; run bounded decoder adaptation; interpret ANLS and exact match as "
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
        "- **Duration (T4):** on blob `c1ea0b6d` (commit `f197e08`, Colab T4, 8 October 2026) (an earlier revision, not re-run for this one) the 12 code cells took **268.7 s** in total on a fresh VM, the 54 s environment build included (adaptation 96.26 s, frozen evaluation 22.93 s). `docs/release-verification.md` lists every recorded run with its revision. No CPU figure exists because the notebook does not run on CPU.",
        "- **Knowledge:** basic Python. The machine-learning ideas the notebook uses (train/validation/test splits, frozen and trainable weights, greedy generation, why held-out ANLS is not a confidence score) are explained in a short primer before Section 6 and in the Glossary.",
        "- **Data:** the default path generates 40 fictional documents and 120 question rows in code under CC0-1.0: 72 train, 18 validation and 30 held-out test rows. Optional BYOD must contain only documents you are authorized to process, at least 17 documents at three questions each (every split needs 8 rows), at most 5,001 zip members and 2 GiB extracted; refusals name the zip, the row and the rule. Do not upload confidential or restricted data to a hosted notebook.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Build or upload the corpus, then validate it\n\n"
                "The default corpus has three questions per document and fixed document-level splits. `validate_dataset` checks "
                "every id, decoded image, question and accepted answer; `check_split_disjoint` refuses leakage. The aggregate "
                "digest must equal `SAMPLE_DIGEST` (the page images depend on the pinned Pillow 11.3.0 build, which the isolated environment provides). Two refusal probes demonstrate duplicate-id and cross-split rejection. With `USE_BYOD`, the zip is checked for its member count and extracted size before it is read, every split must hold at least `MIN_RECORDS` (8) rows, and a refused zip prints one message naming the zip, the row and the rule, with no traceback. A refusal clears the run state (`DATA_READY` becomes `False` and the result variables are dropped) and moves earlier result files into a `outputs/previous_run_<time>/` folder, kept rather than deleted, so nothing from a previous run can be read as the result for your data; Sections 5–8 refuse to run until this cell prints the input manifest. Fix the named file and run this cell again. On the default sample the cell also prints how often each vendor, city and layout occurs per split."
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
                "os.makedirs('outputs', exist_ok=True)\n"
                "# The file the learner gave, by name; '(no file)' until an upload arrives.\n"
                "byod_label = (Path(BYOD_PATH.strip()).expanduser().name or BYOD_PATH.strip()) if BYOD_PATH.strip() else '(no file)'\n\n\n"
                'def byod_file(path, kind, suffixes=()):\n'
                '    """BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog."""\n'
                '    global byod_label\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if not source.is_file():\n'
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise FileNotFoundError(f'BYOD is on but its path field is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set the path field.') from None\n"
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '        name, payload = next(iter(uploaded.items()))\n'
                '        byod_label = Path(name).name\n'
                "        source = Path('work') / Path(name).name\n"
                '        source.parent.mkdir(parents=True, exist_ok=True)\n'
                '        source.write_bytes(payload)\n'
                '    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n'
                '        raise ValueError(f\'{{source.name}}: expected a {{kind}} ending in {{" or ".join(suffixes)}}.\')\n'
                '    return source\n'
                '\n'
                '\n'
                "def load_byod_splits():\n"
                "    \"\"\"Read, bound and split the BYOD zip. ValueError / FileNotFoundError here is a refusal naming the zip, row and rule.\"\"\"\n"
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
                "    return splits\n\n\n"
                "class BYODRefusal(ValueError):\n"
                "    \"\"\"A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``.\"\"\"\n\n"
                "    dimer_refusal = True\n\n"
                "    def __init__(self, file_name, reason):\n"
                "        super().__init__(f\"{{file_name}}: {{reason}}\")\n"
                "        self.file_name = file_name\n"
                "        self.reason = reason\n\n"
                "    def _render_traceback_(self):\n"
                "        return [f\"Refused: {{self}}\"]\n\n\n"
                "# DAT20: a refused BYOD input is one Refused: line, not a crash, and Run all stops here. Genuine failures (and any failure of the default corpus) still raise.\n"
                "byod_refusal = None\n"
                "DATA_READY = False\n"
                "try:\n"
                "    splits = load_byod_splits() if USE_BYOD else build_sample_dataset()\n"
                "    data_source = 'BYOD' if USE_BYOD else CORPUS_NAME\n"
                "    validated = {{name: validate_dataset(rows) for name, rows in splits.items()}}\n"
                "    splits = {{name: value['records'] for name, value in validated.items()}}\n"
                "    split_counts = check_split_disjoint(splits)\n"
                "except (ValueError, FileNotFoundError) as exc:\n"
                "    if not USE_BYOD:\n"
                "        raise\n"
                "    byod_refusal = str(exc)\n"
                "    import shutil\n"
                "    import time\n"
                "    for stale_name in ('splits', 'train_records', 'val_records', 'test_records', 'all_records', 'corpus_digest', 'input_manifest', 'empty', 'majority', 'frozen', 'adapt_result', 'adapted', 'comparison', 'result'):\n"
                "        globals().pop(stale_name, None)\n"
                "    previous = sorted(Path('outputs').glob('{stem}_*'))\n"
                "    archive = Path('outputs') / ('previous_run_' + time.strftime('%Y%m%d-%H%M%S')) if previous else None\n"
                "    if archive is not None:\n"
                "        archive.mkdir(parents=True, exist_ok=True)\n"
                "        for stale in previous:\n"
                "            shutil.move(str(stale), str(archive / stale.name))\n"
                "    kept = f'Earlier result files were moved, not deleted, to {{archive}}/ so they cannot be read as results for your data. ' if archive is not None else ''\n"
                "    reason = byod_refusal.rstrip('.')\n"
                "    reason = reason[len(byod_label) + 2:] if reason.startswith(byod_label + ': ') else reason  # the file is named once\n"
                "    raise BYODRefusal(byod_label, f'{{reason}}. Nothing was scored. ' + kept + 'Fix the named file or row, then run this cell again; Sections 5-8 refuse to run until this cell prints the input manifest.') from None\n"
                "if byod_refusal is None:\n"
                "    all_records = [record for rows in splits.values() for record in rows]\n"
                "    corpus_digest = dataset_digest(all_records)\n"
                "    if not USE_BYOD and corpus_digest != SAMPLE_DIGEST:\n"
                "        import PIL\n"
                "        raise AssertionError(f'corpus digest {{corpus_digest[:16]}}... differs from SAMPLE_DIGEST {{SAMPLE_DIGEST[:16]}}...: the page images depend on the Pillow build (pinned pillow==11.3.0; this environment runs Pillow {{PIL.__version__}}); run in the isolated environment Section 1 builds')\n"
                "    findings = []\n"
                "    try:\n"
                "        validate_dataset([*splits['train'][:8], {{**splits['train'][0]}}])\n"
                "    except ValueError as exc:\n"
                "        findings.append({{'probe': 'duplicate-id', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "    try:\n"
                "        check_split_disjoint({{'train': splits['train'], 'test': [splits['train'][0]]}})\n"
                "    except ValueError as exc:\n"
                "        findings.append({{'probe': 'document-leakage', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "    assert len(findings) == 2\n"
                "    input_manifest = {{'source': data_source, 'license': CORPUS_LICENSE if not USE_BYOD else 'caller-owned', 'splits': split_counts, 'digest': corpus_digest, 'expected_digest': None if USE_BYOD else SAMPLE_DIGEST, 'document_disjoint': True, 'findings': findings}}\n"
                "    Path('outputs/{stem}_input_manifest.json').write_text(json.dumps(input_manifest, indent=2), encoding='utf-8')\n"
                "    print(json.dumps(input_manifest, indent=2))\n"
                "    if not USE_BYOD:\n"
                "        print(json.dumps({{'field_frequencies_per_split': sample_field_frequencies(splits)}}, indent=2))\n"
                "    DATA_READY = True"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Three questions about the same invoice share its layout and values, so a question-level split would let the model see a test document in training. In the recorded run of an earlier revision of this notebook, blob `c1ea0b6d` (commit `f197e08`, Colab T4, 8 October 2026), not re-run for this revision, the corpus split 72 / 18 / 30 rows with no document shared; the corpus digest matched `SAMPLE_DIGEST`, and both refusal probes (duplicate id, document leakage) were rejected.</details>'
            ),
        },
        {
            "md": (
                "## 5. Record non-neural baselines and the frozen model\n\n"
                "The empty baseline proves the scorer's zero floor. The raw first-answer frequency baseline (printed as `majority`) "
                "ignores page and question and answers every row with the most frequent first-listed training answer. The frozen "
                "checkpoint is evaluated before any update. Similarities below 0.5 get no ANLS credit, but a sufficiently similar "
                "wrong string still gets partial credit (an invoice number one digit off, `INV-2026-1008` for `INV-2026-1009`, scores "
                "12/13 ≈ 0.92), so every accepted answer must be a genuinely acceptable one; exact match is stricter. The cell also shows "
                "one test page with its question, accepted answers and the frozen prediction. These are one-split point estimates. The printed `adapted` flag must be `False`: it says which weights were scored, and the cell stops if a re-run somehow scored an adapted pipeline (it reloads the frozen one first)."
                '\n\n**Predict before running:** what ANLS will the empty answer and the raw first-answer frequency baseline score, and how close to 1.0 will the frozen model be on these clean synthetic invoices?'
            ),
            "code": (
                "import torch\n\n"
                "if not globals().get('DATA_READY'):\n"
                "    raise RuntimeError('Section 4 has not produced a validated corpus (a BYOD input was refused, or Section 4 has not run): fix the input and run Section 4 again before this cell.')\n"

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
                '    reloaded = None  # release the Section 8 reload before loading the base again\n'
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
                "    raise RuntimeError('the pipeline scored here carries an adapter; these are not frozen-model numbers')\n"
                "example = frozen['rows'][0]\n"
                "display(test_records[0]['image'].resize((384, 512)))\n"
                "print({{'example': {{'question': example['question'], 'accepted_answers': example['answers'], 'frozen_prediction': example['prediction'], 'anls': example['anls'], 'exact_match': example['exact_match']}}}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>In the recorded run of an earlier revision of this notebook, blob `c1ea0b6d` (commit `f197e08`, Colab T4, 8 October 2026), not re-run for this revision, the empty baseline scored ANLS 0.0 (the scorer's floor), the raw first-answer frequency baseline 0.2949 (its answer, an invoice number, is close to the test invoice numbers) with exact match 0.0, and the frozen model **1.0** ANLS and exact match: it already answered every synthetic test question exactly. That run used the earlier punctuation-insensitive normalisation; re-scoring the two baselines with this revision's rule gives the same 0.0 and 0.2949 (a CPU check of this revision's code, not a hosted run). Keep that in mind for Section 7.</details>"
            ),
        },
        {
            "md": (
                "## 6. Adapt the final two decoder blocks\n\n"
                "**Primer (plain language).** *Training* changes the model's weights so its answers on the training rows get closer to the "
                "accepted answers. Only the *trainable* weights change, here decoder blocks 10 and 11 (the decoder is the part that writes "
                "the answer, one token at a time); every other weight stays *frozen*. The *learning rate* sets how big each weight change is. "
                "One *epoch* is one pass over the training rows. *Validation* rows are never trained on: after each epoch they are scored, "
                "and the epoch with the best validation ANLS is the one kept, including epoch 0, the unchanged base weights. The *test* rows "
                "are scored once, in Section 7, to estimate performance after that selection. Answers are produced by *greedy generation*: "
                "at each step the model emits its single most likely next token.\n\n"
                "`adapt` freezes every parameter except decoder blocks 10 and 11. AdamW runs for two bounded epochs with "
                "gradient clipping; epoch 0 records the frozen validation score and the highest validation ANLS wins. Training "
                "is transactional, and the test split is not consulted during selection. Because `adapt` updates `pipe` in place, this cell "
                "(and Section 5) first reload the frozen pipeline from the verified snapshot when `pipe` was already adapted or freed by an "
                "earlier run (releasing Section 8's `reloaded` model first), so a re-run with other values never stacks on a previous "
                "adaptation. The cell also keeps a CPU copy of the trainable tensors before training, so Section 8 can report how far "
                "the selected weights moved."
                '\n\n**Predict before running:** the validation split already scores near the top for the frozen model. Which epoch will validation selection keep?'
            ),
            "code": (
                "if not globals().get('DATA_READY'):\n"
                "    raise RuntimeError('Section 4 has not produced a validated corpus (a BYOD input was refused, or Section 4 has not run): fix the input and run Section 4 again before this cell.')\n"
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
                '    reloaded = None  # release the Section 8 reload before loading the base again\n'
                '    gc.collect()\n'
                '    if torch.cuda.is_available():\n'
                '        torch.cuda.empty_cache()\n'
                '    pipe = Pix2StructDocVQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n'
                "    print({{'reloaded_frozen_pipeline': True, 'reason': 'the previous pipeline had been adapted in place or freed'}})\n"
                '\n'
                "base_names = set(pipe._trainable_names(TRAINABLE_DECODER_LAYERS))\n"
                "base_tensors = {{name: tensor.detach().cpu().clone() for name, tensor in pipe._model.state_dict().items() if name in base_names}}\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, seed=0, progress=report_epoch)\n"
                "assert adapt_result['n_train'] == len(train_records)\n"
                "assert adapt_result['n_val'] == len(val_records)\n"
                "assert all(name.startswith(('decoder.layer.10.', 'decoder.layer.11.')) for name in adapt_result['trainable_names'])\n"
                "print({{k: adapt_result[k] for k in ('n_trainable', 'n_total', 'best_epoch', 'selection', 'seconds')}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the recorded run of an earlier revision of this notebook, blob `c1ea0b6d` (commit `f197e08`, Colab T4, 8 October 2026), not re-run for this revision, the frozen validation score (epoch 0) was ANLS 0.9815 / exact match 0.9444, and epochs 1 and 2 both reached 1.0 / 1.0, so the earliest best epoch, **1**, was kept. Two decoder blocks (18,878,976 of 282,285,696 parameters) trained in 96.26 s on a T4.</details>'
            ),
        },
        {
            "md": (
                "## 7. Evaluate the selected adapter on held-out documents\n\n"
                "The selected model is evaluated once on the held-out test rows. The comparison reports absolute ANLS and "
                "exact match plus deltas versus the frozen checkpoint and baselines. Improvement is not an execution gate: "
                "regressions remain evidence and must be recorded.\n\n"
                "**How to read a delta near the ceiling.** The cell prints the ANLS headroom the frozen model left (1 − frozen ANLS). "
                "At the ceiling (`at_ceiling`: frozen ANLS 1.0) no improvement is possible, but a tie or a regression still is. Near the "
                "ceiling (`sample_saturated`: frozen ANLS at or above 0.95) a gain is possible but cannot exceed the headroom. In the recorded run of an earlier revision of this notebook, blob `c1ea0b6d` (commit `f197e08`, Colab T4, 8 October 2026), not re-run for this revision, "
                "the frozen model already answered every held-out synthetic question exactly (ANLS 1.0, exact match 1.0), so no gain was "
                "possible and the delta was 0.0. The cell also reports which weights were scored and the held-out rows the adapted model "
                "did not answer exactly (Section 8 writes all of them to `outputs/{stem}_predictions.csv`). A zero delta at the ceiling "
                "is not a broken run; a larger gain needs data the frozen model gets wrong — your own pages through BYOD, or a harder "
                "test set."
                '\n\n**Predict before running:** given Section 5, what is the largest test gain the adapted model *could* show here?'
            ),
            "code": (
                "if not globals().get('DATA_READY'):\n"
                "    raise RuntimeError('Section 4 has not produced a validated corpus (a BYOD input was refused, or Section 4 has not run): fix the input and run Section 4 again before this cell.')\n"
                "adapted = pipe.evaluate(test_records)\n"
                "frozen_anls_headroom = round(1.0 - frozen['anls'], 4)\n"
                "at_ceiling = frozen['anls'] >= 1.0\n"
                "sample_saturated = frozen['anls'] >= 0.95\n"
                "print({{'scored_weights': {{'frozen_adapted_flag': frozen['adapted'], 'adapted_adapted_flag': adapted['adapted']}}, 'sample_saturated': sample_saturated, 'at_ceiling': at_ceiling, 'frozen_anls_headroom': frozen_anls_headroom}})\n"
                "if at_ceiling:\n"
                "    print('Frozen ANLS is 1.0 on these held-out rows (at ceiling): no improvement is possible here, but a tie or a regression still is.')\n"
                "elif sample_saturated:\n"
                "    print(f'Frozen ANLS is {{frozen[\"anls\"]}} (near ceiling): a gain of at most {{frozen_anls_headroom}} ANLS is possible on these rows, and a regression is also possible.')\n"
                "not_exact = [{{'id': row['id'], 'question': row['question'], 'prediction': row['prediction'], 'accepted': row['answers'], 'anls': row['anls']}} for row in adapted['rows'] if row['anls'] < 1.0]\n"
                "print({{'held_out_rows_not_answered_exactly': not_exact[:10] or 'none: every held-out answer matched an accepted answer exactly', 'count': len(not_exact)}})\n"
                "comparison = {{\n"
                "    'anls': {{'empty': empty['anls'], 'majority': majority['anls'], 'frozen': frozen['anls'], 'adapted': adapted['anls']}},\n"
                "    'exact_match': {{'empty': empty['exact_match'], 'majority': majority['exact_match'], 'frozen': frozen['exact_match'], 'adapted': adapted['exact_match']}},\n"
                "    'delta_vs_frozen': {{'anls': round(adapted['anls'] - frozen['anls'], 4), 'exact_match': round(adapted['exact_match'] - frozen['exact_match'], 4)}},\n"
                "    'n_test': len(test_records),\n"
                "    'estimation': 'one seeded synthetic document split; no dispersion estimate',\n"
                "    'sample_saturated': sample_saturated,\n"
                "    'at_ceiling': at_ceiling,\n"
                "    'frozen_anls_headroom': frozen_anls_headroom,\n"
                "}}\n"
                "print(json.dumps(comparison, indent=2))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>None: in the recorded run of an earlier revision of this notebook, blob `c1ea0b6d` (commit `f197e08`, Colab T4, 8 October 2026), not re-run for this revision, the frozen model already scored 1.0, so no improvement was possible (a regression still was), and the delta was 0.0 for both ANLS and exact match. At the ceiling this sample checks that adaptation, export and reload work, not that adaptation helps. A harder test set (real scans, other templates and layouts) is needed to measure a gain.</details>'
            ),
        },
        {
            "md": (
                "## 8. Export, reload, and check serialization/reconstruction parity\n\n"
                "The artifact contains only trained decoder tensors in safetensors. Its manifest binds them to the exact base "
                "model id, revision and weight digest, plus the artifact size and digest. The live model is released before a "
                "fresh base loads the adapter; eight held-out answers must match before results are written. Because the frozen model answers these rows the same way at the ceiling, answer parity alone cannot tell whether the adapter reached the model, so the cell also records every trained tensor before the live model is freed and requires the reloaded tensors to be identical — a base loaded without the adapter fails that check whenever training changed any tensor. This is **serialization/reconstruction parity (8 answers, 28 tensors)**: it shows the exported adapter rebuilds the selected model, not that training learned anything. Next to it the cell records `selected_vs_base`, the largest absolute and the L2 difference between the selected tensors and the base weights (both 0 when validation kept epoch 0). With the Section 6 defaults the artifact goes to `outputs/{stem}_adapter`; other settings write to their own `outputs/{stem}_adapter_e…_lr…_k…` folder and result files, so the default artifact is kept."
                '\n\n**Predict before running:** after freeing the model and reloading the adapter on a fresh base, will the eight held-out answers match exactly?'
            ),
            "code": (
                "import csv\n"
                "import gc\n"
                "import platform\n"
                "import transformers\n\n"
                "if not globals().get('DATA_READY'):\n"
                "    raise RuntimeError('Section 4 has not produced a validated corpus (a BYOD input was refused, or Section 4 has not run): fix the input and run Section 4 again before this cell.')\n"
                "run_tag = '' if (EPOCHS, LEARNING_RATE, TRAINABLE_DECODER_LAYERS) == (2, 2e-4, 2) else f'_e{{EPOCHS}}_lr{{LEARNING_RATE:g}}_k{{TRAINABLE_DECODER_LAYERS}}'\n"
                "artifact_dir = Path('outputs/{stem}_adapter') if not run_tag else Path(f'outputs/{stem}_adapter{{run_tag}}')\n"
                "result_path = Path('outputs/{stem}_result.json') if not run_tag else Path(f'outputs/{stem}_result{{run_tag}}.json')\n"
                "predictions_path = Path('outputs/{stem}_predictions.csv') if not run_tag else Path(f'outputs/{stem}_predictions{{run_tag}}.csv')\n"
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
                "tensor_parity = {{'identical_tensors': sum(torch.equal(expected_tensors[name], reloaded_state[name].detach().cpu()) for name in trained_names), 'tensors_of': len(trained_names)}}\n"
                "del reloaded_state\n"
                "assert tensor_parity['identical_tensors'] == tensor_parity['tensors_of'], tensor_parity\n"
                "reload_parity = {{'identical_answers': sum(a == e for a, e in zip(actual_answers, expected_answers)), 'answers_of': len(expected_answers), **tensor_parity}}\n"
                "selected_vs_base = {{'best_epoch': adapt_result['best_epoch'], 'max_abs_diff': max(float((expected_tensors[name] - base_tensors[name]).abs().max()) for name in trained_names), 'l2_diff': sum(float(((expected_tensors[name] - base_tensors[name]).double() ** 2).sum()) for name in trained_names) ** 0.5}}\n"
                "result = {{'comparison': comparison, 'frozen': frozen, 'adapted': adapted, 'adaptation': adapt_result, 'reload_parity': reload_parity, 'selected_vs_base': selected_vs_base, 'input_manifest': input_manifest, 'notebook_source': NOTEBOOK_SOURCE, 'repository_revision': NOTEBOOK_SOURCE['repository_revision'], 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'model_license': MODEL_LICENSE, 'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': reloaded.device, 'gpu': gpu_name}}}}\n"
                "result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')\n"
                "with open(predictions_path, 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['id', 'document_id', 'question', 'prediction', 'accepted_answers', 'anls', 'exact_match'])\n"
                "    for row in adapted['rows']:\n"
                "        writer.writerow([row['id'], row['document_id'], row['question'], row['prediction'], '|'.join(row['answers']), row['anls'], row['exact_match']])\n"
                "print({{'artifact_files': sorted(path.name for path in artifact_dir.iterdir()), 'serialization_reconstruction_parity': reload_parity, 'selected_vs_base': selected_vs_base, 'outputs': sorted(os.listdir('outputs'))}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>Yes: in the recorded run of an earlier revision of this notebook, blob `c1ea0b6d` (commit `f197e08`, Colab T4, 8 October 2026), not re-run for this revision, all 8 answers were identical after the reload and 28 of 28 trained tensors were identical (serialization/reconstruction parity, not proof of learning; that run did not yet record `selected_vs_base`). That run printed the record with a single `of` key that showed the tensor count (28) instead of the answer count; this revision keeps them apart as `answers_of` (8) and `tensors_of` (28).</details>"
            ),
        },
        {
            "md": (
                "### Optional experiment: one trainable decoder block instead of two (off by default)\n\n"
                "**Predict → change one thing → run → observe → explain.** Set `RUN_BLOCK_COMPARISON = True` and run this cell: it loads a "
                "**fresh frozen pipeline** from the verified snapshot (so nothing stacks on the adapter you just exported), adapts only "
                "decoder block 11 with the Section 6 settings, compares the two runs on **validation** (best validation ANLS and exact match, "
                "kept epoch), chooses on validation, and saves the one-block adapter to its own folder. It does not re-score the test rows: "
                "Section 7 already used them once, and scoring a second configuration on them would make the comparison exploratory (a "
                "final estimate for a newly chosen setting needs an untouched test set). Predict first: will one block match two on "
                "validation, how much smaller will its artifact be (count the tensors), and will validation selection keep the same "
                "epoch? No recorded run of this experiment exists, so its result is unknown. It releases Section 8's `reloaded` model "
                "before loading a fresh one. The default artifact and `result.json` are untouched; the experiment costs one more adaptation (96.26 s on a T4 for two blocks in blob `c1ea0b6d` (commit `f197e08`, Colab T4, 8 October 2026), an earlier revision)."
            ),
            "code": (
                "RUN_BLOCK_COMPARISON = False  # @param {{type:\"boolean\"}}\n\n"
                "if not RUN_BLOCK_COMPARISON:\n"
                "    print({{'block_comparison_experiment': 'skipped (set RUN_BLOCK_COMPARISON = True to run it); nothing was trained or written'}})\n"
                "elif not globals().get('DATA_READY'):\n"
                "    raise RuntimeError('Section 4 has not produced a validated corpus (a BYOD input was refused, or Section 4 has not run): fix the input and run Section 4 again before this cell.')\n"
                "else:\n"
                "    reloaded = None  # release the Section 8 reload before loading another model\n"
                "    gc.collect()\n"
                "    torch.cuda.empty_cache()\n"
                "    experiment_pipe = Pix2StructDocVQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)  # fresh frozen weights; the exported artifact is not touched\n"
                "    one_block = experiment_pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=1, seed=0, progress=report_epoch)\n"
                "    experiment_dir = Path('outputs/{stem}_adapter_experiment_1block')\n"
                "    experiment_pipe.save_artifact(experiment_dir, metadata={{'tutorial': '{stem}', 'experiment': 'one trainable decoder block', 'data_source': data_source, 'corpus_digest': corpus_digest}})\n"
                "    experiment_manifest = json.loads((experiment_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "    default_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "    def best_val(run):\n"
                "        return next(entry['val'] for entry in run['history'] if entry['epoch'] == run['best_epoch'])\n"
                "    one_val, two_val = best_val(one_block), best_val(adapt_result)\n"
                "    chosen = 'one block' if (one_val['anls'], one_val['exact_match']) > (two_val['anls'], two_val['exact_match']) else 'two blocks (the default; kept on a tie)'\n"
                "    summary = {{\n"
                "        'one_block': {{'best_epoch': one_block['best_epoch'], 'n_trainable': one_block['n_trainable'], 'best_val_anls': one_val['anls'], 'best_val_exact_match': one_val['exact_match'], 'artifact_tensors': len(experiment_manifest['tensors']), 'artifact_bytes': experiment_manifest['files'][0]['bytes']}},\n"
                "        'two_blocks': {{'best_epoch': adapt_result['best_epoch'], 'n_trainable': adapt_result['n_trainable'], 'best_val_anls': two_val['anls'], 'best_val_exact_match': two_val['exact_match'], 'artifact_tensors': len(default_manifest['tensors']), 'artifact_bytes': default_manifest['files'][0]['bytes']}},\n"
                "        'chosen_on_validation': chosen,\n"
                "        'test': 'not re-scored: Section 7 already used the test rows once; a final estimate for a newly chosen setting needs an untouched test set',\n"
                "        'default_artifact_unchanged': True,\n"
                "    }}\n"
                "    print({{'block_comparison_experiment': summary}})\n"
                "    del experiment_pipe\n"
                "    gc.collect()\n"
                "    torch.cuda.empty_cache()"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>No recorded run of this experiment exists, so there is no reference number. Its result is unknown until you run it. What to look for: the one-block artifact holds block 11 only, exactly half the tensors of the two-block one (14 of 28) and about half the bytes; then compare the best validation scores and the kept epochs. If the validation scores tie, that is evidence about how easy the sample is, not about the number of blocks.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Under DIMER Notebook Specification 2.5 [8], a successful run proves that this exact carrier can verify the pinned snapshot, create and validate its fixed corpus, "
        "measure frozen and adapted held-out behavior, serialize only the intended tensors, and reproduce selected answers "
        "after a fresh reload. It does not prove DocVQA benchmark quality, real-document generalization, calibration, robustness "
        "to handwriting or unseen layouts, or production fitness. ANLS and exact match are corpus averages, not answer "
        "confidence. A negative delta is valid evidence. The generated names and transactions are fictional; BYOD users own "
        "consent, licensing, retention and access control.\n\n"
        "Successful execution proves that the recorded repository revision, carried in this notebook without the repository being "
        "reachable, completes the stated E2E path. It does **not** establish benchmark superiority or production readiness.\n\n"
        "**Next experiments.** Every re-run of the Section 5 or Section 6 cell first reloads the frozen pipeline from the verified "
        "snapshot when `pipe` was adapted or freed (releasing Section 8's `reloaded` model), so each experiment starts from the same "
        "base weights; its epoch 0 scores those base weights on the validation rows (Section 5 scored them on the test rows). To "
        "compare settings, change `EPOCHS`, `LEARNING_RATE` or `TRAINABLE_DECODER_LAYERS` in Section 6, re-run only Section 6, and "
        "compare the validation history (`adapt_result['history']`) and the kept epoch. Choose one setting on validation, then score "
        "it on the test rows once (Sections 7 and 8; a non-default setting writes its own adapter folder and result files, so the "
        "default artifact is kept). Scoring several settings on the same test rows makes those comparisons exploratory, and a final "
        "estimate for the chosen setting then needs an untouched test set. Also: run the one-block comparison cell above (validation "
        "only); add an external document-domain test set (never DocVQA train or validation pages, which the checkpoint was "
        "fine-tuned on [2]) or your own pages through BYOD (Section 4 cell, then Sections 5–8) — the only way a gain larger than the "
        "default sample's headroom can show; hold out a layout or vendor family as a separate experiment; repeat across seeds; "
        "stratify by field type and layout; and read the held-out rows Section 7 lists.\n\n"
        '## Troubleshooting\n\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"This notebook needs a CUDA GPU" (Section 1) or "The E2E default path requires a CUDA GPU" (Section 5)** — choose **Runtime → Change runtime type → T4 GPU** and Run all again; Section 1 stops before any download.\n'
        '- **The delta in Section 7 is 0.0** — expected when the frozen model already scores 1.0 (`at_ceiling: True`): no improvement is possible, though a tie or a regression is; it is not a broken run. A gain can only show on pages the frozen model gets wrong.\n'
        '- **A re-run scores "frozen" numbers that differ from the first pass** — Sections 5 and 6 reload the frozen pipeline whenever `pipe` was adapted or freed (they print `reloaded_frozen_pipeline`); `frozen["adapted"]` must be `False`.\n'
        '- **"The isolated environment\'s Python process exited"** or **CUDA out of memory** — restart the session and choose **Run all**. Sections 5 and 6 and the optional experiment release the Section 8 `reloaded` model before loading another; if memory still runs out after several re-runs, restart.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again.\n'
        "- **The corpus digest differs from `SAMPLE_DIGEST`** — the page images depend on the Pillow build's bundled font rendering; use the pinned environment Section 1 builds.\n"
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one …"** — set `BYOD_PATH` to the zip in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **BYOD: "exceed the BYOD ceiling", "not a zip archive", "hold fewer than MIN_RECORDS", or a `load_byod_dataset` / `validate_dataset` refusal** — Section 4 shows one line, `Refused: <file>: …`, with no traceback (Run all stops there), naming the zip, the `records.csv` row and the rule (member count or extracted size, too few rows in a split, duplicate id, a file that is not a decodable image, empty question or answer), moves earlier result files into `outputs/previous_run_<time>/` (kept, not deleted), and Sections 5–8 refuse to run until Section 4 accepts the zip; 17 documents at three questions each is the minimum.\n\n'
        '## Glossary\n\n'
        '- **OCR-free document QA** — answering a question about a page image without a separate text-recognition step.\n'
        '- **Patches** — the composite image (question header plus page) is cut into at most 2,048 patches the encoder reads.\n'
        '- **ANLS** — average normalised Levenshtein similarity [3]: per question, 1 minus the normalised edit distance to the closest accepted answer, then averaged. Similarities below 0.5 get no credit; a sufficiently similar wrong string can still get partial credit (`INV-2026-1008` for `INV-2026-1009` scores 12/13), so every accepted answer must be genuinely acceptable. This notebook implements the ANLS rules of MP-DocVQA-Framework `metrics.py` at `656d4e5` [4], consistent with the RRC DocVQA task description [5]; the login-protected RRC evaluator was not inspected: answers are compared lower-cased with leading and trailing spaces removed; punctuation and inner spaces count, and an empty answer scores 0.\n'
        '- **Exact match** — the share of answers identical to an accepted answer after the same normalisation (lower-cased, trimmed).\n'
        '- **Empty / raw first-answer frequency baseline** — answering nothing; answering every question with the most frequent first-listed training answer (counted as an exact string; printed as `majority`).\n'
        '- **Document-level split** — all questions about one document stay in one split, so the test pages are new document instances from the same generator (shared templates, vendors and layouts), not unseen templates.\n'
        '- **At / near the ceiling** — at frozen ANLS 1.0 no improvement is possible (a tie or a regression still is); near it (`sample_saturated`, frozen ANLS at or above 0.95) a gain cannot exceed the headroom, 1 − frozen ANLS.\n'
        '- **Epoch / validation selection** — one pass over the training rows; keeping the epoch with the best validation ANLS (epoch 0, the frozen model, included).\n'
        '- **Adapter / serialization-reconstruction parity** — the trained decoder tensors only, bound to the base digest; the reloaded model gives identical answers and tensors. It shows the export rebuilds the selected model, not that training learned anything (`selected_vs_base` shows how far the weights moved).\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: your pages and questions through the same cells.\n\n'
        '## Conclusion (your notes)\n\nWrite your conclusion from **your** run (optional); the recorded run is only a reference:\n\n'
        '- Empty ___, raw first-answer frequency ___, frozen ___, adapted ___ (test ANLS); the delta was ___ and the headroom ___.\n'
        '- The kept epoch was ___ because ___.\n'
        '- What this sample can and cannot show about adaptation: ___.\n'
        '- One harder test set I would build next: ___.\n\n'
        "## References\n\n"
        "[1] K. Lee, M. Joshi, I. Turc, H. Hu, F. Liu, J. Eisenschlos, U. Khandelwal, P. Shaw, M.-W. Chang, and K. Toutanova, \"Pix2Struct: Screenshot parsing as pretraining for visual language understanding,\" in *Proc. 40th Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 202, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2210.03347\n\n"
        "[2] M. Mathew, D. Karatzas, and C. V. Jawahar, \"DocVQA: A dataset for VQA on document images,\" in *Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)*, 2021. [Online]. Available: https://doi.org/10.48550/arXiv.2007.00398\n\n"
        "[3] A. F. Biten, R. Tito, A. Mafla, L. Gomez, M. Rusiñol, E. Valveny, C. V. Jawahar, and D. Karatzas, \"Scene text visual question answering,\" in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019. [Online]. Available: https://doi.org/10.48550/arXiv.1905.13648\n\n"
        "[4] R. Tito, \"MP-DocVQA-Framework: `metrics.py` (DocVQA ANLS and accuracy evaluator),\" GitHub, commit 656d4e5, 2023. [Online]. Available: https://github.com/rubenpt91/MP-DocVQA-Framework/blob/656d4e5a316720c83a1b1679fb39f424265655d7/metrics.py\n\n"
        "[5] Robust Reading Competition, \"Document Visual Question Answering: Tasks (evaluation metric),\" Computer Vision Center, UAB. [Online]. Available: https://rrc.cvc.uab.es/?ch=17&com=tasks\n\n"
        "[6] Google, \"google/pix2struct-docvqa-base\" model card, Hugging Face, revision {MODEL_REVISION}. [Online]. Available: https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/README.md\n\n"
        "[7] K. Valcorza, \"pix2struct-docvqa-pipeline\" (code, model card and the code-generated sample corpus), GitHub. [Online]. Available: https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline; model card: https://github.com/kurtvalcorza/pix2struct-docvqa-pipeline/blob/main/MODEL_CARD.md\n\n"
        "[8] DIMER, \"DIMER Notebook Specification 2.5,\" fleet specifications, ml-worker repository, `integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md`."
    ),
}
