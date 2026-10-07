"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings (prefix PSD) fixed in the generator on
top of the fleet-sweep fixes. They need only CI's dependencies: the Section 4 BYOD branch, the Section 8 tensor-parity
block and the experiment cell are executed with the notebook's own source and stand-ins (synthetic PNGs, no model),
and the rest are static checks on the generated notebook."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on single lines

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import textwrap
import types
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook_review", TOOLS / "build_notebook.py")
TEMPLATE = _load("notebook_template_review", TOOLS / "notebook_template.py").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, marker: str) -> str:
    found = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(found) == 1, f"exactly one code cell must contain {marker!r}"
    return found[0]


# --- PSD-m3: spec 2.2 everywhere -----------------------------------------------------------------------------------


def test_psd_m3_spec_2_2_declared_everywhere(nb: dict) -> None:
    """PSD-m3: metadata, opening cell, registry, validator and generator agree on NOTEBOOK_SPEC 2.2 (§32 item 6)."""
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert build.NOTEBOOK_SPEC == "2.2"
    validator = _load("validate_release_assets_review", TOOLS / "validate_release_assets.py")
    assert validator.NOTEBOOK_SPEC == "2.2"
    assert "DIMER Notebook Specification 2.2 — **standalone** (§4)" in _src(nb["cells"][0])
    assert "DIMER Notebook Specification 2.2" in (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    text = _markdown(nb) + "\n".join(_src(c) for c in _code_cells(nb))
    assert "Specification 2.0" not in text and "NOTEBOOK_SPEC 2.0" not in text


# --- PSD-M4: Infrastructure titles, guided markers, a coded default-off experiment --------------------------------


def test_psd_m4_every_carried_and_install_cell_is_titled_infrastructure(nb: dict) -> None:
    """PSD-M4 acceptance: every carried-module, install and snapshot cell starts with `# @title Infrastructure`, is
    collapsed, and the carried text after the title line still equals the module (PAR1 via embedded_module_text)."""
    titled = 0
    for cell in _code_cells(nb):
        s = _src(cell)
        if "# dimer: kernel cell" in s or cell["metadata"].get("dimer", {}).get("embedded_module") or "MANIFEST = {" in s:
            assert s.startswith("# @title Infrastructure"), s[:80]
            assert cell["metadata"].get("cellView") == "form"
            titled += 1
    assert titled >= 5
    ctx = build.load_context(ROOT, TEMPLATE, nb["metadata"]["dimer"]["generated_from"]["revision"])
    for cell, module in zip([c for c in _code_cells(nb) if c["metadata"].get("dimer", {}).get("embedded_module")], ctx["modules"], strict=True):
        assert _src(cell).startswith(build.EMBEDDED_TITLE_PREFIX)
        assert build.embedded_module_text(_src(cell)).rstrip("\n") + "\n" == ctx["embedded"][module]
    assert build.embedded_module_text("x = 1\n") == "x = 1\n"


def test_psd_m4_guided_markers_all_present(nb: dict) -> None:
    """PSD-M4 acceptance: the review's static probe markers are all true and at least four cells are form-collapsed."""
    text = "\n".join(_src(c) for c in nb["cells"])
    markers = {
        "how_to_use": r"how to use this notebook", "roadmap": r"roadmap", "glossary": r"glossary", "troubleshooting": r"troubleshoot",
        "infrastructure_label": r"infrastructure", "predict_prompt": r"\bpredict\b(?!ion)|make a prediction",
        "check_your_reasoning": r"check your reasoning|<details", "what_to_notice": r"what to notice|expected result|look for",
        "conclusion_template": r"conclusion template|your conclusion|write.*conclusion",
    }
    missing = [k for k, p in markers.items() if not re.search(p, text, re.I)]
    assert not missing, missing
    assert sum(c.get("metadata", {}).get("cellView") == "form" for c in nb["cells"]) >= 4


def test_psd_m4_coded_experiment_cell_is_off_by_default_and_uses_a_fresh_pipeline(nb: dict) -> None:
    """PSD-M4 acceptance / PSD-S3: the one-block comparison is a coded experiment gated off by default; under Run all
    it only prints `skipped`; when on, it adapts a fresh pipeline (never `reloaded`) into its own artifact folder."""
    source = _cell_with(nb, "RUN_BLOCK_COMPARISON = False  # @param")
    assert "experiment_pipe = Pix2StructDocVQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)" in source
    assert "reloaded.adapt" not in source and "pipe.adapt(" not in source.replace("experiment_pipe.adapt(", "")
    assert "_adapter_experiment_1block" in source and "trainable_decoder_layers=1" in source
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "<experiment>", "exec"), {})
    assert "skipped" in out.getvalue()


# --- PSD-M2: the ceiling is explained in the notebook's own markdown and reported at run time ------------------------


def test_psd_m2_ceiling_explained_and_reported(nb: dict) -> None:
    """PSD-M2 acceptance (explanation branch): the review's `ceiling_warning_in_markdown` regex matches the Section 7
    markdown, and the cell computes `sample_saturated`, prints the scored weights and lists the inexact held-out rows."""
    seven_md = next(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown" and _src(c).startswith("## 7."))
    assert re.search(r"ceiling effect|at ceiling|already answers|saturat|cannot show an? (adaptation )?gain", seven_md, re.I)
    assert "How to read a zero or near-ceiling delta" in seven_md
    seven = _cell_with(nb, "adapted = pipe.evaluate(test_records)")
    assert "sample_saturated = frozen['anls'] >= 0.95" in seven
    assert "'frozen_adapted_flag': frozen['adapted']" in seven
    assert "held_out_rows_not_answered_exactly" in seven and "'sample_saturated': sample_saturated" in seven
    # the branch logic, with stand-in metrics
    block = seven.split("comparison = {", 1)[0]
    rows = [{"id": "a", "question": "q", "prediction": "x", "answers": ["x"], "anls": 1.0}, {"id": "b", "question": "q", "prediction": "y", "answers": ["z"], "anls": 0.0}]
    ns = {"pipe": types.SimpleNamespace(evaluate=lambda r: {"anls": 0.5, "adapted": True, "rows": rows}), "test_records": [], "frozen": {"anls": 1.0, "adapted": False}}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(compile(block, "<seven>", "exec"), ns)
    assert ns["sample_saturated"] is True and ns["not_exact"] == [{"id": "b", "question": "q", "prediction": "y", "accepted": ["z"], "anls": 0.0}]
    assert "cannot show an adaptation gain" in out.getvalue()


# --- PSD-M3: rerun routes named; frozen scores guarded ---------------------------------------------------------------


def test_psd_m3_rerun_instructions_name_cells_and_frozen_flag_is_printed(nb: dict) -> None:
    """PSD-M3 / PSD-S1: the BYOD and next-experiment texts name the cells to re-run; Section 5 prints `adapted` and
    stops if the scored pipeline carries an adapter; both Section 5 and 6 reload before scoring or training."""
    md = _markdown(nb)
    assert re.search(r"re-?run (from )?(section|cell)", md, re.I)
    assert "re-run Section 4 and every code cell of Sections 5–8 in order" in md
    assert "re-run the Section 6, 7 and 8 cells in order" in md
    five = _cell_with(nb, "frozen = pipe.evaluate(test_records)")
    assert "'adapted'" in five and "if frozen['adapted']:" in five
    assert five.index("if globals().get('pipe') is None or pipe.adapter is not None:") < five.index("frozen = pipe.evaluate")
    six = _cell_with(nb, "adapt_result = pipe.adapt(")
    assert six.index("if globals().get('pipe') is None or pipe.adapter is not None:") < six.index("adapt_result = pipe.adapt(")


# --- PSD-m1: tensor parity after reload ------------------------------------------------------------------------------


class _T:
    def __init__(self, v: float) -> None:
        self.v = v

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return _T(self.v)


def _parity_block(nb: dict) -> tuple[str, str]:
    eight = _cell_with(nb, "del pipe\n")
    before = "trained_names = " + eight.split("trained_names = ", 1)[1].split("del pipe\n", 1)[0]
    after = "reloaded_state = " + eight.split("reloaded_state = ", 1)[1].split("reload_parity = ", 1)[0]
    return before, after


def _run_parity(nb: dict, live: dict, reloaded: dict) -> dict:
    before, after = _parity_block(nb)
    ns = {
        "adapt_result": {"trainable_names": sorted(live)},
        "pipe": types.SimpleNamespace(_model=types.SimpleNamespace(state_dict=lambda: {k: _T(v) for k, v in live.items()})),
        "reloaded": types.SimpleNamespace(_model=types.SimpleNamespace(state_dict=lambda: {k: _T(v) for k, v in reloaded.items()})),
        "torch": types.SimpleNamespace(equal=lambda a, b: a.v == b.v),
    }
    exec(compile(before, "<before>", "exec"), ns)
    exec(compile(after, "<after>", "exec"), ns)
    return ns["tensor_parity"]


def test_psd_m1_tensor_parity_detects_a_base_loaded_without_the_adapter(nb: dict) -> None:
    """PSD-m1 acceptance: the Section 8 check records the trained tensors before `del pipe` and fails when the reloaded
    model holds the base tensors instead; it passes when every trained tensor is identical."""
    live = {"decoder.layer.10.w": 1.5, "decoder.layer.11.w": 2.5, "decoder.final_layer_norm.weight": 0.5}
    assert _run_parity(nb, live, dict(live)) == {"identical_tensors": 3, "of": 3}
    base = {"decoder.layer.10.w": 1.0, "decoder.layer.11.w": 2.0, "decoder.final_layer_norm.weight": 0.5}
    with pytest.raises(AssertionError, match="identical_tensors.: 1"):
        _run_parity(nb, live, base)
    eight = _cell_with(nb, "del pipe\n")
    assert eight.index("expected_tensors = ") < eight.index("del pipe\n") < eight.index("tensor_parity = ")
    assert "assert actual_answers == expected_answers" in eight


# --- PSD-m4 / PSD-m5 ---------------------------------------------------------------------------------------------------


def test_psd_m4_cuda_check_runs_in_section_1_before_any_download(nb: dict) -> None:
    """PSD-m4: the Section 1 'record the runtime' cell stops with a GPU-runtime instruction before Section 3 downloads;
    the Prerequisites state the recorded T4 duration."""
    cells = _code_cells(nb)
    record = next(i for i, c in enumerate(cells) if "# @title Infrastructure: record the runtime" in _src(c))
    download = next(i for i, c in enumerate(cells) if "stage_missing_files(" in _src(c) and "MANIFEST = {" in _src(c))
    assert record < download
    assert "if not torch.cuda.is_available():" in _src(cells[record]) and "Change runtime type -> T4 GPU" in _src(cells[record])
    ns = {"torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}
    guard = _src(cells[record]).split("print(", 1)[1].split("\n", 1)[1]
    with pytest.raises(RuntimeError, match="before any model download"):
        exec(compile(guard, "<guard>", "exec"), ns)
    prereq = next(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown" and _src(c).startswith("## Prerequisites"))
    assert "**212.1 s**" in prereq and "Tesla T4" in prereq
    assert re.search(r"\b\d+\s*(s|sec|seconds|min|minutes|hours?)\b", prereq)


def test_psd_m5_preinstalled_variable_documented_and_digest_assert_names_pillow(nb: dict) -> None:
    """PSD-m5: DIMER_NOTEBOOK_CI_PREINSTALLED is explained in markdown and the corpus-digest check names the Pillow pin."""
    code = "\n".join(_src(c) for c in _code_cells(nb))
    assert "DIMER_NOTEBOOK_CI_PREINSTALLED" in code and "`DIMER_NOTEBOOK_CI_PREINSTALLED=1` lets an" in _markdown(nb)
    four = _cell_with(nb, "USE_BYOD = False  # @param")
    assert "pinned pillow==11.3.0" in four and "PIL.__version__" in four
    assert "assert corpus_digest == SAMPLE_DIGEST, (corpus_digest, SAMPLE_DIGEST)" not in four


# --- PSD-m2: BYOD branch replay with the review's inputs ---------------------------------------------------------------


def _png(colour: str, size=(96, 96)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def _zip(entries: list[tuple[str, bytes | str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in entries:
            z.writestr(n, d)
    return buf.getvalue()


def _csv(n_docs: int, folder: str = "", drop_col: str | None = None) -> str:
    cols = ["id", "document_id", "file", "question", "answers"]
    if drop_col:
        cols = [c for c in cols if c != drop_col]
    lines = [",".join(cols)]
    for d in range(n_docs):
        for q in range(3):
            row = {"id": f"d{d}-q{q}", "document_id": f"d{d}", "file": f"{folder}page{d}.png", "question": f"question {q}?", "answers": f"answer {d} {q}"}
            lines.append(",".join(row[c] for c in cols))
    return "\n".join(lines)


def _pages(n: int, folder: str = "") -> list[tuple[str, bytes]]:
    return [(f"{folder}page{i}.png", _png("white")) for i in range(n)]


def _byod_branch(nb: dict) -> str:
    source = _cell_with(nb, "USE_BYOD = False  # @param")
    body = source.split("\nif USE_BYOD:\n", 1)[1].split("\nelse:\n", 1)[0]
    return textwrap.dedent(body)


def _run_byod(nb: dict, work: Path, payload: bytes) -> dict:
    from pix2struct_docvqa_pipeline.samples import MAX_RECORDS, MIN_RECORDS, load_byod_dataset, split_dataset

    zip_path = work / "mine.zip"
    zip_path.write_bytes(payload)
    ns = {
        "Path": Path, "zipfile": zipfile, "MAX_RECORDS": MAX_RECORDS, "MIN_RECORDS": MIN_RECORDS, "SPLIT_SEED": 42,
        "load_byod_dataset": load_byod_dataset, "split_dataset": split_dataset, "BYOD_PATH": str(zip_path),
        "byod_file": lambda path, kind, suffixes=(): Path(path),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_byod_branch(nb), "<byod>", "exec"), ns)
    return ns


def test_psd_m2_byod_valid_nested_zip_is_split(nb: dict, tmp_path: Path) -> None:
    """PSD-m2 (review `byod_replay.valid_20docs_nested_folders`): a 20-document zip with folders is accepted and split."""
    ns = _run_byod(nb, tmp_path, _zip([("data/records.csv", _csv(20, folder="images/"))] + _pages(20, "data/images/")))
    assert {k: len(v) for k, v in ns["splits"].items()} == {"test": 12, "validation": 9, "train": 39}
    # the stated minimum (17 documents at three questions each) is accepted; 16 is not
    ns = _run_byod(nb, tmp_path, _zip([("records.csv", _csv(17))] + _pages(17)))
    assert {k: len(v) for k, v in ns["splits"].items()} == {"test": 9, "validation": 9, "train": 33}
    with pytest.raises(ValueError, match="hold fewer than MIN_RECORDS"):
        _run_byod(nb, tmp_path, _zip([("records.csv", _csv(16))] + _pages(16)))


@pytest.mark.parametrize(
    ("name", "entries", "match"),
    [
        ("no_records_csv", _pages(20), r"BYOD data must include records\.csv"),
        ("missing_answers_column", [("records.csv", _csv(20, drop_col="answers"))] + _pages(20), r"records\.csv must have id, document_id, file, question, answers"),
        ("non_image_page", [("records.csv", _csv(20))] + _pages(19) + [("page19.png", b"not an image")], r"records\.csv row 'd19-q0': file 'page19\.png' is not a decodable image"),
        ("small_10docs_30rows", [("records.csv", _csv(10))] + _pages(10), r"mine\.zip: split\(s\) \{'test': 6, 'validation': 6\} hold fewer than MIN_RECORDS = 8 rows .*17 documents"),
    ],
)
def test_psd_m2_byod_refusals_name_the_file_row_and_rule(nb: dict, tmp_path: Path, name: str, entries: list, match: str) -> None:
    """PSD-m2 acceptance: each expected failure raises ValueError naming the failed rule (no StopIteration, no bare
    UnidentifiedImageError, no split-less '6 records' message)."""
    with pytest.raises(ValueError, match=match):
        _run_byod(nb, tmp_path, _zip(entries))


def test_psd_m2_byod_member_count_and_expanded_size_are_capped(nb: dict, tmp_path: Path) -> None:
    """PSD-m2 (§20): more members than MAX_RECORDS + 1, or more than 2 GiB extracted, is refused before the zip is read."""
    from pix2struct_docvqa_pipeline.samples import MAX_RECORDS

    too_many = _zip([("records.csv", _csv(20))] + [(f"f{i}.txt", b"x") for i in range(MAX_RECORDS + 1)])
    with pytest.raises(ValueError, match=rf"mine\.zip: {MAX_RECORDS + 2} files .* exceed the BYOD ceiling of {MAX_RECORDS + 1} files"):
        _run_byod(nb, tmp_path, too_many)
    with pytest.raises(ValueError, match=r"mine\.zip: not a zip archive"):
        _run_byod(nb, tmp_path, b"not a zip")
    branch = _byod_branch(nb)
    assert "BYOD_MAX_EXPANDED_BYTES = MAX_RECORDS + 1, 2 * 1024 ** 3" in branch and "expanded > BYOD_MAX_EXPANDED_BYTES" in branch


def test_psd_m2_cancelled_upload_is_named_not_stopiteration(nb: dict) -> None:
    """PSD-m2 (review `byod_replay.cancelled_upload`): the helper checks the upload count before `next(iter(...))`."""
    source = _cell_with(nb, "USE_BYOD = False  # @param")
    assert "Upload exactly one" in source and source.index("if len(uploaded) != 1:") < source.index("next(iter(uploaded.items()))")
