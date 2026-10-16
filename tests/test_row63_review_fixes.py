"""Regression tests for the row-63 review (2026-10-16) of tutorials/pix2struct_docvqa_colab.ipynb.

R63-M1 (reload_parity keys) is covered in test_review_fixes.py::test_psd_m1_tensor_parity_detects_a_base_loaded_without_the_adapter.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import io
import json
import os
import re
from pathlib import Path

import pytest

from pix2struct_docvqa_pipeline import anls, exact_match, normalize_answer
from pix2struct_docvqa_pipeline.metrics import majority_answer_baseline

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "pix2struct_docvqa_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, needle: str) -> str:
    found = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(found) == 1, needle
    return found[0]


# --- R63-m4: ANLS / exact match follow the official DocVQA evaluator -------------------------------------------------
# Source: rubenpt91/MP-DocVQA-Framework metrics.py @ 656d4e5 (Evaluator: _preprocess_str = lower().strip();
# ANLS = max over golds of 1 - editdistance / max(len); kept if >= 0.5 else 0; empty prediction scores 0) and the RRC
# DocVQA task page ("Answers are not case sensitive. Answers are space sensitive.").


@pytest.mark.parametrize(
    ("prediction", "golds", "expected"),
    [
        ("Acme Corp", ["acme corp"], 1.0),  # case-insensitive
        ("  acme corp\n", ["Acme Corp"], 1.0),  # leading/trailing whitespace stripped
        ("NW 2026 0417", ["NW-2026-0417"], 1 - 2 / 12),  # punctuation counts: 2 substitutions over 12 chars
        ("acme  corp", ["acme corp"], 1 - 1 / 10),  # inner spaces count (space sensitive)
        ("ACME, Corp", ["acme corp"], 1 - 1 / 10),  # the comma is one insertion
        ("Acme Co", ["Acme Corp"], 7 / 9),
        ("1,099.20", ["$1,099.20", "1099.20"], 8 / 9),  # max over golds
        ("ab", ["abcde"], 0.0),  # similarity 0.4 < 0.5 -> 0
        ("abcd", ["abcdefgh"], 0.5),  # exactly 0.5 is kept (>= threshold)
        ("", ["x"], 0.0),  # empty prediction scores 0
        ("   ", ["x"], 0.0),  # whitespace-only prediction normalises to empty
    ],
)
def test_r63_m4_anls_matches_official_hand_worked_cases(prediction: str, golds: list[str], expected: float) -> None:
    assert anls(prediction, golds) == pytest.approx(expected)


def test_r63_m4_normalisation_and_exact_match_follow_official_rules() -> None:
    assert normalize_answer(" NW-2026-0417 ") == "nw-2026-0417"
    assert exact_match("NW 2026 0417", ["NW-2026-0417"]) is False
    assert exact_match(" nw-2026-0417", ["X", "NW-2026-0417"]) is True


def test_r63_m4_majority_description_matches_its_counting() -> None:
    rec = lambda i, a: {"id": f"r{i}", "document_id": f"d{i}", "question": "q", "answers": [a]}  # noqa: E731
    out = majority_answer_baseline([rec(0, "Blue")], [rec(1, "Blue"), rec(2, "blue"), rec(3, "Red"), rec(4, "Blue")])
    assert out["answer"] == "Blue" and "exact string" in out["baseline"] and "normalized" not in out["baseline"]


def test_r63_m4_notebook_describes_the_official_normalisation(nb: dict) -> None:
    md = _markdown(nb)
    assert "official DocVQA evaluation [4], [5]" in md
    assert "https://rrc.cvc.uab.es/?ch=17&com=tasks" in md and "656d4e5a316720c83a1b1679fb39f424265655d7" in md
    assert "punctuation removed" not in json.dumps(nb)


# --- R63-m1: quoted numbers carry their revision; stale prose removed --------------------------------------------------


def test_r63_m1_quotes_are_labelled_and_stale_prose_is_gone(nb: dict) -> None:
    md = _markdown(nb)
    for stale in ("has no recorded number yet", "ten code cells", "26 September 2026", "212.1 s", "75,519,448", "about 95 s"):
        assert stale not in md, stale
    unlabelled = re.findall(r"(?i)(?:in|on) the recorded run(?! of an earlier revision)|the recorded run reported", md)
    assert not unlabelled, unlabelled
    assert md.count("blob `c1ea0b6d`") >= 6
    assert "docs/release-verification.md" in md


# --- R63-m2: NOTEBOOK_SPEC 2.5 declaration, kind, use notice, IEEE references -------------------------------------------


def test_r63_m2_kind_use_notice_and_references(nb: dict) -> None:
    assert nb["metadata"]["dimer"]["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    notices = [i for i, c in enumerate(nb["cells"][:first_code]) if c["cell_type"] == "markdown" and _src(c).startswith("> **⚠️ Notice.**")]
    assert notices == [1]
    notice = _src(nb["cells"][1])
    for clause in ("For research, training and evaluation only.", "Data you bring stays in your runtime", '"as is", without warranty of any kind', "accept no liability"):
        assert clause in notice.replace("\n> ", " ")
    assert "[!WARNING]" not in notice
    md = _markdown(nb)
    refs = md.split("## References", 1)[1]
    entries = re.findall(r"^\[(\d+)\] ", refs, re.M)
    assert entries == [str(i) for i in range(1, 9)]
    body = md.split("## References", 1)[0]
    for n in entries:
        assert re.search(rf"\[{n}\]|\[\d\], \[{n}\]", body), f"reference [{n}] is never cited"
    assert "https://doi.org/10.48550/arXiv.2210.03347" in refs


# --- R63-m3: a BYOD refusal is one message, no traceback, no stale results ------------------------------------------------


def _section4_namespace(tmp_path: Path, raw_cell: str) -> dict:
    from pix2struct_docvqa_pipeline import samples

    ns = {name: getattr(samples, name) for name in ("MAX_RECORDS", "MIN_RECORDS", "load_byod_dataset", "split_dataset", "validate_dataset", "check_split_disjoint", "dataset_digest", "build_sample_dataset", "SAMPLE_DIGEST")}
    ns.update({"CORPUS_NAME": "sample", "CORPUS_LICENSE": "CC0"})
    return ns


def test_r63_m3_byod_refusal_prints_one_message_and_clears_stale_results(nb: dict, tmp_path: Path, monkeypatch) -> None:
    source = _cell_with(nb, "USE_BYOD = False  # @param")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    source = source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {str(tmp_path / 'mine.zip')!r}  # @param", 1)
    (tmp_path / "mine.zip").write_bytes(b"not a zip")
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs/pix2struct_docvqa_adapter")
    Path("outputs/pix2struct_docvqa_result.json").write_text("{}", encoding="utf-8")
    Path("outputs/pix2struct_docvqa_adapter/manifest.json").write_text("{}", encoding="utf-8")
    ns = _section4_namespace(tmp_path, source)
    ns.update({"frozen": {"anls": 1.0}, "splits": {"stale": []}})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "<section4>", "exec"), ns)  # no exception: a refusal is not a crash
    printed = out.getvalue()
    assert printed.count("BYOD input refused:") == 1 and "mine.zip: not a zip archive" in printed and "run this cell again" in printed
    assert ns["byod_refusal"] and "splits" not in ns and "frozen" not in ns and "input_manifest" not in ns
    assert sorted(os.listdir("outputs")) == []
    five = _cell_with(nb, "majority = majority_answer_baseline(")
    guard = five.split("if globals().get('byod_refusal'):", 1)[1].split("\nif not torch", 1)[0]
    with pytest.raises(RuntimeError, match="Section 4 refused the BYOD input"):
        exec(compile("if globals().get('byod_refusal'):" + guard, "<guard>", "exec"), ns)


def test_r63_m3_default_corpus_failure_still_raises(nb: dict, tmp_path: Path, monkeypatch) -> None:
    source = _cell_with(nb, "USE_BYOD = False  # @param")
    monkeypatch.chdir(tmp_path)
    ns = _section4_namespace(tmp_path, source)

    def broken():
        raise ValueError("records[0]: corrupted default corpus")

    ns["build_sample_dataset"] = broken
    with pytest.raises(ValueError, match="corrupted default corpus"):
        exec(compile(source, "<section4>", "exec"), ns)


# --- R63-m5 / S1-S3 -------------------------------------------------------------------------------------------------------


def test_r63_m5_s1_s2_s3_prose(nb: dict) -> None:
    md = _markdown(nb)
    assert "final layer norm" not in md and "exactly half the tensors of the two-block one (14 of 28)" in md
    assert "was fine-tuned on DocVQA [2], [6]" in md and "never DocVQA train or validation pages" in md
    assert "Compare settings on the validation history (`adapt_result['history']`)" in md.replace('"\n        "', "")
    assert "`DIMER_ISOLATED_ENV` overrides the folder" in md
