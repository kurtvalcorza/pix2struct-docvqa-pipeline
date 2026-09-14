"""Offline tests for the public validation, metric and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from pix2struct_docvqa_pipeline import (
    ANLS_THRESHOLD,
    DEFAULT_MAX_NEW_TOKENS,
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_QUESTION_CHARS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    anls,
    evaluation_report,
    exact_match,
    normalize_answer,
    validate_inputs,
)

QUESTIONS = ["What is the invoice number?", "Who is the customer?"]


def _image(width: int = 850, height: int = 1100) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(answer: str, question: str = QUESTIONS[0], truncated: bool = False) -> dict:
    return {"answer": answer, "question": question, "truncated": truncated}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), QUESTIONS, names=["form.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["question_chars"] == [1, MAX_QUESTION_CHARS]
    assert manifest["inputs"] == [{"id": "form.png", "mode": "RGB", "size": [850, 1100]}]
    assert manifest["questions"] == QUESTIONS
    assert manifest["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_request() -> None:
    manifest = validate_inputs(_image(), ["  Who   is it? "], max_new_tokens=64)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["questions"] == ["Who is it?"]
    assert manifest["generation"]["max_new_tokens"] == 64


def test_validate_inputs_rejects_like_answer() -> None:
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs(_image(), "Who?")
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs(_image(), [])
    with pytest.raises(ValueError, match="MAX_QUESTION_CHARS"):
        validate_inputs(_image(), ["x" * (MAX_QUESTION_CHARS + 1)])
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs(_image(), QUESTIONS, max_new_tokens=0)
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8), QUESTIONS)
    with pytest.raises(ValueError, match="exactly one entry"):
        validate_inputs(_image(), QUESTIONS, names=["a", "b"])


def test_normalize_answer_and_exact_match() -> None:
    assert normalize_answer("  $1,099.20 ") == "1 099 20"
    assert exact_match("ACME, Corp", ["acme corp"]) is True
    assert exact_match("Acme Co", ["acme corp"]) is False


def test_anls_threshold_and_max_over_golds() -> None:
    assert anls("NW-2026-0417", ["NW-2026-0417"]) == 1.0
    assert anls("Acme Co", ["Acme Corp"]) == pytest.approx(7 / 9)
    assert anls("12", ["Acme Corp"]) == 0.0  # similarity below ANLS_THRESHOLD scores 0
    assert anls("1,099.20", ["$1,099.20", "1099.20"]) == 1.0
    assert 0 < ANLS_THRESHOLD < 1
    with pytest.raises(ValueError, match="golds"):
        anls("x", [])


def test_evaluation_report_not_measurable_without_golds() -> None:
    report = evaluation_report([_result("NW-2026-0417")], sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_questions"] == 1 and report["truncated"] == [False]
    assert "ANLS" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "no score" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_golds() -> None:
    results = [_result("NW-2026-0417"), _result("Blue Yonder", QUESTIONS[1], truncated=True)]
    report = evaluation_report(results, [["NW-2026-0417"], ["Blue Yonder Airlines"]])
    assert report["verdict"] == "sample-sanity"
    assert report["truncated"] == [False, True]
    by_id = {metric["id"]: metric for metric in report["metrics"]}
    assert by_id["exact_match"]["value"] == 0.5
    assert by_id["anls"]["threshold"] == ANLS_THRESHOLD
    assert by_id["anls"]["value"] == pytest.approx((1.0 + anls("Blue Yonder", ["Blue Yonder Airlines"])) / 2)
    assert [entry["exact_match"] for entry in report["per_question"]] == [True, False]
    assert report["per_question"][1]["golds"] == ["Blue Yonder Airlines"]


def test_evaluation_report_rejects_mismatched_or_empty_golds() -> None:
    with pytest.raises(ValueError, match="golds has"):
        evaluation_report([_result("a")], [["a"], ["b"]])
    with pytest.raises(ValueError, match="non-empty sequence"):
        evaluation_report([_result("a")], [[]])
    with pytest.raises(ValueError, match="results"):
        evaluation_report([], None)
