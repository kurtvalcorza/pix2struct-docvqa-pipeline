from __future__ import annotations

import csv
import io
import zipfile

import pytest
from PIL import Image

from pix2struct_docvqa_pipeline.metrics import (
    empty_baseline,
    majority_answer_baseline,
    qa_metrics,
)
from pix2struct_docvqa_pipeline.pipeline import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_WEIGHTS_NAME,
    MODEL_ID,
    MODEL_REVISION,
    WEIGHT_FILE,
    WEIGHT_SHA256,
    Pix2StructDocVQAPipeline,
)
from pix2struct_docvqa_pipeline.samples import (
    SAMPLE_DIGEST,
    SAMPLE_SPLIT,
    build_sample_dataset,
    check_split_disjoint,
    dataset_digest,
    load_byod_dataset,
    split_dataset,
    validate_dataset,
)


def _record(index: int, document: int | None = None) -> dict:
    return {
        "id": f"row-{index}",
        "document_id": f"doc-{document if document is not None else index}",
        "image": Image.new("RGB", (32, 32), "white"),
        "question": "What is the total?",
        "answers": [f"PHP {index}"],
    }


class _FakeModel:
    def named_parameters(self):
        for layer in range(12):
            yield f"decoder.layer.{layer}.weight", object()
        yield "decoder.lm_head.weight", object()


def test_generated_sample_is_pinned_and_document_disjoint() -> None:
    splits = build_sample_dataset()
    assert {name: len(rows) for name, rows in splits.items()} == SAMPLE_SPLIT
    assert check_split_disjoint(splits) == SAMPLE_SPLIT
    records = [record for rows in splits.values() for record in rows]
    assert dataset_digest(records) == SAMPLE_DIGEST
    assert len({record["document_id"] for record in records}) == 40


def test_validate_dataset_normalizes_and_rejects_bad_records() -> None:
    records = [_record(index) for index in range(8)]
    records[0]["question"] = "  What   is the total? "
    records[0]["answers"] = [" PHP 0 ", "PHP 0"]
    checked = validate_dataset(records)
    assert checked["records"][0]["question"] == "What is the total?"
    assert checked["records"][0]["answers"] == ["PHP 0"]

    duplicated = [*records, {**_record(9), "id": "row-0"}]
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(duplicated)
    with pytest.raises(ValueError, match="answers"):
        validate_dataset([{**record, "answers": []} for record in records])


def test_split_dataset_keeps_documents_together() -> None:
    records = [_record(index, document=index // 2) for index in range(20)]
    splits = split_dataset(records, seed=7)
    assert sum(map(len, splits.values())) == len(records)
    check_split_disjoint(splits)


def test_metrics_and_non_neural_baselines() -> None:
    records = [_record(index) for index in range(8)]
    exact = qa_metrics([f"PHP {index}" for index in range(8)], records)
    assert exact["anls"] == 1.0
    assert exact["exact_match"] == 1.0
    assert empty_baseline(records)["anls"] == 0.0
    majority = majority_answer_baseline(records, [{**_record(10), "answers": ["fixed"]}])
    assert majority["answer"] == "fixed"
    with pytest.raises(ValueError, match="predictions"):
        qa_metrics([], records)


def test_byod_zip_loader_and_duplicate_basename_refusal(tmp_path) -> None:
    image = Image.new("RGB", (32, 32), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(["id", "document_id", "file", "question", "answers"])
    for index in range(8):
        writer.writerow([f"q-{index}", f"d-{index}", "page.png", "Total?", "PHP 10|10"])

    archive_path = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("records.csv", rows.getvalue())
        archive.writestr("page.png", buffer.getvalue())
    loaded = load_byod_dataset(archive_path)
    assert validate_dataset(loaded)["n_records"] == 8

    duplicate_path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate_path, "w") as archive:
        archive.writestr("records.csv", rows.getvalue())
        archive.writestr("a/page.png", buffer.getvalue())
        archive.writestr("b/page.png", buffer.getvalue())
    with pytest.raises(ValueError, match="duplicate basename"):
        load_byod_dataset(duplicate_path)


def test_trainable_scope_is_only_the_requested_final_decoder_blocks() -> None:
    pipe = Pix2StructDocVQAPipeline(
        lambda *_args: {"answer": ""},
        _model=_FakeModel(),
        _processor=object(),
        _font_bytes=b"font",
    )
    assert pipe._trainable_names(2) == [
        "decoder.layer.10.weight",
        "decoder.layer.11.weight",
    ]
    with pytest.raises(ValueError, match="1..12"):
        pipe._trainable_names(0)


def test_artifact_manifest_is_bound_to_the_pinned_base(tmp_path) -> None:
    pipe = Pix2StructDocVQAPipeline(lambda *_args: {"answer": ""})
    manifest = {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_FORMAT_VERSION,
        "base_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_file": WEIGHT_FILE,
            "weight_sha256": WEIGHT_SHA256,
        },
        "adapter": {"trainable_decoder_layers": 2},
        "files": [{"path": ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["decoder.layer.10.weight", "decoder.layer.11.weight"],
    }
    assert pipe._check_artifact_manifest(tmp_path, manifest) == tmp_path / ARTIFACT_WEIGHTS_NAME
    manifest["base_model"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="different base"):
        pipe._check_artifact_manifest(tmp_path, manifest)
