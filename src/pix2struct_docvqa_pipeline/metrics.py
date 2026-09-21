"""Corpus metrics and non-adapted baselines for document visual question answering."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import anls, exact_match

METRIC_DEFINITIONS = {
    "anls": "mean answer normalized Levenshtein similarity with similarities below 0.5 set to zero",
    "exact_match": "fraction whose normalized prediction equals an accepted answer",
}


def qa_metrics(predictions: Sequence[str], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(predictions) != len(records):
        raise ValueError(f"{len(predictions)} predictions for {len(records)} records")
    if not records:
        raise ValueError("records must not be empty")
    rows = []
    for prediction, record in zip(predictions, records, strict=True):
        answers = list(record["answers"])
        rows.append(
            {
                "id": record["id"],
                "document_id": record["document_id"],
                "question": record["question"],
                "prediction": str(prediction),
                "answers": answers,
                "anls": anls(str(prediction), answers),
                "exact_match": exact_match(str(prediction), answers),
            }
        )
    return {
        "n": len(rows),
        "anls": round(statistics.fmean(row["anls"] for row in rows), 4),
        "exact_match": round(statistics.fmean(float(row["exact_match"]) for row in rows), 4),
        "rows": rows,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def empty_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = qa_metrics([""] * len(records), records)
    out["baseline"] = "empty answer"
    return out


def majority_answer_baseline(
    records: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not reference:
        raise ValueError("reference records must not be empty")
    answer = Counter(str(r["answers"][0]) for r in reference).most_common(1)[0][0]
    out = qa_metrics([answer] * len(records), records)
    out["baseline"] = "most frequent normalized training answer, ignoring image and question"
    out["answer"] = answer
    return out
