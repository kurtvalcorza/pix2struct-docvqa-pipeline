"""Deterministic synthetic document-QA data and BYOD validation for the E2E carrier."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .pipeline import MAX_QUESTION_CHARS, normalize_answer, validate_image

CORPUS_NAME = "DIMER synthetic business-document QA sample"
CORPUS_LICENSE = "CC0-1.0 (generated in code; no external document or personal data)"
CORPUS_VERSION = "1"
SAMPLE_SEED = 42
SAMPLE_DOCUMENTS = {"train": 24, "validation": 6, "test": 10}
QUESTIONS_PER_DOCUMENT = 3
SAMPLE_SPLIT = {name: count * QUESTIONS_PER_DOCUMENT for name, count in SAMPLE_DOCUMENTS.items()}
SAMPLE_DIGEST = "b9e7b0b27ae120979c988da745c8d0324de4267925f8c53b7a2f7f66317a438a"
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MAX_ANSWERS = 8
MAX_ANSWER_CHARS = 128
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

_VENDORS = ("Northwind Office", "Blue Fern Bakery", "Harbor Tools", "Cedar Health", "Atlas Transit")
_CITIES = ("Manila", "Cebu", "Davao", "Baguio", "Iloilo", "Quezon City")
_CONTACTS = ("Ana Reyes", "Miguel Santos", "Lea Cruz", "Paolo Lim", "Maya Flores")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return _sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    parts = sorted(
        f"{r['id']}:{r['document_id']}:{image_digest(r['image'])}:{r['question']}:{'|'.join(r['answers'])}"
        for r in records
    )
    return _sha256("\n".join(parts).encode())


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _document(index: int) -> tuple[Image.Image, dict[str, str]]:
    vendor = _VENDORS[index % len(_VENDORS)]
    city = _CITIES[(index * 3) % len(_CITIES)]
    contact = _CONTACTS[(index * 2) % len(_CONTACTS)]
    invoice = f"INV-{2026 + index // 90}-{1000 + index:04d}"
    po = f"PO-{70000 + index * 17}"
    total = f"PHP {1250 + index * 137:,.2f}"
    date = f"2026-{1 + index % 9:02d}-{1 + (index * 7) % 27:02d}"
    fields = {
        "vendor": vendor,
        "city": city,
        "contact": contact,
        "invoice": invoice,
        "po": po,
        "total": total,
        "date": date,
    }
    image = Image.new("RGB", (768, 1024), "white")
    draw = ImageDraw.Draw(image)
    title, body, small = _font(34), _font(24), _font(18)
    accent = (35 + index * 13 % 120, 70 + index * 17 % 120, 120 + index * 19 % 100)
    draw.rectangle((0, 0, 768, 120), fill=accent)
    draw.text((36, 32), vendor.upper(), font=title, fill="white")
    draw.text((36, 82), f"{city} branch", font=small, fill="white")
    if index % 2:
        labels = [
            ("INVOICE", invoice),
            ("DATE", date),
            ("PURCHASE ORDER", po),
            ("CONTACT", contact),
            ("TOTAL DUE", total),
        ]
        y = 180
        for label, value in labels:
            draw.text((48, y), label, font=small, fill="gray")
            draw.text((280, y), value, font=body, fill="black")
            draw.line((48, y + 38, 720, y + 38), fill=(210, 210, 210), width=2)
            y += 105
    else:
        draw.text((48, 175), "INVOICE", font=title, fill=accent)
        draw.text((510, 184), invoice, font=body, fill="black")
        draw.text((48, 270), f"Date: {date}", font=body, fill="black")
        draw.text((48, 330), f"Purchase order: {po}", font=body, fill="black")
        draw.text((48, 390), f"Contact: {contact}", font=body, fill="black")
        draw.rectangle((430, 540, 720, 640), outline=accent, width=4)
        draw.text((455, 565), f"TOTAL  {total}", font=body, fill="black")
    draw.text((48, 920), "Synthetic training page — no real person or transaction", font=small, fill="gray")
    return image, fields


def build_sample_dataset(seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    order = list(range(sum(SAMPLE_DOCUMENTS.values())))
    random.Random(seed).shuffle(order)
    result: dict[str, list[dict[str, Any]]] = {}
    start = 0
    specs = (
        ("What is the invoice number?", "invoice"),
        ("What is the purchase order number?", "po"),
        ("What is the total due?", "total"),
    )
    for split, n_docs in SAMPLE_DOCUMENTS.items():
        rows = []
        for index in order[start : start + n_docs]:
            image, fields = _document(index)
            document_id = f"synthetic-{index:03d}"
            for q_index, (question, field) in enumerate(specs):
                rows.append(
                    {
                        "id": f"{document_id}-q{q_index}",
                        "document_id": document_id,
                        "image": image.copy(),
                        "question": question,
                        "answers": [fields[field]],
                        "source": "generated",
                    }
                )
        result[split] = rows
        start += n_docs
    return result


def _open(image: Any, where: str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str | Path):
        try:
            loaded = Image.open(image)
            loaded.load()
            return loaded
        except Exception as exc:
            raise ValueError(f"{where}: cannot decode image") from exc
    raise ValueError(f"{where}: image must be a PIL image or a path")


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
) -> dict[str, Any]:
    if isinstance(records, Mapping | str | bytes) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence of document-QA mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} required")
    checked, ids = [], set()
    for index, record in enumerate(records):
        required = ("id", "document_id", "image", "question", "answers")
        if not isinstance(record, Mapping) or any(k not in record for k in required):
            raise ValueError(f"records[{index}] must contain id/document_id/image/question/answers")
        rid, document_id = record["id"], record["document_id"]
        if not isinstance(rid, str) or not _ID_RE.fullmatch(rid) or rid in ids:
            raise ValueError(f"records[{index}]: id must be unique and match {_ID_RE.pattern}")
        if not isinstance(document_id, str) or not _ID_RE.fullmatch(document_id):
            raise ValueError(f"records[{index}]: document_id must match {_ID_RE.pattern}")
        image = validate_image(_open(record["image"], f"records[{index}].image"))
        question = " ".join(str(record["question"]).split())
        if not question or len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"records[{index}]: question must contain 1..{MAX_QUESTION_CHARS} characters")
        answers = record["answers"]
        invalid_answers = isinstance(answers, str | bytes) or not isinstance(answers, Sequence)
        if invalid_answers or not 1 <= len(answers) <= MAX_ANSWERS:
            raise ValueError(f"records[{index}]: answers must contain 1..{MAX_ANSWERS} strings")
        normalized = []
        for answer in answers:
            if not isinstance(answer, str) or not answer.strip() or len(answer) > MAX_ANSWER_CHARS:
                raise ValueError(
                    f"records[{index}]: each answer must contain 1..{MAX_ANSWER_CHARS} characters"
                )
            if normalize_answer(answer) not in {normalize_answer(a) for a in normalized}:
                normalized.append(answer.strip())
        ids.add(rid)
        checked.append(
            {
                "id": rid,
                "document_id": document_id,
                "image": image,
                "question": question,
                "answers": normalized,
                "source": record.get("source", "byod"),
            }
        )
    return {
        "records": checked,
        "n_records": len(checked),
        "n_documents": len({r["document_id"] for r in checked}),
        "digest": dataset_digest(checked),
    }


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    seen: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            key = str(record["document_id"])
            if key in seen and seen[key] != split:
                raise ValueError(f"document {key!r} appears in both {seen[key]} and {split}")
            seen[key] = split
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    checked = validate_dataset(records)["records"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        groups.setdefault(record["document_id"], []).append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test, n_val = max(1, round(len(keys) * test_fraction)), round(len(keys) * val_fraction)
    if len(keys) - n_test - n_val < 1:
        raise ValueError("too few distinct documents to split")
    selected = {
        "test": keys[:n_test],
        "validation": keys[n_test : n_test + n_val],
        "train": keys[n_test + n_val :],
    }
    return {name: [row for key in chosen for row in groups[key]] for name, chosen in selected.items()}


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    members: dict[str, bytes] = {}
    if source.is_dir():
        for file in sorted(source.rglob("*")):
            if file.is_file():
                if file.name in members:
                    raise ValueError(f"duplicate basename {file.name!r}")
                members[file.name] = file.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    name = Path(info.filename).name
                    if name in members:
                        raise ValueError(f"duplicate basename {name!r}")
                    members[name] = archive.read(info)
    else:
        raise ValueError(f"{source} is neither a directory nor a zip")
    if "records.csv" not in members:
        raise ValueError("BYOD data must include records.csv")
    rows = list(csv.DictReader(io.StringIO(members["records.csv"].decode("utf-8-sig"))))
    required = ("id", "document_id", "file", "question", "answers")
    if not rows or any(column not in rows[0] for column in required):
        raise ValueError("records.csv must have id, document_id, file, question, answers")
    images: dict[str, Image.Image] = {}
    records = []
    for row in rows:
        name = Path(row["file"]).name
        if name not in members:
            raise ValueError(f"records.csv names missing file {name!r}")
        if name not in images:
            image = Image.open(io.BytesIO(members[name]))
            image.load()
            images[name] = image.convert("RGB")
        records.append(
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "image": images[name].copy(),
                "question": row["question"],
                "answers": [a.strip() for a in row["answers"].split("|") if a.strip()],
                "source": "byod",
            }
        )
    return records


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "document_id", "file", "question", "answers", "source"])
        for record in records:
            writer.writerow(
                [
                    record["id"],
                    record["document_id"],
                    f"{record['document_id']}.png",
                    record["question"],
                    "|".join(record["answers"]),
                    record.get("source", ""),
                ]
            )
    return out


def manifest_json(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    return json.dumps(
        {
            "corpus": CORPUS_NAME,
            "version": CORPUS_VERSION,
            "license": CORPUS_LICENSE,
            "splits": check_split_disjoint(splits),
            "digest": dataset_digest([record for rows in splits.values() for record in rows]),
        },
        indent=2,
    )
