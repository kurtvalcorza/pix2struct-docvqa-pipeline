"""OCR-free document question answering with the pinned ``google/pix2struct-docvqa-base`` checkpoint.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the Pix2Struct architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed. The question is rendered as a
text header on top of the page (the Pix2Struct VQA input convention) with Pillow's bundled font, so no
font is fetched from the Hub at inference time.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont

MODEL_ID = "google/pix2struct-docvqa-base"
MODEL_REVISION = "63f6b3de436e39f75c7a486881a9c2c14a7f4e89"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "pix2struct-docvqa-base"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Generation ceilings. DocVQA answers are short spans (the checkpoint's text_config max_length is 20);
# the default leaves room for a long address or title, the ceiling bounds runaway generation.
MAX_NEW_TOKENS = 128
DEFAULT_MAX_NEW_TOKENS = 32
DECODING = "greedy"
# Question ceiling. The question is rendered as a header line (wrapped at 80 characters by the
# processor) above the page; a very long question shrinks the page's share of the patch budget.
MAX_QUESTION_CHARS = 256
# Input ceilings. The processor extracts at most MAX_PATCHES 16x16 patches (preprocessor_config.json)
# after scaling the image to fill that budget, so pixel count only guards memory during resizing.
MAX_PATCHES = 2048
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
# ANLS (DocVQA's official metric): a normalised Levenshtein similarity below this threshold scores 0.
ANLS_THRESHOLD = 0.5
_PUNCT_RE = re.compile(r"[^\w\s]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def header_font_bytes() -> bytes:
    """Pillow's bundled Aileron Regular (CC0) as TrueType bytes: the header font for the rendered question.

    The upstream image processor otherwise fetches ``ybelkada/fonts/Arial.TTF`` from the Hub at
    inference time — an unpinned, unlisted download of a proprietary font. The bundled subset covers
    the printable ASCII range, which is what a question is expected to use.
    """
    font = ImageFont.load_default(size=36)
    data = getattr(font, "font_bytes", None)
    if not data:
        raise RuntimeError("Pillow's bundled TrueType font is unavailable (FreeType support missing)")
    return bytes(data)


def normalize_answer(text: str) -> str:
    """DocVQA-style normalisation: lower-case, punctuation removed, whitespace collapsed."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def anls(prediction: str, golds: Sequence[str], *, threshold: float = ANLS_THRESHOLD) -> float:
    """Average Normalised Levenshtein Similarity for one question (Biten et al., ICDAR 2019).

    ``1 - lev(pred, gold) / max(len(pred), len(gold))`` over normalised strings, maximised over the
    accepted ``golds``; a similarity below ``threshold`` scores 0 so a near-miss is not rewarded.
    """
    if not golds:
        raise ValueError("golds must contain at least one accepted answer")
    pred = normalize_answer(prediction)
    best = 0.0
    for gold in golds:
        ref = normalize_answer(gold)
        longest = max(len(pred), len(ref))
        similarity = 1.0 if longest == 0 else 1.0 - _levenshtein(pred, ref) / longest
        best = max(best, similarity)
    return best if best >= threshold else 0.0


def exact_match(prediction: str, golds: Sequence[str]) -> bool:
    """Whether the normalised prediction equals any normalised accepted answer."""
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(gold) for gold in golds)


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one page image as PIL.Image.Image (any mode, converted to RGB) plus one question string",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "question_chars": [1, MAX_QUESTION_CHARS],
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "decoding": f"{DECODING} (do_sample=False), deterministic on a fixed device and dtype",
    "preprocessing": (
        "the question is rendered as a black-on-white header (Pillow's bundled font, wrapped at 80 "
        "characters) above the page; the composite is scaled to fill at most MAX_PATCHES 16x16 patches "
        "(aspect ratio preserved), normalised per image, and flattened into patch tokens with row/column "
        "positions; the decoder generates the answer text"
    ),
    "output": "one answer string (the model's decoded text), no score",
}


def _check_inputs(image: Any, question: Any, max_new_tokens: Any) -> tuple[Image.Image, str, int]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``answer`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    if not isinstance(question, str):
        raise TypeError("question must be a str")
    checked_question = " ".join(question.split())
    if not checked_question:
        raise ValueError("question must contain at least one non-whitespace character")
    if len(checked_question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"question has {len(checked_question)} chars > MAX_QUESTION_CHARS {MAX_QUESTION_CHARS}"
        )
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    return rgb, checked_question, max_new_tokens


def validate_inputs(
    image: Image.Image,
    questions: Sequence[str],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Every question is checked exactly as ``answer`` would check it; rejection is reported by raising,
    and a caller that wants the finding recorded catches the exception and stores ``str(exc)`` under
    ``findings``.
    """
    if isinstance(questions, str) or not isinstance(questions, Sequence) or not questions:
        raise TypeError("questions must be a non-empty sequence of str")
    checked = [_check_inputs(image, question, max_new_tokens)[1] for question in questions]
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (answer takes one page image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "questions": checked,
        "generation": {"max_new_tokens": int(max_new_tokens), "do_sample": False, "decoding": DECODING},
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    results: Sequence[Mapping[str, Any]],
    golds: Sequence[Sequence[str]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``golds`` (one sequence of accepted answers per result, in order) the report carries the
    mean ``anls`` and the ``exact_match`` rate over the questions plus one per-question entry, verdict
    ``sample-sanity``; without golds it is ``not-measurable`` and says what labelled data would make
    the task measurable.
    """
    if not results:
        raise ValueError("results must contain at least one answer result")
    base = {
        "task": "document page image + question -> answer text (OCR-free)",
        "score_semantics": (
            "the answer is generated text and carries no score, probability or correctness signal; a "
            "fluent answer is not evidence that it is read from the page. Greedy decoding makes the "
            "output reproducible on a fixed device and dtype, a reproducibility property, not a quality one"
        ),
        "sample_kind": sample_kind,
        "n_questions": len(results),
        "truncated": [bool(result.get("truncated")) for result in results],
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if golds is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no accepted answers were supplied for the evaluated questions",
            "needs": (
                "question/answer pairs with accepted answers on pages from the deployment domain "
                "(DocVQA-style annotations) scored with ANLS; no such labelled set ships with this repository"
            ),
        }
    if len(golds) != len(results):
        raise ValueError(f"golds has {len(golds)} entries for {len(results)} results")
    per_question = []
    for result, accepted in zip(results, golds, strict=True):
        if isinstance(accepted, str) or not accepted:
            raise ValueError("each golds entry must be a non-empty sequence of accepted answers")
        prediction = str(result["answer"])
        per_question.append(
            {
                "question": result.get("question"),
                "prediction": prediction,
                "golds": list(accepted),
                "anls": anls(prediction, accepted),
                "exact_match": exact_match(prediction, accepted),
            }
        )
    metrics = [
        {
            "id": "anls",
            "value": sum(entry["anls"] for entry in per_question) / len(per_question),
            "threshold": ANLS_THRESHOLD,
            "normalisation": "lower-cased, punctuation removed, whitespace collapsed; max over golds",
            "estimation": f"{len(per_question)} question(s) on one page, no dispersion estimate",
        },
        {
            "id": "exact_match",
            "value": sum(entry["exact_match"] for entry in per_question) / len(per_question),
            "normalisation": "lower-cased, punctuation removed, whitespace collapsed",
            "estimation": f"{len(per_question)} question(s) on one page, no dispersion estimate",
        },
    ]
    return {
        **base,
        "metrics": metrics,
        "per_question": per_question,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(per_question)} authored question(s) on one tutorial page whose content you rendered "
            "yourself; plumbing evidence, not a DocVQA benchmark"
        ),
        "needs": (
            "a labelled question/answer set on pages from the deployment domain (scans, forms, layouts) "
            "for any accuracy claim; the DocVQA benchmark itself is registration-gated and not bundled"
        ),
    }


@dataclass
class Pix2StructDocVQAPipeline:
    """``_runner(image, question, max_new_tokens)`` returns ``{"answer": str, "new_tokens": int}``."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Pix2StructDocVQAPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        font_bytes = header_font_bytes()
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = Pix2StructProcessor.from_pretrained(location, **common)
        if not getattr(processor.image_processor, "is_vqa", False):
            raise RuntimeError("snapshot image processor is not the VQA variant (is_vqa=False); refusing")
        model = Pix2StructForConditionalGeneration.from_pretrained(location, dtype=torch.float32, **common)
        model = model.eval().to(resolved_device)

        def runner(image: Image.Image, question: str, max_new_tokens: int) -> dict[str, Any]:
            # The image processor is called directly: Pix2StructProcessor.__call__ drops the
            # font_bytes kwarg, and font_bytes is what replaces the default Hub font download
            # (see header_font_bytes). The VQA processor renders the question as the header.
            inputs = processor.image_processor(
                image, header_text=question, return_tensors="pt", font_bytes=font_bytes
            ).to(resolved_device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            # Encoder-decoder: the output holds only decoder tokens (decoder_start + answer + eos).
            answer_ids = generated[0]
            decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return {"answer": decoded, "new_tokens": int(answer_ids.shape[0]) - 1}

        return cls(runner, resolved_device, "float32", source)

    def answer(
        self,
        image: Image.Image,
        question: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Answer one question about one page image; ``answer`` is the decoded text, stripped."""
        rgb, checked_question, checked_tokens = _check_inputs(image, question, max_new_tokens)
        raw = self._runner(rgb, checked_question, checked_tokens)
        if not isinstance(raw, dict) or "answer" not in raw:
            raise RuntimeError("runner must return a dict with 'answer'")
        new_tokens = int(raw.get("new_tokens", 0))
        return {
            "answer": str(raw["answer"]).strip(),
            "question": checked_question,
            "image_size": list(rgb.size),
            "new_tokens": new_tokens,
            "truncated": new_tokens >= checked_tokens,
            "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
