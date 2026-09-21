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
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont

MODEL_ID = "google/pix2struct-docvqa-base"
MODEL_REVISION = "63f6b3de436e39f75c7a486881a9c2c14a7f4e89"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "pix2struct-docvqa-base"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = "067f7f314d87fa56daa5bcfaf36fa0b33ceebf7b7d4fae6a1e51ab7af64ee0b5"
DECODER_LAYERS = 12
DEFAULT_TRAINABLE_DECODER_LAYERS = 2
MAX_TARGET_TOKENS = 32
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 50
ARTIFACT_FORMAT = "org.valcorza.pix2struct-docvqa-base.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"

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
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    _font_bytes: bytes | None = field(default=None, repr=False)

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
        for param in model.parameters():
            param.requires_grad_(False)

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

        return cls(
            runner,
            resolved_device,
            "float32",
            source,
            _model=model,
            _processor=processor,
            _font_bytes=font_bytes,
        )

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

    # ---- adaptation contract -----------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None or self._font_bytes is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._processor

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Answer and score a validated document-QA corpus with ANLS and exact match."""
        from .metrics import qa_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        predictions = [
            self.answer(record["image"], record["question"], max_new_tokens=max_new_tokens)["answer"]
            for record in checked
        ]
        metrics = qa_metrics(predictions, checked)
        metrics.update(
            {
                "max_new_tokens": max_new_tokens,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trainable_decoder_layers: int) -> list[str]:
        if (
            isinstance(trainable_decoder_layers, bool)
            or not isinstance(trainable_decoder_layers, int)
            or not 1 <= trainable_decoder_layers <= DECODER_LAYERS
        ):
            raise ValueError(f"trainable_decoder_layers must be an int in 1..{DECODER_LAYERS}")
        model, _ = self._require_model()
        first = DECODER_LAYERS - trainable_decoder_layers
        prefixes = tuple(f"decoder.layer.{index}." for index in range(first, DECODER_LAYERS))
        names = [name for name, _param in model.named_parameters() if name.startswith(prefixes)]
        if not names:
            raise RuntimeError("no decoder-layer parameters matched the pinned Pix2Struct architecture")
        return names

    def _training_inputs(self, records: Sequence[Mapping[str, Any]]) -> Any:
        model, processor = self._require_model()
        device = next(model.parameters()).device
        return processor.image_processor(
            images=[record["image"] for record in records],
            header_text=[record["question"] for record in records],
            max_patches=MAX_PATCHES,
            return_tensors="pt",
            font_bytes=self._font_bytes,
        ).to(device)

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 3,
        lr: float = 2e-4,
        batch_size: int = 1,
        trainable_decoder_layers: int = DEFAULT_TRAINABLE_DECODER_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Tune only the last decoder blocks and retain the best validation-ANLS epoch."""
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not 0.0 < lr <= 1e-3:
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be an int in 1..16")
        names = self._trainable_names(trainable_decoder_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )

        import torch

        torch.manual_seed(seed)
        model, processor = self._require_model()
        device = next(model.parameters()).device
        wanted = set(names)
        initial_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name in wanted
        }
        best_state = {name: value.clone() for name, value in initial_state.items()}
        started = time.perf_counter()

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {
                key: value
                for key, value in self.evaluate(val_checked).items()
                if key in ("anls", "exact_match", "n")
            }

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {
            "epoch": 0,
            "train_loss": None,
            "val": score_val(),
            "note": "frozen model",
        }
        history.append(entry)
        if progress:
            progress(entry)
        best_score = entry["val"]["anls"] if entry["val"] else -math.inf
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [param for param in model.parameters() if param.requires_grad]
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                model.encoder.eval()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    indexes = order[start : start + batch_size]
                    batch = [train_checked[index] for index in indexes]
                    inputs = self._training_inputs(batch)
                    tokenized = processor.tokenizer(
                        [record["answers"][0] for record in batch],
                        padding=True,
                        truncation=True,
                        max_length=MAX_TARGET_TOKENS,
                        return_tensors="pt",
                    ).to(device)
                    labels = tokenized["input_ids"].clone()
                    labels[labels == processor.tokenizer.pad_token_id] = -100
                    optimiser.zero_grad(set_to_none=True)
                    output = model(**inputs, labels=labels)
                    output.loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(output.loss.detach()))
                model.eval()
                entry = {
                    "epoch": epoch,
                    "train_loss": sum(losses) / len(losses),
                    "val": score_val(),
                }
                history.append(entry)
                if progress:
                    progress(entry)
                current = entry["val"]["anls"] if entry["val"] else math.inf
                if current > best_score or not entry["val"]:
                    best_score = current
                    best_state = {
                        name: value.detach().clone()
                        for name, value in model.state_dict().items()
                        if name in wanted
                    }
                    best_epoch = epoch
        except BaseException:
            restored = dict(model.state_dict())
            restored.update(initial_state)
            model.load_state_dict(restored, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_decoder_layers": trainable_decoder_layers,
            "trainable_names": names,
            "n_trainable": sum(param.numel() for param in params),
            "n_total": sum(param.numel() for param in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation ANLS" if val_checked else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ---------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Save the adapted decoder tensors as safetensors, bound to the pinned base digest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
            if name in names
        }
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {
                key: value
                for key, value in self.adapter.items()
                if key not in ("history", "trainable_names")
            },
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _check_artifact_manifest(self, root: Path, manifest: Mapping[str, Any]) -> Path:
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not "
                f"{ARTIFACT_FORMAT_VERSION!r}"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file") != WEIGHT_FILE:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        adapter = manifest.get("adapter")
        layers = adapter.get("trainable_decoder_layers") if isinstance(adapter, Mapping) else None
        if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= DECODER_LAYERS:
            raise ValueError("artifact manifest does not record valid trainable decoder layers")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify manifest, digest and exact tensor set before loading an adapter."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_decoder_layers"]))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith("decoder.layer."):
                raise ValueError(f"artifact tensor {key} is not an adaptable decoder tensor")
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({key: value.to(state[key].dtype) for key, value in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Pix2StructDocVQAPipeline:
        pipeline = cls.from_pretrained(
            device=device,
            weights_dir=weights_dir,
            allow_download=allow_download,
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
