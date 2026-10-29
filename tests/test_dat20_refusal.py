"""DAT20: a refused BYOD input in Section 4 shows exactly one line, ``Refused: <file>: <reason>``, with no traceback,
and the cell raises so Run all stops there (before this change it printed and carried on). The router now carries the
canonical refusal channel (dat20-router-refusal/1). The in-cell invalidation is kept; a genuine failure keeps its
traceback. Section 4 is executed from the committed notebook."""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

from test_row63_review_fixes import _cell_with, _section4_namespace, nb  # noqa: F401  (fixture)

REFUSAL_CELL = "pix2struct_docvqa-17"
MARKER = "USE_BYOD = False  # @param"


def _section4(nb: dict, tmp_path: Path, path: str, ns: dict | None = None):  # noqa: F811
    source = _cell_with(nb, MARKER).replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    source = source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)
    ns = _section4_namespace(tmp_path, source) if ns is None else ns
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "<section4>", "exec"), ns)
    return ns, out.getvalue()


def _refused(nb: dict, tmp_path: Path, path: str, ns: dict | None = None) -> tuple[str, dict]:  # noqa: F811
    ns = _section4_namespace(tmp_path, "") if ns is None else ns
    with pytest.raises(ValueError) as caught:
        _section4(nb, tmp_path, path, ns)
    exc = caught.value
    assert type(exc).__name__ == "BYODRefusal" and exc.dimer_refusal is True and exc.__suppress_context__
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    return rendered[0], ns


def test_the_scripted_refusal_input_is_one_line_naming_the_file(nb, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    line, ns = _refused(nb, tmp_path, str(hostname))
    assert line.startswith("Refused: hostname: expected a zip with records.csv and page images ending in .zip") and "hostname: hostname" not in line
    assert "BYOD input refused" not in line and ns["DATA_READY"] is False and ns["byod_refusal"]


def test_a_directory_and_no_upload_dialog_are_refused_by_name(nb, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    (tmp_path / "etc").mkdir()
    line, _ = _refused(nb, tmp_path, str(tmp_path / "etc"))
    assert line.startswith("Refused: etc: BYOD path ")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    line, _ = _refused(nb, tmp_path, "")
    assert line.startswith("Refused: (no file): BYOD is on but its path field is empty")


def test_the_in_cell_invalidation_is_kept(nb, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "pix2struct_docvqa_result.json").write_text("{}", encoding="utf-8")
    ns = _section4_namespace(tmp_path, "")
    ns.update({"frozen": {"anls": 1.0}, "splits": {"stale": []}, "result": {"stale": True}})
    line, ns = _refused(nb, tmp_path, str(tmp_path / "missing.zip"), ns)
    assert line.startswith("Refused: missing.zip: BYOD path ") and "moved, not deleted" in line
    assert not {"splits", "frozen", "result", "input_manifest"} & set(ns)
    assert not (tmp_path / "outputs" / "pix2struct_docvqa_result.json").exists()


def test_a_genuine_failure_keeps_its_type(nb, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mine.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # an empty but valid zip
    ns = _section4_namespace(tmp_path, "")

    def broken(path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug"):
        _section4(nb, tmp_path, str(tmp_path / "mine.zip"), ns)


def test_the_refusal_cell_id_and_the_router_channel(nb):  # noqa: F811
    cell = next(c for c in nb["cells"] if c.get("id") == REFUSAL_CELL)
    source = "".join(cell["source"])
    assert source == _cell_with(nb, MARKER) and "BYOD_PATH = ''" in source and "raise BYODRefusal(byod_label, " in source
    assert "print(f'BYOD input refused" not in source
    router = _cell_with(nb, "class IsolatedCellRefusal(Exception):")
    assert 'if getattr(exc, "dimer_refusal", False):' in router and 'elif kind == "refusal":' in router
    assert 'return [f"Refused: {self}"]' in router
