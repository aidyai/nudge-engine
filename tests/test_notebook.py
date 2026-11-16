"""The Colab notebook runs commands we cannot try from here, so this checks every one of them can be found.

The first bug a user hit was `python scripts/x.py` failing with "No module named engine". These tests make
that kind of mistake fail in CI, before a person sees it in a cell.
"""

import importlib
import importlib.util
import json
import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "nudge_engine_setup.ipynb"


def commands() -> list[str]:
    """Every shell line in the code cells: `!cmd` lines, and lines inside `%%bash` cells."""
    out = []
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.lstrip().startswith("%%bash"):
            out += [ln.strip() for ln in src.splitlines()[1:] if ln.strip() and not ln.startswith("#")]
        out += [ln.strip()[1:] for ln in src.splitlines() if ln.strip().startswith("!")]
    return out


def uv_runs() -> list[list[str]]:
    result = []
    for line in commands():
        if "uv run" not in line:
            continue
        # drop {placeholders} the notebook fills in, and anything after a shell pipe or redirect
        words = shlex.split(re.sub(r"\{[^}]*\}", "x", line.split("|")[0].split(">")[0]), posix=True)
        result.append(words[words.index("run") + 1 :])
    return result


def test_the_notebook_has_commands_to_check():
    assert len(uv_runs()) >= 8


def test_python_is_never_run_by_file_path():
    """`python path/to/file.py` hides the repo from the script. Use `python -m package.module`."""
    for words in uv_runs():
        if "python" in words:
            rest = words[words.index("python") + 1 :]
            assert not (rest and rest[0].endswith(".py")), f"run it with -m instead: {' '.join(words)}"


def test_every_python_module_the_notebook_runs_can_be_imported():
    for words in uv_runs():
        if "python" in words and "-m" in words:
            module = words[words.index("-m") + 1]
            assert importlib.util.find_spec(module) is not None, f"cannot find module {module}"


def test_every_modal_file_and_entrypoint_exists():
    for words in uv_runs():
        if "modal" not in words or words[words.index("modal") + 1 : words.index("modal") + 2] not in (["run"], ["deploy"]):
            continue
        target = words[words.index("modal") + 2]
        path, _, entry = target.partition("::")
        assert (ROOT / path).is_file(), f"no such file: {path}"
        if entry:
            module = importlib.import_module(path.removesuffix(".py").replace("/", "."))
            assert hasattr(module, entry), f"{path} has no entrypoint {entry}"


def test_the_notebook_adds_the_repo_to_the_python_path():
    first_code = next(c for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code")
    assert 'os.environ["PYTHONPATH"]' in "".join(first_code["source"])


@pytest.mark.parametrize("script", ["scripts.check_services", "engine.seed", "evals.report"])
def test_the_modules_the_notebook_uses_import_from_the_repo_root(script):
    assert importlib.import_module(script)
