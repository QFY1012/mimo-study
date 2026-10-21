"""Build the Colab notebooks and plain scripts from colab/src/.

- Inlines the official judge prompts (XiaomiMiMo/verl, Apache-2.0) into webdev_common.py.
- Replaces the `# __LIBRARY__` cell with code that writes the shared library next to the
  notebook's work dir and imports it, so each notebook is self-contained.
- Writes colab/<name>.ipynb (for the browser) and colab/dist/<name>.py (for `colab exec` / python).

Run: python colab/build.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
NOTEBOOKS = {"stage0_eval": "webdev_stage0_eval", "stage1_train": "webdev_stage1_train"}


def library_files() -> dict:
    common = (SRC / "webdev_common.py").read_text()
    common = common.replace("__QUERY_FIT__", repr((HERE / "prompts" / "query_fit.md").read_text()))
    common = common.replace("__GROUP_PICK__", repr((HERE / "prompts" / "group_pick.md").read_text()))
    return {"webdev_common.py": common, "lora_sync.py": (SRC / "lora_sync.py").read_text()}


def library_cell(files: dict) -> str:
    lines = ["# 共享库：渲染、判官、奖励（两个 notebook 共用同一份代码）",
             "LIB_FILES = {"]
    lines += [f"    {name!r}: {src!r}," for name, src in files.items()]
    lines += ["}",
              'LIB_DIR = WORKDIR / "lib"',
              "LIB_DIR.mkdir(parents=True, exist_ok=True)",
              "for _name, _src in LIB_FILES.items():",
              "    (LIB_DIR / _name).write_text(_src)",
              "if str(LIB_DIR) not in sys.path:",
              "    sys.path.insert(0, str(LIB_DIR))",
              "import importlib",
              "import webdev_common as wc",
              "import lora_sync",
              "importlib.reload(wc)",
              "importlib.reload(lora_sync)"]
    return "\n".join(lines)


def to_cells(script: str) -> list:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})

    for line in script.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    for c in cells:
        c["source"] = [l + "\n" for l in c["source"]]
        if c["source"]:
            c["source"][-1] = c["source"][-1].rstrip("\n")
    return cells


def main():
    lib = library_cell(library_files())
    (HERE / "dist").mkdir(exist_ok=True)
    for src_name, out_name in NOTEBOOKS.items():
        script = (SRC / f"{src_name}.py").read_text()
        assert "# __LIBRARY__" in script, src_name
        script = script.replace("# __LIBRARY__", lib)
        (HERE / "dist" / f"{out_name}.py").write_text(script)
        nb = {
            "cells": to_cells(script),
            "metadata": {
                "accelerator": "GPU",
                "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
                "kernelspec": {"display_name": "Python 3", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 0,
        }
        (HERE / f"{out_name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
        print("built", out_name, f"({len(nb['cells'])} cells)", file=sys.stderr)


if __name__ == "__main__":
    main()
