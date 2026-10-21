"""Build the Colab notebook from colab/src/webdev_stage0.py.

Inlines the official judge prompts (from XiaomiMiMo/verl, Apache-2.0) and splits the
percent-format script into notebook cells. Run: python colab/build.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"


def render_script() -> str:
    src = (HERE / "src" / "webdev_stage0.py").read_text()
    src = src.replace("__QUERY_FIT__", repr((PROMPTS / "query_fit.md").read_text()))
    src = src.replace("__GROUP_PICK__", repr((PROMPTS / "group_pick.md").read_text()))
    return src


def to_cells(script: str) -> list[dict]:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": [l + "\n" for l in lines]})

    for line in script.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    for c in cells:
        if c["source"]:
            c["source"][-1] = c["source"][-1].rstrip("\n")
    return cells


def main():
    script = render_script()
    (HERE / "webdev_stage0_rendered.py").write_text(script)
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
    (HERE / "webdev_stage0.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print("wrote", HERE / "webdev_stage0.ipynb", "and", HERE / "webdev_stage0_rendered.py", file=sys.stderr)


if __name__ == "__main__":
    main()
