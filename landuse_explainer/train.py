"""This project has no training - it visualizes a model trained in `landuse_tagger`.

`train.py` exists for convention; it just forwards to evaluate.py.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    print("`landuse_explainer` has no training step; running evaluate.py instead")
    runpy.run_path(str(HERE / "evaluate.py"), run_name="__main__")


if __name__ == "__main__":
    sys.exit(main())
