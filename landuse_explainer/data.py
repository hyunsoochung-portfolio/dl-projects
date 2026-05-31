"""Reuses the EuroSAT pipeline from `landuse_tagger`."""
from __future__ import annotations

import sys
from pathlib import Path

_MOD04 = Path(__file__).resolve().parent.parent / "landuse_tagger"
sys.path.insert(0, str(_MOD04))

from data import (  # noqa: E402,F401
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    load_tf_datasets,
)

CHECKPOINT_PATH = _MOD04 / "artifacts" / "eurosat_cnn.keras"
