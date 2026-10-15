"""Общие параметры всех этапов эксперимента."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODEL_NAME = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DATASET_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
MAX_LENGTH = 128
SEED = 42
LABELS = {0: "Negative", 1: "Positive"}
MODEL_DIR = ROOT / "fine_tuned_model"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
