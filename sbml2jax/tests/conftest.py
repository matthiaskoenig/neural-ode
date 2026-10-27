from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "examples" / "models"
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def models_dir() -> Path:
    return MODELS_DIR


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR
