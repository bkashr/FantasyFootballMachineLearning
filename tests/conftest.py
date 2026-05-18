import os
import sys
from pathlib import Path

import pytest

# Make the project root importable when pytest is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_db(tmp_path):
    """Fresh schema-initialized SQLite file per test."""
    import database

    db_path = str(tmp_path / "test.db")
    database.init_db(db_path)
    return db_path
