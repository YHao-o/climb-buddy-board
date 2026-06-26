import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db as guide_db


@pytest.fixture
def conn(tmp_path):
    c = guide_db.get_connection(tmp_path / "test.db")
    guide_db.init_db(c)
    yield c
    c.close()
