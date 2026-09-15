import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zenith.clock import FixedClock  # noqa: E402
from zenith.db import Database  # noqa: E402
from zenith.store import Store  # noqa: E402

# A Wednesday, so weekday resolution is easy to reason about.
NOW = "2026-09-16T10:00:00"


@pytest.fixture
def clock():
    return FixedClock(NOW)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "zenith.db"


@pytest.fixture
def store(db_path, clock):
    return Store(Database(db_path), clock)
