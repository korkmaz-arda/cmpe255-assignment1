"""S01/S05/S06 - the app renders every view from cached results, with no fallback figures."""

from __future__ import annotations

from pathlib import Path

import pytest

from skills_lab.catalog import View
from skills_lab.data.acquire import missing_datasets

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app" / "main.py"

pytestmark = pytest.mark.skipif(
    bool(missing_datasets()),
    reason="run python scripts/prepare_data.py to enable app smoke tests",
)


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(APP), default_timeout=900).run()


def test_app_starts_on_the_catalog(app):
    assert not app.exception
    assert app.session_state["view"] == View.CATALOG.value


def test_every_view_renders_without_error(app):
    for view in View:
        app.button(key=f"nav-{view.value}").click().run()
        assert not app.exception, f"{view.value} raised {app.exception}"
        assert app.session_state["view"] == view.value


def test_catalog_jump_targets_all_reach_a_workspace(app):
    app.button(key="nav-catalog").click().run()
    jump_keys = [b.key for b in app.button if b.key and b.key.startswith("jump-")]
    assert jump_keys
    for key in jump_keys:
        app.button(key="nav-catalog").click().run()
        app.button(key=key).click().run()
        assert app.session_state["view"] != View.CATALOG.value
        assert not app.exception


def test_running_a_skill_shows_a_real_payload(app):
    app.button(key="nav-catalog").click().run()
    app.button(key="run-model-evaluation").click().run()
    assert not app.exception
    assert app.session_state.get("pending_skill") == "model-evaluation"
    assert app.code  # the JSON payload block
