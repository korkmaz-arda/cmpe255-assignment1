"""Application smoke tests, plus the core/UI separation invariant."""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

from basket import autoresearch, config, pipeline

ROOT = Path(__file__).resolve().parent.parent


def test_core_package_imports_without_streamlit():
    """The analytical core must be usable, and testable, with no UI framework installed."""
    script = """
import builtins
real = builtins.__import__

def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'streamlit', 'plotly'}:
        raise ImportError(f'{name} is not installed')
    return real(name, *args, **kwargs)

builtins.__import__ = guard
import basket.config, basket.data, basket.mining, basket.rules, basket.graph
import basket.recommend, basket.benchmark, basket.pipeline, basket.engine
import basket.autoresearch, basket.reference
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_no_streamlit_import_in_the_core_package():
    for path in (ROOT / "basket").glob("*.py"):
        source = path.read_text()
        assert "import streamlit" not in source, path.name
        assert "import plotly" not in source, path.name


@pytest.fixture
def running_app(prepared):
    pipeline.run(min_support=0.05, min_confidence=0.3, min_lift=1.0, max_len=4,
                 include_reference=False)
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=90)
    app.run()
    return app


def test_workspace_renders_with_a_seeded_basket(running_app):
    assert not running_app.exception, running_app.exception
    body = " ".join(str(m.value) for m in running_app.markdown)
    assert "Which product should we suggest next" in body
    # Opens on a non-empty basket, so suggestions are visible immediately.
    assert running_app.session_state["basket"]


def test_admin_view_renders_every_section(running_app):
    running_app.session_state["view"] = "Data-science admin"
    for tab in ("Algorithm benchmark", "Rules explorer", "Re-mine"):
        running_app.session_state["admin_tab"] = tab
        running_app.run()
        assert not running_app.exception, (tab, running_app.exception)
        # The section actually rendered, rather than silently falling back to the
        # workspace — the mistake a mirrored view key used to hide.
        headings = [s.value for s in running_app.subheader]
        assert "Basket" not in headings, tab


def test_rules_explorer_searches_and_reports_no_match(running_app):
    running_app.session_state["view"] = "Data-science admin"
    running_app.session_state["admin_tab"] = "Rules explorer"
    running_app.session_state["rules_query"] = "zzz-no-such-product"
    running_app.run()
    assert not running_app.exception
    assert any("No rules mention" in i.value for i in running_app.info)


def test_parameter_search_reports_when_it_has_not_run(running_app):
    running_app.session_state["view"] = "Data-science admin"
    running_app.session_state["admin_tab"] = "Parameter search"
    running_app.run()
    assert not running_app.exception
    assert any("not been run yet" in i.value for i in running_app.info)


def test_first_run_state_explains_what_to_do(tmp_path, monkeypatch):
    """With no artifacts, the app must say how to make some rather than crash."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("BASKET_ARTIFACT_DIR", str(tmp_path / "empty"))
    app = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=60)
    app.run()
    assert not app.exception
    body = " ".join(str(m.value) for m in app.markdown)
    assert "basket.pipeline" in body


def test_glossary_defines_every_metric_the_ui_shows():
    from app import glossary
    for term in glossary.METRIC_TERMS + glossary.ALGORITHM_TERMS:
        assert glossary.define(term), term
    # Confidence must be stated as a conditional rate, never as a share of all baskets.
    confidence = glossary.define("confidence").lower()
    assert "conditional" in confidence
    assert "not '30% of all baskets'" in confidence.replace("’", "'")
    # Lift must not be described causally.
    assert "not say buying one makes" in glossary.define("lift")
