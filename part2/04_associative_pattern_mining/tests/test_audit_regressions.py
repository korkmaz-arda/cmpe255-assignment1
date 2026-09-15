"""Regression tests for defects found in the user-facing audit.

Each of these was a real bug that unit tests of the analytical core could not
see, because it lived in what the user is shown. They are checked against the
rendered app, not against source text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.workspace import isolation
from basket import config, pipeline

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app" / "main.py")


def _app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=90)
    app.run()
    assert not app.exception, app.exception
    return app


@pytest.fixture
def served(prepared):
    pipeline.run(min_support=0.05, min_confidence=0.3, min_lift=1.0, max_len=4,
                 include_reference=True)


@pytest.fixture
def empty_rule_set(prepared):
    """Thresholds nothing can clear: the degenerate state a user can reach from Re-mine."""
    pipeline.run(min_support=0.05, min_confidence=0.99, min_lift=50.0, max_len=4,
                 include_reference=True)
    assert json.loads(config.RULES_JSON.read_text())["rules"] == []


def _plotly_spec(app: AppTest) -> dict:
    charts = app.get("plotly_chart")
    assert charts, "no chart rendered"
    return json.loads(charts[0].proto.spec)


def _node_trace(spec: dict) -> dict:
    return next(t for t in spec["data"] if t.get("mode") == "markers+text")


# --------------------------------------------------------------------------- #
# Affinity graph
# --------------------------------------------------------------------------- #

EDGES = [
    {"source": 1, "target": 2, "lift": 9.0},
    {"source": 2, "target": 3, "lift": 8.0},
    {"source": 4, "target": 5, "lift": 7.0},
]
NODES = {1, 2, 3, 4, 5}


def test_isolation_draws_only_edges_touching_the_selection():
    drawn, _, selected = isolation(EDGES, NODES, 1, visible_limit=35)
    assert selected == 1
    assert drawn == [EDGES[0]]


def test_isolation_dims_every_node_that_is_not_a_neighbour():
    """Used to dim nothing: highlights were computed from all edges, not touching ones."""
    _, highlighted, _ = isolation(EDGES, NODES, 1, visible_limit=35)
    assert highlighted == {1, 2}
    assert NODES - highlighted == {3, 4, 5}


def test_stale_selection_is_dropped():
    """A product no longer in the graph (e.g. after a re-mine) must not stay 'isolated'."""
    drawn, highlighted, selected = isolation(EDGES, NODES, 999, visible_limit=2)
    assert selected is None
    assert drawn == EDGES[:2]
    assert highlighted == NODES


def test_node_tooltip_shows_department_frequency_and_illustrative_price(served):
    """Used to show a raw product_id: Plotly ignores hovertext once hovertemplate is set."""
    node = _node_trace(_plotly_spec(_app()))
    assert "%{customdata[1]}" in node["hovertemplate"]
    product_id, tooltip = node["customdata"][0]
    assert isinstance(product_id, int)
    assert "appears in" in tooltip and "of baskets" in tooltip
    assert "illustrative price" in tooltip


def test_selecting_a_node_dims_the_rest_in_the_rendered_chart(served):
    app = _app()
    edges = json.loads(config.GRAPH_JSON.read_text())["edges"]
    app.session_state["selected_node"] = edges[0]["source"]
    app.run()
    node = _node_trace(_plotly_spec(app))
    opacities = node["marker"]["opacity"]
    assert 0.25 in opacities and 1.0 in opacities


def test_stale_selection_does_not_render_an_isolation_banner(served):
    app = _app()
    app.session_state["selected_node"] = 987654321
    app.run()
    assert not app.exception
    assert not any("rule(s) touching" in i.value for i in app.info)
    assert app.session_state["selected_node"] is None


# --------------------------------------------------------------------------- #
# Empty rule set
# --------------------------------------------------------------------------- #

def test_empty_rule_set_shows_undefined_metrics_as_dashes(empty_rule_set):
    """Top lift over zero rules is undefined; '0.00x' reads as a measurement."""
    app = _app()
    app.session_state["view"] = "Data-science admin"
    app.session_state["admin_tab"] = "Algorithm benchmark"
    app.run()
    tiles = {m.label: m.value for m in app.metric}
    assert tiles["Active rules"] == "0"
    assert tiles["Top lift"] == "—"
    assert any("No rules survived" in w.value for w in app.warning)


def test_explorer_blames_the_rule_set_not_the_search_box(empty_rule_set):
    app = _app()
    app.session_state["view"] = "Data-science admin"
    app.session_state["admin_tab"] = "Rules explorer"
    app.run()
    messages = [i.value for i in app.info] + [w.value for w in app.warning]
    assert not any("No rules mention" in m for m in messages)
    assert any("rule set is empty" in m for m in messages)


def test_empty_basket_does_not_point_at_presets_that_do_not_exist(empty_rule_set):
    app = _app()
    assert app.session_state["basket"] == []
    infos = [i.value for i in app.info]
    assert not any("quick-start" in m for m in infos)


# --------------------------------------------------------------------------- #
# Action feedback
# --------------------------------------------------------------------------- #

def test_remine_reports_the_new_rule_count_after_the_rerun(served):
    """Used to be wiped: st.success was followed immediately by st.rerun."""
    app = _app()
    app.session_state["view"] = "Data-science admin"
    app.session_state["admin_tab"] = "Re-mine"
    app.run()
    next(b for b in app.button if b.label == "Re-mine now").click().run()
    assert not app.exception
    outcomes = [s.value for s in app.success] + [w.value for w in app.warning]
    assert any("Re-mined" in m for m in outcomes), outcomes
    assert app.session_state["remining"] is False
    # Shown once, then cleared, so it does not linger on later interactions.
    assert app.session_state["action_message"] is None


def test_remine_to_zero_rules_warns_instead_of_celebrating(served, monkeypatch):
    """No rule surviving is a real outcome and must be reported as one.

    The fixture's strongest rules have confidence 1.0 and lift 7.0, so nothing in
    the UI's slider ranges empties it; the pipeline is steered to an empty result
    so the UI's handling of that outcome is what gets tested.
    """
    real_run = pipeline.run

    def run_to_empty(**kwargs):
        return real_run(**{**kwargs, "min_lift": 50.0})

    monkeypatch.setattr(pipeline, "run", run_to_empty)
    app = _app()
    app.session_state["view"] = "Data-science admin"
    app.session_state["admin_tab"] = "Re-mine"
    app.run()
    next(b for b in app.button if b.label == "Re-mine now").click().run()
    assert not app.exception
    assert any("no rule" in w.value for w in app.warning)
    assert not any("Re-mined:" in s.value for s in app.success)


@pytest.mark.parametrize(
    "label, value, param",
    [
        ("Minimum support", 0.01, "min_support"),
        ("Minimum confidence", 0.9, "min_confidence"),
        ("Minimum lift", 5.0, "min_lift"),
    ],
)
def test_each_remine_slider_reaches_the_pipeline(served, label, value, param):
    """Wiring, checked end to end: the value set in the UI is the value mined with."""
    app = _app()
    app.session_state["view"] = "Data-science admin"
    app.session_state["admin_tab"] = "Re-mine"
    app.run()
    {s.label: s for s in app.slider}[label].set_value(value)
    app.run()
    next(b for b in app.button if b.label == "Re-mine now").click().run()
    assert not app.exception
    assert json.loads(config.RULES_JSON.read_text())["params"][param] == pytest.approx(value)


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #

def test_benchmark_labels_production_tiles_and_benchmark_table_separately(served):
    app = _app()
    app.session_state["view"] = "Data-science admin"
    app.session_state["admin_tab"] = "Algorithm benchmark"
    app.run()
    headings = " ".join(m.value for m in app.markdown)
    assert "Production rule set" in headings
    assert "Benchmark pass" in headings


def test_top_lift_is_shown_with_the_evidence_behind_it(served):
    app = _app()
    app.session_state["view"] = "Data-science admin"
    app.session_state["admin_tab"] = "Algorithm benchmark"
    app.run()
    assert any("orders" in c.value and "top-lift rule" in c.value for c in app.caption)


def test_search_is_labelled_exploratory_and_not_applied(served, monkeypatch):
    from basket import autoresearch
    monkeypatch.setattr(config, "SEARCH_BASELINE",
                        {"min_support": 0.05, "max_len": 4,
                         "min_confidence": 0.2, "min_lift": 1.0})
    monkeypatch.setattr(config, "BUNDLE_MIN_SUPPORT", 0.10)
    autoresearch.save(autoresearch.execute(n_orders=105))
    app = _app()
    app.session_state["view"] = "Data-science admin"
    app.session_state["admin_tab"] = "Parameter search"
    app.run()
    assert any("Exploratory, not applied" in i.value for i in app.info)


def test_report_figures_are_measured_from_the_data(served):
    """The report used to hard-code 39,123 products, 'Banana 14%' and 'around 0.1%'."""
    meta = json.loads(config.RUN_META_JSON.read_text())["corpus"]
    assert meta["split_distinct_products"] == 11        # the fixture's ordered products
    assert meta["split_orders"] == 105
    source = (ROOT / "app" / "crispdm.py").read_text()
    for literal in ("39,123", "Banana", "14%", "around 0.1%"):
        assert literal not in source, literal
