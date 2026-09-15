"""Application smoke tests.

These render the real Streamlit script against a fixture-trained artifact set and
assert it produces no exceptions — enough to catch import errors, bad API usage
and broken bindings without any browser automation.
"""

from __future__ import annotations

import pytest

from segmentation import autoresearch, config, pipeline

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP = str(config.PROJECT_ROOT / "app" / "main.py")


@pytest.fixture
def app(clean_sample, monkeypatch):
    """A trained artifact set plus an AutoResearch record, then a booted app."""
    monkeypatch.setattr(config, "MIN_TRAIN_SAMPLE", 50)
    pipeline.run(k=4, seed=3, frame=clean_sample)
    autoresearch.save(autoresearch.AutoResearch(clean_sample, k=3, seed=1).run())
    return AppTest.from_file(APP, default_timeout=180).run()


def _assert_clean(at):
    assert not at.exception, [str(e.value) for e in at.exception]


def test_explorer_renders(app):
    _assert_clean(app)
    assert app.session_state["view_control"] == "Segment explorer"
    # A prediction is issued automatically on first load.
    assert app.session_state["prediction"] is not None
    assert app.session_state["prediction"]["persona"]["name"]


def test_explorer_shows_a_slider_for_every_attribute(app):
    keys = {slider.key for slider in app.slider}
    for field in config.BASE_FEATURES:
        assert f"input-{field}" in keys, f"no control for {field}"


def test_persona_filter_toggles(app):
    app.session_state["selected_cluster"] = 1
    app.run()
    _assert_clean(app)
    assert app.session_state["selected_cluster"] == 1


def test_tsne_projection_renders(app):
    """Drive the widget, not a session key.

    Setting a plain session key here would be silently overwritten by the
    widget on rerun, so the test would pass without ever rendering t-SNE.
    """
    app.segmented_control(key="projection_control").set_value("t-SNE").run()
    _assert_clean(app)
    assert app.session_state["projection_control"] == "t-SNE"
    caption = " ".join(c.value for c in app.caption)
    assert "t-SNE" in caption and "out-of-sample" in caption


def _open_admin(app, tab: str):
    app.segmented_control(key="view_control").set_value("Admin console").run()
    _assert_clean(app)
    app.segmented_control(key="admin_tab_control").set_value(tab).run()
    _assert_clean(app)
    return app


def test_admin_benchmarks_renders_real_content(app):
    """Assert the panel actually rendered, not merely that nothing raised."""
    _open_admin(app, "Benchmarks")
    columns = [list(d.value.columns) for d in app.dataframe]
    leaderboard = next(c for c in columns if "Algorithm" in c)
    assert {"Silhouette", "Davies–Bouldin", "Calinski–Harabasz", "Noise"} <= set(leaderboard)
    assert any("k" in c and "WCSS" in c for c in columns), "elbow sweep table missing"
    assert any(m.label == "Silhouette" for m in app.metric)


def test_admin_autoresearch_renders_real_content(app):
    _open_admin(app, "AutoResearch")
    text = " ".join(m.value for m in app.markdown)
    assert "backbone" in text.lower()
    assert any("Starting silhouette" == m.label for m in app.metric)


def test_admin_radar_renders_real_content(app):
    _open_admin(app, "Radar profiles")
    assert app.radio, "persona focus control missing"
    assert "All clusters" in app.radio[0].options


def test_autoresearch_filters_narrow_the_stream(app):
    _open_admin(app, "AutoResearch")
    app.selectbox(key="decision_filter").set_value("accepted").run()
    _assert_clean(app)
    headings = " ".join(m.value for m in app.markdown)
    assert "Experiment stream" in headings


def test_metric_glossary_is_present_for_readers(app):
    """Every metric on screen must be explained somewhere in plain language."""
    from segmentation import config

    _open_admin(app, "Benchmarks")
    body = " ".join(m.value for m in app.markdown)
    for entry in config.METRIC_GLOSSARY.values():
        assert entry["name"] in body, f"{entry['name']} is never explained"


def test_app_survives_missing_artifacts(monkeypatch):
    """Empty state must degrade to guidance, not to a traceback."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("No trained model" in w.value for w in at.warning)


def test_compact_collapses_blank_lines():
    from app import theme

    markup = "<div class='card'>\n  <b>x</b>\n  \n</div>"
    assert theme.compact(markup) == "<div class='card'> <b>x</b> </div>"
    assert "\n" not in theme.compact(markup)


@pytest.mark.parametrize("tab", ["Benchmarks", "AutoResearch", "Radar profiles"])
def test_no_html_block_contains_a_blank_line(app, tab):
    """A blank line inside an HTML block leaks raw tags onto the page.

    Streamlit renders HTML through a markdown parser, which treats a blank line
    as a paragraph break: it closes the block early and prints the trailing
    ``</div>`` as visible text. Conditional interpolations cause this whenever
    the condition is unmet and leaves a whitespace-only line behind.
    """
    _open_admin(app, tab)
    offenders = [
        m.value for m in app.markdown
        if "<div" in m.value and any(not line.strip() for line in m.value.splitlines())
    ]
    assert not offenders, f"{len(offenders)} HTML block(s) would leak raw tags"


def test_backbone_cards_are_well_formed_with_and_without_the_champion_badge(app):
    _open_admin(app, "AutoResearch")
    cards = [
        m.value for m in app.markdown
        if m.value.lstrip().startswith("<div class='persona-card'")
    ]
    assert len(cards) >= 2
    assert any("CHAMPION" in c for c in cards), "no champion badge rendered"
    assert any("CHAMPION" not in c for c in cards), "no plain card to check"
    for card in cards:
        assert card.rstrip().endswith("</div>")
        assert card.count("<div") == card.count("</div>")


# --------------------------------------------------------------------------- #
# Regressions from the user-facing audit
# --------------------------------------------------------------------------- #

def _plotly_spec(app):
    import json

    return json.loads(app.get("plotly_chart")[0].proto.spec)


def _decode(value):
    import base64

    import numpy as np

    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
    return np.asarray(value)


def _live_marker_x(app):
    markers = [t for t in _plotly_spec(app)["data"] if t.get("name") == "Your customer"]
    return float(_decode(markers[0]["x"])[0]) if markers else None


def test_live_marker_matches_the_result_card_on_first_load(app):
    """The scatter once drew before the classifier computed the prediction, so the
    first load showed a result card with no marker on the canvas."""
    _assert_clean(app)
    assert _live_marker_x(app) == pytest.approx(app.session_state["prediction"]["pca_x"])


def test_live_marker_is_never_one_submission_behind(app):
    """After a new submission the marker once stayed at the previous customer's
    position, contradicting the result card beside it."""
    before = app.session_state["prediction"]["pca_x"]
    for field, value in dict(income_k=95.0, spending_score=92.0, total_spend=2400.0,
                             discount_sensitivity=0.03, web_visits_month=2.0).items():
        app.slider(key=f"input-{field}").set_value(value)
    submit = [b for b in app.button if b.label == "Classify customer"][0]
    submit.click().run()
    _assert_clean(app)
    after = app.session_state["prediction"]["pca_x"]
    assert after != pytest.approx(before), "the submission should move the customer"
    assert _live_marker_x(app) == pytest.approx(after)


def test_live_marker_is_absent_on_tsne(app):
    app.segmented_control(key="projection_control").set_value("t-SNE").run()
    _assert_clean(app)
    assert _live_marker_x(app) is None


def test_empty_experiment_filter_explains_itself(app):
    _open_admin(app, "AutoResearch")
    phases = app.selectbox(key="phase_filter").options
    decisions = app.selectbox(key="decision_filter").options
    # Search for a combination that matches nothing in this run.
    for phase in phases[1:]:
        for decision in decisions[1:]:
            app.selectbox(key="phase_filter").set_value(phase)
            app.selectbox(key="decision_filter").set_value(decision).run()
            heading = [m.value for m in app.markdown if "Experiment stream" in m.value][0]
            if "· 0 of" in heading:
                assert any("No steps match" in i.value for i in app.info)
                return
    pytest.skip("every filter combination matched at least one step in this run")


def test_autoresearch_tile_is_not_presented_as_comparable(app):
    tile = [m for m in app.metric if m.label.startswith("AutoResearch best")][0]
    assert "separate search" in tile.label
    assert "not comparable" in tile.help


def _crispdm_app():
    import streamlit as st

    from app import crispdm, theme
    from segmentation import engine as engine_module

    theme.inject()
    crispdm.RENDERERS[st.session_state["phase"]](engine_module.load())


@pytest.mark.parametrize("phase_index", range(6))
def test_crispdm_phase_renders_from_live_artifacts(app, phase_index):
    """The report lives in a modal, which kept it out of every earlier test."""
    from app import crispdm

    report = AppTest.from_function(_crispdm_app, default_timeout=120)
    report.session_state["phase"] = crispdm.PHASES[phase_index]
    report.run()
    _assert_clean(report)
    body = " ".join(m.value for m in report.markdown if "<style>" not in m.value)
    assert len(body) > 200, "phase rendered no substantive content"


def test_crispdm_evaluation_explains_persona_verification(app):
    from app import crispdm

    report = AppTest.from_function(_crispdm_app, default_timeout=120)
    report.session_state["phase"] = crispdm.PHASES[4]
    report.run()
    body = " ".join(m.value for m in report.markdown)
    assert "satisfies *every* claim" in body


@pytest.mark.parametrize("phase_index", range(6))
def test_crispdm_phase_survives_missing_artifacts(phase_index):
    from app import crispdm

    report = AppTest.from_function(_crispdm_app, default_timeout=120)
    report.session_state["phase"] = crispdm.PHASES[phase_index]
    report.run()
    _assert_clean(report)


def test_unmatched_segment_is_labelled_as_such_in_the_ui(clean_sample, monkeypatch):
    """A segment no persona fits must not be shown as a weak persona match."""
    from segmentation import personas as personas_module

    monkeypatch.setattr(config, "MIN_TRAIN_SAMPLE", 50)
    pipeline.run(k=4, seed=3, frame=clean_sample)
    records = json_load(config.PERSONAS_JSON)
    records[0].update(
        personas_module.describe_unmatched(records[0]["cluster"], records[0]["profile_position"], 0)
    )
    records[0]["weak_match"] = True
    config.PERSONAS_JSON.write_text(__import__("json").dumps(records))

    at = AppTest.from_file(APP, default_timeout=180).run()
    _assert_clean(at)
    captions = [c.value for c in at.caption]
    unmatched_captions = [c for c in captions if "No persona fits this segment" in c]
    assert len(unmatched_captions) == 1, "exactly the one unmatched segment should say so"
    weak = [c for c in captions if "Weak persona match" in c]
    assert len(weak) == sum(1 for r in records if r.get("matched", True) and r["weak_match"]), (
        "the unmatched segment must not also be captioned as a weak persona match"
    )


def json_load(path):
    import json

    return json.loads(path.read_text())
