"""Browser end-to-end checks for behavior that only exists in the browser.

Run with: pytest -m e2e   (requires `python -m playwright install chromium`)
"""

from datetime import date, timedelta

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def row(page, title):
    return page.locator(".task-row", has_text=title)


def add_task(server, text, **extra):
    return server.api("POST", "/api/tasks", {"capture": text, **extra})


def shift(iso, days):
    return (date.fromisoformat(iso) + timedelta(days=days)).isoformat()


def test_quick_capture_preview_and_create(server, open_app):
    page = open_app()
    capture = page.locator("#capture-input")
    submit = page.locator("#capture-submit")

    capture.fill("!high #ops @tomorrow")
    expect(page.locator("#capture-chips")).to_contain_text("Title is empty")
    expect(submit).to_be_disabled()

    capture.fill("Deploy API !high #ops ~1.5h @tomorrow")
    chips = page.locator("#capture-chips")
    expect(chips.locator("[data-kind=priority]")).to_contain_text("High")
    expect(chips.locator("[data-kind=tag]")).to_contain_text("#ops")
    expect(chips.locator("[data-kind=estimate]")).to_contain_text("90 min")
    expect(chips.locator("[data-kind=due]")).to_contain_text(shift(server.today(), 1))
    expect(submit).to_be_enabled()
    capture.press("Enter")

    new_row = page.locator("[data-bucket=tomorrow] .task-row", has_text="Deploy API")
    expect(new_row).to_be_visible()
    expect(new_row.locator(".chip-tag")).to_have_text("#ops")
    expect(capture).to_have_value("")
    task = server.api("GET", "/api/workspace")["tasks"][0]
    assert (task["title"], task["priority"], task["estimated_minutes"]) == ("Deploy API", "high", 90)


def test_board_drag_changes_status(server, open_app):
    task = add_task(server, "Drag me")
    page = open_app()
    page.locator(".view-tab[data-view=board]").click()
    card = page.locator(".column[data-status=todo] .card", has_text="Drag me")
    expect(card).to_be_visible()
    card.drag_to(page.locator(".column[data-status=in_progress]"))
    expect(page.locator(".column[data-status=in_progress] .card", has_text="Drag me")).to_be_visible()
    assert server.api("GET", f"/api/tasks/{task['id']}")["status"] == "in_progress"

    page.locator(".column[data-status=in_progress] .card", has_text="Drag me").drag_to(
        page.locator(".column[data-status=completed]"))
    expect(page.locator(".column[data-status=completed] .card", has_text="Drag me")).to_be_visible()
    done = server.api("GET", f"/api/tasks/{task['id']}")
    assert done["status"] == "completed" and done["completed_at"]
    page.wait_for_function("window.__feedback.includes('complete')")


def test_board_quick_add_creates_in_column(server, open_app):
    page = open_app()
    page.locator(".view-tab[data-view=board]").click()
    page.locator(".column[data-status=review] .quick-add").fill("Needs review")
    page.locator(".column[data-status=review] .quick-add").press("Enter")
    expect(page.locator(".column[data-status=review] .card", has_text="Needs review")).to_be_visible()


def test_matrix_drop_places_task_in_quadrant(server, open_app):
    today = server.today()
    task = add_task(server, "Overdue chore")
    server.api("PATCH", f"/api/tasks/{task['id']}", {"due_date": shift(today, -2), "priority": "medium"})
    page = open_app()
    page.locator(".view-tab[data-view=matrix]").click()
    card = page.locator("[data-quadrant=delegate] .card", has_text="Overdue chore")
    expect(card).to_be_visible()

    card.drag_to(page.locator("[data-quadrant=schedule]"))
    expect(page.locator("[data-quadrant=schedule] .card", has_text="Overdue chore")).to_be_visible()
    moved = server.api("GET", f"/api/tasks/{task['id']}")
    assert (moved["priority"], moved["due_date"]) == ("high", shift(today, 1))

    page.locator("[data-quadrant=schedule] .card", has_text="Overdue chore").drag_to(page.locator("[data-quadrant=do_first]"))
    expect(page.locator("[data-quadrant=do_first] .card", has_text="Overdue chore")).to_be_visible()
    page.locator("[data-quadrant=do_first] .card", has_text="Overdue chore").drag_to(page.locator("[data-quadrant=eliminate]"))
    expect(page.locator("[data-quadrant=eliminate] .card", has_text="Overdue chore")).to_be_visible()
    moved = server.api("GET", f"/api/tasks/{task['id']}")
    assert (moved["priority"], moved["due_date"]) == ("low", None)


def test_keyboard_shortcuts(server, open_app):
    page = open_app()
    page.locator("body").click()
    page.keyboard.press("Control+k")
    expect(page.locator(".palette-overlay")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator(".palette-overlay")).to_have_count(0)

    page.keyboard.press("?")
    expect(page.locator(".shortcuts-overlay")).to_be_visible()
    expect(page.locator(".shortcuts-overlay")).to_contain_text("~1.5h")
    page.keyboard.press("Escape")
    expect(page.locator(".shortcuts-overlay")).to_have_count(0)

    page.locator("body").click()
    page.keyboard.press("/")
    expect(page.locator("#search-input")).to_be_focused()
    page.keyboard.type("a?b/")
    expect(page.locator("#search-input")).to_have_value("a?b/")
    expect(page.locator(".shortcuts-overlay")).to_have_count(0)

    page.locator("#capture-input").click()
    page.keyboard.type("What? /really")
    expect(page.locator("#capture-input")).to_have_value("What? /really")
    expect(page.locator(".shortcuts-overlay")).to_have_count(0)


def test_command_palette_navigation(server, open_app):
    add_task(server, "Quarterly report")
    page = open_app()
    page.keyboard.press("Control+k")
    palette_input = page.locator(".palette-input")
    palette_input.fill("accent")
    items = page.locator(".palette-item")
    expect(items).to_have_count(5)
    expect(items.nth(0)).to_have_class("palette-item highlighted")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    expect(items.nth(2)).to_have_class("palette-item highlighted")
    page.keyboard.press("ArrowUp")
    expect(items.nth(1)).to_have_class("palette-item highlighted")
    page.keyboard.press("Enter")
    expect(page.locator("html")).to_have_attribute("data-accent", "cyan")

    page.keyboard.press("Control+k")
    page.locator(".palette-input").fill("zzzz-nothing")
    expect(page.locator(".palette-empty")).to_have_text("No results")
    page.locator(".palette-input").fill("calendar")
    page.keyboard.press("Enter")
    expect(page.locator(".view-tab[data-view=calendar]")).to_have_class("view-tab active")

    page.keyboard.press("Control+k")
    page.locator(".palette-input").fill("quarterly")
    page.keyboard.press("Enter")
    expect(page.locator(".editor-overlay .title-input")).to_have_value("Quarterly report")

    page.reload()
    expect(page.locator("html")).to_have_attribute("data-accent", "cyan")  # persisted per browser


def test_multi_tab_sync(server, open_app):
    first = open_app()
    second = open_app()
    first.locator("#capture-input").fill("Synced task #shared")
    first.locator("#capture-input").press("Enter")
    expect(row(second, "Synced task")).to_be_visible()

    row(second, "Synced task").locator(".complete-check").click()
    expect(row(first, "Synced task").locator(".complete-check")).to_have_attribute("aria-checked", "true")

    # A change made outside any browser also reaches both tabs.
    task = server.api("GET", "/api/workspace")["tasks"][0]
    server.api("PATCH", f"/api/tasks/{task['id']}", {"title": "Renamed elsewhere"})
    expect(row(first, "Renamed elsewhere")).to_be_visible()
    expect(row(second, "Renamed elsewhere")).to_be_visible()


def test_trash_restore_and_purge(server, open_app):
    task = add_task(server, "Disposable")
    page = open_app()
    target = row(page, "Disposable")
    target.hover()
    target.locator("button[aria-label='Move to trash']").click()
    expect(row(page, "Disposable")).to_have_count(0)
    expect(page.locator("[data-nav=trash] .count")).to_have_text("1")

    page.locator("[data-nav=trash]").click()
    expect(row(page, "Disposable")).to_be_visible()
    row(page, "Disposable").hover()
    row(page, "Disposable").locator("button[aria-label='Restore']").click()
    expect(row(page, "Disposable")).to_have_count(0)
    page.locator("[data-nav=smart-all]").click()
    expect(row(page, "Disposable")).to_be_visible()

    row(page, "Disposable").hover()
    row(page, "Disposable").locator("button[aria-label='Move to trash']").click()
    page.locator("[data-nav=trash]").click()
    row(page, "Disposable").hover()
    row(page, "Disposable").locator("button[aria-label='Delete forever']").click()  # confirm auto-accepted
    expect(page.locator(".empty-state")).to_contain_text("Trash is empty")
    with pytest.raises(Exception):
        server.api("GET", f"/api/tasks/{task['id']}")


def test_optimistic_completion_reconciles_with_server(server, open_app):
    add_task(server, "Optimistic")
    page = open_app()
    check = row(page, "Optimistic").locator(".complete-check")

    pending = []
    page.route("**/api/tasks/*", lambda route: pending.append(route) if route.request.method == "PATCH" else route.continue_())
    check.click()
    # The checkbox flips before the server has replied (the PATCH is still held).
    expect(check).to_have_attribute("aria-checked", "true")
    page.wait_for_function("window.__feedback.includes('complete')")
    assert pending and server.api("GET", "/api/workspace")["tasks"][0]["status"] == "todo"
    pending[0].continue_()
    expect(page.locator("#sync-status.live")).to_be_visible()
    page.wait_for_timeout(300)
    expect(row(page, "Optimistic").locator(".complete-check")).to_have_attribute("aria-checked", "true")
    assert server.api("GET", "/api/workspace")["tasks"][0]["status"] == "completed"
    page.unroute("**/api/tasks/*")

    # Failure: the optimistic change is rolled back and an error is shown.
    page.route("**/api/tasks/*", lambda route: route.fulfill(status=500, json={"error": "boom"})
               if route.request.method == "PATCH" else route.continue_())
    row(page, "Optimistic").locator(".complete-check").click()
    expect(page.locator(".toast.error")).to_contain_text("boom")
    expect(row(page, "Optimistic").locator(".complete-check")).to_have_attribute("aria-checked", "true")
    assert server.api("GET", "/api/workspace")["tasks"][0]["status"] == "completed"


def test_completion_feedback_and_mute(server, open_app):
    add_task(server, "Celebrate")
    add_task(server, "Quiet")
    page = open_app()
    row(page, "Celebrate").locator(".complete-check").click()
    page.wait_for_function("window.__feedback.includes('complete')")
    expect(page.locator(".confetti-canvas")).to_have_count(1)

    row(page, "Celebrate").locator(".complete-check").click()
    page.wait_for_function("window.__feedback.includes('uncomplete')")

    page.locator("#mute-toggle").click()
    expect(page.locator("#mute-toggle")).to_have_attribute("aria-pressed", "true")
    before = page.evaluate("window.__feedback.length")
    row(page, "Quiet").locator(".complete-check").click()
    expect(row(page, "Quiet").locator(".complete-check")).to_have_attribute("aria-checked", "true")
    page.wait_for_timeout(300)
    assert page.evaluate("window.__feedback.length") == before
    page.reload()
    expect(page.locator("#mute-toggle")).to_have_attribute("aria-pressed", "true")  # persisted


def test_pomodoro_logs_actual_focus_time(server, open_app):
    task = add_task(server, "Deep work")
    page = open_app(install_clock=True)
    page.locator("#pomodoro-button").click()
    dialog = page.locator(".pomodoro-overlay")
    expect(dialog.locator(".timer-display")).to_have_text("25:00")
    select = dialog.locator("select")
    expect(select.locator("option", has_text="Deep work")).to_have_count(1)
    select.select_option(str(task["id"]))

    # Under one focused minute logs nothing.
    dialog.get_by_role("button", name="Start").click()
    page.clock.run_for("00:45")
    dialog.get_by_role("button", name="Reset").click()
    page.wait_for_timeout(300)
    assert server.api("GET", f"/api/tasks/{task['id']}")["time_spent_minutes"] == 0

    # Partial session: 10 focused minutes, then reset.
    dialog.get_by_role("button", name="Start").click()
    page.clock.run_for("10:00")
    expect(dialog.locator(".timer-display")).to_have_text("15:00")
    dialog.get_by_role("button", name="Reset").click()
    expect(dialog.locator(".timer-message")).to_contain_text("Logged 10 focused minutes")
    assert server.api("GET", f"/api/tasks/{task['id']}")["time_spent_minutes"] == 10

    # Paused time is not counted.
    dialog.get_by_role("button", name="Start").click()
    page.clock.run_for("05:00")
    dialog.get_by_role("button", name="Pause").click()
    page.clock.run_for("30:00")
    expect(dialog.locator(".timer-display")).to_have_text("20:00")

    # Full session completion.
    dialog.get_by_role("button", name="Resume").click()
    page.clock.run_for("20:01")
    expect(dialog.locator(".timer-message")).to_contain_text("Focus session complete")
    expect(dialog).to_contain_text("Focus sessions completed: 1")
    page.wait_for_function("window.__feedback.includes('fanfare')")
    assert server.api("GET", f"/api/tasks/{task['id']}")["time_spent_minutes"] == 35

    # Switching mode resets and stops the countdown; breaks log nothing.
    dialog.get_by_role("button", name="Short Break 5m").click()
    expect(dialog.locator(".timer-display")).to_have_text("05:00")
    dialog.get_by_role("button", name="Start").click()
    page.clock.run_for("05:01")
    expect(dialog.locator(".timer-message")).to_contain_text("Break over")
    assert server.api("GET", f"/api/tasks/{task['id']}")["time_spent_minutes"] == 35


def test_calendar_click_to_schedule_uses_clicked_day(server, open_app):
    today = server.today()
    page = open_app()
    page.locator(".view-tab[data-view=calendar]").click()
    target_day = shift(today, 2)
    if target_day[:7] != today[:7]:
        target_day = today
    cell = page.locator(f".day[data-date='{target_day}']")
    cell.click(position={"x": 5, "y": 60})
    composer = page.locator(".day-composer-input")
    composer.fill("Dentist !high")
    composer.press("Enter")
    expect(cell.locator(".cal-chip", has_text="Dentist")).to_be_visible()

    cell.click(position={"x": 5, "y": 60})
    page.locator(".day-composer-input").fill("Override @tomorrow")
    page.locator(".day-composer-input").press("Enter")
    tasks = {t["title"]: t for t in server.api("GET", "/api/workspace")["tasks"]}
    assert tasks["Dentist"]["due_date"] == target_day and tasks["Dentist"]["priority"] == "high"
    assert tasks["Override"]["due_date"] == shift(today, 1)


def test_detail_editor_saves_subtasks_and_tags(server, open_app):
    server.api("POST", "/api/tags", {"name": "alpha"})
    add_task(server, "Editable")
    page = open_app()
    row(page, "Editable").click()
    editor = page.locator(".editor-overlay")
    editor.locator(".title-input").fill("Edited title")
    editor.get_by_label("New subtask").fill("First step")
    editor.get_by_label("New subtask").press("Enter")
    editor.get_by_label("New subtask").fill("Second step")
    editor.get_by_label("New subtask").press("Enter")
    editor.get_by_label("Toggle First step").check()
    expect(editor).to_contain_text("1/2 done")
    editor.locator(".tag-picker .chip", has_text="#alpha").click()
    editor.get_by_role("button", name="Save").click()
    expect(page.locator(".editor-overlay")).to_have_count(0)
    edited = row(page, "Edited title")
    expect(edited.locator(".chip-subtasks")).to_contain_text("1/2")
    expect(edited.locator(".chip-tag")).to_have_text("#alpha")

    row(page, "Edited title").click()
    page.keyboard.press("Escape")
    expect(page.locator(".editor-overlay")).to_have_count(0)


def test_analytics_empty_store_shows_no_score(server, open_app):
    page = open_app()
    page.locator(".view-tab[data-view=analytics]").click()
    expect(page.locator(".metric").nth(2)).to_contain_text("—")
    expect(page.locator(".metric").nth(2)).not_to_contain_text("/100")
    add_task(server, "First")
    expect(page.locator(".metric").nth(2)).to_contain_text("/100")


def test_analytics_view_renders_server_telemetry(server, open_app):
    t = add_task(server, "Done one")
    server.api("PATCH", f"/api/tasks/{t['id']}", {"status": "completed"})
    add_task(server, "Open one")
    page = open_app()
    page.locator(".view-tab[data-view=analytics]").click()
    expect(page.locator(".metric").first).to_contain_text("50%")
    expect(page.locator(".heatmap .heat")).to_have_count(30)
    expect(page.locator(".bar-row[data-priority=medium]")).to_contain_text("1/2 · 50%")


def test_bulk_actions_and_sidebar_collapse(server, open_app):
    for name in ("Bulk A", "Bulk B", "Bulk C"):
        add_task(server, name)
    page = open_app()
    page.locator(".select-all input").check()
    expect(page.locator("#bulk-bar")).to_contain_text("3 selected")
    page.get_by_label("Set priority for selected").select_option("urgent")
    expect(page.locator(".task-row .chip-priority", has_text="Urgent")).to_have_count(3)
    page.get_by_role("button", name="Complete", exact=True).click()
    expect(page.locator("[data-bucket=completed] .task-row")).to_have_count(3)
    page.get_by_role("button", name="Trash", exact=True).click()
    expect(page.locator(".empty-state")).to_be_visible()
    expect(page.locator("#bulk-bar")).to_be_hidden()

    page.locator("#sidebar-toggle").click()
    expect(page.locator("#app")).to_have_class("app sidebar-collapsed")
    expect(page.locator(".tags-section")).to_be_hidden()
    page.locator("#sidebar-toggle").click()
    expect(page.locator(".tags-section")).to_be_visible()
