from datetime import date

import pytest

from zenith.capture import parse_capture, resolve_due_token

TODAY = date(2026, 9, 16)  # Wednesday


def parse(text):
    return parse_capture(text, TODAY)


@pytest.mark.parametrize("word,expected", [
    ("urgent", "urgent"), ("crit", "urgent"), ("critical", "urgent"), ("u", "urgent"),
    ("high", "high"), ("h", "high"), ("important", "high"),
    ("medium", "medium"), ("med", "medium"), ("m", "medium"), ("normal", "medium"),
    ("low", "low"), ("l", "low"), ("HIGH", "high"),
])
def test_priority_synonyms(word, expected):
    result = parse(f"Do thing !{word}")
    assert result.priority == expected
    assert result.title == "Do thing"


def test_unknown_priority_defaults_to_medium_and_is_stripped():
    result = parse("Do thing !whenever")
    assert result.priority == "medium"
    assert result.title == "Do thing"


def test_only_first_priority_token_is_used():
    result = parse("!low Do thing !urgent")
    assert result.priority == "low"
    assert result.title == "Do thing !urgent"


def test_tags_lowercased_deduplicated_and_removed():
    result = parse("Ship #Frontend release #frontend #back_end #ui-kit")
    assert result.tags == ["frontend", "back_end", "ui-kit"]
    assert result.title == "Ship release"


@pytest.mark.parametrize("token,minutes", [
    ("~45", 45), ("~30m", 30), ("~30min", 30), ("~2h", 120), ("~1.5h", 90), ("~1.5hr", 90),
    ("~0.25h", 15), ("~2.5", 3), ("~0.01h", 1),
])
def test_estimate(token, minutes):
    result = parse(f"Write report {token}")
    assert result.estimated_minutes == minutes
    assert result.title == "Write report"


@pytest.mark.parametrize("token,expected", [
    ("today", date(2026, 9, 16)), ("tod", date(2026, 9, 16)),
    ("tomorrow", date(2026, 9, 17)), ("tom", date(2026, 9, 17)),
    ("yesterday", date(2026, 9, 15)), ("nextweek", date(2026, 9, 23)),
    ("2026-12-31", date(2026, 12, 31)),
    ("friday", date(2026, 9, 18)), ("fri", date(2026, 9, 18)),
    ("monday", date(2026, 9, 21)), ("mon", date(2026, 9, 21)),
    ("wed", date(2026, 9, 23)),  # same weekday as today -> next week, not today
    ("Tuesday", date(2026, 9, 22)),
])
def test_due_tokens(token, expected):
    assert resolve_due_token(token, TODAY) == expected
    result = parse(f"Call bank @{token}")
    assert result.due_date == expected
    assert result.title == "Call bank"


@pytest.mark.parametrize("token", ["someday", "2026-02-30", "5pm", "fridayish"])
def test_unrecognized_due_token_yields_no_date_but_is_stripped(token):
    result = parse(f"Call bank @{token} soon")
    assert result.due_date is None
    assert result.title == "Call bank soon"


def test_full_line_whitespace_collapsed():
    result = parse("  Review   PR !high #backend ~30m   @tomorrow  please ")
    assert result.title == "Review PR please"
    assert (result.priority, result.tags, result.estimated_minutes, result.due_date) == (
        "high", ["backend"], 30, date(2026, 9, 17))
    kinds = [c["kind"] for c in result.chips]
    assert sorted(kinds) == ["due", "estimate", "priority", "tag"]


def test_tokens_must_start_a_word():
    result = parse("Email bob@example.com about C#dev and 50!stuff")
    assert result.due_date is None and result.tags == [] and result.priority == "medium"
    assert result.title == "Email bob@example.com about C#dev and 50!stuff"


def test_empty_residual_title():
    assert parse("!high #x @today ~5m").title == ""
    assert parse("").title == ""


def test_tags_scanned_on_original_input():
    # A tag-looking fragment that only appears after stripping another token is not a tag.
    result = parse("Plan @tomorrow#notatag")
    assert result.tags == []
    assert result.title == "Plan"


@pytest.mark.parametrize("text,due,title", [
    ("Call mom @friday.", date(2026, 9, 18), "Call mom ."),
    ("Ship it @tomorrow, then rest", date(2026, 9, 17), "Ship it , then rest"),
    ("Renew passport (@2026-10-01)", None, "Renew passport (@2026-10-01)"),  # token must start a word
    ("Renew passport @2026-10-01)", date(2026, 10, 1), "Renew passport )"),
    ("Pay @someday!", None, "Pay !"),
])
def test_due_token_ignores_trailing_punctuation(text, due, title):
    result = parse(text)
    assert result.due_date == due
    assert result.title == title
