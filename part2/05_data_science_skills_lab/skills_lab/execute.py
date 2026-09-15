"""F12 - run the analysis behind a catalog skill and return its real result payload.

Routing is total: every skill id maps to a computation. Nothing here returns a canned
"success" message, so a payload on screen always reflects work that actually ran.
"""

from __future__ import annotations

from typing import Any, Mapping

from skills_lab.catalog import SKILLS_BY_ID, View
from skills_lab.stats import two_proportion_test


class UnknownSkillError(KeyError):
    """Raised for a skill id that is not in the catalog."""


class NeedsUserCountsError(ValueError):
    """Raised when a skill needs numbers from the user rather than a dataset.

    Only the A/B skill is in this position: this project ships no stored experiment
    record, so there is nothing to compute until the user supplies visitor and
    conversion counts.
    """


def execute(
    skill_id: str,
    results: Mapping[str, Any],
    ab_inputs: Mapping[str, int] | None = None,
) -> dict:
    """Return the computed payload for ``skill_id``.

    ``results`` holds the already-computed benchmark results keyed by view value.
    ``ab_inputs`` carries the counts currently entered in the A/B calculator; the A/B
    skill is the one skill whose computation depends on user input rather than a dataset.
    """
    if skill_id not in SKILLS_BY_ID:
        raise UnknownSkillError(skill_id)
    skill = SKILLS_BY_ID[skill_id]

    if skill.id == "ab-testing":
        if not ab_inputs:
            raise NeedsUserCountsError(
                "The A/B test has no dataset to run on - it works from counts you enter. "
                "Enter your visitor and conversion counts in the A/B panel below, then "
                "run the test there (or run this skill again afterwards)."
            )
        result = two_proportion_test(**dict(ab_inputs))
        return {
            "skill": skill.name,
            "computation": "Two-proportion z-test on the counts you entered",
            "inputs": dict(ab_inputs),
            "result": result.to_dict(),
        }

    result = results.get(skill.view.value)
    if result is None:
        raise ValueError(
            f"The {skill.workspace_label} benchmark has not been computed yet."
        )
    return {
        "skill": skill.name,
        "computation": skill.workspace_label,
        "result": result.payload(),
    }


def routed_views() -> set[View]:
    return {skill.view for skill in SKILLS_BY_ID.values()}
