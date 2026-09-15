"""Two-proportion significance testing for the A/B workspace.

The test is the classical pooled two-proportion z-test with a normal approximation.
Everything reported here is computed from the counts the user enters; there is no stored
or illustrative experiment record anywhere in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from math import sqrt

from scipy import stats

from skills_lab.config import ALPHA

ASSUMPTIONS = [
    "Visitors are independent and randomly assigned to exactly one variant.",
    "Sample sizes were fixed in advance; the result is read once, not peeked at repeatedly.",
    "Each visitor contributes at most one conversion (a Bernoulli outcome).",
    "The normal approximation needs roughly 10+ conversions and 10+ non-conversions per arm.",
]


@dataclass
class ABResult:
    control_visitors: int
    control_conversions: int
    treatment_visitors: int
    treatment_conversions: int
    control_rate: float
    treatment_rate: float
    pooled_rate: float
    standard_error: float
    z_score: float
    p_value: float
    absolute_lift_pp: float
    relative_lift_pct: float
    ci_low_pp: float
    ci_high_pp: float
    alpha: float
    significant: bool
    recommendation: str
    rationale: str
    observed_power: float
    power_note: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ABInputError(ValueError):
    """Raised for inputs that cannot form a valid two-proportion test."""


def _validate(visitors: int, conversions: int, label: str) -> None:
    if visitors <= 0:
        raise ABInputError(f"{label} visitors must be greater than zero.")
    if conversions < 0:
        raise ABInputError(f"{label} conversions cannot be negative.")
    if conversions > visitors:
        raise ABInputError(f"{label} conversions cannot exceed {label.lower()} visitors.")


def two_proportion_test(
    control_visitors: int,
    control_conversions: int,
    treatment_visitors: int,
    treatment_conversions: int,
    *,
    alpha: float = ALPHA,
    minimum_practical_effect_pp: float = 0.5,
) -> ABResult:
    """Pooled two-proportion z-test with a Wald confidence interval on the difference.

    ``minimum_practical_effect_pp`` is the smallest difference (in percentage points)
    the user considers worth shipping; it drives the magnitude part of the guidance and
    never changes the statistics.
    """
    _validate(control_visitors, control_conversions, "Control")
    _validate(treatment_visitors, treatment_conversions, "Treatment")

    p_control = control_conversions / control_visitors
    p_treatment = treatment_conversions / treatment_visitors
    pooled = (control_conversions + treatment_conversions) / (
        control_visitors + treatment_visitors
    )

    standard_error = sqrt(
        pooled * (1 - pooled) * (1 / control_visitors + 1 / treatment_visitors)
    )
    if standard_error == 0:
        z_score, p_value = 0.0, 1.0
    else:
        z_score = (p_treatment - p_control) / standard_error
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    absolute_lift_pp = (p_treatment - p_control) * 100
    relative_lift_pct = (p_treatment - p_control) / max(p_control, 1e-12) * 100

    # Unpooled standard error is the right one for an interval around the difference.
    se_diff = sqrt(
        p_control * (1 - p_control) / control_visitors
        + p_treatment * (1 - p_treatment) / treatment_visitors
    )
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_low_pp = absolute_lift_pp - z_crit * se_diff * 100
    ci_high_pp = absolute_lift_pp + z_crit * se_diff * 100

    significant = bool(p_value < alpha)

    # Post-hoc ("observed") power. Descriptive only - see power_note.
    if se_diff > 0:
        effect_z = abs(p_treatment - p_control) / se_diff
        observed_power = float(
            stats.norm.cdf(effect_z - z_crit) + stats.norm.cdf(-effect_z - z_crit)
        )
    else:
        observed_power = 0.0

    recommendation, rationale = _recommend(
        significant=significant,
        absolute_lift_pp=absolute_lift_pp,
        ci_low_pp=ci_low_pp,
        ci_high_pp=ci_high_pp,
        mde_pp=minimum_practical_effect_pp,
    )

    warnings: list[str] = []
    for label, visitors, conversions in (
        ("control", control_visitors, control_conversions),
        ("treatment", treatment_visitors, treatment_conversions),
    ):
        if min(conversions, visitors - conversions) < 10:
            warnings.append(
                f"The normal approximation is unreliable for the {label} arm "
                f"(fewer than 10 conversions or non-conversions)."
            )

    return ABResult(
        control_visitors=control_visitors,
        control_conversions=control_conversions,
        treatment_visitors=treatment_visitors,
        treatment_conversions=treatment_conversions,
        control_rate=p_control,
        treatment_rate=p_treatment,
        pooled_rate=pooled,
        standard_error=standard_error,
        z_score=float(z_score),
        p_value=float(p_value),
        absolute_lift_pp=absolute_lift_pp,
        relative_lift_pct=relative_lift_pct,
        ci_low_pp=ci_low_pp,
        ci_high_pp=ci_high_pp,
        alpha=alpha,
        significant=significant,
        recommendation=recommendation,
        rationale=rationale,
        observed_power=observed_power,
        power_note=(
            "Observed (post-hoc) power is a restatement of the p-value, not independent "
            "evidence. It is shown as context only and takes no part in the decision."
        ),
        warnings=warnings,
    )


def _recommend(
    *,
    significant: bool,
    absolute_lift_pp: float,
    ci_low_pp: float,
    ci_high_pp: float,
    mde_pp: float,
) -> tuple[str, str]:
    """Guidance from effect direction, magnitude and interval width - not p alone."""
    interval = f"95% CI [{ci_low_pp:+.2f}, {ci_high_pp:+.2f}] pp"

    if not significant:
        if ci_low_pp < -mde_pp and ci_high_pp > mde_pp:
            return (
                "Inconclusive - keep control",
                f"The interval spans both a meaningful loss and a meaningful gain "
                f"({interval}). The experiment cannot distinguish them; collect more data.",
            )
        return (
            "No meaningful difference - keep control",
            f"The interval is consistent with no effect and excludes large effects in "
            f"either direction ({interval}). There is no basis for switching.",
        )

    if absolute_lift_pp < 0:
        return (
            "Keep control - treatment is worse",
            f"Treatment converts below control and the interval stays negative ({interval}).",
        )

    if ci_low_pp >= mde_pp:
        return (
            "Ship treatment",
            f"The effect is positive and the whole interval clears the "
            f"{mde_pp:.2f} pp practical threshold ({interval}).",
        )

    return (
        "Positive but possibly too small - hold",
        f"The gain is statistically distinguishable from zero, but the interval reaches "
        f"below the {mde_pp:.2f} pp you called practically meaningful ({interval}). "
        f"Weigh the rollout cost against a possibly marginal gain.",
    )
