import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# STATIFY RESULTS INTELLIGENCE ENGINE
# ============================================================


# ============================================================
# P-VALUE INTERPRETATION
# ============================================================

def p_value_decision(p_value, alpha=0.05):

    if p_value < alpha:

        return {
            "decision": "Reject the null hypothesis",
            "significant": True,
            "short": "Statistically significant",
            "explanation": (
                f"The p-value ({p_value:.4f}) is smaller "
                f"than the significance level α = {alpha:.2f}. "
                "There is sufficient statistical evidence "
                "to reject the null hypothesis."
            )
        }

    return {
        "decision": "Fail to reject the null hypothesis",
        "significant": False,
        "short": "Not statistically significant",
        "explanation": (
            f"The p-value ({p_value:.4f}) is greater than "
            f"or equal to the significance level α = {alpha:.2f}. "
            "There is insufficient statistical evidence "
            "to reject the null hypothesis."
        )
    }


# ============================================================
# COHEN'S D
# ============================================================

def cohens_d(group1, group2):

    group1 = pd.Series(group1).dropna()
    group2 = pd.Series(group2).dropna()

    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        return None

    pooled_sd = np.sqrt(
        (
            (n1 - 1) * group1.var(ddof=1)
            +
            (n2 - 1) * group2.var(ddof=1)
        )
        /
        (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return None

    d = (
        group1.mean() -
        group2.mean()
    ) / pooled_sd

    return d


def interpret_cohens_d(d):

    if d is None:
        return "Effect size could not be calculated."

    magnitude = abs(d)

    if magnitude < 0.2:

        interpretation = "negligible"

    elif magnitude < 0.5:

        interpretation = "small"

    elif magnitude < 0.8:

        interpretation = "moderate"

    else:

        interpretation = "large"

    return (
        f"Cohen's d = {d:.3f}, indicating a "
        f"{interpretation} standardized difference."
    )


# ============================================================
# MEAN DIFFERENCE + CONFIDENCE INTERVAL
# ============================================================

def mean_difference_ci(
    group1,
    group2,
    confidence=0.95
):

    group1 = pd.Series(group1).dropna()
    group2 = pd.Series(group2).dropna()

    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        return None

    mean1 = group1.mean()
    mean2 = group2.mean()

    difference = mean1 - mean2

    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)

    # Welch standard error
    standard_error = np.sqrt(
        var1 / n1 +
        var2 / n2
    )

    if standard_error == 0:
        return None

    # Welch-Satterthwaite degrees of freedom
    numerator = (
        var1 / n1 +
        var2 / n2
    ) ** 2

    denominator = (
        ((var1 / n1) ** 2) / (n1 - 1)
        +
        ((var2 / n2) ** 2) / (n2 - 1)
    )

    df = numerator / denominator

    alpha = 1 - confidence

    critical = stats.t.ppf(
        1 - alpha / 2,
        df
    )

    margin = critical * standard_error

    lower = difference - margin
    upper = difference + margin

    return {
        "difference": difference,
        "lower": lower,
        "upper": upper,
        "standard_error": standard_error,
        "degrees_of_freedom": df,
        "confidence": confidence
    }


# ============================================================
# MEAN DIFFERENCE INTERPRETATION
# ============================================================

def interpret_mean_difference(
    group1_name,
    group2_name,
    mean1,
    mean2,
    ci
):

    difference = mean1 - mean2

    if difference > 0:

        direction = (
            f"{group1_name} had a higher mean than "
            f"{group2_name}"
        )

    elif difference < 0:

        direction = (
            f"{group1_name} had a lower mean than "
            f"{group2_name}"
        )

    else:

        direction = (
            "The two groups had identical sample means"
        )

    if ci is None:

        return direction

    if ci["lower"] <= 0 <= ci["upper"]:

        precision = (
            "The confidence interval includes zero, "
            "so a zero population mean difference remains "
            "plausible."
        )

    else:

        precision = (
            "The confidence interval does not include zero, "
            "supporting evidence of a population mean difference."
        )

    return (
        f"{direction}. "
        f"The estimated mean difference was "
        f"{difference:.3f}. {precision}"
    )


# ============================================================
# CORRELATION INTERPRETATION
# ============================================================

def interpret_correlation(r):

    if r is None:
        return "Correlation could not be interpreted."

    magnitude = abs(r)

    if magnitude < 0.10:
        strength = "negligible"

    elif magnitude < 0.30:
        strength = "weak"

    elif magnitude < 0.50:
        strength = "moderate"

    elif magnitude < 0.70:
        strength = "strong"

    else:
        strength = "very strong"

    if r > 0:
        direction = "positive"

    elif r < 0:
        direction = "negative"

    else:
        direction = "no"

    if r == 0:

        return (
            "The observed correlation is zero."
        )

    return (
        f"The correlation is {strength} and {direction} "
        f"(r = {r:.3f})."
    )


# ============================================================
# R-SQUARED
# ============================================================

def correlation_r_squared(r):

    if r is None:
        return None

    return r ** 2


# ============================================================
# R-SQUARED INTERPRETATION
# ============================================================

def interpret_r_squared(r_squared):

    if r_squared is None:
        return None

    percentage = r_squared * 100

    return (
        f"The correlation corresponds to approximately "
        f"{percentage:.1f}% shared variance between the "
        "two variables."
    )


# ============================================================
# ODDS RATIO INTERPRETATION
# ============================================================

def interpret_odds_ratio(odds_ratio):

    if odds_ratio is None:
        return None

    if odds_ratio > 1:

        percentage = (
            (odds_ratio - 1) * 100
        )

        return (
            f"The estimated odds are approximately "
            f"{percentage:.1f}% higher in the exposed/first "
            "category relative to the reference category."
        )

    elif odds_ratio < 1:

        percentage = (
            (1 - odds_ratio) * 100
        )

        return (
            f"The estimated odds are approximately "
            f"{percentage:.1f}% lower in the exposed/first "
            "category relative to the reference category."
        )

    return (
        "The odds are approximately equal between "
        "the two comparison categories."
    )


# ============================================================
# GENERAL RESEARCH INTERPRETATION
# ============================================================

def research_interpretation(
    significant,
    relationship,
    context=""
):

    if significant:

        return (
            f"There is statistical evidence of {relationship}. "
            f"{context}"
        )

    return (
        f"The analysis did not provide sufficient statistical "
        f"evidence of {relationship}. {context}"
    )


# ============================================================
# PRACTICAL SIGNIFICANCE WARNING
# ============================================================

def practical_significance_warning(
    p_value,
    effect_size=None
):

    if p_value < 0.05 and effect_size is not None:

        if abs(effect_size) < 0.2:

            return (
                "Although the result is statistically significant, "
                "the estimated effect size is very small. "
                "Consider whether the difference is practically "
                "important in the research context."
            )

        if abs(effect_size) >= 0.8:

            return (
                "The result is statistically significant and the "
                "effect size is large, suggesting potentially "
                "meaningful practical importance."
            )

    return None