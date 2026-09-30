import numpy as np
import pandas as pd

from scipy import stats


# ============================================================
# STATIFY ASSUMPTION INTELLIGENCE ENGINE
# ============================================================


# ============================================================
# NORMALITY — SHAPIRO-WILK
# ============================================================

def shapiro_normality(series):

    data = pd.Series(series).dropna()

    n = len(data)

    if n < 3:
        return {
            "test": "Shapiro-Wilk",
            "available": False,
            "message": "At least 3 observations are required."
        }

    # Shapiro is not ideal for extremely large samples
    if n > 5000:

        sample = data.sample(
            5000,
            random_state=42
        )

    else:

        sample = data

    statistic, p_value = stats.shapiro(
        sample
    )

    return {
        "test": "Shapiro-Wilk",
        "available": True,
        "statistic": statistic,
        "p_value": p_value,
        "n": n,
        "normal": p_value >= 0.05
    }


# ============================================================
# IQR OUTLIER DETECTION
# ============================================================

def iqr_outliers(series):

    data = pd.Series(series).dropna()

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = data[
        (data < lower) |
        (data > upper)
    ]

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "count": len(outliers),
        "percentage": (
            len(outliers) / len(data) * 100
            if len(data) > 0
            else 0
        ),
        "values": outliers
    }


# ============================================================
# LEVENE'S TEST
# ============================================================

def levene_test(groups):

    cleaned_groups = [
        pd.Series(group).dropna()
        for group in groups
    ]

    cleaned_groups = [
        group
        for group in cleaned_groups
        if len(group) > 0
    ]

    if len(cleaned_groups) < 2:

        return {
            "available": False
        }

    statistic, p_value = stats.levene(
        *cleaned_groups,
        center="median"
    )

    return {
        "available": True,
        "test": "Levene's test",
        "statistic": statistic,
        "p_value": p_value,
        "equal_variance": p_value >= 0.05
    }


# ============================================================
# SAMPLE SIZE
# ============================================================

def sample_size_assessment(
    n
):

    if n < 10:

        status = "Very small"

        explanation = (
            "The sample is very small. Statistical tests "
            "may have low power and assumption checks may "
            "be unreliable."
        )

    elif n < 30:

        status = "Small"

        explanation = (
            "The sample is relatively small. Distributional "
            "assumptions deserve particular attention."
        )

    elif n < 100:

        status = "Moderate"

        explanation = (
            "The sample size is moderate and generally "
            "provides more stable estimates than very small "
            "samples."
        )

    else:

        status = "Large"

        explanation = (
            "The sample is relatively large. Statistical "
            "inference may be more robust to moderate "
            "departures from normality."
        )

    return {
        "n": n,
        "status": status,
        "explanation": explanation
    }


# ============================================================
# NORMALITY INTERPRETATION
# ============================================================

def interpret_normality(
    result,
    alpha=0.05
):

    if not result.get("available", False):

        return result["message"]

    p = result["p_value"]

    if p >= alpha:

        return (
            f"The Shapiro-Wilk test did not provide sufficient "
            f"evidence of departure from normality "
            f"(p = {p:.4f})."
        )

    return (
        f"The Shapiro-Wilk test indicates evidence of departure "
        f"from normality (p = {p:.4f}). This does not automatically "
        f"mean that a parametric test is invalid; sample size, "
        f"outliers, skewness, and the robustness of the chosen "
        f"method should also be considered."
    )


# ============================================================
# VARIANCE INTERPRETATION
# ============================================================

def interpret_variance(
    result,
    alpha=0.05
):

    if not result.get("available", False):

        return (
            "Variance equality could not be assessed."
        )

    p = result["p_value"]

    if p >= alpha:

        return (
            f"Levene's test did not provide sufficient evidence "
            f"that the group variances differ "
            f"(p = {p:.4f})."
        )

    return (
        f"Levene's test indicates evidence of unequal group "
        f"variances (p = {p:.4f}). Consider a method that does "
        f"not require equal variances, such as Welch's test, "
        f"where appropriate."
    )


# ============================================================
# ASSUMPTION PROFILE
# ============================================================

def build_assumption_profile(
    series,
    groups=None
):

    data = pd.Series(series).dropna()

    normality = shapiro_normality(
        data
    )

    outliers = iqr_outliers(
        data
    )

    sample = sample_size_assessment(
        len(data)
    )

    variance = None

    if groups is not None:

        variance = levene_test(
            groups
        )

    return {
        "normality": normality,
        "outliers": outliers,
        "sample_size": sample,
        "variance": variance
    }


# ============================================================
# TEST RECOMMENDATION
# ============================================================

def recommend_two_group_test(
    normality_ok,
    variance_equal,
    outlier_problem=False
):

    if normality_ok and variance_equal and not outlier_problem:

        return {
            "test": "Independent samples t-test",
            "reason": (
                "The data do not show major diagnostic problems "
                "for a standard two-group parametric comparison."
            )
        }

    if normality_ok and not variance_equal:

        return {
            "test": "Welch's t-test",
            "reason": (
                "The group variances appear unequal. Welch's "
                "t-test is preferable because it does not require "
                "equal population variances."
            )
        }

    return {
        "test": "Mann-Whitney U",
        "reason": (
            "The diagnostics indicate substantial issues with "
            "the assumptions of the standard t-test. A rank-based "
            "alternative may be more appropriate, depending on "
            "the research question and data structure."
        )
    }


# ============================================================
# ANOVA RECOMMENDATION
# ============================================================

def recommend_anova(
    normality_ok,
    variance_equal,
    outlier_problem=False
):

    if normality_ok and variance_equal and not outlier_problem:

        return {
            "test": "One-Way ANOVA",
            "reason": (
                "The diagnostic profile is broadly compatible "
                "with the assumptions of classical One-Way ANOVA."
            )
        }

    if normality_ok and not variance_equal:

        return {
            "test": "Welch's ANOVA",
            "reason": (
                "The group variances appear unequal. Welch's ANOVA "
                "provides a more appropriate alternative to "
                "classical ANOVA when homogeneity of variance "
                "is questionable."
            )
        }

    return {
        "test": "Kruskal-Wallis",
        "reason": (
            "The data show substantial departures from the "
            "assumptions of classical ANOVA. A rank-based "
            "alternative may be appropriate."
        )
    }


# ============================================================
# ASSUMPTION WARNING
# ============================================================

def assumption_warning(
    normality,
    outliers,
    variance=None
):

    warnings = []

    if normality.get(
        "available",
        False
    ):

        if not normality["normal"]:

            warnings.append(
                "Normality may be questionable."
            )

    if outliers["count"] > 0:

        warnings.append(
            f"{outliers['count']} potential outlier(s) "
            "were detected using the IQR rule."
        )

    if variance is not None:

        if variance.get(
            "available",
            False
        ):

            if not variance["equal_variance"]:

                warnings.append(
                    "Homogeneity of variance may be questionable."
                )

    return warnings