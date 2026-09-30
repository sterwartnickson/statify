import pandas as pd
import numpy as np

from scipy import stats


# ============================================================
# STATIFY DIAGNOSTICS ENGINE
# ============================================================


# ============================================================
# BASIC SAMPLE INFORMATION
# ============================================================

def sample_information(series):

    data = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(data) == 0:
        return None

    return {
        "n": len(data),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "min": data.min(),
        "max": data.max()
    }


# ============================================================
# SHAPIRO-WILK NORMALITY
# ============================================================

def normality_test(series):

    data = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(data) < 3:
        return None

    # Shapiro-Wilk becomes less practical with extremely
    # large samples, so use a random sample when necessary.
    if len(data) > 5000:

        test_data = data.sample(
            5000,
            random_state=42
        )

    else:

        test_data = data

    statistic, p_value = stats.shapiro(
        test_data
    )

    if p_value < 0.05:

        interpretation = (
            "The Shapiro-Wilk test provides evidence "
            "against normality."
        )

    else:

        interpretation = (
            "The Shapiro-Wilk test does not provide "
            "strong evidence against normality."
        )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "n": len(test_data),
        "Interpretation": interpretation
    }


# ============================================================
# IQR OUTLIER CHECK
# ============================================================

def outlier_diagnostics(series):

    data = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(data) < 4:
        return None

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
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Outlier Count": len(outliers),
        "Outlier Rate": len(outliers) / len(data)
    }


# ============================================================
# LEVENE'S TEST
# ============================================================

def variance_homogeneity(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = [
        values["outcome"].values
        for _, values in data.groupby("group")
    ]

    if len(groups) < 2:
        return None

    if any(len(g) < 2 for g in groups):
        return None

    statistic, p_value = stats.levene(
        *groups,
        center="median"
    )

    if p_value < 0.05:

        interpretation = (
            "There is evidence that the group variances "
            "are not equal."
        )

    else:

        interpretation = (
            "There is insufficient evidence of unequal "
            "group variances."
        )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "Interpretation": interpretation
    }


# ============================================================
# GROUP NORMALITY
# ============================================================

def group_normality(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    results = []

    for group_name, group_data in data.groupby("group"):

        values = group_data["outcome"]

        result = normality_test(
            values
        )

        if result is not None:

            results.append({
                "Group": group_name,
                "n": result["n"],
                "Statistic": result["Statistic"],
                "p_value": result["p_value"],
                "Interpretation": result["Interpretation"]
            })

    return pd.DataFrame(
        results
    )


# ============================================================
# TWO-GROUP DIAGNOSTIC
# ============================================================

def diagnose_two_group_comparison(
    outcome,
    group
):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) != 2:

        return {
            "valid": False,
            "message": (
                "The grouping variable must contain "
                "exactly two groups."
            )
        }

    group1 = data[
        data["group"] == groups[0]
    ]["outcome"]

    group2 = data[
        data["group"] == groups[1]
    ]["outcome"]

    normality = group_normality(
        data["outcome"],
        data["group"]
    )

    variance = variance_homogeneity(
        data["outcome"],
        data["group"]
    )

    outliers1 = outlier_diagnostics(
        group1
    )

    outliers2 = outlier_diagnostics(
        group2
    )

    recommendations = []

    # --------------------------------------------------------
    # VARIANCE
    # --------------------------------------------------------

    if variance is not None:

        if variance["p_value"] < 0.05:

            recommendations.append(
                "Consider Welch's t-test because "
                "the evidence suggests unequal variances."
            )

        else:

            recommendations.append(
                "The equal-variance assumption is not "
                "strongly contradicted by Levene's test."
            )

    # --------------------------------------------------------
    # NORMALITY
    # --------------------------------------------------------

    normality_flags = 0

    if not normality.empty:

        normality_flags = (
            normality["p_value"] < 0.05
        ).sum()

    if normality_flags > 0:

        recommendations.append(
            "At least one group shows evidence against "
            "normality. Inspect the Q-Q plots and sample "
            "sizes before deciding whether a parametric "
            "test remains appropriate."
        )

    else:

        recommendations.append(
            "The normality tests do not provide strong "
            "evidence against normality in the groups."
        )

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    outlier_count = 0

    if outliers1 is not None:
        outlier_count += outliers1["Outlier Count"]

    if outliers2 is not None:
        outlier_count += outliers2["Outlier Count"]

    if outlier_count > 0:

        recommendations.append(
            f"{outlier_count} potential outlier(s) were "
            "detected. Investigate whether they are genuine "
            "observations, data-entry errors, or influential values."
        )

    else:

        recommendations.append(
            "No potential outliers were detected using "
            "the 1.5 × IQR rule."
        )

    return {
        "valid": True,
        "groups": groups,
        "group_sizes": {
            str(groups[0]): len(group1),
            str(groups[1]): len(group2)
        },
        "normality": normality,
        "variance": variance,
        "outliers": {
            str(groups[0]): outliers1,
            str(groups[1]): outliers2
        },
        "recommendations": recommendations
    }


# ============================================================
# MULTI-GROUP DIAGNOSTIC
# ============================================================

def diagnose_multi_group_comparison(
    outcome,
    group
):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) < 3:

        return {
            "valid": False,
            "message": (
                "At least three groups are required "
                "for this diagnostic."
            )
        }

    normality = group_normality(
        data["outcome"],
        data["group"]
    )

    variance = variance_homogeneity(
        data["outcome"],
        data["group"]
    )

    outlier_results = {}

    for group_name, group_data in data.groupby("group"):

        outlier_results[str(group_name)] = (
            outlier_diagnostics(
                group_data["outcome"]
            )
        )

    recommendations = []

    # --------------------------------------------------------
    # VARIANCE
    # --------------------------------------------------------

    if variance is not None:

        if variance["p_value"] < 0.05:

            recommendations.append(
                "There is evidence of unequal group variances. "
                "A robust alternative to ordinary ANOVA should "
                "be considered."
            )

        else:

            recommendations.append(
                "Levene's test does not provide strong evidence "
                "of unequal variances."
            )

    # --------------------------------------------------------
    # NORMALITY
    # --------------------------------------------------------

    if not normality.empty:

        abnormal_groups = normality[
            normality["p_value"] < 0.05
        ]

        if len(abnormal_groups) > 0:

            recommendations.append(
                f"{len(abnormal_groups)} group(s) show evidence "
                "against normality. Inspect Q-Q plots and sample "
                "sizes before rejecting ANOVA."
            )

        else:

            recommendations.append(
                "The group-level normality tests do not provide "
                "strong evidence against normality."
            )

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    total_outliers = 0

    for result in outlier_results.values():

        if result is not None:

            total_outliers += result["Outlier Count"]

    if total_outliers > 0:

        recommendations.append(
            f"{total_outliers} potential outlier(s) were detected "
            "across the groups. Investigate their validity and "
            "possible influence."
        )

    else:

        recommendations.append(
            "No potential outliers were detected using "
            "the 1.5 × IQR rule."
        )

    return {
        "valid": True,
        "groups": groups,
        "normality": normality,
        "variance": variance,
        "outliers": outlier_results,
        "recommendations": recommendations
    }


# ============================================================
# CHI-SQUARE EXPECTED FREQUENCY DIAGNOSTIC
# ============================================================

def chi_square_diagnostics(
    variable1,
    variable2
):

    data = pd.DataFrame({
        "variable1": variable1,
        "variable2": variable2
    }).dropna()

    table = pd.crosstab(
        data["variable1"],
        data["variable2"]
    )

    if (
        table.shape[0] < 2
        or table.shape[1] < 2
    ):
        return None

    chi2, p_value, dof, expected = (
        stats.chi2_contingency(
            table
        )
    )

    expected_df = pd.DataFrame(
        expected,
        index=table.index,
        columns=table.columns
    )

    total_cells = expected.size

    small_cells = (
        expected < 5
    ).sum()

    smallest_expected = expected.min()

    small_cell_percentage = (
        small_cells /
        total_cells
    )

    if (
        smallest_expected < 1
        or small_cell_percentage > 0.20
    ):

        recommendation = (
            "The expected-frequency conditions for the "
            "ordinary chi-square approximation may be problematic. "
            "For a 2 × 2 table, Fisher's exact test should be "
            "considered."
        )

    else:

        recommendation = (
            "The expected frequencies appear adequate for "
            "the ordinary chi-square approximation."
        )

    return {
        "Observed": table,
        "Expected": expected_df,
        "Chi-square": chi2,
        "p_value": p_value,
        "Degrees of Freedom": dof,
        "Small Expected Cells": small_cells,
        "Small Cell Percentage": small_cell_percentage,
        "Minimum Expected Frequency": smallest_expected,
        "Recommendation": recommendation
    }


# ============================================================
# DIAGNOSTIC SUMMARY
# ============================================================

def diagnostic_status(
    p_value,
    alpha=0.05
):

    if p_value is None:

        return "Unable to assess"

    if p_value < alpha:

        return "⚠️ Evidence of a potential assumption issue"

    return "✓ No strong evidence of an issue"