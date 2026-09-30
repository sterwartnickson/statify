import pandas as pd
import numpy as np

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from scipy import stats


# ============================================================
# STATIFY POST-HOC ENGINE
# ============================================================


# ============================================================
# TUKEY HSD
# ============================================================

def tukey_hsd(
    data,
    outcome,
    group
):

    analysis_data = data[
        [outcome, group]
    ].dropna()

    if analysis_data.empty:
        return None

    if analysis_data[group].nunique() < 3:
        return None

    result = pairwise_tukeyhsd(
        endog=analysis_data[outcome],
        groups=analysis_data[group],
        alpha=0.05
    )

    table = pd.DataFrame(
        data=result._results_table.data[1:],
        columns=result._results_table.data[0]
    )

    return {
        "method": "Tukey HSD",
        "table": table,
        "raw_result": result
    }


# ============================================================
# BONFERRONI PAIRWISE COMPARISONS
# ============================================================

def bonferroni_pairwise(
    data,
    outcome,
    group
):

    analysis_data = data[
        [outcome, group]
    ].dropna()

    groups = (
        analysis_data[group]
        .dropna()
        .unique()
    )

    if len(groups) < 3:
        return None

    results = []

    for i in range(len(groups)):

        for j in range(i + 1, len(groups)):

            group1 = analysis_data[
                analysis_data[group] == groups[i]
            ][outcome]

            group2 = analysis_data[
                analysis_data[group] == groups[j]
            ][outcome]

            result = stats.ttest_ind(
                group1,
                group2,
                equal_var=False
            )

            mean_difference = (
                group1.mean() -
                group2.mean()
            )

            results.append({
                "Group 1": groups[i],
                "Group 2": groups[j],
                "Mean Difference": mean_difference,
                "Raw p-value": result.pvalue
            })

    results_df = pd.DataFrame(
        results
    )

    rejected, corrected_p, _, _ = (
        multipletests(
            results_df["Raw p-value"],
            method="bonferroni"
        )
    )

    results_df["Adjusted p-value"] = corrected_p
    results_df["Significant"] = rejected

    return results_df


# ============================================================
# EFFECT SIZE — ETA SQUARED
# ============================================================

def eta_squared(
    anova_table
):

    if anova_table is None:
        return None

    # Find residual/error row
    residual_candidates = [
        index
        for index in anova_table.index
        if str(index).lower() in [
            "residual",
            "resid",
            "error"
        ]
    ]

    if not residual_candidates:
        return None

    residual = residual_candidates[0]

    ss_error = anova_table.loc[
        residual,
        "sum_sq"
    ]

    total_ss = (
        anova_table["sum_sq"]
        .sum()
    )

    if total_ss == 0:
        return None

    return 1 - (
        ss_error /
        total_ss
    )


# ============================================================
# EFFECT SIZE INTERPRETATION
# ============================================================

def interpret_eta_squared(
    eta
):

    if eta is None:
        return "Eta squared could not be calculated."

    if eta < 0.01:

        magnitude = "negligible"

    elif eta < 0.06:

        magnitude = "small"

    elif eta < 0.14:

        magnitude = "moderate"

    else:

        magnitude = "large"

    return (
        f"η² = {eta:.3f}, indicating a "
        f"{magnitude} proportion of variance "
        "associated with the factor."
    )


# ============================================================
# ANOVA DECISION
# ============================================================

def anova_posthoc_guidance(
    p_value,
    alpha=0.05
):

    if p_value < alpha:

        return {
            "significant": True,
            "message": (
                "The omnibus ANOVA is statistically significant. "
                "At least one group mean differs from another. "
                "Post-hoc comparisons are appropriate to identify "
                "which groups differ."
            )
        }

    return {
        "significant": False,
        "message": (
            "The omnibus ANOVA is not statistically significant. "
            "There is insufficient evidence that the group means "
            "differ overall. Routine post-hoc testing is therefore "
            "not generally necessary."
        )
    }


# ============================================================
# POST-HOC RESEARCH INTERPRETATION
# ============================================================

def interpret_tukey_results(
    tukey_table
):

    if tukey_table is None:
        return []

    interpretations = []

    for _, row in tukey_table.iterrows():

        group1 = row["group1"]
        group2 = row["group2"]

        difference = float(
            row["meandiff"]
        )

        p_adjusted = float(
            row["p-adj"]
        )

        reject = row["reject"]

        if reject:

            if difference > 0:

                direction = (
                    f"{group1} had a higher mean than "
                    f"{group2}"
                )

            else:

                direction = (
                    f"{group1} had a lower mean than "
                    f"{group2}"
                )

            interpretations.append(
                f"{direction}. The adjusted p-value was "
                f"{p_adjusted:.4f}, indicating a statistically "
                "significant pairwise difference."
            )

        else:

            interpretations.append(
                f"No statistically significant difference "
                f"was detected between {group1} and {group2} "
                f"(adjusted p = {p_adjusted:.4f})."
            )

    return interpretations


# ============================================================
# SIMPLE GROUP SUMMARY
# ============================================================

def group_descriptive_summary(
    data,
    outcome,
    group
):

    summary = (
        data
        .groupby(group)[outcome]
        .agg(
            Count="count",
            Mean="mean",
            SD="std",
            Median="median"
        )
        .reset_index()
    )

    return summary
