import pandas as pd
import numpy as np

from scipy import stats


# ============================================================
# STATIFY NON-PARAMETRIC ENGINE
# ============================================================


# ============================================================
# MANN-WHITNEY U
# ============================================================

def mann_whitney_analysis(
    outcome,
    group
):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) != 2:
        return None

    g1 = data[
        data["group"] == groups[0]
    ]["outcome"]

    g2 = data[
        data["group"] == groups[1]
    ]["outcome"]

    result = stats.mannwhitneyu(
        g1,
        g2,
        alternative="two-sided"
    )

    return {
        "test": "Mann-Whitney U",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "group1": str(groups[0]),
        "group2": str(groups[1]),
        "n1": len(g1),
        "n2": len(g2),
        "median1": g1.median(),
        "median2": g2.median()
    }


# ============================================================
# WILCOXON SIGNED-RANK
# ============================================================

def wilcoxon_analysis(
    before,
    after
):

    data = pd.DataFrame({
        "before": before,
        "after": after
    }).dropna()

    if len(data) < 2:
        return None

    result = stats.wilcoxon(
        data["before"],
        data["after"]
    )

    return {
        "test": "Wilcoxon signed-rank test",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "n": len(data),
        "median_before": data["before"].median(),
        "median_after": data["after"].median()
    }


# ============================================================
# KRUSKAL-WALLIS
# ============================================================

def kruskal_analysis(
    outcome,
    group
):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    grouped = [
        values["outcome"].values
        for _, values in data.groupby("group")
    ]

    if len(grouped) < 3:
        return None

    result = stats.kruskal(
        *grouped
    )

    return {
        "test": "Kruskal-Wallis H test",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "number_of_groups": len(grouped)
    }


# ============================================================
# RANK-BISERIAL CORRELATION
# ============================================================

def rank_biserial_correlation(
    outcome,
    group
):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) != 2:
        return None

    g1 = data[
        data["group"] == groups[0]
    ]["outcome"]

    g2 = data[
        data["group"] == groups[1]
    ]["outcome"]

    result = stats.mannwhitneyu(
        g1,
        g2,
        alternative="two-sided"
    )

    u = result.statistic

    n1 = len(g1)
    n2 = len(g2)

    rbc = (
        (2 * u) /
        (n1 * n2)
    ) - 1

    return rbc


# ============================================================
# INTERPRET RANK-BISERIAL
# ============================================================

def interpret_rank_biserial(r):

    if r is None:
        return "Effect size could not be calculated."

    magnitude = abs(r)

    if magnitude < 0.10:

        strength = "negligible"

    elif magnitude < 0.30:

        strength = "small"

    elif magnitude < 0.50:

        strength = "moderate"

    else:

        strength = "large"

    direction = (
        "positive"
        if r > 0
        else "negative"
        if r < 0
        else "zero"
    )

    return (
        f"Rank-biserial correlation = {r:.3f}. "
        f"This represents a {strength}, {direction} "
        "effect."
    )


# ============================================================
# WHY NON-PARAMETRIC?
# ============================================================

def explain_nonparametric_reason(
    normality_problem=False,
    outlier_problem=False,
    ordinal_data=False,
    small_sample=False,
    skewed_data=False
):

    reasons = []

    if normality_problem:

        reasons.append(
            "The outcome shows evidence of substantial "
            "departure from normality."
        )

    if outlier_problem:

        reasons.append(
            "Potential influential outliers were detected, "
            "which may affect mean-based methods."
        )

    if ordinal_data:

        reasons.append(
            "The outcome is ordinal, so rank-based methods "
            "are naturally appropriate."
        )

    if small_sample:

        reasons.append(
            "The sample size is small, making distributional "
            "assumptions more difficult to assess reliably."
        )

    if skewed_data:

        reasons.append(
            "The outcome distribution appears strongly skewed."
        )

    if not reasons:

        reasons.append(
            "A non-parametric method may be useful when the "
            "research question concerns ranks or distributions "
            "rather than relying primarily on population means."
        )

    return reasons


# ============================================================
# MANN-WHITNEY EXPLANATION
# ============================================================

def explain_mann_whitney():

    return {
        "what": (
            "The Mann-Whitney U test is a rank-based method "
            "for comparing two independent groups."
        ),

        "why": (
            "It is often considered when a two-group comparison "
            "using a t-test is questionable because the outcome "
            "is strongly non-normal, ordinal, or affected by "
            "outliers."
        ),

        "not_what": (
            "It should not automatically be described as a test "
            "of medians. Its interpretation depends on the "
            "shapes and distributions of the two groups."
        ),

        "interpretation": (
            "A statistically significant result indicates evidence "
            "that the distributions of the two groups differ."
        )
    }


# ============================================================
# WILCOXON EXPLANATION
# ============================================================

def explain_wilcoxon():

    return {
        "what": (
            "The Wilcoxon signed-rank test is a rank-based method "
            "for paired or matched observations."
        ),

        "why": (
            "It can be used when paired differences do not satisfy "
            "the assumptions required for a paired t-test, "
            "particularly when the differences are substantially "
            "non-normal."
        ),

        "interpretation": (
            "A statistically significant result provides evidence "
            "that the paired measurements differ systematically."
        )
    }


# ============================================================
# KRUSKAL-WALLIS EXPLANATION
# ============================================================

def explain_kruskal():

    return {
        "what": (
            "The Kruskal-Wallis test is a rank-based method for "
            "comparing three or more independent groups."
        ),

        "why": (
            "It is often considered when one-way ANOVA assumptions "
            "are problematic, particularly with strongly non-normal "
            "or ordinal outcomes."
        ),

        "interpretation": (
            "A significant result indicates that at least one "
            "group distribution differs from another."
        ),

        "important": (
            "A significant Kruskal-Wallis result does not identify "
            "which groups differ. Appropriate post-hoc pairwise "
            "comparisons are required."
        )
    }


# ============================================================
# GENERAL NON-PARAMETRIC WARNING
# ============================================================

def nonparametric_warning():

    return (
        "Non-parametric does not mean assumption-free. "
        "Rank-based tests still have assumptions and their "
        "interpretation depends on the study design and the "
        "shape of the distributions."
    )
def dunn_posthoc(
    data,
    outcome,
    group,
    correction="bonferroni"
):

    import scikit_posthocs as sp

    analysis_data = data[
        [outcome, group]
    ].dropna()

    result = sp.posthoc_dunn(
        analysis_data,
        val_col=outcome,
        group_col=group,
        p_adjust=correction
    )

    return result
    