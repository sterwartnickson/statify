import pandas as pd
import numpy as np

from scipy import stats


# ============================================================
# STATIFY ROBUST TESTS ENGINE
# ============================================================


# ============================================================
# WELCH'S INDEPENDENT T-TEST
# ============================================================

def welch_t_test(
    group1,
    group2
):

    group1 = pd.Series(group1).dropna()
    group2 = pd.Series(group2).dropna()

    if len(group1) < 2 or len(group2) < 2:
        return None

    result = stats.ttest_ind(
        group1,
        group2,
        equal_var=False
    )

    mean_difference = (
        group1.mean() -
        group2.mean()
    )

    return {
        "test": "Welch's independent t-test",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "mean1": group1.mean(),
        "mean2": group2.mean(),
        "mean_difference": mean_difference,
        "n1": len(group1),
        "n2": len(group2)
    }


# ============================================================
# WELCH'S ANOVA
# ============================================================

def welch_anova(
    data,
    outcome,
    group
):

    analysis_data = data[
        [outcome, group]
    ].dropna()

    grouped = [
        values[outcome].values
        for _, values in
        analysis_data.groupby(group)
    ]

    if len(grouped) < 3:
        return None

    try:

        result = stats.f_oneway(
            *grouped,
            equal_var=False
        )

    except TypeError:

        # Fallback for older SciPy versions
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        # Welch ANOVA approximation is unavailable
        # through this older SciPy API.
        return {
            "test": "Welch's ANOVA",
            "statistic": np.nan,
            "p_value": np.nan,
            "number_of_groups": len(grouped),
            "available": False,
            "message": (
                "Your SciPy version does not support "
                "Welch's ANOVA through scipy.stats.f_oneway. "
                "Please update SciPy."
            )
        }

    return {
        "test": "Welch's ANOVA",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "number_of_groups": len(grouped),
        "available": True
    }


# ============================================================
# GAMES-HOWELL POST-HOC TEST
# ============================================================

def games_howell_posthoc(
    data,
    outcome,
    group
):

    import pingouin as pg

    analysis_data = data[
        [outcome, group]
    ].dropna()

    result = pg.pairwise_gameshowell(
        data=analysis_data,
        dv=outcome,
        between=group
    )

    return result


# ============================================================
# WELCH EXPLANATION (EDUCATIONAL TEXT)
# ============================================================

def explain_welch():

    return {
        "what": (
            "Welch's method is a family of statistical tests used to "
            "compare group means without assuming that the groups have "
            "equal variances. This makes it a safer default than the "
            "classic Student's t-test or one-way ANOVA when variances "
            "differ across groups."
        ),
        "welch_t": (
            "For two groups, Welch's t-test compares the means of the "
            "two groups while adjusting the degrees of freedom to "
            "account for unequal variances and/or unequal sample sizes."
        ),
        "welch_anova": (
            "For three or more groups, Welch's ANOVA extends the same "
            "idea: it tests whether at least one group mean differs "
            "from the others, without requiring that all groups share "
            "the same variance."
        ),
        "why": (
            "Standard t-tests and ANOVA assume equal variances "
            "(homogeneity of variance) across groups. When this "
            "assumption is violated, those tests can give misleading "
            "p-values. Welch's versions are more robust and are "
            "recommended whenever variances are unequal, even when "
            "sample sizes are similar."
        ),
        "important": (
            "A significant Welch's ANOVA result only tells you that "
            "group means differ somewhere \u2014 it does not identify "
            "which specific groups differ. Use a post-hoc test such as "
            "Games-Howell to make pairwise comparisons."
        )
    }


# ============================================================
# WELCH GUIDANCE (RESULT-BASED MESSAGE)
# ============================================================

def welch_guidance(p_value):

    if p_value is None or (
        isinstance(p_value, float) and np.isnan(p_value)
    ):
        return (
            "Welch's test could not produce a valid p-value. Please "
            "check your data and grouping variable."
        )

    if p_value < 0.05:
        return (
            f"The result is statistically significant "
            f"(p = {p_value:.4f}), suggesting that at least one group "
            f"mean is different from the others."
        )

    return (
        f"The result is not statistically significant "
        f"(p = {p_value:.4f}), suggesting no strong evidence of a "
        f"difference between group means."
    )