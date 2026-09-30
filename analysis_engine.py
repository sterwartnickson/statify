import pandas as pd
import numpy as np
from scipy import stats


# ============================================================
# STATIFY ANALYSIS ENGINE
# ============================================================


def independent_t_test(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) != 2:
        return None

    g1 = data[data["group"] == groups[0]]["outcome"]
    g2 = data[data["group"] == groups[1]]["outcome"]

    result = stats.ttest_ind(
        g1,
        g2,
        equal_var=True
    )

    return {
        "test": "Independent-samples t-test",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "group1": str(groups[0]),
        "group2": str(groups[1]),
        "mean1": g1.mean(),
        "mean2": g2.mean(),
        "n1": len(g1),
        "n2": len(g2)
    }


# ============================================================
# WELCH T-TEST
# ============================================================

def welch_t_test(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) != 2:
        return None

    g1 = data[data["group"] == groups[0]]["outcome"]
    g2 = data[data["group"] == groups[1]]["outcome"]

    result = stats.ttest_ind(
        g1,
        g2,
        equal_var=False
    )

    return {
        "test": "Welch's t-test",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "group1": str(groups[0]),
        "group2": str(groups[1]),
        "mean1": g1.mean(),
        "mean2": g2.mean(),
        "n1": len(g1),
        "n2": len(g2)
    }


# ============================================================
# PAIRED T-TEST
# ============================================================

def paired_t_test(before, after):

    data = pd.DataFrame({
        "before": before,
        "after": after
    }).dropna()

    result = stats.ttest_rel(
        data["before"],
        data["after"]
    )

    difference = (
        data["after"] -
        data["before"]
    )

    return {
        "test": "Paired t-test",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "n": len(data),
        "mean_before": data["before"].mean(),
        "mean_after": data["after"].mean(),
        "mean_difference": difference.mean()
    }


# ============================================================
# MANN-WHITNEY
# ============================================================

def mann_whitney(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) != 2:
        return None

    g1 = data[data["group"] == groups[0]]["outcome"]
    g2 = data[data["group"] == groups[1]]["outcome"]

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
        "n2": len(g2)
    }


# ============================================================
# ONE-WAY ANOVA
# ============================================================

def one_way_anova(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = [
        values["outcome"].values
        for _, values in data.groupby("group")
    ]

    if len(groups) < 3:
        return None

    result = stats.f_oneway(*groups)

    return {
        "test": "One-way ANOVA",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "number_of_groups": len(groups)
    }


# ============================================================
# KRUSKAL-WALLIS
# ============================================================

def kruskal_wallis(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = [
        values["outcome"].values
        for _, values in data.groupby("group")
    ]

    if len(groups) < 3:
        return None

    result = stats.kruskal(*groups)

    return {
        "test": "Kruskal-Wallis",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "number_of_groups": len(groups)
    }


# ============================================================
# PEARSON
# ============================================================

def pearson(x, y):

    data = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    result = stats.pearsonr(
        data["x"],
        data["y"]
    )

    return {
        "test": "Pearson correlation",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "n": len(data)
    }


# ============================================================
# SPEARMAN
# ============================================================

def spearman(x, y):

    data = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    result = stats.spearmanr(
        data["x"],
        data["y"]
    )

    return {
        "test": "Spearman correlation",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "n": len(data)
    }


# ============================================================
# CHI-SQUARE
# ============================================================

def chi_square(variable1, variable2):

    data = pd.DataFrame({
        "variable1": variable1,
        "variable2": variable2
    }).dropna()

    table = pd.crosstab(
        data["variable1"],
        data["variable2"]
    )

    result = stats.chi2_contingency(
        table
    )

    return {
        "test": "Chi-square test of independence",
        "statistic": result[0],
        "p_value": result[1],
        "degrees_of_freedom": result[2],
        "table": table
    }


# ============================================================
# FISHER'S EXACT
# ============================================================

def fisher_exact(variable1, variable2):

    data = pd.DataFrame({
        "variable1": variable1,
        "variable2": variable2
    }).dropna()

    table = pd.crosstab(
        data["variable1"],
        data["variable2"]
    )

    if table.shape != (2, 2):
        return None

    result = stats.fisher_exact(
        table
    )

    return {
        "test": "Fisher's exact test",
        "odds_ratio": result[0],
        "p_value": result[1],
        "table": table
    }


# ============================================================
# TWO-WAY ANOVA
# ============================================================

def two_way_anova(
    data,
    outcome,
    factor1,
    factor2
):

    analysis_data = data[
        [
            outcome,
            factor1,
            factor2
        ]
    ].dropna()

    if analysis_data.empty:
        return None

    try:

        import statsmodels.api as sm

        from statsmodels.formula.api import ols

        formula = (
            f'Q("{outcome}") ~ '
            f'C(Q("{factor1}")) + '
            f'C(Q("{factor2}")) + '
            f'C(Q("{factor1}")):'
            f'C(Q("{factor2}"))'
        )

        model = ols(
            formula,
            data=analysis_data
        ).fit()

        table = sm.stats.anova_lm(
            model,
            typ=2
        )

        return {
            "test": "Two-way ANOVA",
            "anova_table": table,
            "model": model,
            "n": len(analysis_data)
        }

    except Exception as error:

        return {
            "test": "Two-way ANOVA",
            "error": str(error)
        }


# ============================================================
# EFFECT SIZE — COHEN'S D
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
            (n1 - 1) * group1.var()
            +
            (n2 - 1) * group2.var()
        )
        /
        (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return None

    return (
        group1.mean() -
        group2.mean()
    ) / pooled_sd


# ============================================================
# P-VALUE INTERPRETATION
# ============================================================

def interpret_p_value(
    p_value,
    alpha=0.05
):

    if p_value < alpha:

        return (
            "Statistically significant: there is evidence "
            "against the null hypothesis."
        )

    return (
        "Not statistically significant: there is insufficient "
        "evidence to reject the null hypothesis."
    )