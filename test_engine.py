import pandas as pd
import numpy as np

from scipy import stats


# ============================================================
# STATIFY STATISTICAL TEST ENGINE
# ============================================================


# ============================================================
# VARIABLE TYPES
# ============================================================

def variable_is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def variable_is_categorical(series):
    return not variable_is_numeric(series)


def number_of_groups(series):
    return series.dropna().nunique()


# ============================================================
# PEARSON CORRELATION
# ============================================================

def pearson_correlation(x, y):

    data = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    if len(data) < 3:
        return None

    statistic, p_value = stats.pearsonr(
        data["x"],
        data["y"]
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "n": len(data)
    }


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

def spearman_correlation(x, y):

    data = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    if len(data) < 3:
        return None

    statistic, p_value = stats.spearmanr(
        data["x"],
        data["y"]
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "n": len(data)
    }


# ============================================================
# INDEPENDENT T-TEST
# ============================================================

def independent_t_test(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) != 2:
        return None

    group1 = data[
        data["group"] == groups[0]
    ]["outcome"]

    group2 = data[
        data["group"] == groups[1]
    ]["outcome"]

    statistic, p_value = stats.ttest_ind(
        group1,
        group2,
        equal_var=True
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "Group 1": str(groups[0]),
        "Group 2": str(groups[1]),
        "Mean 1": group1.mean(),
        "Mean 2": group2.mean(),
        "n 1": len(group1),
        "n 2": len(group2)
    }


# ============================================================
# WELCH'S T-TEST
# ============================================================

def welch_t_test(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) != 2:
        return None

    group1 = data[
        data["group"] == groups[0]
    ]["outcome"]

    group2 = data[
        data["group"] == groups[1]
    ]["outcome"]

    statistic, p_value = stats.ttest_ind(
        group1,
        group2,
        equal_var=False
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "Group 1": str(groups[0]),
        "Group 2": str(groups[1]),
        "Mean 1": group1.mean(),
        "Mean 2": group2.mean(),
        "n 1": len(group1),
        "n 2": len(group2)
    }


# ============================================================
# PAIRED T-TEST
# ============================================================

def paired_t_test(before, after):

    data = pd.DataFrame({
        "before": before,
        "after": after
    }).dropna()

    if len(data) < 2:
        return None

    statistic, p_value = stats.ttest_rel(
        data["before"],
        data["after"]
    )

    differences = (
        data["after"] -
        data["before"]
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "n": len(data),
        "Mean Before": data["before"].mean(),
        "Mean After": data["after"].mean(),
        "Mean Difference": differences.mean()
    }


# ============================================================
# MANN-WHITNEY U
# ============================================================

def mann_whitney_test(outcome, group):

    data = pd.DataFrame({
        "outcome": outcome,
        "group": group
    }).dropna()

    groups = data["group"].unique()

    if len(groups) != 2:
        return None

    group1 = data[
        data["group"] == groups[0]
    ]["outcome"]

    group2 = data[
        data["group"] == groups[1]
    ]["outcome"]

    statistic, p_value = stats.mannwhitneyu(
        group1,
        group2,
        alternative="two-sided"
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "Group 1": str(groups[0]),
        "Group 2": str(groups[1]),
        "n 1": len(group1),
        "n 2": len(group2)
    }


# ============================================================
# CHI-SQUARE
# ============================================================

def chi_square_test(variable1, variable2):

    data = pd.DataFrame({
        "variable1": variable1,
        "variable2": variable2
    }).dropna()

    contingency_table = pd.crosstab(
        data["variable1"],
        data["variable2"]
    )

    if (
        contingency_table.shape[0] < 2
        or contingency_table.shape[1] < 2
    ):
        return None

    statistic, p_value, degrees_freedom, expected = (
        stats.chi2_contingency(
            contingency_table
        )
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "Degrees of Freedom": degrees_freedom,
        "Observed": contingency_table,
        "Expected": expected
    }


# ============================================================
# FISHER'S EXACT TEST
# ============================================================

def fisher_exact_test(variable1, variable2):

    data = pd.DataFrame({
        "variable1": variable1,
        "variable2": variable2
    }).dropna()

    contingency_table = pd.crosstab(
        data["variable1"],
        data["variable2"]
    )

    if contingency_table.shape != (2, 2):
        return None

    odds_ratio, p_value = stats.fisher_exact(
        contingency_table
    )

    return {
        "Odds Ratio": odds_ratio,
        "p_value": p_value,
        "Observed": contingency_table
    }


# ============================================================
# ONE-WAY ANOVA
# ============================================================

def one_way_anova(outcome, group):

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

    statistic, p_value = stats.f_oneway(
        *grouped
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "Groups": len(grouped)
    }


# ============================================================
# WELCH ANOVA
# ============================================================

def welch_anova(outcome, group):

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

    try:

        result = stats.f_oneway(
            *grouped
        )

        return {
            "Statistic": result.statistic,
            "p_value": result.pvalue,
            "Groups": len(grouped)
        }

    except Exception:

        return None


# ============================================================
# KRUSKAL-WALLIS
# ============================================================

def kruskal_wallis_test(outcome, group):

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

    statistic, p_value = stats.kruskal(
        *grouped
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "Groups": len(grouped)
    }


# ============================================================
# TWO-WAY ANOVA
# ============================================================

def two_way_anova(data, outcome, factor1, factor2):

    required_columns = [
        outcome,
        factor1,
        factor2
    ]

    analysis_data = data[
        required_columns
    ].dropna()

    if (
        analysis_data[factor1].nunique() < 2
        or analysis_data[factor2].nunique() < 2
    ):
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

        return table

    except Exception:

        return None


# ============================================================
# ASSUMPTION DIAGNOSTICS
# ============================================================

def shapiro_normality(series):

    data = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(data) < 3:
        return None

    # Shapiro is generally recommended only for
    # reasonably sized samples.
    sample = data

    if len(sample) > 5000:
        sample = sample.sample(
            5000,
            random_state=42
        )

    statistic, p_value = stats.shapiro(
        sample
    )

    return {
        "Statistic": statistic,
        "p_value": p_value,
        "n": len(sample)
    }


def levene_test(outcome, group):

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

    statistic, p_value = stats.levene(
        *groups,
        center="median"
    )

    return {
        "Statistic": statistic,
        "p_value": p_value
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
            ((n1 - 1) * group1.var())
            +
            ((n2 - 1) * group2.var())
        )
        /
        (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return None

    return (
        (group1.mean() - group2.mean())
        / pooled_sd
    )


# ============================================================
# P-VALUE INTERPRETATION
# ============================================================

def interpret_p_value(
    p_value,
    alpha=0.05
):

    if p_value is None or pd.isna(p_value):

        return "The p-value could not be determined."

    if p_value < alpha:

        return (
            f"The result is statistically significant at "
            f"the {alpha:.2f} significance level. "
            "There is evidence against the null hypothesis."
        )

    return (
        f"The result is not statistically significant at "
        f"the {alpha:.2f} significance level. "
        "There is insufficient evidence to reject the "
        "null hypothesis."
    )


# ============================================================
# TEST RECOMMENDATION ENGINE
# ============================================================

def recommend_tests(
    x,
    y
):

    x_numeric = variable_is_numeric(x)
    y_numeric = variable_is_numeric(y)

    recommendations = []

    # ========================================================
    # NUMERICAL + NUMERICAL
    # ========================================================

    if x_numeric and y_numeric:

        recommendations.append({
            "Test": "Pearson correlation",
            "Type": "Parametric",
            "Purpose": "Measure linear association",
            "Reason": (
                "Both variables are numerical and the "
                "research question concerns their linear "
                "relationship."
            ),
            "Alternative": (
                "Spearman correlation can be considered when "
                "the relationship is monotonic but not well "
                "represented by a linear correlation, or when "
                "rank-based analysis is more appropriate."
            )
        })

        recommendations.append({
            "Test": "Spearman correlation",
            "Type": "Non-parametric",
            "Purpose": "Measure monotonic association",
            "Reason": (
                "Spearman correlation uses ranks rather than "
                "assuming the same linear relationship required "
                "by Pearson correlation."
            ),
            "Alternative": (
                "Pearson correlation may be preferable when "
                "a linear relationship is scientifically justified "
                "and its assumptions are reasonably satisfied."
            )
        })

    # ========================================================
    # NUMERICAL + CATEGORICAL
    # ========================================================

    elif x_numeric and not y_numeric:

        groups = number_of_groups(y)

        if groups == 2:

            recommendations.append({
                "Test": "Independent-samples t-test",
                "Type": "Parametric",
                "Purpose": "Compare the means of two independent groups",
                "Reason": (
                    "The outcome is numerical and the grouping "
                    "variable contains exactly two independent groups."
                ),
                "Alternative": (
                    "Mann-Whitney U may be considered when the "
                    "t-test assumptions are not adequately supported, "
                    "particularly with substantial outliers or "
                    "strongly non-normal group distributions."
                )
            })

            recommendations.append({
                "Test": "Welch's t-test",
                "Type": "Parametric",
                "Purpose": "Compare two independent group means",
                "Reason": (
                    "Welch's t-test is particularly useful when "
                    "the two groups may have unequal variances."
                ),
                "Alternative": (
                    "Mann-Whitney U provides a rank-based alternative "
                    "when a mean-based parametric comparison is not "
                    "appropriate."
                )
            })

            recommendations.append({
                "Test": "Mann-Whitney U",
                "Type": "Non-parametric",
                "Purpose": "Compare two independent groups using ranks",
                "Reason": (
                    "This method does not require normally distributed "
                    "outcomes in the same way as the t-test and can be "
                    "useful when the data contain substantial skewness "
                    "or influential outliers."
                ),
                "Alternative": (
                    "The independent-samples or Welch's t-test may "
                    "be preferable when the outcome is suitable for "
                    "mean-based inference."
                )
            })

        elif groups >= 3:

            recommendations.append({
                "Test": "One-way ANOVA",
                "Type": "Parametric",
                "Purpose": "Compare means across three or more groups",
                "Reason": (
                    "The outcome is numerical and the grouping "
                    "variable contains three or more groups."
                ),
                "Alternative": (
                    "Kruskal-Wallis may be considered when the "
                    "assumptions required for ANOVA are not "
                    "adequately supported."
                )
            })

            recommendations.append({
                "Test": "Kruskal-Wallis",
                "Type": "Non-parametric",
                "Purpose": "Compare three or more independent groups",
                "Reason": (
                    "Kruskal-Wallis is a rank-based alternative "
                    "when the assumptions of one-way ANOVA are "
                    "not adequately supported."
                ),
                "Alternative": (
                    "One-way ANOVA is generally preferable when "
                    "its assumptions are reasonably satisfied and "
                    "comparison of group means is the scientific "
                    "objective."
                )
            })

    # ========================================================
    # CATEGORICAL + NUMERICAL
    # ========================================================

    elif not x_numeric and y_numeric:

        groups = number_of_groups(x)

        if groups == 2:

            recommendations.append({
                "Test": "Independent-samples t-test",
                "Type": "Parametric",
                "Purpose": "Compare means between two groups",
                "Reason": (
                    "The categorical variable defines two groups "
                    "and the outcome is numerical."
                ),
                "Alternative": (
                    "Mann-Whitney U may be considered if the "
                    "parametric assumptions are not adequately supported."
                )
            })

            recommendations.append({
                "Test": "Welch's t-test",
                "Type": "Parametric",
                "Purpose": "Compare means when variances may differ",
                "Reason": (
                    "Welch's method is less dependent on the "
                    "assumption of equal population variances."
                ),
                "Alternative": (
                    "Mann-Whitney U provides a rank-based alternative."
                )
            })

            recommendations.append({
                "Test": "Mann-Whitney U",
                "Type": "Non-parametric",
                "Purpose": "Compare two independent groups using ranks",
                "Reason": (
                    "Useful when the numerical outcome is strongly "
                    "skewed or contains influential observations "
                    "that make a mean-based comparison questionable."
                ),
                "Alternative": (
                    "Welch's t-test is often preferable when "
                    "mean differences are the scientific target "
                    "and its assumptions are reasonably supported."
                )
            })

        elif groups >= 3:

            recommendations.append({
                "Test": "One-way ANOVA",
                "Type": "Parametric",
                "Purpose": "Compare means across three or more groups",
                "Reason": (
                    "The categorical variable defines three or "
                    "more independent groups."
                ),
                "Alternative": (
                    "Kruskal-Wallis can be considered when ANOVA "
                    "assumptions are not adequately supported."
                )
            })

            recommendations.append({
                "Test": "Kruskal-Wallis",
                "Type": "Non-parametric",
                "Purpose": "Compare three or more independent groups",
                "Reason": (
                    "This rank-based method can be useful when "
                    "strong skewness, outliers, or other issues "
                    "make ANOVA less appropriate."
                ),
                "Alternative": (
                    "One-way ANOVA may be preferable when its "
                    "assumptions are reasonably satisfied."
                )
            })

    # ========================================================
    # CATEGORICAL + CATEGORICAL
    # ========================================================

    else:

        recommendations.append({
            "Test": "Chi-square test of independence",
            "Type": "Categorical association",
            "Purpose": "Test association between categorical variables",
            "Reason": (
                "Both variables are categorical."
            ),
            "Alternative": (
                "Fisher's exact test is particularly useful for "
                "a 2 × 2 table when expected cell frequencies "
                "are small."
            )
        })

        if (
            number_of_groups(x) == 2
            and number_of_groups(y) == 2
        ):

            recommendations.append({
                "Test": "Fisher's exact test",
                "Type": "Exact test",
                "Purpose": "Test association in a 2 × 2 table",
                "Reason": (
                    "Both variables contain two categories, "
                    "making a 2 × 2 contingency table possible. "
                    "Fisher's exact test is useful when expected "
                    "cell frequencies are small."
                ),
                "Alternative": (
                    "Chi-square may be appropriate when the "
                    "expected-frequency conditions are adequately met."
                )
            })

    return pd.DataFrame(
        recommendations
    )