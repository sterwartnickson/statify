import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.formula.api import ols


# ============================================================
# STATIFY TWO-WAY ANOVA ENGINE
# ============================================================


# ============================================================
# TWO-WAY ANOVA
# ============================================================

def two_way_anova(
    data,
    outcome,
    factor_a,
    factor_b
):

    analysis_data = data[
        [outcome, factor_a, factor_b]
    ].dropna().copy()

    if analysis_data.empty:
        return None

    if analysis_data[factor_a].nunique() < 2:
        return None

    if analysis_data[factor_b].nunique() < 2:
        return None

    formula = (
        f'Q("{outcome}") ~ '
        f'C(Q("{factor_a}")) * '
        f'C(Q("{factor_b}"))'
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
        "model": model,
        "anova_table": table,
        "data": analysis_data
    }


# ============================================================
# PARTIAL ETA SQUARED
# ============================================================

def partial_eta_squared(
    anova_table
):

    results = {}

    residual_ss = anova_table.loc[
        "Residual",
        "sum_sq"
    ]

    for effect in anova_table.index:

        if effect == "Residual":
            continue

        effect_ss = anova_table.loc[
            effect,
            "sum_sq"
        ]

        denominator = (
            effect_ss +
            residual_ss
        )

        if denominator == 0:
            results[effect] = np.nan

        else:
            results[effect] = (
                effect_ss /
                denominator
            )

    return results


# ============================================================
# EFFECT SIZE INTERPRETATION
# ============================================================

def interpret_partial_eta_squared(
    eta
):

    if pd.isna(eta):
        return "Effect size could not be calculated."

    if eta < 0.01:

        magnitude = "very small"

    elif eta < 0.06:

        magnitude = "small"

    elif eta < 0.14:

        magnitude = "moderate"

    else:

        magnitude = "large"

    return (
        f"Partial η² = {eta:.3f}, indicating a "
        f"{magnitude} effect."
    )


# ============================================================
# INTERACTION INTERPRETATION
# ============================================================

def interaction_interpretation(
    p_value,
    factor_a,
    factor_b,
    alpha=0.05
):

    if p_value < alpha:

        return (
            f"There is evidence of a statistically significant "
            f"interaction between {factor_a} and {factor_b}. "
            f"This means the effect of {factor_a} on the outcome "
            f"appears to depend on the level of {factor_b}, or "
            f"vice versa."
        )

    return (
        f"There is insufficient statistical evidence of an "
        f"interaction between {factor_a} and {factor_b}. "
        f"The effect of one factor does not appear to vary "
        f"systematically across levels of the other factor."
    )


# ============================================================
# MAIN EFFECT INTERPRETATION
# ============================================================

def main_effect_interpretation(
    p_value,
    factor,
    alpha=0.05
):

    if p_value < alpha:

        return (
            f"There is evidence that {factor} is associated "
            f"with differences in the outcome after accounting "
            f"for the other factor and the interaction."
        )

    return (
        f"There is insufficient evidence that {factor} is "
        f"associated with differences in the outcome after "
        f"accounting for the other terms in the model."
    )


# ============================================================
# ANOVA TABLE PREPARATION
# ============================================================

def prepare_anova_table(
    table
):

    output = table.reset_index()

    output = output.rename(
        columns={
            "index": "Effect",
            "sum_sq": "Sum of Squares",
            "df": "df",
            "F": "F statistic",
            "PR(>F)": "p-value"
        }
    )

    return output


# ============================================================
# CELL SUMMARY
# ============================================================

def cell_means(
    data,
    outcome,
    factor_a,
    factor_b
):

    summary = (
        data
        .groupby(
            [factor_a, factor_b]
        )[outcome]
        .agg(
            Count="count",
            Mean="mean",
            SD="std"
        )
        .reset_index()
    )

    return summary


# ============================================================
# INTERACTION STATUS
# ============================================================

def interaction_status(
    p_value,
    alpha=0.05
):

    if p_value < alpha:
        return "Significant interaction"

    return "No statistically significant interaction"