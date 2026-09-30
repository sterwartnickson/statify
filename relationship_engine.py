import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


# ============================================================
# STATIFY — RELATIONSHIP DISCOVERY ENGINE
# ============================================================


def correlation_analysis(
    data,
    numeric_columns,
    method="pearson"
):

    results = []

    for i in range(len(numeric_columns)):

        for j in range(i + 1, len(numeric_columns)):

            x_name = numeric_columns[i]
            y_name = numeric_columns[j]

            pair = data[
                [x_name, y_name]
            ].dropna()

            if len(pair) < 3:
                continue

            x = pd.to_numeric(
                pair[x_name],
                errors="coerce"
            )

            y = pd.to_numeric(
                pair[y_name],
                errors="coerce"
            )

            valid = pd.concat(
                [x, y],
                axis=1
            ).dropna()

            if len(valid) < 3:
                continue

            if method == "pearson":

                r, p = pearsonr(
                    valid.iloc[:, 0],
                    valid.iloc[:, 1]
                )

            else:

                r, p = spearmanr(
                    valid.iloc[:, 0],
                    valid.iloc[:, 1]
                )

            results.append({
                "Variable 1": x_name,
                "Variable 2": y_name,
                "Correlation": r,
                "p-value": p,
                "N": len(valid),
                "Method": method
            })

    return pd.DataFrame(results)


# ============================================================
# CORRELATION STRENGTH
# ============================================================

def correlation_strength(r):

    r = abs(r)

    if r < 0.10:
        return "Negligible"

    elif r < 0.30:
        return "Weak"

    elif r < 0.50:
        return "Moderate"

    elif r < 0.70:
        return "Strong"

    else:
        return "Very strong"


# ============================================================
# CORRELATION DIRECTION
# ============================================================

def correlation_direction(r):

    if r > 0:
        return "Positive"

    elif r < 0:
        return "Negative"

    return "No direction"


# ============================================================
# ADD INTERPRETATION
# ============================================================

def interpret_correlations(
    correlation_table,
    alpha=0.05
):

    if correlation_table.empty:
        return correlation_table

    results = correlation_table.copy()

    results["Strength"] = (
        results["Correlation"]
        .apply(correlation_strength)
    )

    results["Direction"] = (
        results["Correlation"]
        .apply(correlation_direction)
    )

    results["Significant"] = (
        results["p-value"] < alpha
    )

    return results


# ============================================================
# MOST IMPORTANT RELATIONSHIPS
# ============================================================

def strongest_relationships(
    correlation_table,
    top_n=10
):

    if correlation_table.empty:
        return correlation_table

    result = correlation_table.copy()

    result["Absolute Correlation"] = (
        result["Correlation"].abs()
    )

    result = result.sort_values(
        "Absolute Correlation",
        ascending=False
    )

    return result.head(top_n)