import pandas as pd
import numpy as np


# ============================================================
# STATIFY — PATTERN & TREND INTELLIGENCE
# ============================================================


def detect_trend(
    data,
    value_column,
    order_column
):

    if (
        order_column is None
        or value_column is None
        or order_column == value_column
        or order_column not in data.columns
        or value_column not in data.columns
    ):
        return None

    temp = data[
        [order_column, value_column]
    ].copy()

    temp[value_column] = pd.to_numeric(
        temp[value_column],
        errors="coerce"
    )

    temp = temp.dropna()

    if len(temp) < 3:
        return None

    # Preserve the user's ordering variable
    temp = temp.sort_values(
        order_column
    )

    y = temp[value_column].values
    x = np.arange(len(y))

    slope = np.polyfit(
        x,
        y,
        1
    )[0]

    first = y[0]
    last = y[-1]

    if first != 0:

        percentage_change = (
            (last - first) /
            abs(first)
        ) * 100

    else:

        percentage_change = np.nan

    if slope > 0:

        direction = "Increasing"

    elif slope < 0:

        direction = "Decreasing"

    else:

        direction = "Stable"

    return {
        "order_column": order_column,
        "value_column": value_column,
        "slope": slope,
        "percentage_change": percentage_change,
        "direction": direction,
        "observations": len(temp)
    }


# ============================================================
# VARIABILITY / VOLATILITY
# ============================================================

def detect_volatility(
    data,
    value_column
):

    series = pd.to_numeric(
        data[value_column],
        errors="coerce"
    ).dropna()

    if len(series) < 3:
        return None

    mean = series.mean()
    std = series.std()

    if mean != 0:

        cv = abs(std / mean) * 100

    else:

        cv = np.nan

    if pd.isna(cv):

        level = "Undetermined"

    elif cv < 10:

        level = "Low"

    elif cv < 25:

        level = "Moderate"

    else:

        level = "High"

    return {
        "variable": value_column,
        "standard_deviation": std,
        "coefficient_of_variation": cv,
        "volatility": level
    }


# ============================================================
# EXTREME VALUES
# ============================================================

def detect_extremes(
    data,
    numeric_columns
):

    results = []

    for column in numeric_columns:

        series = pd.to_numeric(
            data[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        results.append({
            "variable": column,
            "minimum": series.min(),
            "maximum": series.max(),
            "range": series.max() - series.min()
        })

    return pd.DataFrame(results)


# ============================================================
# AUTOMATIC PATTERN SUMMARY
# ============================================================

def generate_pattern_summary(
    data,
    numeric_columns
):

    patterns = []

    for column in numeric_columns:

        volatility = detect_volatility(
            data,
            column
        )

        if volatility is None:
            continue

        patterns.append({
            "variable": column,
            "pattern": (
                f"{column} shows "
                f"{volatility['volatility'].lower()} "
                f"variability "
                f"(CV = "
                f"{volatility['coefficient_of_variation']:.2f}%)."
            )
        })

    return patterns