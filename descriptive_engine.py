import pandas as pd
import numpy as np


# ============================================================
# STATIFY DESCRIPTIVE STATISTICS ENGINE
# ============================================================


def numerical_summary(series):
    """
    Calculate descriptive statistics for a numerical variable.
    """

    data = pd.to_numeric(series, errors="coerce").dropna()

    if len(data) == 0:
        return {}

    summary = {
        "Count": int(data.count()),
        "Missing": int(series.isna().sum()),
        "Mean": data.mean(),
        "Median": data.median(),
        "Standard Deviation": data.std(),
        "Variance": data.var(),
        "Minimum": data.min(),
        "Q1": data.quantile(0.25),
        "Q2 (Median)": data.quantile(0.50),
        "Q3": data.quantile(0.75),
        "Maximum": data.max(),
        "IQR": data.quantile(0.75) - data.quantile(0.25),
        "Skewness": data.skew(),
        "Kurtosis": data.kurt(),
    }

    return summary


def categorical_summary(series):
    """
    Generate a frequency table for categorical variables.
    """

    counts = series.value_counts(dropna=False)

    percentages = series.value_counts(
        normalize=True,
        dropna=False
    ) * 100

    table = pd.DataFrame({
        "Category": counts.index.astype(str),
        "Frequency": counts.values,
        "Percentage": percentages.values.round(2)
    })

    return table


def get_frequency_table(series):
    """
    Generate frequency and percentage information.
    """

    counts = series.value_counts(dropna=False)

    percentages = (
        series.value_counts(
            normalize=True,
            dropna=False
        ) * 100
    )

    table = pd.DataFrame({
        "Frequency": counts,
        "Percentage": percentages.round(2)
    })

    return table.reset_index()


def interpret_skewness(skewness):
    """
    Provide a simple interpretation of skewness.
    """

    if pd.isna(skewness):
        return "Skewness could not be determined."

    if abs(skewness) < 0.5:
        return "The distribution appears approximately symmetric."

    elif abs(skewness) < 1:
        return "The distribution shows moderate skewness."

    elif skewness >= 1:
        return "The distribution is strongly right-skewed."

    else:
        return "The distribution is strongly left-skewed."


def interpret_kurtosis(kurtosis):
    """
    Provide a simple interpretation of kurtosis.
    """

    if pd.isna(kurtosis):
        return "Kurtosis could not be determined."

    if abs(kurtosis) < 0.5:
        return "Kurtosis is close to the normal-distribution reference."

    elif kurtosis > 0.5:
        return "The distribution is relatively heavy-tailed."

    else:
        return "The distribution is relatively light-tailed."


def numerical_interpretation(summary):
    """
    Generate a basic statistical interpretation.
    """

    if not summary:
        return []

    interpretations = []

    mean = summary["Mean"]
    median = summary["Median"]
    skewness = summary["Skewness"]

    interpretations.append(
        f"The mean is {mean:.3f}, while the median is {median:.3f}."
    )

    interpretations.append(
        interpret_skewness(skewness)
    )

    interpretations.append(
        interpret_kurtosis(summary["Kurtosis"])
    )

    interpretations.append(
        f"The middle 50% of observations lie between "
        f"{summary['Q1']:.3f} and {summary['Q3']:.3f}."
    )

    return interpretations