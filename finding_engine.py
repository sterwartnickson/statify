import pandas as pd
import numpy as np


# ============================================================
# STATIFY — FINDING DETECTION ENGINE
# ============================================================


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

def numerical_summary(data, numeric_columns):

    findings = []

    for column in numeric_columns:

        series = pd.to_numeric(
            data[column],
            errors="coerce"
        ).dropna()

        if len(series) == 0:
            continue

        mean = series.mean()
        median = series.median()
        std = series.std()

        minimum = series.min()
        maximum = series.max()

        skewness = series.skew()

        findings.append({
            "variable": column,
            "n": len(series),
            "mean": mean,
            "median": median,
            "std": std,
            "minimum": minimum,
            "maximum": maximum,
            "skewness": skewness
        })

    return pd.DataFrame(findings)


# ============================================================
# SKEWNESS FINDINGS
# ============================================================

def detect_skewness(
    data,
    numeric_columns
):

    findings = []

    for column in numeric_columns:

        series = pd.to_numeric(
            data[column],
            errors="coerce"
        ).dropna()

        if len(series) < 3:
            continue

        skew = series.skew()

        if abs(skew) < 0.5:

            interpretation = (
                "The distribution appears approximately symmetric."
            )

        elif abs(skew) < 1:

            interpretation = (
                "The distribution shows moderate skewness."
            )

        else:

            interpretation = (
                "The distribution shows substantial skewness."
            )

        findings.append({
            "variable": column,
            "skewness": skew,
            "interpretation": interpretation
        })

    return pd.DataFrame(findings)


# ============================================================
# RANGE / VARIABILITY
# ============================================================

def detect_variability(
    data,
    numeric_columns
):

    findings = []

    for column in numeric_columns:

        series = pd.to_numeric(
            data[column],
            errors="coerce"
        ).dropna()

        if len(series) < 2:
            continue

        mean = series.mean()
        std = series.std()

        if mean != 0:

            cv = (
                abs(std / mean) * 100
            )

        else:

            cv = np.nan

        findings.append({
            "variable": column,
            "standard_deviation": std,
            "coefficient_of_variation": cv
        })

    return pd.DataFrame(findings)


# ============================================================
# MISSING DATA FINDINGS
# ============================================================

def detect_missing_patterns(
    data
):

    results = []

    total_rows = len(data)

    if total_rows == 0:
        return pd.DataFrame()

    for column in data.columns:

        missing = data[column].isna().sum()

        percentage = (
            missing /
            total_rows *
            100
        )

        results.append({
            "variable": column,
            "missing_count": missing,
            "missing_percentage": percentage
        })

    result = pd.DataFrame(results)

    return result.sort_values(
        "missing_percentage",
        ascending=False
    )


# ============================================================
# CATEGORICAL DISTRIBUTION
# ============================================================

def categorical_summary(
    data,
    categorical_columns
):

    results = []

    for column in categorical_columns:

        counts = (
            data[column]
            .value_counts(
                dropna=False
            )
        )

        total = counts.sum()

        for category, count in counts.items():

            percentage = (
                count /
                total *
                100
            )

            results.append({
                "variable": column,
                "category": category,
                "count": count,
                "percentage": percentage
            })

    return pd.DataFrame(results)


# ============================================================
# IMBALANCED CATEGORIES
# ============================================================

def detect_category_imbalance(
    data,
    categorical_columns
):

    findings = []

    for column in categorical_columns:

        counts = (
            data[column]
            .value_counts(
                normalize=True,
                dropna=False
            ) * 100
        )

        if len(counts) < 2:
            continue

        largest = counts.iloc[0]
        smallest = counts.iloc[-1]

        findings.append({
            "variable": column,
            "largest_category_percentage": largest,
            "smallest_category_percentage": smallest
        })

    return pd.DataFrame(findings)


# ============================================================
# AUTOMATED FINDING GENERATOR
# ============================================================

def generate_findings(
    data,
    numeric_columns,
    categorical_columns
):

    findings = []

    # --------------------------------------------------------
    # Numerical variables
    # --------------------------------------------------------

    summary = numerical_summary(
        data,
        numeric_columns
    )

    for _, row in summary.iterrows():

        findings.append({
            "type": "Numerical",
            "variable": row["variable"],
            "finding": (
                f"{row['variable']} has a mean of "
                f"{row['mean']:.2f} and a median of "
                f"{row['median']:.2f}, with a standard "
                f"deviation of {row['std']:.2f}."
            )
        })


    # --------------------------------------------------------
    # Skewness
    # --------------------------------------------------------

    skewness = detect_skewness(
        data,
        numeric_columns
    )

    for _, row in skewness.iterrows():

        if abs(row["skewness"]) >= 0.5:

            findings.append({
                "type": "Distribution",
                "variable": row["variable"],
                "finding": (
                    f"{row['variable']} shows "
                    f"skewness of "
                    f"{row['skewness']:.2f}. "
                    f"{row['interpretation']}"
                )
            })


    # --------------------------------------------------------
    # Missing data
    # --------------------------------------------------------

    missing = detect_missing_patterns(
        data
    )

    for _, row in missing.iterrows():

        if row["missing_percentage"] > 0:

            findings.append({
                "type": "Missing Data",
                "variable": row["variable"],
                "finding": (
                    f"{row['variable']} contains "
                    f"{int(row['missing_count'])} missing "
                    f"observations "
                    f"({row['missing_percentage']:.2f}%)."
                )
            })


    # --------------------------------------------------------
    # Category imbalance
    # --------------------------------------------------------

    imbalance = detect_category_imbalance(
        data,
        categorical_columns
    )

    for _, row in imbalance.iterrows():

        difference = (
            row["largest_category_percentage"] -
            row["smallest_category_percentage"]
        )

        if difference >= 30:

            findings.append({
                "type": "Category Balance",
                "variable": row["variable"],
                "finding": (
                    f"{row['variable']} has an uneven "
                    f"category distribution. The difference "
                    f"between the largest and smallest "
                    f"category proportions is approximately "
                    f"{difference:.1f} percentage points."
                )
            })

    return findings