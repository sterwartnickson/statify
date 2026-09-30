import pandas as pd
import numpy as np


# ============================================================
# STATIFY DATA ENGINE
# ============================================================


def load_dataset(uploaded_file):
    """
    Load CSV or Excel data into a pandas DataFrame.
    """

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError(
            "Unsupported file format. Please upload a CSV or Excel file."
        )

    return df


def get_dataset_overview(df):
    """
    Generate high-level information about the dataset.
    """

    rows, columns = df.shape

    numeric_variables = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_variables = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    missing_cells = int(df.isna().sum().sum())

    duplicate_rows = int(df.duplicated().sum())

    return {
        "rows": rows,
        "columns": columns,
        "numeric_variables": len(numeric_variables),
        "categorical_variables": len(categorical_variables),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
    }


def detect_variable_type(series):
    """
    Attempt to classify a variable into a useful statistical type.
    """

    # Numeric variables
    if pd.api.types.is_numeric_dtype(series):

        unique_values = series.dropna().nunique()

        if unique_values == 2:
            return "Binary numeric"

        if pd.api.types.is_integer_dtype(series):

            if unique_values <= 10:
                return "Discrete numeric"

            return "Integer numeric"

        return "Continuous numeric"

    # Datetime variables
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date / Time"

    # Text / categorical variables
    unique_values = series.dropna().nunique()

    if unique_values == 2:
        return "Binary categorical"

    if unique_values <= 10:
        return "Categorical"

    return "Text / High-cardinality"


def profile_variables(df):
    """
    Create a detailed profile for every variable.
    """

    profiles = []

    for column in df.columns:

        series = df[column]

        missing = int(series.isna().sum())

        non_missing = int(series.notna().sum())

        unique = int(series.nunique(dropna=True))

        variable_type = detect_variable_type(series)

        missing_percentage = (
            (missing / len(df)) * 100
            if len(df) > 0
            else 0
        )

        profiles.append({
            "Variable": column,
            "Type": variable_type,
            "Missing": missing,
            "Missing %": round(missing_percentage, 2),
            "Unique": unique,
            "Non-missing": non_missing,
        })

    return pd.DataFrame(profiles)


def detect_data_quality_issues(df):
    """
    Detect basic data-quality issues.
    """

    issues = []

    # Missing values
    for column in df.columns:

        missing_percentage = (
            df[column].isna().mean() * 100
        )

        if missing_percentage > 50:

            issues.append({
                "Variable": column,
                "Issue": "High missingness",
                "Details": (
                    f"{missing_percentage:.1f}% "
                    "of observations are missing."
                ),
                "Severity": "High",
            })

        elif missing_percentage > 10:

            issues.append({
                "Variable": column,
                "Issue": "Moderate missingness",
                "Details": (
                    f"{missing_percentage:.1f}% "
                    "of observations are missing."
                ),
                "Severity": "Medium",
            })

        elif missing_percentage > 0:

            issues.append({
                "Variable": column,
                "Issue": "Missing values",
                "Details": (
                    f"{missing_percentage:.1f}% "
                    "of observations are missing."
                ),
                "Severity": "Low",
            })

    # Duplicate observations
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        issues.append({
            "Variable": "Dataset",
            "Issue": "Duplicate observations",
            "Details": (
                f"{duplicate_count} duplicate rows detected."
            ),
            "Severity": "Medium",
        })

    # Constant variables
    for column in df.columns:

        unique_values = df[column].nunique(dropna=True)

        if unique_values <= 1:

            issues.append({
                "Variable": column,
                "Issue": "Constant variable",
                "Details": (
                    "The variable contains only one "
                    "unique non-missing value."
                ),
                "Severity": "Medium",
            })

    return pd.DataFrame(issues)