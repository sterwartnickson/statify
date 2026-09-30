import pandas as pd
import numpy as np


# ============================================================
# STATIFY CLEANING ENGINE
# ============================================================


# ============================================================
# DATA COPY
# ============================================================

def create_working_copy(df):
    """
    Create a safe copy of the original dataset.
    """
    return df.copy(deep=True)


# ============================================================
# DUPLICATES
# ============================================================

def duplicate_summary(df):

    duplicate_mask = df.duplicated()

    return {
        "count": int(duplicate_mask.sum()),
        "percentage": (
            duplicate_mask.mean() * 100
            if len(df) > 0
            else 0
        )
    }


def remove_duplicates(df):

    before = len(df)

    cleaned = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    removed = before - len(cleaned)

    return cleaned, {
        "action": "Remove duplicates",
        "details": f"Removed {removed} duplicate row(s).",
        "affected_rows": removed
    }


# ============================================================
# MISSING VALUES
# ============================================================

def missing_summary(df):

    summary = pd.DataFrame({
        "Variable": df.columns,
        "Missing": [
            int(df[col].isna().sum())
            for col in df.columns
        ],
        "Missing %": [
            round(
                df[col].isna().mean() * 100,
                2
            )
            for col in df.columns
        ]
    })

    return summary.sort_values(
        "Missing %",
        ascending=False
    )


# ============================================================
# DROP MISSING ROWS
# ============================================================

def drop_missing_rows(
    df,
    column=None
):

    before = len(df)

    if column is None:

        cleaned = (
            df
            .dropna()
            .reset_index(drop=True)
        )

        description = (
            "Removed rows containing one or more missing values."
        )

    else:

        cleaned = (
            df
            .dropna(subset=[column])
            .reset_index(drop=True)
        )

        description = (
            f"Removed rows where {column} was missing."
        )

    removed = before - len(cleaned)

    return cleaned, {
        "action": "Remove missing observations",
        "details": (
            f"{description} "
            f"{removed} row(s) were removed."
        ),
        "affected_rows": removed
    }


# ============================================================
# IMPUTATION
# ============================================================

def impute_missing(
    df,
    column,
    method
):

    cleaned = df.copy()

    missing_before = int(
        cleaned[column].isna().sum()
    )

    if missing_before == 0:

        return cleaned, {
            "action": "Imputation",
            "details": (
                f"No missing observations were present "
                f"in {column}."
            ),
            "affected_rows": 0
        }

    original_dtype = cleaned[column].dtype

    # A column can look like whole numbers (e.g. satisfaction scores
    # 1-5) but still be stored as float64, because pandas cannot hold
    # NaN in an integer column and silently upcasts it. So instead of
    # trusting the dtype, check whether every existing value is a
    # whole number.
    existing_values = cleaned[column].dropna()

    is_whole_number_column = (
        pd.api.types.is_numeric_dtype(original_dtype)
        and not existing_values.empty
        and (existing_values % 1 == 0).all()
    )

    if method == "Mean":

        if not pd.api.types.is_numeric_dtype(
            cleaned[column]
        ):
            raise ValueError(
                "Mean imputation requires a numerical variable."
            )

        replacement = cleaned[column].mean()

    elif method == "Median":

        if not pd.api.types.is_numeric_dtype(
            cleaned[column]
        ):
            raise ValueError(
                "Median imputation requires a numerical variable."
            )

        replacement = cleaned[column].median()

    elif method == "Mode":

        mode = cleaned[column].mode(
            dropna=True
        )

        if mode.empty:
            raise ValueError(
                "A mode could not be calculated."
            )

        replacement = mode.iloc[0]

    else:

        raise ValueError(
            "Unsupported imputation method."
        )

    # Match the replacement value's type to the variable's original
    # data type: whole-number variables get a rounded whole-number
    # replacement, decimal variables keep their decimal precision.
    if method in ("Mean", "Median") and is_whole_number_column:

        replacement = int(
            round(replacement)
        )

    cleaned[column] = (
        cleaned[column]
        .fillna(replacement)
    )

    if is_whole_number_column:

        # All missing values in this column are now filled, so it's
        # safe to convert to a genuine integer dtype (rather than
        # leaving it as float64 with trailing ".0"s).
        cleaned[column] = cleaned[column].astype(
            "int64"
        )

    return cleaned, {
        "action": "Missing-value imputation",
        "details": (
            f"Imputed {missing_before} missing value(s) "
            f"in {column} using the {method.lower()}. "
            f"Replacement value: {replacement}"
        ),
        "affected_rows": missing_before
    }


# ============================================================
# TYPE CONVERSION
# ============================================================

def convert_to_numeric(
    df,
    column
):

    cleaned = df.copy()

    before_missing = int(
        cleaned[column].isna().sum()
    )

    cleaned[column] = pd.to_numeric(
        cleaned[column],
        errors="coerce"
    )

    after_missing = int(
        cleaned[column].isna().sum()
    )

    newly_missing = (
        after_missing -
        before_missing
    )

    return cleaned, {
        "action": "Convert variable to numeric",
        "details": (
            f"Converted {column} to numeric. "
            f"{max(newly_missing, 0)} value(s) could not "
            "be converted and became missing."
        ),
        "affected_rows": max(
            newly_missing,
            0
        )
    }


def convert_to_category(
    df,
    column
):

    cleaned = df.copy()

    cleaned[column] = (
        cleaned[column]
        .astype("category")
    )

    return cleaned, {
        "action": "Convert variable to categorical",
        "details": (
            f"Converted {column} to categorical format."
        ),
        "affected_rows": len(cleaned)
    }


def convert_to_datetime(
    df,
    column
):

    cleaned = df.copy()

    before_missing = int(
        cleaned[column].isna().sum()
    )

    cleaned[column] = pd.to_datetime(
        cleaned[column],
        errors="coerce"
    )

    after_missing = int(
        cleaned[column].isna().sum()
    )

    failed = (
        after_missing -
        before_missing
    )

    return cleaned, {
        "action": "Convert variable to date",
        "details": (
            f"Converted {column} to date/time format. "
            f"{max(failed, 0)} value(s) could not be parsed."
        ),
        "affected_rows": max(
            failed,
            0
        )
    }


# ============================================================
# RANGE VALIDATION
# ============================================================

def detect_out_of_range(
    df,
    column,
    minimum=None,
    maximum=None
):

    data = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    mask = pd.Series(
        False,
        index=df.index
    )

    if minimum is not None:

        mask = (
            mask |
            (data < minimum)
        )

    if maximum is not None:

        mask = (
            mask |
            (data > maximum)
        )

    return df.loc[
        mask
    ].copy()


# ============================================================
# SET IMPOSSIBLE VALUES TO MISSING
# ============================================================

def replace_out_of_range_with_missing(
    df,
    column,
    minimum=None,
    maximum=None
):

    cleaned = df.copy()

    numeric = pd.to_numeric(
        cleaned[column],
        errors="coerce"
    )

    mask = pd.Series(
        False,
        index=cleaned.index
    )

    if minimum is not None:

        mask = (
            mask |
            (numeric < minimum)
        )

    if maximum is not None:

        mask = (
            mask |
            (numeric > maximum)
        )

    count = int(
        mask.sum()
    )

    cleaned.loc[
        mask,
        column
    ] = np.nan

    return cleaned, {
        "action": "Range validation",
        "details": (
            f"Set {count} out-of-range value(s) "
            f"in {column} to missing. "
            f"Allowed range: {minimum} to {maximum}."
        ),
        "affected_rows": count
    }


# ============================================================
# OUTLIERS
# ============================================================

def detect_iqr_outliers(
    df,
    column
):

    data = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    valid = data.dropna()

    if len(valid) < 4:

        return {
            "count": 0,
            "indices": [],
            "lower_bound": np.nan,
            "upper_bound": np.nan
        }

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)

    iqr = q3 - q1

    lower = (
        q1 -
        1.5 * iqr
    )

    upper = (
        q3 +
        1.5 * iqr
    )

    mask = (
        (data < lower) |
        (data > upper)
    )

    return {
        "count": int(mask.sum()),
        "indices": df.index[
            mask
        ].tolist(),
        "lower_bound": lower,
        "upper_bound": upper
    }


# ============================================================
# OPTIONAL OUTLIER REMOVAL
# ============================================================

def remove_iqr_outliers(
    df,
    column
):

    result = detect_iqr_outliers(
        df,
        column
    )

    cleaned = (
        df
        .drop(
            index=result["indices"]
        )
        .reset_index(drop=True)
    )

    return cleaned, {
        "action": "Remove potential outliers",
        "details": (
            f"Removed {result['count']} observation(s) "
            f"from {column} using the 1.5 × IQR rule."
        ),
        "affected_rows": result["count"]
    }


# ============================================================
# RENAME VARIABLE
# ============================================================

def rename_variable(
    df,
    old_name,
    new_name
):

    new_name = new_name.strip()

    if not new_name:

        raise ValueError(
            "The new variable name cannot be empty."
        )

    if new_name in df.columns:

        raise ValueError(
            "That variable name already exists."
        )

    cleaned = df.rename(
        columns={
            old_name: new_name
        }
    )

    return cleaned, {
        "action": "Rename variable",
        "details": (
            f"Renamed {old_name} to {new_name}."
        ),
        "affected_rows": 0
    }


# ============================================================
# DROP VARIABLE
# ============================================================

def drop_variable(
    df,
    column
):

    cleaned = df.drop(
        columns=[column]
    )

    return cleaned, {
        "action": "Drop variable",
        "details": (
            f"Removed variable {column} from the working dataset."
        ),
        "affected_rows": 0
    }