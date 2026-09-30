
"""Domain-aware cleaning engine for Statify Batch 2."""
import pandas as pd


def _role_columns(df, variable_map, roles):
    if variable_map is None or variable_map.empty:
        return []
    return [
        c for c in variable_map.loc[
            variable_map["Domain Role"].isin(roles), "Variable"
        ].tolist()
        if c in df.columns
    ]


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _clean_numeric_text(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.copy()
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(r"[$€£]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def domain_cleaning_plan(df, domain, variable_map=None):
    """Build conservative, review-first domain cleaning recommendations."""
    if variable_map is None:
        variable_map = pd.DataFrame()

    rows = []

    def add(action_id, area, variable, issue, action, risk="Review first"):
        rows.append({
            "Action ID": action_id,
            "Area": area,
            "Variable": variable,
            "Issue": issue,
            "Recommended Action": action,
            "Risk": risk,
        })

    duplicates = int(df.duplicated().sum())
    if duplicates:
        add(
            "DUP-01", "Duplicates", "<all columns>",
            f"{duplicates} exact duplicate row(s) detected.",
            "Review and remove exact duplicate rows if they are unintended.",
        )

    if domain == "🛒 Sales / E-commerce & Retail":
        for c in _role_columns(df, variable_map, ["Quantity"]):
            n = int((_numeric(df[c]) < 0).sum())
            if n:
                add("SAL-01", "Range", c, f"{n} negative quantity value(s).",
                    "Review; negative quantities may represent returns/adjustments.")
        for c in (
            _role_columns(df, variable_map, ["Price"])
            + _role_columns(df, variable_map, ["Revenue"])
        ):
            n = int((_numeric(df[c]) < 0).sum())
            if n:
                add("SAL-02", "Range", c, f"{n} negative value(s).",
                    "Review refunds, returns and accounting adjustments before changing.")
        for c in _role_columns(df, variable_map, ["Order ID"]):
            n = int(df[c].duplicated(keep=False).sum())
            if n:
                add("SAL-03", "Identifiers", c,
                    f"{n} row(s) share an order identifier.",
                    "Review whether one order can legitimately contain multiple line items.")

    elif domain == "💰 Finance":
        # Negative financial amounts can be legitimate, so do not flag them as errors.
        for c in _role_columns(df, variable_map, ["Amount", "Income", "Expense", "Balance"]):
            s = df[c]
            if not pd.api.types.is_numeric_dtype(s):
                converted = _clean_numeric_text(s)
                parse_failures = int(
                    s.notna().sum() - converted.notna().sum()
                )
                if parse_failures:
                    add("FIN-01", "Numeric Format", c,
                        f"{parse_failures} non-missing value(s) could not be read as numeric.",
                        "Convert standard currency-formatted text to numeric, then review failures.")
        add(
            "FIN-02", "Sign Convention", "<amount variables>",
            "Negative amounts were not automatically treated as errors.",
            "Check the dataset's debit/credit/refund convention before changing signs.",
            "Important",
        )

    elif domain == "👥 Demography":
        for c in _role_columns(df, variable_map, ["Age"]):
            s = _numeric(df[c])
            n = int(((s < 0) | (s > 120)).sum())
            if n:
                add("DEM-01", "Range", c,
                    f"{n} age value(s) fall outside 0–120.",
                    "Review impossible or miscoded ages; do not automatically delete.")
        for c in _role_columns(
            df, variable_map, ["Population", "Children", "Births", "Deaths"]
        ):
            n = int((_numeric(df[c]) < 0).sum())
            if n:
                add("DEM-02", "Range", c,
                    f"{n} negative count value(s).",
                    "Review coding and measurement definition before replacement.")

    elif domain == "🏥 Health":
        for c in _role_columns(df, variable_map, ["Age"]):
            s = _numeric(df[c])
            n = int(((s < 0) | (s > 120)).sum())
            if n:
                add("HLT-01", "Range", c,
                    f"{n} age value(s) fall outside 0–120.",
                    "Review impossible or miscoded ages; retain until verified.")
        for role in ["Weight", "Height", "BMI"]:
            for c in _role_columns(df, variable_map, [role]):
                n = int((_numeric(df[c]) < 0).sum())
                if n:
                    add("HLT-02", "Range", c,
                        f"{n} negative {role.lower()} value(s).",
                        "Review measurement/unit/coding errors before replacement.")
        for c in _role_columns(df, variable_map, ["Blood Pressure"]):
            bad = (
                ~df[c].astype("string").str.contains(r"\d", regex=True, na=False)
            ).sum()
            if bad:
                add("HLT-03", "Format", c,
                    f"{int(bad)} value(s) contain no detectable blood-pressure number.",
                    "Review BP formatting before analysis.")

    elif domain == "🌾 Agriculture":
        for role in ["Area", "Yield", "Production", "Rainfall", "Fertilizer"]:
            for c in _role_columns(df, variable_map, [role]):
                n = int((_numeric(df[c]) < 0).sum())
                if n:
                    add("AGR-01", "Range", c,
                        f"{n} negative {role.lower()} value(s).",
                        "Review units, measurement and coding before replacement.")
        for c in _role_columns(df, variable_map, ["Season"]):
            add("AGR-02", "Categories", c,
                "Season is a domain-sensitive categorical variable.",
                "Standardize spelling/case only after reviewing observed categories.")

    # General numeric-looking text detection.
    for c in df.columns:
        if df[c].dtype == "object":
            converted = _clean_numeric_text(df[c])
            nonmissing = int(df[c].notna().sum())
            convertible = int(converted.notna().sum())
            if nonmissing >= 5 and convertible / max(nonmissing, 1) >= 0.8:
                add("GEN-01", "Type", c,
                    "Most non-missing values appear numeric but the column is stored as text.",
                    "Convert numeric-looking text to numeric.",
                    "Usually safe after review")

    return pd.DataFrame(rows)


def apply_domain_action(df, action_row):
    """Apply one explicitly selected action to a copy of the working dataset."""
    cleaned = df.copy()
    action_id = str(action_row.get("Action ID", ""))
    variable = str(action_row.get("Variable", ""))

    if action_id == "DUP-01":
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        return cleaned, f"Removed {before - len(cleaned)} exact duplicate row(s)."

    if action_id == "GEN-01" and variable in cleaned.columns:
        original = cleaned[variable]
        converted = _clean_numeric_text(original)
        if converted.notna().sum() >= original.notna().sum() * 0.8:
            old_missing = int(original.isna().sum())
            cleaned[variable] = converted
            new_missing = int(cleaned[variable].isna().sum())
            return cleaned, (
                f"Converted '{variable}' to numeric. "
                f"{new_missing - old_missing} additional value(s) could not be parsed."
            )

    return cleaned, "No automatic change was applied; this recommendation requires manual review."
