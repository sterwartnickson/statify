"""
STATIFY DOMAIN INTELLIGENCE ENGINE
Domain-aware profiling and data-quality rules for:
Sales/E-commerce & Retail, Finance, Demography, Health, Agriculture.

This module is deliberately conservative:
it flags suspicious observations and recommends actions rather than
silently deleting research/business records.
"""

import re
import numpy as np
import pandas as pd


DOMAIN_PROFILES = {
    "🛒 Sales / E-commerce & Retail": {
        "keywords": {
            "order": ["order", "order_id", "invoice", "transaction"],
            "customer": ["customer", "customer_id", "client", "buyer"],
            "product": ["product", "sku", "item", "category", "brand"],
            "quantity": ["quantity", "qty", "units", "items_sold"],
            "price": ["price", "unit_price", "selling_price", "cost"],
            "revenue": ["revenue", "sales", "sales_amount", "turnover", "amount"],
            "discount": ["discount", "discount_rate"],
            "return": ["return", "returned", "refund"],
            "date": ["date", "order_date", "sale_date", "transaction_date"],
        },
        "checks": [
            ("quantity", "negative", "Quantity should normally not be negative; investigate returns/refunds separately."),
            ("price", "negative", "Price/cost should normally not be negative."),
            ("revenue", "negative", "Revenue/sales amount below zero should be investigated."),
            ("discount", "range01", "Discount rates expressed as proportions should normally be between 0 and 1."),
            ("order", "duplicates", "Repeated order IDs may indicate duplicate records; verify before removing."),
        ],
    },
    "💰 Finance": {
        "keywords": {
            "transaction": ["transaction", "transaction_id", "txn", "reference"],
            "account": ["account", "account_id", "customer_account"],
            "income": ["income", "salary", "revenue", "earnings"],
            "expense": ["expense", "expenditure", "cost"],
            "balance": ["balance", "closing_balance", "opening_balance"],
            "amount": ["amount", "value", "payment", "deposit", "withdrawal"],
            "interest": ["interest", "interest_rate"],
            "date": ["date", "transaction_date", "posting_date", "value_date"],
        },
        "checks": [
            ("amount", "negative", "Negative transaction amounts may be valid for debits/refunds; classify them rather than deleting automatically."),
            ("interest", "range01", "Interest rates stored as proportions should normally be between 0 and 1."),
            ("transaction", "duplicates", "Repeated transaction/reference IDs should be investigated before deduplication."),
            ("balance", "numeric", "Balances should be numeric and should be checked for unexpected formatting or currency symbols."),
        ],
    },
    "👥 Demography": {
        "keywords": {
            "age": ["age", "age_years", "respondent_age"],
            "sex": ["sex", "gender", "men", "women"],
            "geography": ["tract", "censustract", "census", "borough", "county"],
            "birth": ["birth", "births", "dob", "date_of_birth"],
            "death": ["death", "deaths", "mortality"],
            "fertility": ["fertility", "children", "children_ever_born", "live_births"],
            "population": ["population", "pop", "totalpop", "household_size"],
            "migration": ["migration", "migrant", "moved"],
            "household": ["household", "household_id", "hhid"],
            "date": ["date", "survey_date", "interview_date", "census_date"],
        },
        "checks": [
            ("age", "age", "Age should be checked for impossible or implausible values; do not remove outliers automatically."),
            ("fertility", "nonnegative", "Birth/children counts should not be negative."),
            ("population", "nonnegative", "Population and household counts should not be negative."),
            ("birth", "date_logic", "Birth dates should not occur after interview/survey dates when both are available."),
            ("household", "duplicates", "Repeated household IDs may be legitimate in member-level data; do not deduplicate blindly."),
        ],
    },
    "🏥 Health": {
        "keywords": {
            "patient": ["patient", "patient_id", "participant", "subject"],
            "age": ["age", "age_years"],
            "sex": ["sex", "gender"],
            "diagnosis": ["diagnosis", "disease", "condition", "icd"],
            "weight": ["weight", "body_weight"],
            "height": ["height", "stature"],
            "bmi": ["bmi", "body_mass_index"],
            "bp": ["blood_pressure", "systolic", "diastolic", "bp"],
            "lab": ["glucose", "cholesterol", "hemoglobin", "haemoglobin", "creatinine"],
            "outcome": ["outcome", "mortality", "recovery", "status"],
            "date": ["date", "visit_date", "admission_date", "discharge_date"],
        },
        "checks": [
            ("age", "age", "Check age for impossible values; clinical or survey-specific limits should be configurable."),
            ("weight", "nonnegative", "Weight should not be negative."),
            ("height", "nonnegative", "Height should not be negative."),
            ("bmi", "nonnegative", "BMI should not be negative."),
            ("bp", "nonnegative", "Blood-pressure measurements should be numeric/non-negative after parsing."),
            ("patient", "duplicates", "Repeated patient IDs may represent repeated visits; do not deduplicate without a visit/design rule."),
        ],
    },
    "🌾 Agriculture": {
        "keywords": {
            "farm": ["farm", "farm_id", "field", "plot"],
            "crop": ["crop", "crop_type", "crop_name"],
            "area": ["area", "farm_size", "plot_size", "hectare", "hectares"],
            "yield": ["yield", "yield_per_ha", "productivity"],
            "production": ["production", "output", "harvest"],
            "rainfall": ["rainfall", "precipitation", "rain"],
            "fertilizer": ["fertilizer", "fertiliser", "fertilizer_amount"],
            "livestock": ["livestock", "cattle", "goats", "sheep", "poultry"],
            "season": ["season", "planting_season", "harvest_season"],
            "date": ["date", "planting_date", "harvest_date"],
        },
        "checks": [
            ("area", "nonnegative", "Farm/plot area should not be negative."),
            ("yield", "nonnegative", "Yield should not be negative."),
            ("production", "nonnegative", "Production/output should not be negative."),
            ("rainfall", "nonnegative", "Rainfall should not be negative."),
            ("fertilizer", "nonnegative", "Fertilizer quantity should not be negative."),
            ("farm", "duplicates", "Repeated farm/plot IDs may be valid for repeated seasons; verify study design first."),
        ],
    },
    "🔬 General / Other": {
        "keywords": {},
        "checks": [],
    },
}


def _normalise_name(name):
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _match_score(df, profile):
    scores = {}
    for role, aliases in profile["keywords"].items():
        for col in df.columns:
            normal = _normalise_name(col)
            if normal in aliases:
                scores[role] = scores.get(role, 0) + 2
            elif any(alias in normal for alias in aliases if len(alias) >= 4):
                scores[role] = scores.get(role, 0) + 1
    return scores


def detect_domain(df):
    """
    Return the most plausible domain, confidence, and evidence.
    Confidence is heuristic, not a scientific probability.
    """
    results = []

    for domain, profile in DOMAIN_PROFILES.items():
        if domain == "🔬 General / Other":
            continue
        scores = _match_score(df, profile)
        total = sum(scores.values())
        matched_roles = list(scores.keys())
        results.append({
            "Domain": domain,
            "Score": total,
            "Matched concepts": ", ".join(matched_roles) if matched_roles else "None",
        })

    results.sort(key=lambda x: x["Score"], reverse=True)

    if not results or results[0]["Score"] == 0:
        return {
            "domain": "🔬 General / Other",
            "confidence": "Low",
            "score": 0,
            "evidence": [],
            "ranking": pd.DataFrame(results),
        }

    top = results[0]
    confidence = "High" if top["Score"] >= 6 else "Moderate" if top["Score"] >= 3 else "Low"

    return {
        "domain": top["Domain"],
        "confidence": confidence,
        "score": top["Score"],
        "evidence": top["Matched concepts"].split(", ") if top["Matched concepts"] != "None" else [],
        "ranking": pd.DataFrame(results),
    }


def classify_variables(df, domain):
    """Classify columns using domain roles plus basic pandas data types."""
    profile = DOMAIN_PROFILES.get(domain, DOMAIN_PROFILES["🔬 General / Other"])
    rows = []

    for col in df.columns:
        normal = _normalise_name(col)
        matched_role = "Unclassified"

        for role, aliases in profile["keywords"].items():
            if normal in aliases or any(alias in normal for alias in aliases if len(alias) >= 4):
                matched_role = role.replace("_", " ").title()
                break

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            data_type = "Date/Time"
        elif pd.api.types.is_numeric_dtype(df[col]):
            data_type = "Numeric"
        else:
            data_type = "Categorical/Text"

        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique(dropna=True))

        rows.append({
            "Variable": col,
            "Domain Role": matched_role,
            "Data Type": data_type,
            "Missing": missing,
            "Unique Values": unique,
        })

    return pd.DataFrame(rows)


def _find_columns(df, aliases):
    found = []
    for col in df.columns:
        normal = _normalise_name(col)
        if normal in aliases or any(alias in normal for alias in aliases if len(alias) >= 4):
            found.append(col)
    return found


def domain_quality_checks(df, domain):
    """Run conservative domain-specific checks and return a review table."""
    profile = DOMAIN_PROFILES.get(domain, DOMAIN_PROFILES["🔬 General / Other"])
    findings = []

    for role, check_type, message in profile["checks"]:
        aliases = profile["keywords"].get(role, [])
        columns = _find_columns(df, aliases)

        for col in columns:
            series = df[col]

            if check_type == "negative":
                numeric = pd.to_numeric(series, errors="coerce")
                count = int((numeric < 0).sum())

            elif check_type == "nonnegative":
                numeric = pd.to_numeric(series, errors="coerce")
                count = int((numeric < 0).sum())

            elif check_type == "age":
                numeric = pd.to_numeric(series, errors="coerce")
                count = int(((numeric < 0) | (numeric > 120)).sum())

            elif check_type == "range01":
                numeric = pd.to_numeric(series, errors="coerce")
                count = int(((numeric < 0) | (numeric > 1)).sum())

            elif check_type == "duplicates":
                count = int(series.duplicated(keep=False).sum())

            elif check_type == "numeric":
                converted = pd.to_numeric(series, errors="coerce")
                nonblank = series.notna().sum()
                count = int(nonblank - converted.notna().sum())

            elif check_type == "date_logic":
                count = 0
                message = "Date-order checks require identifying the relevant date variables; review dates for logical ordering."

            else:
                count = 0

            status = "Review" if count > 0 else "Pass"

            findings.append({
                "Variable": col,
                "Check": check_type.replace("_", " ").title(),
                "Flagged Records": count,
                "Status": status,
                "Recommendation": message,
            })

    if not findings:
        return pd.DataFrame(columns=[
            "Variable", "Check", "Flagged Records", "Status", "Recommendation"
        ])

    return pd.DataFrame(findings)


def cleaning_recommendations(df, domain):
    """Generate human-readable recommendations without changing the dataset."""
    checks = domain_quality_checks(df, domain)
    recommendations = []

    for _, row in checks.iterrows():
        if row["Flagged Records"] > 0:
            recommendations.append({
                "Priority": "High" if row["Check"] in {"Negative", "Age"} else "Review",
                "Variable": row["Variable"],
                "Issue": row["Check"],
                "Records": int(row["Flagged Records"]),
                "Recommended Action": row["Recommendation"],
            })

    # Generic recommendations remain useful in every domain.
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing:
            recommendations.append({
                "Priority": "Review",
                "Variable": col,
                "Issue": "Missing values",
                "Records": missing,
                "Recommended Action": "Choose an imputation, recoding, or row-exclusion strategy based on the study/business context.",
            })

    return pd.DataFrame(recommendations)
