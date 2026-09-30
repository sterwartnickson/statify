"""
performance_engine.py
----------------------
Domain-adaptive performance intelligence for Statify.

Given a dataset and a domain the user selects (Sales, Finance, Health,
Agriculture, Demography, or General), this module:

  1. Detects which columns play which "role" for that field
     (e.g. in Sales: product, salesperson, customer, revenue, quantity).
  2. Builds ranked "top performer" tables from those roles
     (e.g. top products by revenue, salespeople by number of customers).
  3. Writes short, plain-language findings describing the results.

It is intentionally driven by keyword matching against column names rather
than hard-coded column names, so it adapts to differently-named datasets
within the same field.
"""

import pandas as pd


# ------------------------------------------------------------
# DOMAIN ROLE VOCABULARIES
# ------------------------------------------------------------
# "groups" = categorical/text columns that identify a performer
# "values" = numeric columns that measure performance
# Keywords are matched case-insensitively against column names.

DOMAIN_ROLES = {
    "🛒 Sales / E-commerce & Retail": {
        "groups": {
            "product": ["product", "item", "sku", "goods"],
            "salesperson": ["salesperson", "sales rep", "sales_rep", "rep", "agent", "employee", "seller"],
            "customer": ["customer", "client", "buyer"],
            "region": ["region", "branch", "store", "location", "area", "territory"],
        },
        "values": {
            "revenue": ["revenue", "sales", "amount", "total", "price", "value"],
            "quantity": ["quantity", "qty", "units", "count", "volume"],
        },
    },
    "💰 Finance": {
        "groups": {
            "account": ["account", "client", "customer"],
            "category": ["category", "type", "expense", "income", "class"],
            "region": ["region", "branch", "location"],
        },
        "values": {
            "amount": ["amount", "value", "balance", "revenue", "profit", "expense", "cost"],
            "transactions": ["transaction", "txn", "count"],
        },
    },
    "🏥 Health": {
        "groups": {
            "patient": ["patient", "subject", "case"],
            "treatment": ["treatment", "procedure", "drug", "therapy", "medication"],
            "provider": ["doctor", "physician", "provider", "clinician", "nurse"],
            "facility": ["hospital", "clinic", "facility", "ward"],
        },
        "values": {
            "outcome": ["outcome", "result", "score", "recovery", "improvement"],
            "cases": ["cases", "count", "visits", "admissions"],
        },
    },
    "🌾 Agriculture": {
        "groups": {
            "crop": ["crop", "produce", "commodity", "variety"],
            "farmer": ["farmer", "grower", "producer"],
            "region": ["region", "farm", "district", "field", "location", "zone"],
        },
        "values": {
            "yield": ["yield", "production", "output", "harvest"],
            "area": ["area", "hectare", "acre"],
        },
    },
    "👥 Demography": {
        "groups": {
            "group": ["group", "category", "segment", "cohort", "age group", "age_group"],
            "region": [
                "region", "area", "district", "province", "country",
                "entity", "nation", "location", "state", "city", "code",
            ],
        },
        "values": {
            "population": ["population", "pop", "count", "total"],
        },
    },
    "🔬 General / Other": {
        "groups": {
            "category": ["category", "group", "name", "type", "label"],
        },
        "values": {
            "value": ["value", "amount", "score", "total", "count"],
        },
    },
}

FALLBACK_DOMAIN = "🔬 General / Other"

# Actor -> target pairs worth cross-ranking (e.g. "which salesperson
# reached the most distinct customers"). Only used when both roles were
# actually detected for the selected domain.
CROSS_ROLE_PAIRS = [
    ("salesperson", "customer"),
    ("salesperson", "product"),
    ("provider", "patient"),
    ("farmer", "crop"),
    ("account", "category"),
]

# "Stock" measures are snapshots (population at a point in time), not
# totals that should be added up across every row of history. Everything
# else (revenue, yield, outcome...) is a "flow" measure and gets summed
# as normal. If a stock role has a usable date/period column, rankings use
# each group's most recent value instead of a sum across all periods.
STOCK_VALUE_ROLES = {"population"}

# Used only when a dataset has NO country-code column to rely on. Exact
# names of common aggregates, plus a few phrases that only ever appear in
# aggregate labels. (Deliberately not bare substrings like "africa" or
# "america", which would also match South Africa, American Samoa, etc.)
AGGREGATE_EXACT_NAMES = {
    "world", "africa", "asia", "europe", "oceania", "americas",
    "north america", "south america", "latin america and the caribbean",
    "northern america", "european union", "sub-saharan africa",
    "middle east and north africa",
}
AGGREGATE_PHRASE_HINTS = [
    "income", "developed", "developing", "countries", "regions",
    "(un)", "(wb)", "(who)", "(fao)", "(ilo)", "(unicef)",
]

DATE_KEYWORDS = ["date", "day", "month", "year", "period", "time", "week", "quarter"]


def pluralize(word):
    """Simple English plural for role names (category -> categories)."""
    word = str(word)
    lower = word.lower()
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return word[:-1] + "ies"
    if lower.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def is_stock_role(value_role):
    """True if a value role represents a snapshot rather than a running total."""
    return value_role in STOCK_VALUE_ROLES


# ------------------------------------------------------------
# COLUMN MATCHING
# ------------------------------------------------------------

def _score_column(column_name, keywords):
    name = str(column_name).strip().lower()
    best = 0
    for kw in keywords:
        kw = kw.lower()
        if name == kw:
            best = max(best, 100)
        elif name.replace("_", " ") == kw:
            best = max(best, 95)
        elif kw in name:
            best = max(best, 60 + len(kw))
    return best


def _best_match(columns, keywords, exclude=None):
    exclude = exclude or set()
    best_col, best_score = None, 0
    for col in columns:
        if col in exclude:
            continue
        score = _score_column(col, keywords)
        if score > best_score:
            best_col, best_score = col, score
    return best_col if best_score > 0 else None


def detect_roles(df, domain):
    """
    Public helper: detect which columns play which role for a domain.

    Returns (group_roles, value_roles), each a dict of role name -> column
    name. Used by both the Performance Intelligence page and the Dashboard
    page so they agree on what "the product", "the revenue", etc. mean.
    """
    return _build_role_map(df, domain)


_ID_HINTS = ("id", "code", "tract", "zip", "postal", "phone", "index", "key", "uuid")
_ERR_HINTS = ("err", "error", "std", "se_", "_se", "margin", "moe")
_VALUE_PREFERENCE = (
    "income", "revenue", "sales", "price", "amount", "salary", "wage", "cost",
    "score", "rate", "poverty", "unemployment", "yield", "age", "weight",
    "population", "pop", "total",
)


def _looks_like_identifier(name, series):
    lowered = str(name).strip().lower()
    if any(h == lowered or lowered.endswith(h) or lowered.startswith(h) for h in _ID_HINTS):
        return True
    observed = series.dropna()
    if observed.empty:
        return False
    # Near-unique whole numbers look like row IDs; near-unique decimals
    # (incomes, prices, measurements) are ordinary continuous data.
    if not bool((observed % 1 == 0).all()):
        return False
    near_unique = observed.nunique() >= 0.95 * len(observed)
    # A running row number is a *dense* sequence (1..N); whole-dollar incomes
    # are near-unique too but spread over a wide range, so they're not IDs.
    dense = (observed.max() - observed.min() + 1) <= 1.5 * observed.nunique()
    return near_unique and dense


def _generic_group_columns(df, exclude, limit=2):
    """Text columns that make sensible chart categories (2-30 distinct values)."""
    chosen = []
    for col in df.columns:
        if col in exclude or pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        unique = series.astype(str).nunique()
        if not (2 <= unique <= 30) or unique > 0.5 * len(series):
            continue
        # Skip a column that just repeats one we already picked (County / Borough).
        duplicate = any(
            df.groupby([col, other], dropna=True).ngroups == unique == df[other].nunique()
            for other in chosen
        )
        if duplicate:
            continue
        chosen.append(col)
        if len(chosen) == limit:
            break
    return chosen


def _generic_value_columns(df, exclude, limit):
    """Numeric measures worth averaging per group (IDs and error columns skipped)."""
    candidates = []
    for col in df.select_dtypes(include="number").columns:
        if col in exclude:
            continue
        lowered = str(col).lower()
        if any(h in lowered for h in _ERR_HINTS):
            continue
        series = df[col].replace([float("inf"), float("-inf")], float("nan")).dropna()
        if series.nunique() < 5 or series.size < 0.3 * len(df):
            continue
        if _looks_like_identifier(col, series):
            continue
        rank = next(
            (i for i, hint in enumerate(_VALUE_PREFERENCE) if hint in lowered),
            len(_VALUE_PREFERENCE),
        )
        candidates.append((rank, list(df.columns).index(col), col))
    candidates.sort()
    return [col for _, _, col in candidates[:limit]]


def _build_role_map(df, domain):
    group_roles, value_roles = _keyword_role_map(df, domain)

    # Fallback so any dataset gets a useful dashboard, even when column names
    # match none of the domain keywords (e.g. County / Borough / Income).
    if not group_roles:
        for col in _generic_group_columns(df, exclude=set()):
            group_roles[str(col).strip().lower()] = col

    if len(value_roles) < 2:
        needed = 2 - len(value_roles)
        for col in _generic_value_columns(df, exclude=set(value_roles.values()), limit=needed):
            value_roles[f"avg_{str(col).strip().lower()}"] = col

    return group_roles, value_roles


def is_average_role(value_role):
    """Fallback measures are averaged per group, not summed."""
    return str(value_role).startswith("avg_")


def _keyword_role_map(df, domain):
    roles = DOMAIN_ROLES.get(domain, DOMAIN_ROLES[FALLBACK_DOMAIN])

    numeric_cols = list(df.select_dtypes(include="number").columns)
    text_cols = [c for c in df.columns if c not in numeric_cols]

    group_roles = {}
    used_text = set()
    for role, keywords in roles.get("groups", {}).items():
        col = _best_match(text_cols, keywords, exclude=used_text)
        if col:
            group_roles[role] = col
            used_text.add(col)

    value_roles = {}
    used_numeric = set()
    for role, keywords in roles.get("values", {}).items():
        col = _best_match(numeric_cols, keywords, exclude=used_numeric)
        if col:
            value_roles[role] = col
            used_numeric.add(col)

    return group_roles, value_roles


# ------------------------------------------------------------
# DATE / PERIOD DETECTION
# ------------------------------------------------------------
# Shared by the Dashboard's trend panel and by the "latest value" logic
# for stock measures below.

def looks_like_year_column(series):
    """True for a numeric column of bare calendar years (1950, 2024...).

    Naively handing such a column to pd.to_datetime treats each number as
    nanoseconds since epoch, collapsing every row onto the same date — so
    this needs to be checked before that happens.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    numeric = series.dropna()
    if numeric.empty:
        return False
    return numeric.between(1000, 2200).mean() > 0.9


def parse_date_series(series):
    """Parse a column into naive datetimes, handling bare-year columns correctly.

    Never raises: anything that can't be parsed comes back as NaT. Timezone-aware
    values are converted to UTC and made naive so they can be compared with
    plain dates and slider values.
    """
    if looks_like_year_column(series):
        try:
            return pd.to_datetime(
                series.astype("Int64").astype(str), format="%Y", errors="coerce"
            )
        except (TypeError, ValueError):
            pass  # e.g. 2020.5 -- fall through to the generic path

    # A plain number column (day 1-31, month 1-12, hours, IDs, counts...) is
    # NOT a date. pd.to_datetime would read each number as nanoseconds since
    # 1970 and collapse every row onto 1970-01-01, so only accept numbers that
    # are plausible Unix timestamps (seconds or milliseconds).
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").replace(
            [float("inf"), float("-inf")], float("nan")
        )
        observed = numeric.dropna()
        if observed.empty:
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        median = float(observed.abs().median())
        if 1e8 <= median < 1e11:
            unit = "s"
        elif 1e11 <= median < 1e14:
            unit = "ms"
        else:
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        try:
            return pd.to_datetime(numeric, unit=unit, errors="coerce")
        except (TypeError, ValueError, OverflowError):
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_localize(None)
    except (TypeError, ValueError, OverflowError):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")


def sanitize_dataframe(df):
    """
    Make an arbitrary uploaded table safe for every page of the app.

    - column names become unique, non-empty strings (duplicate or numeric
      headers otherwise break groupby / selection code)
    - +/-inf in numeric columns becomes NaN
    - cells holding lists / dicts / sets become text (they are unhashable)
    - timezone-aware datetime columns become timezone-naive
    - boolean columns become "True"/"False" categories
    """
    df = df.copy()

    used, names = set(), []
    for position, col in enumerate(df.columns):
        base = str(col).strip() or f"Column {position + 1}"
        name, n = base, 1
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        names.append(name)
    df.columns = names

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols):
        df[numeric_cols] = df[numeric_cols].replace([float("inf"), float("-inf")], float("nan"))

    for col in df.columns:
        series = df[col]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            df[col] = series.dt.tz_localize(None)
        elif pd.api.types.is_bool_dtype(series):
            # True/False columns are grouping variables, not measurements;
            # pandas treats them as "numeric", which breaks IQR/summary maths.
            df[col] = series.map({True: "True", False: "False"}).astype(object)
        elif series.dtype == object:
            has_container = series.map(
                lambda v: isinstance(v, (list, dict, set, tuple))
            ).any()
            if has_container:
                df[col] = series.map(
                    lambda v: str(v) if isinstance(v, (list, dict, set, tuple)) else v
                )

    return df


def find_date_column(df):
    """Best-effort detection of a usable date/period column."""
    candidates = [
        col for col in df.columns
        if any(kw in str(col).lower() for kw in DATE_KEYWORDS)
    ]

    for col in candidates:
        try:
            parsed = parse_date_series(df[col])
        except (ValueError, TypeError):
            continue
        # Needs real variety: a column where every row parses to the same
        # moment (or that is mostly unparseable) is not a usable date axis.
        if parsed.notna().mean() >= 0.6 and parsed.dropna().nunique() >= 2:
            return col

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dropna().nunique() >= 2:
                return col

    return None


# ------------------------------------------------------------
# DEMOGRAPHY AGGREGATE-ROW FILTERING
# ------------------------------------------------------------

def _find_code_column(df):
    """Best-effort: a short ISO-style country-code column, if present."""
    for col in df.columns:
        name = str(col).strip().lower()
        if name in ("code", "iso code", "iso_code", "country code", "country_code"):
            return col
    return None


def filter_aggregate_rows(df, domain, group_role, group_col):
    """
    For Demography's 'region' ranking, drop rows that represent aggregate
    entities (e.g. 'World', 'Asia (UN)', 'Less developed regions') rather
    than a single real place, so a country ranking isn't dominated by
    totals-of-totals.

    If the dataset has a country-code column, real places are the rows with
    a proper 3-letter code and the code alone decides. Otherwise it falls
    back to recognising well-known aggregate names. Never used outside
    Demography's 'region' role, and never allowed to filter a dataset down
    to nothing.
    """
    if domain != "👥 Demography" or group_role != "region":
        return df

    code_col = _find_code_column(df)
    if code_col is not None:
        codes = df[code_col].astype(str).str.strip()
        is_aggregate = ~codes.str.match(r"^[A-Za-z]{3}$", na=False)
    else:
        names = df[group_col].astype(str).str.strip().str.lower()
        is_aggregate = names.isin(AGGREGATE_EXACT_NAMES) | names.apply(
            lambda n: any(hint in n for hint in AGGREGATE_PHRASE_HINTS)
        )

    filtered = df[~is_aggregate]
    return filtered if not filtered.empty else df


# ------------------------------------------------------------
# RANKING TABLES
# ------------------------------------------------------------

def _latest_value_per_group(df, group_col, value_col, date_col):
    """One row per group, holding its value as of the most recent period."""
    working = df.dropna(subset=[group_col, value_col, date_col]).copy()
    if working.empty:
        return working

    working["_statify_parsed_date"] = parse_date_series(working[date_col])
    working = working.dropna(subset=["_statify_parsed_date"])
    if working.empty:
        return working

    # Collapse duplicate (group, date) rows first, in case several rows
    # share the same group and period.
    collapsed = (
        working.groupby([group_col, "_statify_parsed_date"], dropna=True)[value_col]
        .sum()
        .reset_index()
    )
    latest_idx = collapsed.groupby(group_col)["_statify_parsed_date"].idxmax()
    return collapsed.loc[latest_idx, [group_col, value_col]]


def build_ranking_table(df, domain, group_role, group_col, value_role, value_col, top_n):
    """
    Shared ranking logic used by both Performance Intelligence and the
    Dashboard, so the two pages always agree on what "top X by Y" means.

    Handles two special cases so results stay meaningful:
      - Stock measures (see STOCK_VALUE_ROLES) use each group's most
        recent value instead of summing every historical row together,
        when a date/period column is available.
      - Demography's 'region' role excludes aggregate rows (World, Asia,
        income-group buckets) — see filter_aggregate_rows.
    """
    working_df = filter_aggregate_rows(df, domain, group_role, group_col)

    if is_stock_role(value_role):
        date_col = find_date_column(working_df)
        if date_col is not None:
            latest = _latest_value_per_group(working_df, group_col, value_col, date_col)
            if not latest.empty:
                table = (
                    latest.groupby(group_col, dropna=True)[value_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .reset_index()
                )
                table.columns = [group_col, value_col]
                return table

    grouped = (
        working_df.dropna(subset=[group_col, value_col])
        .groupby(group_col, dropna=True)[value_col]
    )
    aggregated = grouped.mean() if is_average_role(value_role) else grouped.sum()
    table = (
        aggregated
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    table.columns = [group_col, value_col]
    return table


def _rank_count_unique(df, group_col, target_col, top_n, label):
    table = (
        df.dropna(subset=[group_col, target_col])
        .groupby(group_col, dropna=True)[target_col]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    table.columns = [group_col, label]
    return table


def _rank_row_count(df, group_col, top_n, label="Records"):
    table = (
        df.dropna(subset=[group_col])
        .groupby(group_col, dropna=True)
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    table.columns = [group_col, label]
    return table


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------

def generate_performance_report(df, selected_domain, top_n=10):
    """
    Build a domain-adaptive performance intelligence report.

    Parameters
    ----------
    df : pandas.DataFrame
        The user's dataset.
    selected_domain : str
        The field the user picked (must match a key in DOMAIN_ROLES,
        falls back to General / Other if unrecognized).
    top_n : int
        How many rows to keep in each ranking table.

    Returns
    -------
    profile  : dict[str, str]   -- role name -> dataset column used
    sections : list[(str, pandas.DataFrame)] -- ranking tables to display
    findings : list[str]        -- plain-language summary bullets
    """

    if df is None or df.empty:
        return {}, [], ["No data is available to analyze."]

    group_roles, value_roles = _build_role_map(df, selected_domain)

    if not group_roles:
        return {}, [], [
            f"No performance-related variables were confidently detected for "
            f"**{selected_domain}**. Try a different field, or rename columns "
            f"to something Statify can recognize (e.g. 'Product', "
            f"'Salesperson', 'Revenue')."
        ]

    profile = dict(group_roles)
    profile.update({f"{role} (measure)": col for role, col in value_roles.items()})

    sections = []
    findings = []

    if value_roles:
        for value_role, value_col in value_roles.items():
            for group_role, group_col in group_roles.items():
                table = build_ranking_table(
                    df, selected_domain, group_role, group_col,
                    value_role, value_col, top_n,
                )
                if table.empty:
                    continue
                title = f"🏆 Top {pluralize(group_role.title())} by {value_col}"
                sections.append((title, table))

                top_row = table.iloc[0]
                findings.append(
                    f"**{top_row[group_col]}** leads on {value_col} "
                    f"with a total of {top_row[value_col]:,.2f}."
                )
    else:
        # No numeric measure detected — fall back to frequency ranking.
        for group_role, group_col in group_roles.items():
            table = _rank_row_count(df, group_col, top_n)
            if table.empty:
                continue
            sections.append((f"🏆 Most Frequent {pluralize(group_role.title())}", table))
            top_row = table.iloc[0]
            findings.append(
                f"**{top_row[group_col]}** appears most often "
                f"({int(top_row['Records'])} records)."
            )

    for actor_role, target_role in CROSS_ROLE_PAIRS:
        if actor_role in group_roles and target_role in group_roles:
            actor_col = group_roles[actor_role]
            target_col = group_roles[target_role]
            if actor_col == target_col:
                continue

            label = f"Unique {target_col}"
            table = _rank_count_unique(df, actor_col, target_col, top_n, label)
            if table.empty:
                continue

            title = f"👤 Most Unique {pluralize(target_role.title())} per {actor_role.title()}"
            sections.append((title, table))

            top_row = table.iloc[0]
            findings.append(
                f"**{top_row[actor_col]}** reached the most distinct "
                f"{target_col} values ({int(top_row[label])})."
            )

    if not findings:
        findings.append(
            "Performance variables were detected, but no rankings could be "
            "computed. Check that the relevant columns contain valid, "
            "non-missing data."
        )

    return profile, sections, findings
