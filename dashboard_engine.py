"""
dashboard_engine.py
--------------------
Builds the data behind Statify's field-adaptive Dashboard page.

Given a dataset and the field the user selected (Sales, Finance, Health,
Agriculture, Demography, or General), this module works out:

  - headline KPI numbers, formatted for a big-number display (e.g. "438K")
  - a set of ranking "panels" (bar / horizontal-bar / donut) covering each
    detected performer role against the field's main measure
  - a trend panel over time, if a usable date/period column exists

It reuses performance_engine's role detection, ranking logic (including its
handling of "stock" measures like population and Demography's aggregate-row
filtering), and date/period detection, so the Dashboard and Performance
Intelligence pages always agree on what a result means.
"""

import itertools

import pandas as pd

from performance_engine import (
    detect_roles,
    is_average_role,
    _looks_like_identifier,
    build_ranking_table,
    find_date_column,
    parse_date_series,
    looks_like_year_column,
    filter_aggregate_rows,
    is_stock_role,
    pluralize,
)

# For each domain, the order of preference for which group role and value
# role should headline the top ranking / donut breakdown.
ROLE_PRIORITY = {
    "🛒 Sales / E-commerce & Retail": (
        ["product", "salesperson", "customer", "region"],
        ["revenue", "quantity"],
    ),
    "💰 Finance": (
        ["account", "category", "region"],
        ["amount", "transactions"],
    ),
    "🏥 Health": (
        ["treatment", "provider", "patient", "facility"],
        ["outcome", "cases"],
    ),
    "🌾 Agriculture": (
        ["crop", "region", "farmer"],
        ["yield", "area"],
    ),
    "👥 Demography": (
        ["group", "region"],
        ["population"],
    ),
    "🔬 General / Other": (
        ["category"],
        ["value"],
    ),
}

PANEL_KIND_CYCLE = ["hbar", "donut", "vbar"]


def _format_number(value):
    """Turn a raw number into a compact display string, e.g. 438491 -> '438K'."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1_000_000:
        text = f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{sign}{text}M"
    if value >= 1_000:
        text = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{sign}{text}K"
    if value.is_integer():
        return f"{sign}{int(value):,}"
    return f"{sign}{value:,.2f}"


def _ordered_roles(roles, domain, role_key_index):
    order = ROLE_PRIORITY.get(domain, (None, None))[role_key_index]
    if not order:
        return list(roles.items())
    ordered = [(r, roles[r]) for r in order if r in roles]
    ordered += [(r, c) for r, c in roles.items() if r not in order]
    return ordered



# ------------------------------------------------------------
# PERIOD HELPERS (trend, KPI change, insights)
# ------------------------------------------------------------

def _period_totals(source_df, date_col, value_series):
    """Total of value_series per period. Returns (series or None, is_year_only)."""
    if source_df.empty:
        return None, False

    is_year_only = looks_like_year_column(source_df[date_col])
    freq = "Y" if is_year_only else "M"
    periods = (
        parse_date_series(source_df[date_col])
        .dt.to_period(freq)
        .dt.to_timestamp()
    )
    frame = pd.DataFrame({"period": periods, "value": value_series}).dropna()
    if frame.empty:
        return None, is_year_only

    return frame.groupby("period")["value"].sum().sort_index(), is_year_only


def _delta_from_series(series, is_year_only):
    """(fractional change, 'month'|'year') of latest period vs the one before."""
    if series is None or len(series) < 2:
        return None
    prev, last = series.iloc[-2], series.iloc[-1]
    if pd.isna(prev) or pd.isna(last) or prev == 0:
        return None
    return (float((last - prev) / abs(prev)), "year" if is_year_only else "month")


def _period_label(timestamp, is_year_only):
    return timestamp.strftime("%Y") if is_year_only else timestamp.strftime("%b %Y")


def _build_insights(
    full_table, group_col, value_col, trend, is_year_only, max_insights=6
):
    """Plain-language observations drawn from the ranking and the trend."""
    insights = []
    period_word = "year" if is_year_only else "month"

    # -- concentration: how much of the total do the leaders account for?
    if full_table is not None and len(full_table) >= 2:
        values = full_table[value_col]
        total = values.sum()
        if (values >= 0).all() and total > 0:
            top_name = full_table[group_col].iloc[0]
            insights.append(
                f"**{top_name}** accounts for {values.iloc[0] / total:.0%} "
                f"of total {value_col}."
            )
            if len(full_table) >= 5:
                insights.append(
                    f"The top 3 {group_col} entries together make up "
                    f"{values.head(3).sum() / total:.0%} of total {value_col}."
                )

    # -- trend: peak, latest movement, steepest changes, unusual periods
    if trend is not None and len(trend) >= 3:
        label = lambda ts: _period_label(ts, is_year_only)
        peak_ts = trend.idxmax()

        if peak_ts == trend.index[-1]:
            insights.append(
                f"{value_col} is at its highest point in the latest "
                f"{period_word} ({label(peak_ts)})."
            )
        else:
            insights.append(
                f"{value_col} peaked in **{label(peak_ts)}** at "
                f"{_format_number(trend.max())}."
            )

        delta = _delta_from_series(trend, is_year_only)
        if delta is not None:
            pct, _ = delta
            word = "rose" if pct >= 0 else "fell"
            insights.append(
                f"In the latest {period_word} ({label(trend.index[-1])}), "
                f"{value_col} {word} {abs(pct):.1%} versus the previous {period_word}."
            )

        if len(trend) >= 4:
            changes = trend.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
            if not changes.empty:
                worst, best = changes.idxmin(), changes.idxmax()
                if changes[worst] <= -0.10:
                    insights.append(
                        f"Steepest drop: **{label(worst)}**, down "
                        f"{abs(changes[worst]):.0%} from the {period_word} before."
                    )
                if changes[best] >= 0.10 and best != worst:
                    insights.append(
                        f"Strongest growth: **{label(best)}**, up "
                        f"{changes[best]:.0%} from the {period_word} before."
                    )

        if len(trend) >= 8 and trend.std() > 0:
            z_scores = (trend - trend.mean()) / trend.std()
            outlier = z_scores.abs().idxmax()
            if abs(z_scores[outlier]) >= 2:
                direction = "well above" if z_scores[outlier] > 0 else "well below"
                insights.append(
                    f"**{label(outlier)}** stands out as unusual, sitting "
                    f"{direction} the typical {period_word} level."
                )

    return insights[:max_insights]


# ------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------

def get_filter_spec(df, selected_domain, max_options=300):
    """
    Work out which filters make sense for this dataset.

    Returns {"date": {...} or None, "categories": [{role, column, options}]}.
    Date bounds are plain ints for year-only columns, datetime.date otherwise,
    so they can be handed straight to a Streamlit slider.
    """
    spec = {"date": None, "categories": []}
    if df is None or df.empty:
        return spec

    group_roles, _ = detect_roles(df, selected_domain)

    date_col = find_date_column(df)
    if date_col is not None:
        parsed = parse_date_series(df[date_col]).dropna()
        if not parsed.empty:
            is_year_only = bool(looks_like_year_column(df[date_col]))
            date_min = int(parsed.min().year) if is_year_only else parsed.min().date()
            date_max = int(parsed.max().year) if is_year_only else parsed.max().date()
            # Compare the values the slider will actually receive. Two different
            # timestamps can collapse to the same day (or year), and Streamlit
            # raises StreamlitInvalidMinMaxError unless min < max.
            if date_min < date_max:
                spec["date"] = {
                    "column": date_col,
                    "is_year_only": is_year_only,
                    "min": date_min,
                    "max": date_max,
                }

    for role, col in list(group_roles.items())[:4]:
        options = sorted(df[col].dropna().astype(str).unique())
        if 2 <= len(options) <= max_options:
            spec["categories"].append({"role": role, "column": col, "options": options})

    return spec


def apply_filters(df, spec, date_range=None, category_selections=None):
    """
    Return df narrowed by the chosen date range and category selections.

    A date range equal to the full span is treated as "no filter", so rows
    with a missing date aren't dropped unless the user actually narrows it.
    An empty category selection means "include everything".
    """
    filtered = df

    date_spec = spec.get("date") if spec else None
    if date_spec and date_range:
        start, end = date_range
        if (start, end) != (date_spec["min"], date_spec["max"]):
            parsed = parse_date_series(filtered[date_spec["column"]])
            if date_spec["is_year_only"]:
                mask = parsed.dt.year.between(int(start), int(end))
            else:
                mask = parsed.dt.normalize().between(
                    pd.Timestamp(start), pd.Timestamp(end)
                )
            filtered = filtered[mask.fillna(False)]

    for column, selected in (category_selections or {}).items():
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]

    return filtered



# ------------------------------------------------------------
# COMPOSITION PANELS (parts of a whole)
# ------------------------------------------------------------
# Many datasets hold columns that are slices of one whole: share columns that
# add up to ~100% (e.g. Drive / Transit / Walk / Bike shares) or counts that add
# up to a total column (Men + Women = TotalPop). Ranking by category can never
# show those, so they get their own "makeup" panels.

def _share_scale(series_list):
    top = max(float(s.max()) for s in series_list)
    return 1.0 if top <= 1.0001 else 100.0


def _detect_share_sets(df, max_window=10):
    numeric = list(df.select_dtypes(include="number").columns)
    if len(df) == 0:
        return []

    def candidate(col):
        obs = df[col].replace([float("inf"), float("-inf")], float("nan")).dropna()
        return (
            len(obs) >= 0.3 * len(df)
            and obs.min() >= 0
            and obs.max() <= 100.0001
            and obs.nunique() >= 3
        )

    # Runs of neighbouring candidate columns: parts of a whole sit side by side.
    runs, current = [], []
    for col in df.columns:
        if col in numeric and candidate(col):
            current.append(col)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    found = []
    for run in runs:
        i = 0
        while i < len(run) - 1:
            best = 0
            for length in range(2, min(max_window, len(run) - i) + 1):
                window = run[i:i + length]
                scale = _share_scale([df[c] for c in window])
                sums = df[window].sum(axis=1, min_count=length)
                valid = sums.dropna()
                if len(valid) < 0.5 * len(df):
                    continue
                exact = ((valid - scale).abs() <= 0.03 * scale).mean() >= 0.9
                # "Partial" wholes: the listed parts never exceed the total and
                # usually cover most of it (e.g. share columns that omit a
                # small "other" group).
                partial = (
                    ((valid >= 0.90 * scale) & (valid <= 1.01 * scale)).mean() >= 0.9
                    and valid.mean() >= 0.93 * scale
                )
                if exact or partial:
                    best = length
            if best:
                found.append(run[i:i + best])
                i += best
            else:
                i += 1
    return found


def _detect_count_sets(df, max_candidates=15):
    """Pairs of whole-number columns that add up to a third column (Men + Women = Total)."""
    candidates = []
    for col in df.select_dtypes(include="number").columns:
        obs = df[col].replace([float("inf"), float("-inf")], float("nan")).dropna()
        if len(obs) < 0.3 * len(df) or obs.min() < 0 or obs.nunique() < 3:
            continue
        if not bool((obs % 1 == 0).all()):
            continue
        if _looks_like_identifier(col, obs):
            continue
        candidates.append(col)
    candidates = candidates[:max_candidates]

    found = []
    for total in candidates:
        for a, b in itertools.combinations([c for c in candidates if c != total], 2):
            frame = df[[a, b, total]].dropna()
            if len(frame) < 0.3 * len(df) or frame[total].sum() == 0:
                continue
            gap = (frame[a] + frame[b] - frame[total]).abs()
            if (gap <= 0.5).mean() >= 0.95 and frame[a].sum() > 0 and frame[b].sum() > 0:
                found.append((total, [a, b]))
    return found


def _composition_panels(df, weight_col=None, max_panels=6):
    panels = []

    for total, parts in _detect_count_sets(df):
        table = pd.DataFrame({
            "Category": parts,
            "Total": [float(df[p].sum()) for p in parts],
        })
        panels.append({
            "title": f"{' vs '.join(parts)} (of {total})",
            "kind": "donut",
            "table": table,
        })

    used = {c for _, parts in _detect_count_sets(df) for c in parts}
    shares = [w for w in _detect_share_sets(df) if not (set(w) & used)]
    shares.sort(key=len, reverse=True)

    for window in shares:
        scale = _share_scale([df[c] for c in window])
        frame = df[window].copy()
        if weight_col is not None and weight_col in df.columns:
            weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
            mask = frame.notna().all(axis=1) & (weights > 0)
            if mask.any():
                values = [
                    float((frame.loc[mask, c] * weights[mask]).sum() / weights[mask].sum())
                    for c in window
                ]
                how = f"weighted by {weight_col}"
            else:
                values = [float(frame[c].mean()) for c in window]
                how = "average"
        else:
            values = [float(frame[c].mean()) for c in window]
            how = "average"

        unit = "%" if scale == 100.0 else "share"
        labels = list(window)
        remainder = scale - sum(values)
        if remainder > 0.02 * scale:
            labels.append("Other / not listed")
            values = values + [remainder]
        table = pd.DataFrame({
            "Category": labels,
            f"Share ({unit})": [round(v, 2) for v in values],
        })
        panels.append({
            "title": f"{', '.join(window)} ({how})",
            "kind": "donut" if len(labels) <= 7 else "hbar",
            "table": table,
        })

    return panels[:max_panels]


def generate_dashboard(df, selected_domain, top_n=6):
    """
    Build the data for the field-adaptive Dashboard page.

    Returns a dict with:
        partial_note : str warning that the latest month is incomplete, or None
        kpis     : list of (label, formatted_value, delta) headline numbers;
                   delta is None or (fractional_change, "month"|"year")
                   comparing the latest period with the one before it
        headline : str describing the top performer, or None
        insights : list of plain-language observations
        panels   : list of {title, kind, table} ranking panels
                   kind is one of "hbar", "vbar", "donut"; table has two
                   columns [category_col, value_col]
        trend    : {title, table} with table columns [Period, value_col],
                   or None if no usable date column was found
    """
    result = {
        "kpis": [], "headline": None, "insights": [], "panels": [],
        "trend": None, "partial_note": None,
    }

    if df is None or df.empty:
        return result

    group_roles, value_roles = detect_roles(df, selected_domain)

    ordered_groups = _ordered_roles(group_roles, selected_domain, 0)
    ordered_values = _ordered_roles(value_roles, selected_domain, 1)

    date_col = find_date_column(df)

    # Rows to use for anything summed across groups per period. For
    # Demography this drops aggregate rows such as "World" so real
    # countries aren't double-counted alongside their own total.
    period_source = df
    if "region" in group_roles:
        period_source = filter_aggregate_rows(
            df, selected_domain, "region", group_roles["region"]
        )

    # ---- KPIs ------------------------------------------------------------
    result["kpis"].append(("Total Records", _format_number(len(df)), None))

    for role, col in ordered_values:
        series, year_only = (None, False)
        if date_col is not None:
            series, year_only = _period_totals(period_source, date_col, period_source[col])
        delta = _delta_from_series(series, year_only)

        if is_stock_role(role) and series is not None and len(series) >= 1:
            # A snapshot measure: report its latest value, not a sum of history.
            result["kpis"].append(
                (f"{col} (latest)", _format_number(series.iloc[-1]), delta)
            )
        elif is_average_role(role):
            result["kpis"].append(
                (f"Average {col}", _format_number(df[col].mean()), None)
            )
        else:
            result["kpis"].append(
                (f"Total {col}", _format_number(df[col].sum()), delta)
            )

    for role, col in ordered_groups[:2]:
        counted = filter_aggregate_rows(df, selected_domain, role, col)
        result["kpis"].append(
            (f"Unique {pluralize(role.title())}", _format_number(counted[col].nunique()), None)
        )
    result["kpis"] = result["kpis"][:4]

    primary_value_role, primary_value_col = (
        ordered_values[0] if ordered_values else (None, None)
    )

    # ---- Ranking panels: one per group role, against the primary value -
    panel_index = 0
    primary_group_col = None
    for group_role, group_col in ordered_groups:
        if primary_value_col:
            table = build_ranking_table(
                df, selected_domain, group_role, group_col,
                primary_value_role, primary_value_col, top_n,
            )
            value_col = primary_value_col
        else:
            table = df[group_col].value_counts().head(top_n).reset_index()
            table.columns = [group_col, "Records"]
            value_col = "Records"

        if table.empty:
            continue

        result["panels"].append({
            "title": (f"Average {value_col} by {group_col}" if is_average_role(primary_value_role) else f"{value_col} by {group_col}"),
            "kind": "hbar" if is_average_role(primary_value_role) else PANEL_KIND_CYCLE[panel_index % len(PANEL_KIND_CYCLE)],
            "table": table,
        })
        panel_index += 1

        if panel_index == 1:
            primary_group_col = group_col
            primary_group_role = group_role
            top_row = table.iloc[0]
            result["headline"] = (
                f"{top_row[group_col]} leads on {value_col} "
                f"with {top_row[value_col]:,.2f}"
            )

    # ---- Extra panels for secondary measures, using the primary group --
    if ordered_groups and len(ordered_values) > 1:
        secondary_group_role, secondary_group_col = ordered_groups[0]
        for role, value_col in ordered_values[1:]:
            table = build_ranking_table(
                df, selected_domain, secondary_group_role, secondary_group_col,
                role, value_col, top_n,
            )
            if table.empty:
                continue
            result["panels"].append({
                "title": (f"Average {value_col} by {secondary_group_col}" if is_average_role(role) else f"{value_col} by {secondary_group_col}"),
                "kind": "hbar" if is_average_role(role) else PANEL_KIND_CYCLE[panel_index % len(PANEL_KIND_CYCLE)],
                "table": table,
            })
            panel_index += 1

    # ---- Composition panels: columns that are slices of a whole ----------
    try:
        weight_col = value_roles.get("population")
        result["panels"].extend(_composition_panels(df, weight_col))
    except Exception:
        pass  # a failed detection must never take the dashboard down

    # ---- Trend panel -----------------------------------------------------
    trend_series, is_year_only = None, False
    if date_col is not None and primary_value_col is not None:
        trend_series, is_year_only = _period_totals(
            period_source, date_col, period_source[primary_value_col]
        )
        if trend_series is not None and len(trend_series) >= 2:
            trend_table = trend_series.reset_index()
            trend_table.columns = ["Period", primary_value_col]
            result["trend"] = {
                "title": f"{primary_value_col} by {'Year' if is_year_only else 'Month'}",
                "table": trend_table,
            }
        else:
            trend_series = None

    # ---- Is the latest month incomplete? ---------------------------------
    if date_col is not None and not is_year_only:
        last_date = parse_date_series(period_source[date_col]).max()
        if pd.notna(last_date) and last_date.day < last_date.days_in_month * 0.8:
            result["partial_note"] = (
                f"Data ends on {last_date:%d %b %Y}, so the latest month is "
                f"incomplete and its change versus the previous month may look "
                f"larger than it really is."
            )

    # ---- Insights ----------------------------------------------------------
    full_table = None
    if primary_group_col is not None and primary_value_col is not None:
        full_table = build_ranking_table(
            df, selected_domain, primary_group_role, primary_group_col,
            primary_value_role, primary_value_col, max(len(df), 1),
        )
    if primary_value_col is not None:
        result["insights"] = _build_insights(
            full_table, primary_group_col, primary_value_col,
            trend_series, is_year_only,
        )

    return result
