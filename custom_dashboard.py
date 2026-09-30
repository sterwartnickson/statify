"""
STATIFY CUSTOM DASHBOARD

Lets the user decide what a chart shows instead of the app guessing:
pick the column(s) to measure, how to group them, how to calculate
(sum, average, ...) and which chart to draw. Works on any dataset.

Two halves:
  * a pure engine  (build_chart_data / validate_for_chart / make_figure)
    that turns a small "spec" dictionary into a figure, and never raises
    anything except ChartSpecError for bad combinations, and
  * render_custom_dashboard(), the Streamlit page.
"""

import hashlib
import uuid

import numpy as np
import pandas as pd

from performance_engine import parse_date_series


class ChartSpecError(ValueError):
    """A chart request that cannot be drawn. The message is user-friendly."""


# ============================================================
# CONSTANTS
# ============================================================

AGGREGATIONS = [
    "Sum", "Average", "Median", "Count", "Distinct count",
    "Min", "Max", "Weighted average",
]
ADDITIVE = {"Sum", "Count"}          # safe to lump leftovers into "Other"
DATE_PARTS = ["Day", "Week", "Month", "Quarter", "Year"]
_PERIOD = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}

MAX_SERIES = 8
SAMPLE_LIMIT = 5000
AUTO_BIN_THRESHOLD = 60
RAW_CHARTS = {"Histogram", "Box plot", "Scatter"}

CHART_HELP = {
    "Bar": "Compare categories side by side.",
    "Horizontal bar": "Like a bar chart, best when category names are long.",
    "Line": "Show change across ordered values, such as dates.",
    "Area": "Like a line, filled in - good for totals over time.",
    "Pie": "Each category's share of the whole (few categories).",
    "Donut": "A pie with a hole - same idea, easier to read.",
    "Treemap": "Rectangles sized by value - good for many categories.",
    "Stacked bar": "Bars split into coloured parts (needs a second dimension).",
    "Grouped bar": "Bars side by side for each part (needs a second dimension).",
    "100% stacked bar": "Each bar scaled to 100% to compare mix, not size.",
    "Heatmap": "A colour grid of two dimensions.",
    "Histogram": "How the values of one numeric column are spread out.",
    "Box plot": "Spread, median and outliers of a number for each category.",
    "Scatter": "Relationship between two numeric columns.",
}
CHART_TYPES = list(CHART_HELP)

# Every colour here has at least 4:1 contrast against the dark page background
# and reads clearly with dark label text on top of it.
PALETTE = [
    "#8DB600", "#7873f5", "#ff6ec4", "#f59e0b", "#22d3ee",
    "#a3e635", "#fb7185", "#c084fc", "#facc15", "#34d399",
]
TEXT_LIGHT = "#f5f3ff"   # labels drawn on the page background
TEXT_DARK = "#12122b"    # labels drawn on top of coloured shapes
GRID = "rgba(230,230,240,0.18)"


def style_dark(fig, height=380):
    """
    Make every piece of text readable on the app's dark background.

    Sets explicit colours for titles, axes, ticks, legends and colour bars so
    the result does not depend on Streamlit's light/dark theme.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_LIGHT, size=13),
        title_font=dict(color=TEXT_LIGHT, size=16),
        legend=dict(font=dict(color=TEXT_LIGHT, size=12),
                    title_font=dict(color=TEXT_LIGHT), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1e1b4b", font=dict(color=TEXT_LIGHT)),
        margin=dict(l=8, r=8, t=56, b=8),
        height=height,
    )
    fig.update_xaxes(
        tickfont=dict(color=TEXT_LIGHT, size=12),
        title_font=dict(color=TEXT_LIGHT, size=13),
        gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
    )
    fig.update_yaxes(
        tickfont=dict(color=TEXT_LIGHT, size=12),
        title_font=dict(color=TEXT_LIGHT, size=13),
        gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
    )
    fig.update_coloraxes(colorbar=dict(
        tickfont=dict(color=TEXT_LIGHT), title_font=dict(color=TEXT_LIGHT)))
    return fig



# ============================================================
# COLUMN HELPERS
# ============================================================

def _is_number(series):
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


def classify_columns(df):
    """Return (numeric_columns, other_columns, datetime_columns)."""
    numeric, other, dates = [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            dates.append(col)
            other.append(col)
        elif _is_number(s):
            numeric.append(col)
        else:
            other.append(col)
    return numeric, other, dates


def find_date_like_text_columns(df, sample=500):
    """Text columns whose values mostly parse as dates."""
    found = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or _is_number(s):
            continue
        head = s.dropna().astype(str).head(sample)
        if head.nunique() < 2:
            continue
        try:
            parsed = pd.to_datetime(head, errors="coerce", utc=True)
        except Exception:
            continue
        if parsed.notna().mean() >= 0.8:
            found.append(col)
    return found


def _numeric_values(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def _need(df, *columns):
    for col in columns:
        if col is not None and col not in df.columns:
            raise ChartSpecError(
                f"The column '{col}' is no longer in the dataset. "
                "Pick your columns again."
            )


def _fmt(value):
    return f"{value:,.4g}"


# ============================================================
# GROUPING KEYS
# ============================================================

def _bin_numbers(numbers, bins=10):
    """Bucket numbers into equal-width ranges. Returns (labels, order)."""
    clean = numbers.dropna()
    if clean.nunique() < 2:
        labels = numbers.map(lambda v: _fmt(v) if pd.notna(v) else np.nan)
        return labels, [_fmt(v) for v in sorted(clean.unique())]

    cats = pd.cut(numbers, bins=bins, duplicates="drop")
    intervals = list(cats.cat.categories)
    mapping = {iv: f"{_fmt(iv.left)} to {_fmt(iv.right)}" for iv in intervals}
    labels = cats.astype(object).map(mapping)
    return labels, [mapping[iv] for iv in intervals]


def _make_key(df, col, spec, allow_bin=True):
    """
    Build the grouping labels for one column.
    Returns (labels, kind, order, note) where kind is
    'date' | 'number' | 'category'.
    """
    s = df[col]
    note = None

    if pd.api.types.is_datetime64_any_dtype(s) or spec.get("treat_as_dates"):
        parsed = parse_date_series(s)
        part = spec.get("date_part") or "Month"
        if part not in _PERIOD:
            part = "Month"
        labels = parsed.dt.to_period(_PERIOD[part]).dt.start_time
        return labels, "date", None, None

    if _is_number(s):
        numbers = _numeric_values(s)
        many = numbers.nunique() > AUTO_BIN_THRESHOLD
        if allow_bin and (spec.get("bin_numbers") or many):
            labels, order = _bin_numbers(numbers)
            if many and not spec.get("bin_numbers"):
                note = (
                    f"'{col}' has {numbers.nunique():,} different numbers, so "
                    "they were grouped into ranges."
                )
            return labels, "category", order, note
        labels = numbers.map(lambda v: f"{v:g}" if pd.notna(v) else np.nan)
        order = [f"{v:g}" for v in sorted(numbers.dropna().unique())]
        return labels, "number", order, None

    labels = s.astype(str).where(s.notna(), "(missing)")
    return labels, "category", None, None


def _series_key(df, col):
    """Grouping labels for the 'split by' column (always shown as text)."""
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.to_period("M").astype(str).where(s.notna(), "(missing)")
    if _is_number(s):
        numbers = _numeric_values(s)
        if numbers.nunique() > 20:
            labels, _ = _bin_numbers(numbers)
            return labels
        return numbers.map(lambda v: f"{v:g}" if pd.notna(v) else np.nan)
    return s.astype(str).where(s.notna(), "(missing)")


# ============================================================
# AGGREGATION
# ============================================================

def _aggregate(frame, keys, agg):
    """
    frame has key columns plus '_v' (numeric value, may be absent for row
    counts) and '_w' (weights, only for weighted averages).
    """
    grouped = frame.groupby(keys, sort=False, dropna=True, observed=True)

    if "_v" not in frame.columns:
        out = grouped.size()
    elif agg == "Count":
        out = grouped["_v"].count()
    elif agg == "Distinct count":
        out = grouped["_v"].nunique()
    elif agg == "Weighted average":
        work = frame.dropna(subset=["_v", "_w"])
        work = work[work["_w"] > 0].copy()
        if work.empty:
            raise ChartSpecError(
                "A weighted average needs a weight column with positive numbers."
            )
        work["_wv"] = work["_v"] * work["_w"]
        g = work.groupby(keys, sort=False, dropna=True, observed=True)
        out = g["_wv"].sum() / g["_w"].sum()
    else:
        method = {
            "Sum": "sum", "Average": "mean", "Median": "median",
            "Min": "min", "Max": "max",
        }.get(agg)
        if method is None:
            raise ChartSpecError(f"'{agg}' is not a supported calculation.")
        out = getattr(grouped["_v"], method)()

    out = out.rename("value").reset_index()
    return out


def _prepare_values(df, column, agg, weight):
    """Numeric value (+ weight) columns for the given calculation."""
    pieces = {}
    if column is not None:
        if agg in ("Count", "Distinct count"):
            pieces["_v"] = df[column]
        else:
            values = _numeric_values(df[column])
            if values.notna().sum() == 0:
                raise ChartSpecError(
                    f"'{column}' has no numbers to calculate with. "
                    "Choose a numeric column, or use 'Count'."
                )
            pieces["_v"] = values
    if agg == "Weighted average":
        if weight is None:
            raise ChartSpecError("Choose a weight column for the weighted average.")
        pieces["_w"] = _numeric_values(df[weight])
    return pieces


# ============================================================
# POST-PROCESSING SHARED BY ALL AGGREGATED CHARTS
# ============================================================

def _lump_or_drop(table, level, keep, agg, notes, what):
    """Keep the given labels of `level`; lump the rest into 'Other' if that is valid."""
    kept = table[table[level].isin(keep)]
    rest = table[~table[level].isin(keep)]
    if rest.empty:
        return table
    if agg in ADDITIVE:
        other_keys = [c for c in table.columns if c not in (level, "value")]
        if other_keys:
            other = rest.groupby(other_keys, as_index=False, sort=False)["value"].sum()
        else:
            other = pd.DataFrame({"value": [rest["value"].sum()]})
        other[level] = "Other"
        notes.append(f"Smaller {what} were combined into 'Other'.")
        return pd.concat([kept, other[table.columns]], ignore_index=True)
    notes.append(
        f"Showing the top {len(keep)} {what}; the rest are hidden because "
        f"a {agg.lower()} can't be combined into 'Other'."
    )
    return kept


def _finish(table, kind_x, order, spec, agg):
    notes = []
    has_series = "series" in table.columns

    if table.empty:
        raise ChartSpecError("There is nothing to show for these choices.")

    # Limit the number of series.
    if has_series:
        totals = table.groupby("series")["value"].sum().sort_values(ascending=False)
        if len(totals) > MAX_SERIES:
            table = _lump_or_drop(
                table, "series", list(totals.index[:MAX_SERIES]), agg, notes, "series"
            )

    # Limit the number of plain categories (never for dates / ordered ranges).
    top_n = spec.get("top_n")
    if top_n and kind_x == "category" and not order:
        totals = table.groupby("x")["value"].sum().sort_values(ascending=False)
        if len(totals) > top_n:
            table = _lump_or_drop(
                table, "x", list(totals.index[:top_n]), agg, notes, "categories"
            )

    # Work out the order of the x axis.
    present = set(table["x"].unique())
    if kind_x == "date":
        x_order = None
        table = table.sort_values("x")
    else:
        if order:
            x_order = [o for o in order if o in present]
        else:
            totals = table.groupby("x")["value"].sum()
            sort = spec.get("sort", "value_desc")
            if sort == "label":
                x_order = sorted(totals.index, key=str)
            elif sort == "value_asc":
                x_order = list(totals.sort_values().index)
            else:
                x_order = list(totals.sort_values(ascending=False).index)
        if "Other" in x_order:
            x_order.remove("Other")
            x_order.append("Other")
        table["x"] = pd.Categorical(table["x"], categories=x_order, ordered=True)
        table = table.sort_values("x")
        table["x"] = table["x"].astype(str)

    if spec.get("as_percent"):
        if (table["value"] < 0).any():
            raise ChartSpecError("'Show as % of total' needs values that are 0 or more.")
        total = table["value"].sum()
        if total > 0:
            table["value"] = table["value"] / total * 100

    table = table.reset_index(drop=True)
    return table, x_order, notes


# ============================================================
# BUILDERS
# ============================================================

def apply_row_filter(df, column, values):
    if column and values and column in df.columns:
        wanted = {str(v) for v in values}
        return df[df[column].astype(str).isin(wanted)]
    return df


def _y_title(agg, measure):
    if measure is None:
        return "Number of rows"
    if agg == "Count":
        return f"Count of {measure}"
    return f"{agg} of {measure}"


def _build_group_mode(df, spec):
    group = spec.get("group")
    if not group:
        raise ChartSpecError("Choose a column to group by.")
    split = spec.get("split") or None
    _need(df, group, split)
    if split and split == group:
        raise ChartSpecError("'Group by' and 'Split by' must be different columns.")

    measure = spec.get("measure")
    measure = None if measure in (None, "", "(count rows)") else measure
    agg = "Count" if measure is None else spec.get("agg", "Sum")
    weight = spec.get("weight")
    _need(df, measure, weight)

    frame = pd.DataFrame(index=df.index)
    labels, kind, order, note = _make_key(df, group, spec)
    frame["x"] = labels
    keys = ["x"]
    if split:
        frame["series"] = _series_key(df, split)
        keys.append("series")
    for name, values in _prepare_values(df, measure, agg, weight).items():
        frame[name] = values

    frame = frame.dropna(subset=keys)
    if "_v" in frame.columns and agg not in ("Count", "Distinct count"):
        frame = frame.dropna(subset=["_v"])

    table = _aggregate(frame, keys, agg)
    table, x_order, notes = _finish(table, kind, order, spec, agg)
    if note:
        notes.insert(0, note)

    return {
        "table": table, "raw": False, "kind_x": kind, "x_order": x_order,
        "has_series": split is not None, "series_order": _series_order(table),
        "x_title": group, "y_title": _y_title(agg, measure),
        "series_title": split, "notes": notes, "agg": agg,
        "n_x": table["x"].nunique(),
    }


def _series_order(table):
    if "series" not in table.columns:
        return None
    totals = table.groupby("series")["value"].sum().sort_values(ascending=False)
    order = list(totals.index)
    if "Other" in order:
        order.remove("Other")
        order.append("Other")
    return order


def _build_columns_mode(df, spec):
    measures = [m for m in (spec.get("measures") or []) if m]
    if not measures:
        raise ChartSpecError("Choose at least one column to compare.")
    _need(df, *measures)
    for m in measures:
        if not _is_number(df[m]):
            raise ChartSpecError(f"'{m}' is not a numeric column, so it can't be compared here.")

    group = spec.get("group") or None
    _need(df, group)
    agg = spec.get("agg", "Sum")
    weight = spec.get("weight") or None
    _need(df, weight)
    convert = bool(spec.get("percent_to_counts")) and weight is not None
    notes = []

    if convert:
        if agg != "Sum":
            agg = "Sum"
            notes.append("Converting percentages to counts always uses 'Sum'.")
        weights_all = _numeric_values(df[weight])

    if group:
        labels, kind, order, note = _make_key(df, group, spec)
        if note:
            notes.append(note)
    else:
        labels, kind, order = None, "category", None

    parts = []
    for m in measures:
        frame = pd.DataFrame(index=df.index)
        values = _numeric_values(df[m])
        if convert:
            values = values * weights_all / 100.0
        if agg in ("Count", "Distinct count"):
            values = df[m]
        frame["_v"] = values
        if agg == "Weighted average":
            if weight is None:
                raise ChartSpecError("Choose a weight column for the weighted average.")
            frame["_w"] = _numeric_values(df[weight])
        if group:
            frame["x"] = labels
            frame = frame.dropna(subset=["x"])
            keys = ["x"]
        else:
            frame["_all"] = 1
            keys = ["_all"]
        if agg not in ("Count", "Distinct count"):
            frame = frame.dropna(subset=["_v"])
        if frame.empty:
            continue
        part = _aggregate(frame, keys, agg)
        if group:
            part["series"] = m
        else:
            part = part.drop(columns="_all")
            part["x"] = m
        parts.append(part)

    if not parts:
        raise ChartSpecError("The chosen columns contain no numbers to show.")

    table = pd.concat(parts, ignore_index=True)
    cols = ["x", "series", "value"] if group else ["x", "value"]
    table = table[cols]

    table, x_order, more = _finish(table, kind, order, spec, agg)
    notes.extend(more)

    return {
        "table": table, "raw": False, "kind_x": kind, "x_order": x_order,
        "has_series": bool(group), "series_order": _series_order(table),
        "x_title": group or "Column",
        "y_title": (
            f"People ({weight} x percent / 100)" if convert
            else f"{agg} of the selected columns"
        ),
        "series_title": "Column" if group else None,
        "notes": notes, "agg": agg, "n_x": table["x"].nunique(),
    }


def _build_raw(df, spec):
    chart = spec["chart"]
    notes = []

    if chart == "Histogram":
        measure = spec.get("measure")
        if measure in (None, "", "(count rows)"):
            raise ChartSpecError("Choose a numeric column for the histogram.")
        _need(df, measure)
        if not _is_number(df[measure]):
            raise ChartSpecError(f"'{measure}' is not numeric, so it can't be shown as a histogram.")
        values = _numeric_values(df[measure]).dropna()
        if values.empty:
            raise ChartSpecError(f"'{measure}' has no numbers to show.")
        table = pd.DataFrame({"value": values})
        split = spec.get("split") or None
        if split:
            _need(df, split)
            labels = _series_key(df, split).loc[values.index]
            top = labels.value_counts().head(6).index
            table["series"] = labels
            table = table[table["series"].isin(top)]
            if labels.nunique() > 6:
                notes.append("Only the 6 largest groups are shown.")
        return {"table": table.reset_index(drop=True), "raw": True, "kind_x": "number",
                "x_order": None, "has_series": split is not None, "series_order": None,
                "x_title": measure, "y_title": "Number of rows",
                "series_title": split, "notes": notes, "agg": None, "n_x": len(table)}

    if chart == "Box plot":
        measure, group = spec.get("measure"), spec.get("group")
        if measure in (None, "", "(count rows)"):
            raise ChartSpecError("Choose a numeric column to summarise.")
        if not group:
            raise ChartSpecError("Choose a column to group by.")
        _need(df, measure, group)
        if not _is_number(df[measure]):
            raise ChartSpecError(f"'{measure}' is not numeric, so a box plot can't be drawn for it.")
        labels, kind, order, note = _make_key(df, group, spec)
        if note:
            notes.append(note)
        table = pd.DataFrame({"x": labels, "value": _numeric_values(df[measure])}).dropna()
        if table.empty:
            raise ChartSpecError("There are no numbers to show for these choices.")
        counts = table["x"].value_counts()
        if len(counts) > 15 and kind == "category" and not order:
            table = table[table["x"].isin(counts.head(15).index)]
            notes.append("Showing the 15 largest groups.")
        if len(table) > 50000:
            table = table.sample(50000, random_state=1)
        x_order = order if order else sorted(table["x"].unique(), key=str)
        x_order = [o for o in x_order if o in set(table["x"])]
        return {"table": table.reset_index(drop=True), "raw": True, "kind_x": kind,
                "x_order": x_order, "has_series": False, "series_order": None,
                "x_title": group, "y_title": measure, "series_title": None,
                "notes": notes, "agg": None, "n_x": table["x"].nunique()}

    # Scatter
    x, y = spec.get("x"), spec.get("y")
    if not x or not y:
        raise ChartSpecError("Choose a numeric column for each axis.")
    if x == y:
        raise ChartSpecError("Choose two different columns for the two axes.")
    _need(df, x, y)
    for col in (x, y):
        if not _is_number(df[col]):
            raise ChartSpecError(f"'{col}' is not numeric, so it can't be used on a scatter plot axis.")
    table = pd.DataFrame({"x": _numeric_values(df[x]), "y": _numeric_values(df[y])})
    split = spec.get("split") or None
    if split:
        _need(df, split)
        labels = _series_key(df, split)
        top = labels.value_counts().head(8).index
        table["series"] = labels.where(labels.isin(top), "Other")
        if labels.nunique() > 8:
            notes.append("Only the 8 largest groups are coloured; the rest are 'Other'.")
    table = table.dropna(subset=["x", "y"])
    if table.empty:
        raise ChartSpecError("These two columns have no rows where both have a number.")
    if len(table) > SAMPLE_LIMIT:
        table = table.sample(SAMPLE_LIMIT, random_state=1)
        notes.append(f"Showing a random sample of {SAMPLE_LIMIT:,} points.")
    return {"table": table.reset_index(drop=True), "raw": True, "kind_x": "number",
            "x_order": None, "has_series": split is not None, "series_order": None,
            "x_title": x, "y_title": y, "series_title": split, "notes": notes,
            "agg": None, "n_x": len(table)}


def build_chart_data(df, spec):
    """Turn a spec into a tidy table plus labels. Raises ChartSpecError only."""
    try:
        if df is None or df.empty:
            raise ChartSpecError("The dataset has no rows.")
        df = apply_row_filter(df, spec.get("filter_col"), spec.get("filter_values"))
        if df.empty:
            raise ChartSpecError("No rows match the filter. Widen or clear it.")
        if spec.get("chart") in RAW_CHARTS:
            return _build_raw(df, spec)
        if spec.get("mode") == "columns":
            return _build_columns_mode(df, spec)
        return _build_group_mode(df, spec)
    except ChartSpecError:
        raise
    except (KeyError, ValueError, TypeError, AttributeError, IndexError) as error:
        raise ChartSpecError(
            "This combination of columns couldn't be summarised. "
            f"Try different columns. ({type(error).__name__})"
        ) from error


# ============================================================
# VALIDATION / SUGGESTIONS / FIGURE
# ============================================================

def validate_for_chart(data, chart):
    if data["raw"]:
        return
    has_series = data["has_series"]

    if chart in ("Stacked bar", "Grouped bar", "100% stacked bar", "Heatmap"):
        if not has_series:
            raise ChartSpecError(
                f"A {chart.lower()} needs two dimensions. Add 'Split by' "
                "(or, when comparing columns, also choose 'Group by')."
            )
    if chart in ("Pie", "Donut", "Treemap"):
        if has_series:
            raise ChartSpecError(
                f"A {chart.lower()} shows one set of values. Remove the 'Split by' "
                "choice, or use a stacked / grouped bar instead."
            )
        if (data["table"]["value"] < 0).any():
            raise ChartSpecError(
                f"A {chart.lower()} can't show negative values. Try a bar chart."
            )


def suggest_charts(data):
    if data["raw"]:
        return []
    if data["kind_x"] == "date":
        return ["Line", "Area", "Bar"] if not data["has_series"] else ["Line", "Stacked bar", "Area"]
    if data["has_series"]:
        return ["Stacked bar", "Grouped bar", "100% stacked bar", "Heatmap"]
    n = data["n_x"]
    if n <= 6:
        return ["Donut", "Pie", "Bar"]
    if n <= 15:
        return ["Bar", "Horizontal bar", "Treemap"]
    return ["Horizontal bar", "Treemap"]


def default_title(spec):
    chart = spec.get("chart", "")
    if chart == "Scatter":
        return f"{spec.get('y')} vs {spec.get('x')}"
    if chart == "Histogram":
        return f"Distribution of {spec.get('measure')}"
    if chart == "Box plot":
        return f"{spec.get('measure')} by {spec.get('group')}"
    if spec.get("mode") == "columns":
        cols = spec.get("measures") or []
        shown = ", ".join(cols[:4]) + ("..." if len(cols) > 4 else "")
        base = f"{shown}" + (f" by {spec['group']}" if spec.get("group") else "")
        return base
    measure = spec.get("measure")
    if measure in (None, "", "(count rows)"):
        base = f"Number of rows by {spec.get('group')}"
    else:
        base = f"{spec.get('agg', 'Sum')} of {measure} by {spec.get('group')}"
    if spec.get("split"):
        base += f" and {spec['split']}"
    return base


def make_figure(data, spec):
    import plotly.express as px

    chart = spec["chart"]
    table = data["table"]
    title = spec.get("title") or default_title(spec)
    labels = {"x": data["x_title"], "value": data["y_title"]}
    if data["series_title"]:
        labels["series"] = data["series_title"]
    color = "series" if data["has_series"] else None
    orders = {}
    if data["x_order"]:
        orders["x"] = data["x_order"]
    if data.get("series_order"):
        orders["series"] = data["series_order"]

    common = dict(labels=labels, category_orders=orders, color_discrete_sequence=PALETTE)

    if chart == "Histogram":
        fig = px.histogram(
            table, x="value", color=color, nbins=int(spec.get("bins") or 20),
            barmode="overlay", opacity=0.75 if color else 1.0,
            labels={"value": data["x_title"], "series": data["series_title"] or ""},
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(yaxis_title="Number of rows")
    elif chart == "Box plot":
        fig = px.box(table, x="x", y="value", labels=labels, category_orders=orders,
                     color_discrete_sequence=PALETTE)
    elif chart == "Scatter":
        kwargs = dict(x="x", y="y", color=color,
                      labels={"x": data["x_title"], "y": data["y_title"],
                              "series": data["series_title"] or ""},
                      color_discrete_sequence=PALETTE, opacity=0.7)
        if spec.get("trendline") and not data["has_series"]:
            try:
                fig = px.scatter(table, trendline="ols", **kwargs)
            except Exception:
                fig = px.scatter(table, **kwargs)
        else:
            fig = px.scatter(table, **kwargs)
    elif chart in ("Bar", "Grouped bar", "Stacked bar", "100% stacked bar"):
        fig = px.bar(table, x="x", y="value", color=color, **common)
        if chart == "Grouped bar" or (chart == "Bar" and color):
            fig.update_layout(barmode="group")
        elif chart in ("Stacked bar", "100% stacked bar"):
            fig.update_layout(barmode="relative")
        if chart == "100% stacked bar":
            fig.update_layout(barnorm="percent")
            fig.update_yaxes(title_text="Share of total (%)")
    elif chart == "Horizontal bar":
        fig = px.bar(table, x="value", y="x", color=color, orientation="h", **common)
        if color:
            fig.update_layout(barmode="group")
        fig.update_layout(yaxis={"categoryorder": "array",
                                 "categoryarray": list(reversed(data["x_order"] or []))}
                          if data["x_order"] else {})
    elif chart == "Line":
        fig = px.line(table, x="x", y="value", color=color, markers=True, **common)
    elif chart == "Area":
        fig = px.area(table, x="x", y="value", color=color, **common)
    elif chart in ("Pie", "Donut"):
        fig = px.pie(table, names="x", values="value",
                     hole=0.55 if chart == "Donut" else 0,
                     category_orders=orders, color_discrete_sequence=PALETTE)
        fig.update_traces(
            sort=False, textposition="outside", textinfo="label+percent",
            outsidetextfont=dict(color=TEXT_LIGHT, size=13),
            marker=dict(line=dict(color="#0f0c29", width=2)),
            automargin=True,
        )
        fig.update_layout(showlegend=False)
    elif chart == "Treemap":
        fig = px.treemap(table, path=["x"], values="value", color_discrete_sequence=PALETTE)
        fig.update_traces(
            textinfo="label+value+percent root",
            textfont=dict(color=TEXT_DARK, size=14),
            marker=dict(line=dict(color="#0f0c29", width=2)),
        )
    elif chart == "Heatmap":
        grid = table.copy()
        if data["kind_x"] == "date":
            grid["x"] = pd.to_datetime(grid["x"]).dt.strftime("%Y-%m-%d")
        pivot = grid.pivot_table(index="series", columns="x", values="value",
                                 aggfunc="sum", sort=False)
        fig = px.imshow(
            pivot, aspect="auto", color_continuous_scale=["#3b3a8f", "#7873f5", "#f59e0b", "#fde68a"],
            labels={"x": data["x_title"], "y": data["series_title"] or "",
                    "color": data["y_title"]},
        )
    else:
        raise ChartSpecError(f"'{chart}' is not a supported chart type.")

    style_dark(fig)
    fig.update_layout(title=dict(text=title, x=0.02))
    if data["has_series"] and chart not in ("Heatmap",):
        fig.update_layout(legend_title_text=data["series_title"] or "")
    return fig


# ============================================================
# STREAMLIT PAGE
# ============================================================

MODE_GROUP = "One measure, grouped by a category  (e.g. sales by salesperson)"
MODE_COLUMNS = "Compare several columns  (e.g. Q1, Q2, Q3, Q4 sales)"


def _show(st, fig, key):
    try:
        st.plotly_chart(fig, width="stretch", theme=None, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def render_custom_dashboard(df, dataset_id="dataset"):
    import streamlit as st

    st.header("🧩 Custom Dashboard")
    st.caption(
        "You decide what to show: pick the column to measure, how to group it, "
        "how to calculate it, and which chart to draw. Add the charts you like "
        "to your own dashboard at the bottom."
    )

    if df is None or df.empty or len(df.columns) == 0:
        st.info("Upload a dataset with at least one row to build charts.")
        return

    signature = hashlib.md5(
        ("|".join(map(str, df.columns)) + str(dataset_id)).encode()
    ).hexdigest()[:8]

    def K(name):
        return f"cd_{signature}_{name}"

    numeric_cols, other_cols, datetime_cols = classify_columns(df)
    all_cols = list(df.columns)
    date_like_text = find_date_like_text_columns(df)

    saved_key = f"cd_saved_{signature}"
    if saved_key not in st.session_state:
        st.session_state[saved_key] = []
    saved = st.session_state[saved_key]

    # ------------------------------------------------------------------
    # 1. What to show
    # ------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("1 · Choose your data")

        mode_label = st.radio(
            "What would you like to show?",
            [MODE_GROUP, MODE_COLUMNS],
            key=K("mode"),
        )
        spec = {"mode": "columns" if mode_label == MODE_COLUMNS else "group"}

        if spec["mode"] == "group":
            c1, c2, c3 = st.columns(3)
            measure = c1.selectbox(
                "Measure - what to calculate",
                ["(count rows)"] + numeric_cols,
                key=K("measure"),
                help="Pick a number column, or count how many rows fall in each group.",
            )
            spec["measure"] = measure
            agg_options = ["Count"] if measure == "(count rows)" else AGGREGATIONS
            spec["agg"] = c2.selectbox(
                "Calculated as", agg_options, key=K(f"agg_{measure == '(count rows)'}")
            )
            default_group = next(
                (i for i, c in enumerate(all_cols) if c in other_cols), 0
            )
            spec["group"] = c3.selectbox(
                "Group by - the categories", all_cols, index=default_group,
                key=K("group"),
            )
            spec["split"] = st.selectbox(
                "Split each group further by (optional)",
                ["(none)"] + [c for c in all_cols if c != spec["group"]],
                key=K("split"),
            )
            if spec["split"] == "(none)":
                spec["split"] = None
            if spec["agg"] == "Weighted average":
                weight_options = [c for c in numeric_cols if c != measure]
                if weight_options:
                    spec["weight"] = st.selectbox(
                        "Weight column (each row counts in proportion to it)",
                        weight_options, key=K("weight"),
                    )
                else:
                    st.warning("A weighted average needs a second numeric column to use as the weight.")
        else:
            spec["measures"] = st.multiselect(
                "Columns to compare",
                numeric_cols,
                key=K("measures"),
                help="Each selected column becomes one bar / slice / series.",
            )
            c1, c2 = st.columns(2)
            spec["agg"] = c1.selectbox(
                "Calculated as",
                [a for a in AGGREGATIONS if a != "Distinct count"],
                key=K("agg_cols"),
            )
            spec["group"] = c2.selectbox(
                "Group by (optional)", ["(none)"] + all_cols, key=K("group_cols")
            )
            if spec["group"] == "(none)":
                spec["group"] = None
            weight_choice = st.selectbox(
                "Weight / size column (optional)",
                ["(none)"] + numeric_cols,
                key=K("weight_cols"),
                help="Used for weighted averages, or to turn percentages into counts.",
            )
            spec["weight"] = None if weight_choice == "(none)" else weight_choice
            if spec["weight"]:
                spec["percent_to_counts"] = st.checkbox(
                    f"These columns are percentages - convert them to counts using {spec['weight']}",
                    key=K("pct"),
                    help="Each value becomes  percentage x weight / 100, then the values are added up.",
                )
            elif spec["agg"] == "Weighted average":
                st.warning("Choose a weight column above to calculate a weighted average.")

    # ------------------------------------------------------------------
    # 2. Chart type
    # ------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("2 · Choose the chart")
        chart = st.selectbox("Chart type", CHART_TYPES, key=K("chart"))
        spec["chart"] = chart
        st.caption(CHART_HELP[chart])

        if chart == "Histogram":
            spec["measure"] = st.selectbox(
                "Numeric column", numeric_cols or ["(none)"], key=K("h_measure")
            )
            spec["split"] = None
            split_choice = st.selectbox(
                "Compare groups (optional)", ["(none)"] + all_cols, key=K("h_split")
            )
            spec["split"] = None if split_choice == "(none)" else split_choice
            spec["bins"] = st.select_slider(
                "Number of bars", options=[5, 10, 15, 20, 30, 40, 50, 75, 100],
                value=20, key=K("bins"),
            )
        elif chart == "Box plot":
            c1, c2 = st.columns(2)
            spec["measure"] = c1.selectbox(
                "Numeric column", numeric_cols or ["(none)"], key=K("b_measure")
            )
            spec["group"] = c2.selectbox("Group by", all_cols, key=K("b_group"))
        elif chart == "Scatter":
            c1, c2, c3 = st.columns(3)
            spec["x"] = c1.selectbox("X axis (numeric)", numeric_cols or ["(none)"], key=K("s_x"))
            y_options = [c for c in numeric_cols if c != spec["x"]] or ["(none)"]
            spec["y"] = c2.selectbox("Y axis (numeric)", y_options, key=K("s_y"))
            color_choice = c3.selectbox(
                "Colour by (optional)", ["(none)"] + all_cols, key=K("s_color")
            )
            spec["split"] = None if color_choice == "(none)" else color_choice
            spec["trendline"] = st.checkbox("Add a trend line", key=K("trend"))

        spec["title"] = st.text_input(
            "Chart title (optional)",
            key=K("title"),
            placeholder=default_title(spec),
            help="Leave this empty and the app writes a title for you "
                 "(shown in grey). Type your own to replace it.",
        ).strip()

    # ------------------------------------------------------------------
    # 3. Options
    # ------------------------------------------------------------------
    with st.expander("3 · Options (optional): sorting, top N, dates, filter"):
        c1, c2, c3 = st.columns(3)
        top_choice = c1.selectbox(
            "Show at most", ["5", "8", "10", "15", "20", "All"], index=2,
            key=K("topn"), help="Extra categories are combined into 'Other' where that makes sense.",
        )
        spec["top_n"] = None if top_choice == "All" else int(top_choice)
        sort_label = c2.selectbox(
            "Sort categories", ["Largest first", "Smallest first", "A to Z"], key=K("sort")
        )
        spec["sort"] = {"Largest first": "value_desc", "Smallest first": "value_asc",
                        "A to Z": "label"}[sort_label]
        spec["as_percent"] = c3.checkbox("Show as % of total", key=K("percent"))

        group_col = spec.get("group")
        if group_col in datetime_cols or group_col in date_like_text:
            d1, d2 = st.columns(2)
            if group_col in date_like_text and group_col not in datetime_cols:
                spec["treat_as_dates"] = d1.checkbox(
                    f"Treat '{group_col}' as dates", value=True, key=K("asdate")
                )
            if group_col in datetime_cols or spec.get("treat_as_dates"):
                spec["date_part"] = d2.selectbox(
                    "Group dates by", DATE_PARTS, index=2, key=K("datepart")
                )
        elif group_col in numeric_cols:
            spec["bin_numbers"] = st.checkbox(
                f"Group the numbers in '{group_col}' into ranges", key=K("bin")
            )

        f1, f2 = st.columns(2)
        filter_col = f1.selectbox("Only include rows where (optional)", ["(none)"] + other_cols,
                                  key=K("fcol"))
        if filter_col != "(none)":
            counts = df[filter_col].dropna().astype(str).value_counts()
            options = list(counts.index[:200])
            if len(counts) > 200:
                st.caption("Showing the 200 most common values.")
            values = f2.multiselect(
                f"{filter_col} is one of", options, key=K(f"fval_{filter_col}")
            )
            spec["filter_col"], spec["filter_values"] = filter_col, values

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    st.subheader("Preview")
    try:
        data = build_chart_data(df, spec)
        validate_for_chart(data, chart)
        fig = make_figure(data, spec)
        _show(st, fig, K("preview"))

        for note in data["notes"]:
            st.caption(f"ℹ️ {note}")
        hints = [h for h in suggest_charts(data) if h != chart]
        if hints:
            st.caption("💡 Other charts that suit this data: " + ", ".join(hints[:3]))

        with st.expander("View the numbers behind this chart"):
            st.dataframe(data["table"], hide_index=True)
            st.download_button(
                "Download these numbers (CSV)",
                data["table"].to_csv(index=False).encode("utf-8"),
                file_name="statify_chart_data.csv",
                mime="text/csv",
                key=K("download"),
            )

        if st.button("➕ Add this chart to my dashboard", key=K("add")):
            entry_spec = {k: v for k, v in spec.items() if v is not None}
            if any(item["spec"] == entry_spec for item in saved):
                st.toast("That chart is already on your dashboard.")
            else:
                saved.append({"id": uuid.uuid4().hex[:8], "spec": entry_spec})
                st.toast("Added to your dashboard below.")
    except ChartSpecError as error:
        st.info(str(error))
    except Exception as error:  # never let a chart choice crash the page
        import traceback
        traceback.print_exc()
        st.warning(
            "That combination couldn't be drawn. Try different columns or a "
            "different chart type."
        )
        with st.expander("Technical details"):
            st.code(f"{type(error).__name__}: {error}")

    # ------------------------------------------------------------------
    # My dashboard
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📌 My dashboard")
    if not saved:
        st.caption("Charts you add above will appear here.")
        return

    def remove(entry_id):
        st.session_state[saved_key] = [
            item for item in st.session_state[saved_key] if item["id"] != entry_id
        ]

    def clear_all():
        st.session_state[saved_key] = []

    st.button("Clear my dashboard", key=K("clear"), on_click=clear_all)

    for row_start in range(0, len(saved), 2):
        row = saved[row_start:row_start + 2]
        columns = st.columns(2)
        for column, item in zip(columns, row):
            with column:
                with st.container(border=True):
                    try:
                        saved_data = build_chart_data(df, item["spec"])
                        validate_for_chart(saved_data, item["spec"]["chart"])
                        saved_fig = make_figure(saved_data, item["spec"])
                        _show(st, saved_fig, K(f"saved_{item['id']}"))
                    except ChartSpecError as error:
                        st.warning(f"This chart can't be drawn any more: {error}")
                    except Exception:
                        st.warning("This chart can't be drawn with the current data.")
                    st.button(
                        "Remove", key=K(f"rm_{item['id']}"),
                        on_click=remove, args=(item["id"],),
                    )
