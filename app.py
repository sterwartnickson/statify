import textwrap
import io
import streamlit as st
import pandas as pd
from cleaning_engine import (
    create_working_copy,
    duplicate_summary,
    remove_duplicates,
    missing_summary,
    drop_missing_rows,
    impute_missing,
    convert_to_numeric,
    convert_to_category,
    convert_to_datetime,
    detect_out_of_range,
    replace_out_of_range_with_missing,
    detect_iqr_outliers as cleaning_iqr_outliers,
    remove_iqr_outliers,
    rename_variable,
    drop_variable
)

from history_engine import (
    create_history,
    add_history,
    history_table
)
from pattern_engine import (
    detect_trend,
    detect_extremes,
    generate_pattern_summary
)
from relationship_engine import (
    correlation_analysis,
    interpret_correlations,
    strongest_relationships
)
from finding_engine import (
    numerical_summary as dataset_numerical_summary,
    detect_skewness,
    detect_missing_patterns,
    generate_findings
)
from robust_tests_engine import (
    welch_anova,
    explain_welch,
    welch_guidance,
    games_howell_posthoc
)
from anova_engine import (
    partial_eta_squared,
    interpret_partial_eta_squared,
    interaction_interpretation,
    main_effect_interpretation,
    cell_means
)
from nonparametric_engine import (
    mann_whitney_analysis,
    kruskal_analysis,
    rank_biserial_correlation,
    interpret_rank_biserial,
    explain_mann_whitney,
    explain_wilcoxon,
    explain_kruskal,
    nonparametric_warning,
    dunn_posthoc
)
from posthoc_engine import (
    tukey_hsd,
    bonferroni_pairwise,
    anova_posthoc_guidance,
    interpret_tukey_results,
    group_descriptive_summary
)
from results_engine import (
    p_value_decision,
    interpret_cohens_d,
    mean_difference_ci,
    interpret_correlation,
    correlation_r_squared,
    interpret_r_squared,
    research_interpretation,
    practical_significance_warning
)
from analysis_engine import (
    welch_t_test,
    paired_t_test,
    pearson,
    spearman,
    chi_square,
    fisher_exact,
    two_way_anova,
    cohens_d
)
from diagnostics_engine import (
    diagnose_two_group_comparison,
    diagnose_multi_group_comparison,
    chi_square_diagnostics,
)
from test_engine import (
    recommend_tests,
    independent_t_test,
    one_way_anova,
    interpret_p_value,
)

from data_engine import (
    load_dataset,
    get_dataset_overview,
    profile_variables,
    detect_data_quality_issues,
)

from domain_cleaning_engine import (
    domain_cleaning_plan,
    apply_domain_action,
)

from domain_engine import (
    detect_domain,
    classify_variables,
    domain_quality_checks,
    cleaning_recommendations,
)

from descriptive_engine import (
    numerical_summary,
    categorical_summary,
    numerical_interpretation,
)

from distribution_engine import (
    prepare_numeric_data,
    distribution_shape,
    fit_distributions,
    best_distribution,
    normality_tests,
    interpret_normality,
)

from visualization_engine import (
    create_histogram,
    create_boxplot,
    create_density_plot,
    create_qq_plot,
    detect_iqr_outliers,
)

from performance_engine import generate_performance_report, sanitize_dataframe
from dashboard_engine import generate_dashboard, get_filter_spec, apply_filters
from custom_dashboard import render_custom_dashboard, style_dark, PALETTE as CHART_PALETTE


# ============================================================
# STATIFY — BATCH 7 NAVIGATION SHELL
# ============================================================

st.set_page_config(
    page_title="Statify",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------
# GRADIENT UI THEME (purely visual — no logic changes)
# ------------------------------------------------------------
st.markdown(
    textwrap.dedent(
        """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Main content container */
    .block-container {
        padding-top: 2rem;
    }

    /* Top toolbar/header bar (was plain white) */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #0f0c29, #302b63) !important;
    }
    header[data-testid="stHeader"] * {
        color: #f5f3ff !important;
        fill: #f5f3ff !important;
    }

    /* File uploader dropzone + Browse button (was plain white) */
    section[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #1e1b4b, #4c1d95) !important;
        border: 1.5px dashed #8DB600 !important;
        border-radius: 12px;
    }
    section[data-testid="stFileUploaderDropzone"] * {
        color: #f5f3ff !important;
    }
    section[data-testid="stFileUploaderDropzone"] button {
        background: #8DB600 !important;
        color: #12122b !important;
        border: none !important;
        border-radius: 8px;
        font-weight: 700;
    }
    section[data-testid="stFileUploaderDropzone"] button * {
        color: #12122b !important;
    }
    div[data-testid="stFileUploaderFile"] {
        background: rgba(141, 182, 0, 0.15);
        border-radius: 8px;
    }
    div[data-testid="stToolbar"] button,
    div[data-testid="stStatusWidget"] button,
    div[data-testid="stToolbar"] button:hover {
        background: #8DB600 !important;
        color: #12122b !important;
        border-radius: 8px;
    }
    div[data-testid="stToolbar"] svg,
    div[data-testid="stStatusWidget"] svg {
        fill: #12122b !important;
    }

    /* Titles — apple green leads, purple/pink as accents */
    h1, h2, h3 {
        background: linear-gradient(90deg, #8DB600, #7873f5, #ff6ec4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }

    /* Body text stays readable on the dark gradient */
    p, span, label, .stMarkdown, .stCaption {
        color: #e6e6f0 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #4c1d95 50%, #831843 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #f5f3ff !important;
    }

    /* Buttons — solid apple green so the color reads clearly, dark readable label */
    .stButton > button, .stDownloadButton > button {
        background: #8DB600 !important;
        color: #12122b !important;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        font-weight: 700;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(141, 182, 0, 0.45);
    }
    .stButton > button *, .stDownloadButton > button * {
        color: #12122b !important;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #7ba300 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(141, 182, 0, 0.6);
    }
    .stButton > button:hover *, .stDownloadButton > button:hover * {
        color: #12122b !important;
    }

    /* Info / success / warning / error boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* Metrics, dataframes, and inputs get soft rounded gradient-tinted cards */
    div[data-testid="stMetric"], .stDataFrame, .stTextArea textarea, .stTextInput input, .stSelectbox > div {
        border-radius: 12px !important;
    }

    /* Dashboard tile cards — Streamlit's native bordered container,
       restyled as dark navy BI-style panels. */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(160deg, #101a3d 0%, #16204a 100%) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] * {
        color: #e6e6f0 !important;
    }

    /* Field selector buttons — active field stands out, others recede */
    button[kind="secondary"] {
        background: rgba(245, 243, 255, 0.08) !important;
        color: #f5f3ff !important;
        border: 1px solid rgba(245, 243, 255, 0.25) !important;
        box-shadow: none !important;
        font-weight: 600;
        white-space: pre-line;
        line-height: 1.3;
    }
    button[kind="secondary"]:hover {
        background: rgba(141, 182, 0, 0.25) !important;
        transform: translateY(-1px);
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #8DB600, #7873f5) !important;
        color: #12122b !important;
        border: none !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.55), 0 6px 16px rgba(141, 182, 0, 0.5) !important;
        font-weight: 800;
        white-space: pre-line;
        line-height: 1.3;
    }
    button[kind="primary"] * , button[kind="secondary"] * {
        color: inherit !important;
    }

    /* Radio nav in sidebar */
    div[role="radiogroup"] label {
        border-radius: 8px;
        padding: 6px 10px;
        margin: 2px 0;
        cursor: pointer;
        transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
    }
    div[role="radiogroup"] label:hover {
        transform: scale(1.06) translateX(6px);
        background: linear-gradient(90deg, rgba(141, 182, 0, 0.35), rgba(120, 115, 245, 0.25));
        box-shadow: 0 4px 14px rgba(141, 182, 0, 0.35);
    }
    div[role="radiogroup"] label:hover p {
        color: #ffffff !important;
        font-weight: 700;
    }
    div[role="radiogroup"] label[data-checked="true"],
    div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #8DB600, #7873f5);
        box-shadow: 0 4px 14px rgba(141, 182, 0, 0.45);
    }
    div[role="radiogroup"] label:has(input:checked) p {
        color: #12122b !important;
        font-weight: 800;
    }

    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #ff6ec4, #7873f5, #8DB600);
        opacity: 0.6;
    }
    </style>
    """
    ).strip(),
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("📊 STATIFY")
st.sidebar.caption("Statistical Research System")

# ------------------------------------------------------------
# GLOBAL FIELD SELECTOR
# ------------------------------------------------------------
# One place to pick the field a dataset belongs to. Every domain-aware
# page (Domain Intelligence, Clean, Performance Intelligence, ...) reads
# st.session_state.statify_domain, so choosing it here drives all of them.
FIELD_OPTIONS = [
    ("🛒", "Sales", "🛒 Sales / E-commerce & Retail"),
    ("💰", "Finance", "💰 Finance"),
    ("🏥", "Health", "🏥 Health"),
    ("🌾", "Agriculture", "🌾 Agriculture"),
    ("👥", "Demography", "👥 Demography"),
    ("🔬", "General", "🔬 General / Other"),
]

if "statify_domain" not in st.session_state:
    st.session_state.statify_domain = "🔬 General / Other"
    st.session_state.statify_domain_confidence = "Default"

st.sidebar.markdown("**🌐 Your Field**")

field_row_1 = st.sidebar.columns(3)
field_row_2 = st.sidebar.columns(3)
field_slots = field_row_1 + field_row_2

for slot, (icon, label, full_domain) in zip(field_slots, FIELD_OPTIONS):
    is_active = st.session_state.statify_domain == full_domain
    with slot:
        if st.button(
            f"{icon}\n{label}",
            key=f"field_btn_{label}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.statify_domain = full_domain
            st.session_state.statify_domain_confidence = "User selected"
            st.rerun()

st.sidebar.caption(f"Active field: **{st.session_state.statify_domain}**")

st.sidebar.divider()

# ------------------------------------------------------------
# PAGE NAVIGATION
# ------------------------------------------------------------
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🖥️ Dashboard",
        "🧩 Custom Dashboard",
        "📁 Data",
        "🧠 Domain Intelligence",
        "🧹 Clean",
        "📈 Explore",
        "🧪 Analyze",
        "📊 Performance Intelligence",
        "📝 Research Intelligence",
        "📄 Export",
    ],
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"],
    key="statify_main_uploader",
)

# ------------------------------------------------------------
# DATASET SESSION MANAGEMENT
# ------------------------------------------------------------
dataset_ready = False

if uploaded_file is not None:
    try:
        raw_df = sanitize_dataframe(load_dataset(uploaded_file))

        dataset_id = (
            uploaded_file.name,
            len(raw_df),
            len(raw_df.columns),
        )

        if (
            "statify_dataset_id" not in st.session_state
            or st.session_state.statify_dataset_id != dataset_id
        ):
            st.session_state.statify_dataset_id = dataset_id
            st.session_state.statify_filename = uploaded_file.name
            st.session_state.original_df = create_working_copy(raw_df)
            st.session_state.working_df = create_working_copy(raw_df)
            st.session_state.cleaning_history = create_history()

            # Domain-aware intelligence
            domain_result = detect_domain(raw_df)
            st.session_state.statify_domain = domain_result["domain"]
            st.session_state.statify_domain_confidence = domain_result["confidence"]
            st.session_state.statify_domain_result = domain_result

        dataset_ready = True

    except Exception as error:
        st.sidebar.error(f"Unable to read dataset: {error}")

elif "working_df" in st.session_state:
    # Keep the current working dataset available while navigating.
    dataset_ready = True

if dataset_ready:
    df = st.session_state.working_df
    working_df = df
    analysis_df = df

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    st.sidebar.success(
        f"Dataset ready: {len(df):,} rows × {len(df.columns)} variables"
    )

    st.sidebar.caption(
        f"Field confidence: {st.session_state.get('statify_domain_confidence', 'Low')}"
    )
else:
    st.sidebar.info("Upload a research dataset to begin.")

# ------------------------------------------------------------
# SHARED PAGE HEADER
# ------------------------------------------------------------
st.title("📊 Statify")
st.caption("Intelligent Statistical Analysis Platform")

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.header("Welcome to Statify")
    st.write(
        "Statify helps you move from raw research data to data cleaning, "
        "exploration, statistical testing, interpretation and research insight."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**1. Data & Clean**\n\nUpload, inspect and prepare your dataset.")
    with c2:
        st.info("**2. Explore & Analyze**\n\nStudy distributions, assumptions, relationships and tests.")
    with c3:
        st.info("**3. Interpret & Export**\n\nTurn results into research insight and reusable outputs.")

    if dataset_ready:
        st.success(
            f"A dataset is active with {len(df):,} observations and "
            f"{len(df.columns)} variables."
        )
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info(
            "Use the uploader in the sidebar, then open 📁 Data to inspect your dataset."
        )

# ============================================================
# ALL DATA-DEPENDENT PAGES
# ============================================================
else:
    try:
        if not dataset_ready:
            st.warning(
                "Upload a CSV or Excel dataset from the sidebar before using this page."
            )
            st.stop()


        if page == "🖥️ Dashboard":
            # ============================================================
            # FIELD-ADAPTIVE DASHBOARD — BI-style panel grid
            # ============================================================
            dash_domain = st.session_state.get("statify_domain", "🔬 General / Other")

            # ---- Header bar -------------------------------------------------
            st.markdown(
                f"""
            <div style="
                background: linear-gradient(90deg, #0f1c3f, #16204a);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 14px;
                padding: 18px 26px;
                margin-bottom: 22px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div style="font-size:1.35rem; font-weight:800; color:#ffffff; letter-spacing:0.5px;">
                    🖥️ STATIFY DASHBOARD
                </div>
                <div style="
                    background: rgba(255,255,255,0.08);
                    border: 1px solid rgba(255,255,255,0.18);
                    border-radius: 20px;
                    padding: 6px 16px;
                    font-size: 0.9rem;
                    color: #e6e6f0;
                    font-weight: 600;
                ">
                    {dash_domain}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # ---- Filters & options ---------------------------------------
            dash_spec = get_filter_spec(df, dash_domain)
            dash_key = abs(hash((
                str(st.session_state.get("statify_dataset_id")),
                tuple(df.columns),
                len(df),
            )))
            dash_date_range = None
            dash_selections = {}

            with st.expander("🎛️ Filters & Options", expanded=False):
                if dash_spec["date"] is not None and dash_spec["date"]["min"] < dash_spec["date"]["max"]:
                    date_info = dash_spec["date"]
                    try:
                        dash_date_range = st.slider(
                            f"{date_info['column']} range",
                            min_value=date_info["min"],
                            max_value=date_info["max"],
                            value=(date_info["min"], date_info["max"]),
                            key=f"dash_date_{dash_key}",
                        )
                    except Exception:
                        # Never let a filter widget take the whole dashboard down.
                        dash_date_range = None

                if dash_spec["categories"]:
                    filter_cols = st.columns(len(dash_spec["categories"]))
                    for filter_col, category in zip(filter_cols, dash_spec["categories"]):
                        with filter_col:
                            dash_selections[category["column"]] = st.multiselect(
                                category["column"],
                                category["options"],
                                key=f"dash_cat_{category['column']}_{dash_key}",
                                help="Leave empty to include everything.",
                            )

                dash_top_n = st.slider(
                    "Items shown per chart",
                    min_value=3,
                    max_value=12,
                    value=6,
                    key=f"dash_topn_{dash_key}",
                )

            dash_df = apply_filters(df, dash_spec, dash_date_range, dash_selections)
            if len(dash_df) < len(df):
                st.caption(f"Showing {len(dash_df):,} of {len(df):,} rows after filters.")

            dash = generate_dashboard(dash_df, dash_domain, top_n=dash_top_n)

            if dash_df.empty:
                st.warning("No rows match the current filters. Widen the date range or clear a selection.")
            elif not dash["kpis"]:
                st.info("Upload a dataset to see your dashboard.")
            else:
                # ---- Big-number KPI row (with change vs previous period) ----
                kpi_cols = st.columns(len(dash["kpis"]))
                for col, (label, value, delta) in zip(kpi_cols, dash["kpis"]):
                    if delta is None:
                        delta_html = '<div style="height:1.3rem;"></div>'
                    else:
                        pct, period_word = delta
                        arrow, color = ("▲", "#34d399") if pct >= 0 else ("▼", "#ff6b6b")
                        delta_html = (
                            f'<div style="font-size:0.85rem; font-weight:700; color:{color}; '
                        f'margin-top:2px;">{arrow} {abs(pct) * 100:.1f}% '
                        f'<span style="color:#a8a8c0; font-weight:500;">vs prev {period_word}</span></div>'
                        )
                    with col:
                        st.markdown(
                            f"""
                        <div style="text-align:center; padding: 6px 0 18px 0;">
                            <div style="font-size:2.4rem; font-weight:800; color:#ffffff; line-height:1.1;">
                                {value}
                            </div>
                            <div style="font-size:0.95rem; color:#a8a8c0; font-weight:600; margin-top:4px;">
                                {label}
                            </div>
                            {delta_html}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                if dash["partial_note"]:
                    st.caption(f"⚠️ {dash['partial_note']}")

                if dash["headline"]:
                    st.markdown(
                        f"""
                    <div style="
                        background: linear-gradient(90deg, #8DB600, #7873f5, #ff6ec4);
                        border-radius: 12px;
                        padding: 12px 22px;
                        margin-bottom: 22px;
                        font-weight: 700;
                        font-size: 0.95rem;
                        color: #12122b;
                    ">
                        🏆 {dash["headline"]}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                if dash["insights"]:
                    with st.container(border=True):
                        st.markdown("**💡 Key Insights**")
                        for insight in dash["insights"]:
                            st.markdown(f"- {insight}")

                # ---- Trend panel (full width, BI-style colored bars) --------
                if dash["trend"] is not None:
                    trend_table = dash["trend"]["table"]
                    period_col, value_col = trend_table.columns
                    with st.container(border=True):
                        st.markdown(f"**📈 {dash['trend']['title']}**")
                        try:
                            import plotly.express as px

                            colors = [
                                "#7873f5" if v >= 0 else "#ff6ec4"
                                for v in trend_table[value_col]
                            ]
                            fig = px.bar(
                                trend_table,
                                x=period_col,
                                y=value_col,
                            )
                            fig.update_traces(marker_color=colors)
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#e6e6f0",
                                margin=dict(l=10, r=10, t=10, b=10),
                                height=260,
                            )
                            style_dark(fig, height=260)
                            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                            try:
                                st.plotly_chart(fig, width="stretch", theme=None)
                            except TypeError:  # older Streamlit
                                st.plotly_chart(fig, use_container_width=True, theme=None)
                        except ImportError:
                            st.bar_chart(trend_table.set_index(period_col)[value_col])
                else:
                    st.caption(
                        "No date/period column was detected, so no time trend is shown. "
                    "Add a date-like column (e.g. 'Date', 'Month') to unlock a trend panel."
                    )

                # ---- Ranking panels grid (3 tiles per row) -------------------
                panels = dash["panels"]
                palette = CHART_PALETTE

                for row_start in range(0, len(panels), 3):
                    row_panels = panels[row_start:row_start + 3]
                    cols = st.columns(len(row_panels))
                    for col, panel in zip(cols, row_panels):
                        with col:
                            with st.container(border=True):
                                st.markdown(f"**{panel['title']}**")
                                table = panel["table"]
                                cat_col, value_col = table.columns

                                try:
                                    import plotly.express as px

                                    if panel["kind"] == "donut":
                                        fig = px.pie(
                                            table,
                                            names=cat_col,
                                            values=value_col,
                                            hole=0.55,
                                            color_discrete_sequence=palette,
                                        )
                                        fig.update_layout(
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            font_color="#e6e6f0",
                                            margin=dict(l=6, r=6, t=6, b=6),
                                            height=260,
                                            showlegend=True,
                                            legend=dict(font=dict(size=10, color="#e6e6f0")),
                                        )
                                    elif panel["kind"] == "hbar":
                                        fig = px.bar(
                                            table,
                                            x=value_col,
                                            y=cat_col,
                                            orientation="h",
                                            color=value_col,
                                            color_continuous_scale=["#7873f5", "#a78bfa", "#f59e0b"],
                                        )
                                        fig.update_layout(
                                            yaxis={"categoryorder": "total ascending"},
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            plot_bgcolor="rgba(0,0,0,0)",
                                            font_color="#e6e6f0",
                                            margin=dict(l=6, r=6, t=6, b=6),
                                            height=260,
                                            coloraxis_showscale=False,
                                        )
                                    else:  # vbar
                                        fig = px.bar(
                                            table,
                                            x=cat_col,
                                            y=value_col,
                                            color=cat_col,
                                            color_discrete_sequence=palette,
                                        )
                                        fig.update_layout(
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            plot_bgcolor="rgba(0,0,0,0)",
                                            font_color="#e6e6f0",
                                            margin=dict(l=6, r=6, t=6, b=6),
                                            height=260,
                                            showlegend=False,
                                        )

                                    style_dark(fig, height=280)
                                    if panel["kind"] == "donut":
                                        fig.update_traces(
                                            textposition="outside",
                                            textinfo="label+percent",
                                            outsidetextfont=dict(color="#f5f3ff", size=12),
                                            marker=dict(line=dict(color="#0f0c29", width=2)),
                                        )
                                        fig.update_layout(showlegend=False)
                                    fig.update_layout(margin=dict(l=8, r=8, t=12, b=8))
                                    try:
                                        st.plotly_chart(fig, width="stretch", theme=None)
                                    except TypeError:  # older Streamlit
                                        st.plotly_chart(fig, use_container_width=True, theme=None)
                                except ImportError:
                                    st.bar_chart(table.set_index(cat_col)[value_col])

                if not panels:
                    st.info("No ranking panels could be generated from the detected variables.")

        elif page == "🧩 Custom Dashboard":
            render_custom_dashboard(
                df, str(st.session_state.get("statify_dataset_id"))
            )

        elif page == "📁 Data":
            # ============================================================
            # DATASET OVERVIEW
            # ============================================================

            st.success(
                f"Successfully loaded: {uploaded_file.name}"
            )

            st.header("📋 Dataset Overview")

            overview = get_dataset_overview(df)


            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric(
                    "Observations",
                    overview["rows"]
                )

            with col2:
                st.metric(
                    "Variables",
                    overview["columns"]
                )

            with col3:
                st.metric(
                    "Numeric",
                    overview["numeric_variables"]
                )

            with col4:
                st.metric(
                    "Categorical",
                    overview["categorical_variables"]
                )

            with col5:
                st.metric(
                    "Missing Cells",
                    overview["missing_cells"]
                )

            with col6:
                st.metric(
                    "Duplicate Rows",
                    overview["duplicate_rows"]
                )


            # ============================================================
            # DATA PREVIEW
            # ============================================================

            st.header("👀 Data Preview")

            st.dataframe(
                df.head(20),
                use_container_width=True,
            )


            # ============================================================
            # VARIABLE INTELLIGENCE
            # ============================================================

            st.header("🔎 Variable Intelligence")

            variable_profile = profile_variables(df)

            st.dataframe(
                variable_profile,
                use_container_width=True,
                hide_index=True,
            )


            # ============================================================
            # DATA QUALITY
            # ============================================================

            st.header("⚠️ Data Quality Check")

            issues = detect_data_quality_issues(df)

            if issues.empty:

                st.success(
                    "No basic data-quality problems were detected."
                )

            else:

                st.warning(
                    f"{len(issues)} potential data-quality issue(s) detected."
                )

                st.dataframe(
                    issues,
                    use_container_width=True,
                    hide_index=True,
                )


        elif page == "🧠 Domain Intelligence":
            # ============================================================
            # DOMAIN-AWARE DATA INTELLIGENCE
            # ============================================================

            st.header("🧠 Domain Intelligence")
            st.write(
                "Statify adapts its data-quality checks to the subject area of "
            "your dataset instead of applying the same cleaning rules to "
            "every dataset."
            )

            detected = detect_domain(df)

            domain_options = list(DOMAIN_PROFILES.keys()) if "DOMAIN_PROFILES" in globals() else [
                "🛒 Sales / E-commerce & Retail",
                "💰 Finance",
                "👥 Demography",
                "🏥 Health",
                "🌾 Agriculture",
                "🔬 General / Other",
            ]

            current_domain = st.session_state.get(
                "statify_domain", detected["domain"]
            )

            selected_domain = st.selectbox(
                "Data domain",
                domain_options,
                index=domain_options.index(current_domain)
                if current_domain in domain_options else len(domain_options) - 1,
                key="statify_selected_domain",
            )

            st.session_state.statify_domain = selected_domain
            st.session_state.statify_domain_confidence = (
                detected["confidence"] if selected_domain == detected["domain"] else "User selected"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detected Domain", detected["domain"])
            with col2:
                st.metric("Detection Confidence", detected["confidence"])
            with col3:
                st.metric("Variables", len(df.columns))

            if selected_domain != detected["domain"]:
                st.info(
                    "You can override automatic detection when your dataset is "
                "specialized or uses unusual variable names."
                )

            st.divider()

            st.subheader("🔎 Domain-Aware Variable Classification")
            variable_map = classify_variables(df, selected_domain)
            st.dataframe(
                variable_map,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("🛡️ Domain-Specific Quality Checks")
            quality = domain_quality_checks(df, selected_domain)

            if quality.empty:
                st.info(
                    "No domain-specific rules were triggered for this dataset. "
                "General Statify quality checks are still available under 📁 Data and 🧹 Clean."
                )
            else:
                flagged = int((quality["Flagged Records"] > 0).sum())
                if flagged:
                    st.warning(
                        f"{flagged} domain-specific check(s) require review."
                    )
                else:
                    st.success("All applicable domain-specific checks passed.")

                st.dataframe(
                    quality,
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("🧹 Recommended Cleaning Plan")
            recommendations = cleaning_recommendations(df, selected_domain)

            if recommendations.empty:
                st.success(
                    "No domain-specific cleaning action is currently recommended."
                )
            else:
                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "Statify flags suspicious records rather than silently deleting them. "
            "A flagged value may be legitimate depending on the dataset's design."
            )

        elif page == "🧹 Clean":
            # ============================================================
            # DATA CLEANING WORKSPACE
            # ============================================================

            st.divider()

            st.header("🧹 Data Cleaning Workspace")

            st.write(
                "Clean and transform a working copy of your dataset. "
            "Your original uploaded data remains unchanged."
            )

            # ========================================================
            # DOMAIN-AWARE CLEANING
            # ========================================================
            st.subheader("🧠 Domain-Aware Cleaning")

            selected_domain = st.session_state.get(
                "statify_domain", "🔬 General / Other"
            )

            domain_variable_map = classify_variables(
                st.session_state.working_df,
                selected_domain
            )

            domain_plan = domain_cleaning_plan(
                st.session_state.working_df,
                selected_domain,
                domain_variable_map
            )

            if domain_plan.empty:
                st.success(
                    f"No domain-specific cleaning actions are currently recommended "
                f"for {selected_domain}."
                )
            else:
                st.info(
                    f"Domain: **{selected_domain}**. "
                "Review each recommendation before applying it."
                )
                st.dataframe(
                    domain_plan,
                    use_container_width=True,
                    hide_index=True
                )

                action_labels = [
                    f"{row['Action ID']} — {row['Variable']} — {row['Recommended Action']}"
                    for _, row in domain_plan.iterrows()
                ]

                selected_action = st.selectbox(
                    "Select a recommended action",
                    action_labels,
                    key="domain_cleaning_action"
                )

                selected_index = action_labels.index(selected_action)
                action_row = domain_plan.iloc[selected_index].to_dict()

                if st.button(
                    "🧹 Apply Selected Domain Action",
                    key="apply_domain_cleaning_action"
                ):
                    cleaned, message = apply_domain_action(
                        st.session_state.working_df,
                        action_row
                    )
                    st.session_state.working_df = cleaned
                    st.session_state.cleaning_history = add_history(
                        st.session_state.cleaning_history,
                        {
                            "action": "Domain-aware cleaning",
                            "details": message,
                            "variable": action_row.get("Variable", "")
                        }
                    )
                    st.success(message)
                    st.rerun()

            st.divider()

            clean_tab1, clean_tab2, clean_tab3, clean_tab4, clean_tab5 = (
                st.tabs(
                    [
                        "Overview",
                        "Missing Data",
                        "Duplicates",
                        "Variables",
                        "Outliers & Range"
                    ]
                )
            )


            with clean_tab1:

                st.subheader(
                    "Working Dataset"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Rows",
                        len(
                            st.session_state.working_df
                        )
                    )

                with col2:
                    st.metric(
                        "Variables",
                        len(
                            st.session_state.working_df.columns
                        )
                    )

                with col3:
                    total_missing = int(
                        st.session_state.working_df
                        .isna()
                        .sum()
                        .sum()
                    )
                    st.metric(
                        "Missing Cells",
                        total_missing
                    )

                st.dataframe(
                    st.session_state.working_df.head(20),
                    use_container_width=True
                )

                if st.button(
                    "↩️ Reset to Original Dataset",
                    key="reset_dataset"
                ):
                    st.session_state.working_df = (
                        st.session_state.original_df.copy()
                    )
                    st.session_state.cleaning_history = []
                    st.success(
                        "The working dataset has been reset."
                    )
                    st.rerun()


            with clean_tab2:

                st.subheader(
                    "🕳️ Missing Data"
                )

                missing_table = missing_summary(
                    st.session_state.working_df
                )

                st.dataframe(
                    missing_table,
                    use_container_width=True,
                    hide_index=True
                )

                missing_variable = st.selectbox(
                    "Select variable",
                    st.session_state.working_df.columns,
                    key="clean_missing_variable"
                )

                missing_action = st.selectbox(
                    "Choose action",
                    [
                        "Keep unchanged",
                        "Impute with Mean",
                        "Impute with Median",
                        "Impute with Mode",
                        "Remove rows where this variable is missing"
                    ],
                    key="missing_action"
                )

                if st.button(
                    "Apply Missing-Data Action",
                    key="apply_missing_action"
                ):

                    try:

                        if missing_action == "Keep unchanged":

                            st.info(
                                "No changes were made."
                            )

                        elif missing_action == "Impute with Mean":

                            cleaned, action = impute_missing(
                                st.session_state.working_df,
                                missing_variable,
                                "Mean"
                            )
                            st.session_state.working_df = cleaned
                            st.session_state.cleaning_history = (
                                add_history(
                                    st.session_state.cleaning_history,
                                    action
                                )
                            )
                            st.success(
                                action["details"]
                            )

                        elif missing_action == "Impute with Median":

                            cleaned, action = impute_missing(
                                st.session_state.working_df,
                                missing_variable,
                                "Median"
                            )
                            st.session_state.working_df = cleaned
                            st.session_state.cleaning_history = (
                                add_history(
                                    st.session_state.cleaning_history,
                                    action
                                )
                            )
                            st.success(
                                action["details"]
                            )

                        elif missing_action == "Impute with Mode":

                            cleaned, action = impute_missing(
                                st.session_state.working_df,
                                missing_variable,
                                "Mode"
                            )
                            st.session_state.working_df = cleaned
                            st.session_state.cleaning_history = (
                                add_history(
                                    st.session_state.cleaning_history,
                                    action
                                )
                            )
                            st.success(
                                action["details"]
                            )

                        else:

                            cleaned, action = drop_missing_rows(
                                st.session_state.working_df,
                                missing_variable
                            )
                            st.session_state.working_df = cleaned
                            st.session_state.cleaning_history = (
                                add_history(
                                    st.session_state.cleaning_history,
                                    action
                                )
                            )
                            st.success(
                                action["details"]
                            )

                    except Exception as error:

                        st.error(
                            str(error)
                        )


            with clean_tab3:

                st.subheader(
                    "♻️ Duplicate Observations"
                )

                duplicates = duplicate_summary(
                    st.session_state.working_df
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Duplicate Rows",
                        duplicates["count"]
                    )

                with col2:
                    st.metric(
                        "Duplicate Rate",
                        f"{duplicates['percentage']:.2f}%"
                    )

                if duplicates["count"] > 0:

                    duplicate_rows = (
                        st.session_state.working_df[
                            st.session_state.working_df
                            .duplicated(
                                keep=False
                            )
                        ]
                    )

                    st.dataframe(
                        duplicate_rows,
                        use_container_width=True
                    )

                    st.warning(
                        "Do not automatically assume duplicated-looking "
                    "records are errors. Some datasets may legitimately "
                    "contain identical observations."
                    )

                    if st.button(
                        "Remove Duplicate Rows",
                        key="remove_duplicates_button"
                    ):

                        cleaned, action = remove_duplicates(
                            st.session_state.working_df
                        )
                        st.session_state.working_df = cleaned
                        st.session_state.cleaning_history = (
                            add_history(
                                st.session_state.cleaning_history,
                                action
                            )
                        )
                        st.success(
                            action["details"]
                        )
                        st.rerun()

                else:

                    st.success(
                        "No duplicate rows detected."
                    )


            with clean_tab4:

                st.subheader(
                    "🛠️ Variable Management"
                )

                variable = st.selectbox(
                    "Select variable",
                    st.session_state.working_df.columns,
                    key="variable_management"
                )

                st.write(
                    f"Current data type: "
                f"**{st.session_state.working_df[variable].dtype}**"
                )

                transformation = st.selectbox(
                    "Transformation",
                    [
                        "No change",
                        "Convert to numeric",
                        "Convert to categorical",
                        "Convert to date",
                        "Rename variable",
                        "Drop variable"
                    ],
                    key="variable_transformation"
                )

                if transformation == "Rename variable":

                    new_variable_name = st.text_input(
                        "New variable name",
                        key="new_variable_name"
                    )

                if st.button(
                    "Apply Variable Transformation",
                    key="apply_variable_transformation"
                ):

                    try:

                        if transformation == "No change":

                            st.info(
                                "No transformation applied."
                            )

                        elif transformation == "Convert to numeric":

                            cleaned, action = convert_to_numeric(
                                st.session_state.working_df,
                                variable
                            )

                        elif transformation == "Convert to categorical":

                            cleaned, action = convert_to_category(
                                st.session_state.working_df,
                                variable
                            )

                        elif transformation == "Convert to date":

                            cleaned, action = convert_to_datetime(
                                st.session_state.working_df,
                                variable
                            )

                        elif transformation == "Rename variable":

                            cleaned, action = rename_variable(
                                st.session_state.working_df,
                                variable,
                                new_variable_name
                            )

                        elif transformation == "Drop variable":

                            cleaned, action = drop_variable(
                                st.session_state.working_df,
                                variable
                            )

                        if transformation != "No change":

                            st.session_state.working_df = cleaned
                            st.session_state.cleaning_history = (
                                add_history(
                                    st.session_state.cleaning_history,
                                    action
                                )
                            )
                            st.success(
                                action["details"]
                            )
                            st.rerun()

                    except Exception as error:

                        st.error(
                            str(error)
                        )


            with clean_tab5:

                st.subheader(
                    "🚨 Outliers & Valid Range"
                )

                numeric_working_columns = (
                    st.session_state.working_df
                    .select_dtypes(
                        include="number"
                    )
                    .columns
                    .tolist()
                )


                if not numeric_working_columns:

                    st.info(
                        "No numerical variables are available."
                    )

                else:

                    range_variable = st.selectbox(
                        "Select numerical variable",
                        numeric_working_columns,
                        key="range_variable"
                    )


                    # ====================================================
                    # OUTLIERS
                    # ====================================================

                    st.markdown(
                        "### Potential Outliers"
                    )

                    outlier_result = cleaning_iqr_outliers(
                        st.session_state.working_df,
                        range_variable
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Potential Outliers",
                            outlier_result["count"]
                        )

                    with col2:

                        st.metric(
                            "Lower IQR Limit",
                            f"{outlier_result['lower_bound']:.3f}"
                        )

                    with col3:

                        st.metric(
                            "Upper IQR Limit",
                            f"{outlier_result['upper_bound']:.3f}"
                        )


                    st.warning(
                        "A statistical outlier is not automatically "
                    "an incorrect observation. Investigate the value "
                    "before deciding whether to remove it."
                    )


                    if outlier_result["count"] > 0:

                        show_outliers = (
                            st.session_state.working_df.loc[
                                outlier_result["indices"]
                            ]
                        )

                        st.dataframe(
                            show_outliers,
                            use_container_width=True
                        )


                        if st.button(
                            "Remove IQR Outliers",
                            key="remove_iqr_outliers"
                        ):

                            cleaned, action = remove_iqr_outliers(
                                st.session_state.working_df,
                                range_variable
                            )

                            st.session_state.working_df = cleaned

                            st.session_state.cleaning_history = (
                                add_history(
                                    st.session_state.cleaning_history,
                                    action
                                )
                            )

                            st.warning(
                                action["details"]
                            )

                            st.rerun()


                    # ====================================================
                    # RESEARCHER-SPECIFIED VALID RANGE
                    # ====================================================

                    st.markdown(
                        "### Researcher-Specified Valid Range"
                    )

                    st.write(
                        "Use this when the variable has a known logical "
                    "or scientific range."
                    )

                    minimum = st.number_input(
                        "Minimum valid value",
                        value=None,
                        placeholder="Example: 0",
                        key="valid_minimum"
                    )

                    maximum = st.number_input(
                        "Maximum valid value",
                        value=None,
                        placeholder="Example: 120",
                        key="valid_maximum"
                    )


                    if st.button(
                        "Check Valid Range",
                        key="check_valid_range"
                    ):

                        invalid = detect_out_of_range(
                            st.session_state.working_df,
                            range_variable,
                            minimum,
                            maximum
                        )

                        if invalid.empty:

                            st.success(
                                "No values outside the specified range "
                            "were detected."
                            )

                        else:

                            st.warning(
                                f"{len(invalid)} observation(s) fall "
                            "outside the specified valid range."
                            )

                            st.dataframe(
                                invalid,
                                use_container_width=True
                            )


                    if st.button(
                        "Set Invalid Values to Missing",
                        key="set_invalid_missing"
                    ):

                        cleaned, action = (
                            replace_out_of_range_with_missing(
                                st.session_state.working_df,
                                range_variable,
                                minimum,
                                maximum
                            )
                        )

                        st.session_state.working_df = cleaned

                        st.session_state.cleaning_history = (
                            add_history(
                                st.session_state.cleaning_history,
                                action
                            )
                        )

                        st.success(
                            action["details"]
                        )

                        st.rerun()


            # ============================================================
            # CLEANING HISTORY
            # ============================================================

            st.divider()

            st.header(
                "📜 Data Cleaning History"
            )

            history_df = history_table(
                st.session_state.cleaning_history
            )

            if history_df.empty:

                st.info(
                    "No cleaning transformations have been applied yet."
                )

            else:

                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )

            # ============================================================
            # DOWNLOAD CLEANED DATA
            # ============================================================

            st.divider()

            st.header(
                "⬇️ Download Cleaned Data"
            )

            st.write(
                "Export your working dataset — including every cleaning "
            "transformation applied above — as a file you can keep or "
            "use elsewhere."
            )

            download_col1, download_col2 = st.columns(2)

            with download_col1:
                st.download_button(
                    "⬇️ Download as CSV",
                    data=st.session_state.working_df.to_csv(index=False).encode("utf-8"),
                    file_name="statify_cleaned_data.csv",
                    mime="text/csv",
                    key="download_cleaned_csv",
                    use_container_width=True,
                )

            with download_col2:
                try:
                    import xlsxwriter  # noqa: F401
                    excel_engine = "xlsxwriter"
                except ImportError:
                    try:
                        import openpyxl  # noqa: F401
                        excel_engine = "openpyxl"
                    except ImportError:
                        excel_engine = None

                if excel_engine:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine=excel_engine) as writer:
                        st.session_state.working_df.to_excel(
                            writer, index=False, sheet_name="Cleaned Data"
                        )
                    st.download_button(
                        "⬇️ Download as Excel",
                        data=excel_buffer.getvalue(),
                        file_name="statify_cleaned_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_cleaned_xlsx",
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "Excel export needs the `xlsxwriter` or `openpyxl` "
                    "package. Run `pip install xlsxwriter` to enable it."
                    )


        elif page == "📈 Explore":

            # ============================================================
            # STATISTICAL EXPLORER
            # ============================================================

            st.divider()

            st.header("📊 Statistical Explorer")

            st.write(
                "Select a variable to explore its statistical properties."
            )

            selected_variable = st.selectbox(
                "Choose a variable",
                df.columns
            )


            selected_series = df[selected_variable]
            # ============================================================
            # DISTRIBUTION INTELLIGENCE
            # ============================================================

            if pd.api.types.is_numeric_dtype(selected_series):

                st.divider()

                st.header("📈 Distribution Intelligence")

                distribution_data = prepare_numeric_data(
                    selected_series
                )

                if len(distribution_data) < 10:

                    st.warning(
                        "At least 10 valid numerical observations "
                    "are recommended for distribution analysis."
                    )

                else:

                    # ----------------------------------------------------
                    # DISTRIBUTION SHAPE
                    # ----------------------------------------------------

                    shape = distribution_shape(
                        distribution_data
                    )

                    st.subheader("Distribution Shape")

                    st.info(
                        f"Statify detected the general shape as: **{shape}**"
                    )

                    # ----------------------------------------------------
                    # NORMALITY TESTS
                    # ----------------------------------------------------

                    st.subheader("Normality Assessment")

                    normality_results = normality_tests(
                        distribution_data
                    )

                    if normality_results:

                        normality_table = pd.DataFrame(
                            normality_results.items(),
                            columns=[
                                "Test",
                                "Value"
                            ]
                        )

                        st.dataframe(
                            normality_table,
                            use_container_width=True,
                            hide_index=True
                        )

                        if "Shapiro-Wilk p-value" in normality_results:

                            interpretation = interpret_normality(
                                normality_results[
                                    "Shapiro-Wilk p-value"
                                ]
                            )

                            st.write(
                                f"**Interpretation:** {interpretation}"
                            )

                    # ----------------------------------------------------
                    # DISTRIBUTION FITTING
                    # ----------------------------------------------------

                    st.subheader(
                        "Probability Distribution Comparison"
                    )

                    distribution_results = fit_distributions(
                        distribution_data
                    )

                    if distribution_results.empty:

                        st.warning(
                            "Statify could not fit the candidate "
                        "probability distributions to this variable."
                        )

                    else:

                        display_results = distribution_results.copy()

                        round_columns = [
                            "AIC",
                            "BIC",
                            "KS Statistic",
                            "KS p-value"
                        ]

                        for column in round_columns:

                            display_results[column] = (
                                display_results[column].round(4)
                            )

                        st.dataframe(
                            display_results,
                            use_container_width=True,
                            hide_index=True
                        )

                        selected_distribution = best_distribution(
                            distribution_results
                        )

                        st.success(
                            f"Best candidate among the evaluated "
                        f"distributions: **{selected_distribution}**"
                        )

                        st.caption(
                            "The recommendation is based primarily on "
                        "AIC. A lower AIC indicates a better relative "
                        "fit among the candidate models evaluated."
                        )
            # ============================================================
            # VISUAL STATISTICAL EXPLORER
            # ============================================================

            st.divider()

            st.header("📊 Visual Statistical Explorer")

            if pd.api.types.is_numeric_dtype(selected_series):

                visual_data = prepare_numeric_data(
                    selected_series
                )

                if len(visual_data) >= 3:

                    # ----------------------------------------------------
                    # HISTOGRAM
                    # ----------------------------------------------------

                    st.subheader("📊 Histogram")

                    histogram = create_histogram(
                        visual_data,
                        selected_variable
                    )

                    st.pyplot(
                        histogram,
                        use_container_width=True
                    )

                    st.caption(
                        "A histogram shows how observations are distributed "
                    "across the range of the variable."
                    )

                    # ----------------------------------------------------
                    # BOXPLOT
                    # ----------------------------------------------------

                    st.subheader("📦 Boxplot")

                    boxplot = create_boxplot(
                        visual_data,
                        selected_variable
                    )

                    st.pyplot(
                        boxplot,
                        use_container_width=True
                    )

                    st.caption(
                        "The boxplot displays the median, quartiles, "
                    "spread, and potential extreme observations."
                    )

                    # ----------------------------------------------------
                    # DENSITY
                    # ----------------------------------------------------

                    st.subheader("〰️ Density Plot")

                    if len(visual_data) >= 5:

                        try:

                            density_plot = create_density_plot(
                                visual_data,
                                selected_variable
                            )

                            st.pyplot(
                                density_plot,
                                use_container_width=True
                            )

                            st.caption(
                                "The density plot provides a smoothed "
                            "estimate of the distribution."
                            )

                        except Exception:

                            st.warning(
                                "A density estimate could not be calculated "
                            "for this variable."
                            )

                    # ----------------------------------------------------
                    # Q-Q PLOT
                    # ----------------------------------------------------

                    st.subheader("📐 Normal Q-Q Plot")

                    qq_plot = create_qq_plot(
                        visual_data,
                        selected_variable
                    )

                    st.pyplot(
                        qq_plot,
                        use_container_width=True
                    )

                    st.caption(
                        "If observations approximately follow the reference "
                    "line, the data may be reasonably consistent with "
                    "a normal distribution."
                    )

                    # ----------------------------------------------------
                    # OUTLIER ANALYSIS
                    # ----------------------------------------------------

                    st.subheader("🚨 Potential Outliers")

                    outlier_results = detect_iqr_outliers(
                        visual_data
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Q1",
                            f"{outlier_results['Q1']:.3f}"
                        )

                    with col2:

                        st.metric(
                            "IQR",
                            f"{outlier_results['IQR']:.3f}"
                        )

                    with col3:

                        st.metric(
                            "Potential Outliers",
                            outlier_results["Outlier Count"]
                        )

                    st.write(
                        f"Lower bound: "
                    f"**{outlier_results['Lower Bound']:.3f}**"
                    )

                    st.write(
                        f"Upper bound: "
                    f"**{outlier_results['Upper Bound']:.3f}**"
                    )

                    if outlier_results["Outlier Count"] > 0:

                        st.warning(
                            f"{outlier_results['Outlier Count']} "
                        "potential outlier(s) were detected using "
                        "the 1.5 × IQR rule."
                        )

                    else:

                        st.success(
                            "No potential outliers were detected "
                        "using the 1.5 × IQR rule."
                        )


            # ============================================================
            # VARIABLE TYPE
            # ============================================================

            st.subheader("Variable Information")

            variable_type = profile_variables(
                df[df.columns]
            )

            selected_info = variable_type[
                variable_type["Variable"] == selected_variable
            ]

            st.dataframe(
                selected_info,
                use_container_width=True,
                hide_index=True
            )


            # ============================================================
            # NUMERICAL VARIABLE
            # ============================================================

            if selected_series.isna().all():

                st.info(
                    "This variable has no observed values (every entry is "
                    "missing), so there is nothing to summarise. Consider "
                    "dropping it on the Clean page."
                )

            elif pd.api.types.is_numeric_dtype(selected_series):

                st.subheader("📐 Descriptive Statistics")

                summary = numerical_summary(
                    selected_series
                )

                metric1, metric2, metric3, metric4 = st.columns(4)

                with metric1:
                    st.metric(
                        "Mean",
                        f"{summary['Mean']:.3f}"
                    )

                with metric2:
                    st.metric(
                        "Median",
                        f"{summary['Median']:.3f}"
                    )

                with metric3:
                    st.metric(
                        "Std. Deviation",
                        f"{summary['Standard Deviation']:.3f}"
                    )

                with metric4:
                    st.metric(
                        "IQR",
                        f"{summary['IQR']:.3f}"
                    )

                st.write("### Complete Statistical Summary")

                summary_table = pd.DataFrame(
                    summary.items(),
                    columns=["Statistic", "Value"]
                )

                st.dataframe(
                    summary_table,
                    use_container_width=True,
                    hide_index=True
                )

                # --------------------------------------------------------
                # INTERPRETATION
                # --------------------------------------------------------

                st.subheader("🧠 Statistical Interpretation")

                interpretations = numerical_interpretation(
                    summary
                )

                for interpretation in interpretations:

                    st.write(
                        f"• {interpretation}"
                    )


            # ============================================================
            # CATEGORICAL VARIABLE
            # ============================================================

            else:

                st.subheader("📋 Frequency Distribution")

                frequency_table = categorical_summary(
                    selected_series
                )

                st.dataframe(
                    frequency_table,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("📊 Category Overview")

                st.bar_chart(
                    frequency_table.set_index(
                        "Category"
                    )["Frequency"]
                )


            # ============================================================
            # FOOTER
            # ============================================================

            st.divider()

            st.caption(
                "Statify — From Data to Statistical Insight"
            )

        elif page == "🧪 Analyze":
            # ============================================================
            # BATCH 8 — SIMPLE ANALYSIS GUIDE
            # ============================================================
            st.header("🧭 Analysis Guide")
            st.caption("Not sure which statistical method to use? Start here.")

            guide_choice = st.radio(
                "What are you trying to find out?",
                [
                    "Compare groups",
                    "Find a relationship",
                    "Check association between categories",
                ],
                horizontal=True,
                key="statify_analysis_goal",
            )

            if guide_choice == "Compare groups":
                st.info(
                    "**You want to know whether groups differ.**\n\n"
                "• 2 independent groups → Independent t-test (or Welch's t-test when variances differ)\n"
                "• 2 paired measurements → Paired t-test\n"
                "• 3 or more groups → One-way ANOVA\n"
                "• Two factors → Two-way ANOVA\n\n"
                "If the usual assumptions are not suitable, Statify can guide you toward a non-parametric alternative."
                )
            elif guide_choice == "Find a relationship":
                st.info(
                    "**You want to know whether two numerical variables move together.**\n\n"
                "• Pearson correlation → measures linear association\n"
                "• Spearman correlation → useful when the relationship is based on ranks or normality is a concern\n"
                "Remember: correlation does not prove causation."
                )
            else:
                st.info(
                    "**You want to know whether two categorical variables are associated.**\n\n"
                "• Chi-square test → common choice for categorical variables\n"
                "• Fisher's exact test → useful for small expected frequencies\n\n"
                "Statify will also help you check whether the test's assumptions are reasonable."
                )

            st.divider()

            # ============================================================
            # STATISTICAL TEST ADVISOR
            # ============================================================

            st.divider()

            st.header("🧪 Statistical Test Advisor")

            st.write(
                "Select two variables and Statify will identify statistical "
            "methods that may be appropriate for examining their relationship."
            )

            test_col1, test_col2 = st.columns(2)

            with test_col1:

                variable_x = st.selectbox(
                    "First variable",
                    df.columns,
                    key="test_variable_x"
                )

            with test_col2:

                variable_y = st.selectbox(
                    "Second variable",
                    df.columns,
                    key="test_variable_y"
                )


            if variable_x == variable_y:

                st.warning(
                    "Please select two different variables."
                )

            else:

                x = df[variable_x]
                y = df[variable_y]

                # --------------------------------------------------------
                # VARIABLE INFORMATION
                # --------------------------------------------------------

                st.subheader("Variable Relationship")

                info_col1, info_col2 = st.columns(2)

                with info_col1:

                    st.write(
                        f"**{variable_x}**"
                    )

                    st.write(
                        f"Type: `{profile_variables(df).loc[
                        profile_variables(df)['Variable'] == variable_x,
                        'Type'
                    ].iloc[0]}`"
                    )

                with info_col2:

                    st.write(
                        f"**{variable_y}**"
                    )

                    st.write(
                        f"Type: `{profile_variables(df).loc[
                        profile_variables(df)['Variable'] == variable_y,
                        'Type'
                    ].iloc[0]}`"
                    )

                # --------------------------------------------------------
                # RECOMMENDATIONS
                # --------------------------------------------------------

                recommendations = recommend_tests(
                    x,
                    y
                )

                st.subheader(
                    "🎯 Recommended Statistical Methods"
                )

                if recommendations.empty:

                    st.warning(
                        "Statify could not identify an appropriate "
                    "method for these variables."
                    )

                else:

                    for _, recommendation in recommendations.iterrows():

                        st.markdown(
                            f"### {recommendation['Test']}"
                        )

                        st.write(
                            f"**Purpose:** "
                        f"{recommendation['Purpose']}"
                        )

                        st.write(
                            f"**Why:** "
                        f"{recommendation['Reason']}"
                        )

                        st.divider()
            # ============================================================
            # ASSUMPTION & DIAGNOSTIC EXPLORER
            # ============================================================

            st.divider()

            st.header("🔬 Assumption & Diagnostic Explorer")

            st.write(
                "Statify examines important statistical assumptions before "
            "you choose a final inferential method."
            )

            diagnostic_type = st.selectbox(
                "Select analysis design",
                [
                    "Two independent groups",
                    "Three or more independent groups",
                    "Categorical association"
                ],
                key="diagnostic_type"
            )


            # ============================================================
            # TWO GROUPS
            # ============================================================

            if diagnostic_type == "Two independent groups":

                st.subheader("Two-Group Comparison")

                outcome_col, group_col = st.columns(2)

                with outcome_col:

                    diagnostic_outcome = st.selectbox(
                        "Numerical outcome",
                        df.columns,
                        key="diagnostic_two_outcome"
                    )

                with group_col:

                    diagnostic_group = st.selectbox(
                        "Grouping variable",
                        df.columns,
                        key="diagnostic_two_group"
                    )

                if diagnostic_outcome == diagnostic_group:

                    st.warning(
                        "The outcome and grouping variable must be different."
                    )

                else:

                    result = diagnose_two_group_comparison(
                        df[diagnostic_outcome],
                        df[diagnostic_group]
                    )

                    if not result["valid"]:

                        st.error(
                            result["message"]
                        )

                    else:

                        st.subheader("📋 Diagnostic Summary")

                        col1, col2 = st.columns(2)

                        with col1:

                            st.write("**Group sizes**")

                            st.write(
                                result["group_sizes"]
                            )

                        with col2:

                            st.write("**Variance assessment**")

                            if result["variance"]:

                                st.write(
                                    f"Levene p-value: "
                                f"**{result['variance']['p_value']:.4f}**"
                                )

                                st.write(
                                    result["variance"]["Interpretation"]
                                )

                        st.subheader("Normality by Group")

                        if not result["normality"].empty:

                            st.dataframe(
                                result["normality"],
                                use_container_width=True
                            )

                        st.subheader("🚨 Outlier Assessment")

                        for group_name, outlier_result in result[
                            "outliers"
                        ].items():

                            if outlier_result:

                                st.write(
                                    f"**{group_name}:** "
                                f"{outlier_result['Outlier Count']} "
                                "potential outlier(s)"
                                )

                        st.subheader("💡 Statify's Diagnostic Guidance")

                        for recommendation in result[
                            "recommendations"
                        ]:

                            st.info(
                                recommendation
                            )


            # ============================================================
            # THREE OR MORE GROUPS
            # ============================================================

            elif diagnostic_type == "Three or more independent groups":

                st.subheader("Multi-Group Comparison")

                outcome_col, group_col = st.columns(2)

                with outcome_col:

                    diagnostic_outcome = st.selectbox(
                        "Numerical outcome",
                        df.columns,
                        key="diagnostic_multi_outcome"
                    )

                with group_col:

                    diagnostic_group = st.selectbox(
                        "Grouping variable",
                        df.columns,
                        key="diagnostic_multi_group"
                    )

                if diagnostic_outcome == diagnostic_group:

                    st.warning(
                        "The outcome and grouping variable must be different."
                    )

                else:

                    result = diagnose_multi_group_comparison(
                        df[diagnostic_outcome],
                        df[diagnostic_group]
                    )

                    if not result["valid"]:

                        st.error(
                            result["message"]
                        )

                    else:

                        st.subheader("Normality by Group")

                        if not result["normality"].empty:

                            st.dataframe(
                                result["normality"],
                                use_container_width=True
                            )

                        st.subheader("Variance Assessment")

                        if result["variance"]:

                            st.write(
                                f"Levene p-value: "
                            f"**{result['variance']['p_value']:.4f}**"
                            )

                            st.write(
                                result["variance"]["Interpretation"]
                            )

                        st.subheader("🚨 Outlier Assessment")

                        for group_name, outlier_result in result[
                            "outliers"
                        ].items():

                            if outlier_result:

                                st.write(
                                    f"**{group_name}:** "
                                f"{outlier_result['Outlier Count']} "
                                "potential outlier(s)"
                                )

                        st.subheader("💡 Statify's Diagnostic Guidance")

                        for recommendation in result[
                            "recommendations"
                        ]:

                            st.info(
                                recommendation
                            )


            # ============================================================
            # CATEGORICAL ASSOCIATION
            # ============================================================

            else:

                st.subheader(
                    "Categorical Association Diagnostic"
                )

                variable_col1, variable_col2 = st.columns(2)

                with variable_col1:

                    categorical_1 = st.selectbox(
                        "First categorical variable",
                        df.columns,
                        key="diagnostic_cat_1"
                    )

                with variable_col2:

                    categorical_2 = st.selectbox(
                        "Second categorical variable",
                        df.columns,
                        key="diagnostic_cat_2"
                    )

                if categorical_1 == categorical_2:

                    st.warning(
                        "Please select two different variables."
                    )

                else:

                    result = chi_square_diagnostics(
                        df[categorical_1],
                        df[categorical_2]
                    )

                    if result is None:

                        st.error(
                            "The selected variables could not form "
                        "a valid contingency table."
                        )

                    else:

                        st.subheader("Observed Frequencies")

                        st.dataframe(
                            result["Observed"],
                            use_container_width=True
                        )

                        st.subheader("Expected Frequencies")

                        st.dataframe(
                            result["Expected"],
                            use_container_width=True
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Chi-square",
                                f"{result['Chi-square']:.3f}"
                            )

                        with col2:

                            st.metric(
                                "p-value",
                                f"{result['p_value']:.4f}"
                            )

                        with col3:

                            st.metric(
                                "Minimum Expected",
                                f"{result['Minimum Expected Frequency']:.2f}"
                            )

                        st.info(
                            result["Recommendation"]
                        )
            # ============================================================
            # STATIFY ANALYSIS WORKSPACE
            # ============================================================

            st.divider()

            st.header("🧪 Statistical Analysis Workspace")

            st.write(
                "Choose the structure of your research question. "
            "Statify will then guide you toward an appropriate "
            "statistical method."
            )


            analysis_design = st.selectbox(
                "1️⃣ What type of analysis are you performing?",
                [
                    "Relationship between two numerical variables",
                    "Compare two independent groups",
                    "Compare three or more independent groups",
                    "Compare paired measurements",
                    "Two-Way ANOVA",
                    "Association between categorical variables"
                ],
                key="analysis_design"
            )


            # ============================================================
            # RELATIONSHIP
            # ============================================================

            if analysis_design == (
                "Relationship between two numerical variables"
            ):

                st.subheader("📈 Correlation Analysis")

                col1, col2 = st.columns(2)

                with col1:

                    x = st.selectbox(
                        "First numerical variable",
                        df.columns,
                        key="corr_x"
                    )

                with col2:

                    y = st.selectbox(
                        "Second numerical variable",
                        df.columns,
                        key="corr_y"
                    )

                method = st.radio(
                    "Method",
                    [
                        "Pearson correlation",
                        "Spearman correlation"
                    ],
                    horizontal=True
                )

                if st.button(
                    "▶ Run Correlation",
                    key="run_correlation"
                ):

                    if x == y:

                        st.warning(
                            "Please select two different variables."
                        )

                    elif method == "Pearson correlation":

                        result = pearson(
                            df[x],
                            df[y]
                        )

                        r = result["statistic"]
                        p = result["p_value"]
                        decision = p_value_decision(p)

                        st.subheader("📊 Correlation Result")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Correlation",
                                f"{r:.3f}"
                            )

                        with col2:
                            st.metric(
                                "p-value",
                                f"{p:.4f}"
                            )

                        with col3:
                            st.metric(
                                "R²",
                                f"{correlation_r_squared(r):.3f}"
                            )

                        st.write(
                            interpret_correlation(r)
                        )

                        st.write(
                            interpret_r_squared(
                                correlation_r_squared(r)
                            )
                        )

                        if decision["significant"]:
                            st.success(
                                decision["decision"]
                            )
                        else:
                            st.warning(
                                decision["decision"]
                            )

                        st.write(
                            decision["explanation"]
                        )

                    else:

                        result = spearman(
                            df[x],
                            df[y]
                        )

                        r = result["statistic"]
                        p = result["p_value"]
                        decision = p_value_decision(p)

                        st.subheader("📊 Correlation Result")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Correlation",
                                f"{r:.3f}"
                            )

                        with col2:
                            st.metric(
                                "p-value",
                                f"{p:.4f}"
                            )

                        with col3:
                            st.metric(
                                "R²",
                                f"{correlation_r_squared(r):.3f}"
                            )

                        st.write(
                            interpret_correlation(r)
                        )

                        st.write(
                            interpret_r_squared(
                                correlation_r_squared(r)
                            )
                        )

                        if decision["significant"]:
                            st.success(
                                decision["decision"]
                            )
                        else:
                            st.warning(
                                decision["decision"]
                            )

                        st.write(
                            decision["explanation"]
                        )


            # ============================================================
            # TWO INDEPENDENT GROUPS
            # ============================================================

            elif analysis_design == (
                "Compare two independent groups"
            ):

                st.subheader("⚖️ Two Independent Groups")

                col1, col2 = st.columns(2)

                with col1:

                    outcome = st.selectbox(
                        "Numerical outcome",
                        df.columns,
                        key="two_group_outcome"
                    )

                with col2:

                    group = st.selectbox(
                        "Grouping variable",
                        df.columns,
                        key="two_group_variable"
                    )

                method = st.selectbox(
                    "Statistical method",
                    [
                        "Independent samples t-test",
                        "Welch's t-test",
                        "Mann-Whitney U"
                    ],
                    key="two_group_method"
                )

                if st.button(
                    "▶ Run Two-Group Analysis",
                    key="run_two_group"
                ):

                    if outcome == group:

                        st.warning(
                            "Outcome and grouping variables must be different."
                        )

                    else:

                        if method == "Independent samples t-test":

                            result = independent_t_test(
                                df[outcome],
                                df[group]
                            )

                        elif method == "Welch's t-test":

                            result = welch_t_test(
                                df[outcome],
                                df[group]
                            )

                        if method in (
                            "Independent samples t-test",
                            "Welch's t-test"
                        ):

                            if result is None:

                                st.error(
                                    "The selected grouping variable must contain "
                                "exactly two groups."
                                )

                            else:

                                st.success(result["test"])

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric(
                                        "Test statistic",
                                        f"{result['statistic']:.4f}"
                                    )

                                with col2:
                                    st.metric(
                                        "p-value",
                                        f"{result['p_value']:.4f}"
                                    )

                                with col3:
                                    st.metric(
                                        "Significance level",
                                        "0.05"
                                    )

                                st.info(
                                    interpret_p_value(
                                        result["p_value"]
                                    )
                                )

                                # ============================================================
                                # DECISION
                                # ============================================================
                                decision = p_value_decision(
                                    result["p_value"]
                                )

                                if decision["significant"]:
                                    st.success(
                                        f"**Decision:** {decision['decision']}"
                                    )
                                else:
                                    st.warning(
                                        f"**Decision:** {decision['decision']}"
                                    )

                                st.write(
                                    decision["explanation"]
                                )

                                # ============================================================
                                # EFFECT SIZE
                                # ============================================================
                                g1 = df[
                                    df[group] == result["group1"]
                                ][outcome].dropna()
                                g2 = df[
                                    df[group] == result["group2"]
                                ][outcome].dropna()

                                d = cohens_d(
                                    g1,
                                    g2
                                )

                                st.subheader("📏 Effect Size")

                                if d is not None:
                                    st.metric(
                                        "Cohen's d",
                                        f"{d:.3f}"
                                    )
                                    st.write(
                                        interpret_cohens_d(d)
                                    )
                                    warning = practical_significance_warning(
                                        result["p_value"],
                                        d
                                    )
                                    if warning:
                                        st.warning(warning)

                                # ============================================================
                                # CONFIDENCE INTERVAL
                                # ============================================================
                                ci = mean_difference_ci(
                                    g1,
                                    g2
                                )
                                if ci:
                                    st.subheader("📐 95% Confidence Interval")
                                    st.write(
                                        f"Estimated mean difference: "
                                    f"**{ci['difference']:.3f}**"
                                    )
                                    st.write(
                                        f"95% CI: "
                                    f"**[{ci['lower']:.3f}, "
                                    f"{ci['upper']:.3f}]**"
                                    )

                                # ============================================================
                                # RESEARCH INTERPRETATION
                                # ============================================================
                                st.subheader("📝 Research Interpretation")

                                relationship = (
                                    f"a difference in {outcome} between "
                                f"{result['group1']} and {result['group2']}"
                                )

                                research_text = research_interpretation(
                                    decision["significant"],
                                    relationship
                                )

                                st.write(
                                    research_text
                                )

                                if method == "Welch's t-test":

                                    with st.expander(
                                        "📚 Why is Welch's test being recommended?"
                                    ):

                                        explanation = explain_welch()

                                        st.write(
                                            "**What is Welch's method?**"
                                        )
                                        st.write(
                                            explanation["what"]
                                        )
                                        st.write(
                                            "**For two groups:**"
                                        )
                                        st.write(
                                            explanation["welch_t"]
                                        )
                                        st.write(
                                            "**For three or more groups:**"
                                        )
                                        st.write(
                                            explanation["welch_anova"]
                                        )
                                        st.write(
                                            "**Why use it?**"
                                        )
                                        st.write(
                                            explanation["why"]
                                        )
                                        st.warning(
                                            explanation["important"]
                                        )

                        else:

                            result = mann_whitney_analysis(
                                df[outcome],
                                df[group]
                            )

                            if result is None:

                                st.error(
                                    "The grouping variable must contain exactly two groups."
                                )

                            else:

                                st.success(
                                    "Mann-Whitney U test completed."
                                )

                                col1, col2 = st.columns(2)

                                with col1:

                                    st.metric(
                                        "U statistic",
                                        f"{result['statistic']:.3f}"
                                    )

                                with col2:

                                    st.metric(
                                        "p-value",
                                        f"{result['p_value']:.4f}"
                                    )

                                decision = p_value_decision(
                                    result["p_value"]
                                )

                                if decision["significant"]:

                                    st.success(
                                        decision["decision"]
                                    )

                                else:

                                    st.info(
                                        decision["decision"]
                                    )

                                st.write(
                                    decision["explanation"]
                                )

                                # ========================================================
                                # GROUP MEDIANS
                                # ========================================================

                                st.subheader(
                                    "📊 Group Summary"
                                )

                                col1, col2 = st.columns(2)

                                with col1:

                                    st.metric(
                                        result["group1"],
                                        f"Median = {result['median1']:.3f}"
                                    )

                                with col2:

                                    st.metric(
                                        result["group2"],
                                        f"Median = {result['median2']:.3f}"
                                    )

                                # ========================================================
                                # EFFECT SIZE
                                # ========================================================

                                r = rank_biserial_correlation(
                                    df[outcome],
                                    df[group]
                                )

                                st.subheader(
                                    "📏 Effect Size"
                                )

                                if r is not None:

                                    st.metric(
                                        "Rank-biserial correlation",
                                        f"{r:.3f}"
                                    )

                                    st.write(
                                        interpret_rank_biserial(r)
                                    )

                                # ========================================================
                                # EXPLANATION
                                # ========================================================

                                explanation = explain_mann_whitney()

                                with st.expander(
                                    "📚 Why is Mann-Whitney being used?"
                                ):

                                    st.write(
                                        explanation["why"]
                                    )

                                    st.warning(
                                        explanation["not_what"]
                                    )


            # ============================================================
            # THREE OR MORE GROUPS
            # ============================================================

            elif analysis_design == (
                "Compare three or more independent groups"
            ):

                st.subheader("📊 Multi-Group Comparison")

                col1, col2 = st.columns(2)

                with col1:

                    outcome = st.selectbox(
                        "Numerical outcome",
                        df.columns,
                        key="multi_outcome"
                    )

                with col2:

                    group = st.selectbox(
                        "Grouping variable",
                        df.columns,
                        key="multi_group"
                    )

                method = st.selectbox(
                    "Statistical method",
                    [
                        "One-Way ANOVA",
                        "Welch's ANOVA",
                        "Kruskal-Wallis"
                    ],
                    key="multi_method"
                )

                if st.button(
                    "▶ Run Multi-Group Analysis",
                    key="run_multi_group"
                ):

                    if outcome == group:

                        st.warning(
                            "Outcome and grouping variables must be different."
                        )

                    else:

                        if method == "One-Way ANOVA":

                            result = one_way_anova(
                                df[outcome],
                                df[group]
                            )

                            if result is None:

                                st.error(
                                    "At least three independent groups are required."
                                )

                            else:

                                st.success(
                                    result["test"]
                                )

                                col1, col2 = st.columns(2)

                                with col1:

                                    st.metric(
                                        "Test statistic",
                                        f"{result['statistic']:.4f}"
                                    )

                                with col2:

                                    st.metric(
                                        "p-value",
                                        f"{result['p_value']:.4f}"
                                    )

                                # ========================================================
                                # GROUP DESCRIPTIVES
                                # ========================================================

                                st.subheader("📋 Group Descriptive Statistics")

                                summary = group_descriptive_summary(
                                    df,
                                    outcome,
                                    group
                                )

                                st.dataframe(
                                    summary,
                                    use_container_width=True
                                )

                                # ========================================================
                                # ANOVA DECISION
                                # ========================================================

                                guidance = anova_posthoc_guidance(
                                    result["p_value"]
                                )

                                if guidance["significant"]:

                                    st.success(
                                        guidance["message"]
                                    )

                                else:

                                    st.info(
                                        guidance["message"]
                                    )

                                # ====================================================
                                # POST-HOC ANALYSIS
                                # ====================================================

                                if guidance["significant"]:

                                    st.subheader(
                                        "🔍 Post-Hoc Pairwise Comparisons"
                                    )

                                    st.write(
                                        "Because the overall ANOVA is significant, "
                                    "Statify compares the groups to determine "
                                    "where the differences occur."
                                    )

                                    posthoc_method = st.selectbox(
                                        "Multiple-comparison method",
                                        [
                                            "Tukey HSD",
                                            "Bonferroni"
                                        ],
                                        key="posthoc_method"
                                    )

                                    if posthoc_method == "Tukey HSD":

                                        posthoc = tukey_hsd(
                                            df,
                                            outcome,
                                            group
                                        )

                                        if posthoc:

                                            st.dataframe(
                                                posthoc["table"],
                                                use_container_width=True
                                            )

                                            st.subheader(
                                                "📝 Pairwise Interpretation"
                                            )

                                            interpretations = (
                                                interpret_tukey_results(
                                                    posthoc["table"]
                                                )
                                            )

                                            for text in interpretations:

                                                st.write(
                                                    "• " + text
                                                )

                                    else:

                                        posthoc = bonferroni_pairwise(
                                            df,
                                            outcome,
                                            group
                                        )

                                        if posthoc:

                                            st.dataframe(
                                                posthoc,
                                                use_container_width=True
                                            )

                                            st.info(
                                                "Bonferroni adjustment controls the "
                                            "family-wise error rate by making the "
                                            "pairwise significance criterion more "
                                            "conservative."
                                            )

                        # ========================================================
                        # WELCH'S ANOVA
                        # ========================================================

                        elif method == "Welch's ANOVA":

                            result = welch_anova(
                                df,
                                outcome,
                                group
                            )

                            if result is None:

                                st.error(
                                    "Welch's ANOVA requires at least three "
                                "independent groups."
                                )

                            elif not result.get(
                                "available",
                                True
                            ):

                                st.error(
                                    result["message"]
                                )

                            else:

                                st.success(
                                    result["test"]
                                )

                                col1, col2 = st.columns(2)

                                with col1:

                                    st.metric(
                                        "Welch F statistic",
                                        f"{result['statistic']:.3f}"
                                    )

                                with col2:

                                    st.metric(
                                        "p-value",
                                        f"{result['p_value']:.4f}"
                                    )

                                st.write(
                                    welch_guidance(
                                        result["p_value"]
                                    )
                                )

                                if result["p_value"] < 0.05:

                                    st.subheader(
                                        "🔍 Games-Howell Post-Hoc Comparisons"
                                    )

                                    st.write(
                                        "Because Welch's ANOVA is significant, "
                                    "Statify examines which group means differ "
                                    "using Games-Howell pairwise comparisons."
                                    )

                                    gh = games_howell_posthoc(
                                        df,
                                        outcome,
                                        group
                                    )

                                    st.dataframe(
                                        gh,
                                        use_container_width=True
                                    )

                                with st.expander(
                                    "📚 Why is Welch's test being recommended?"
                                ):

                                    explanation = explain_welch()

                                    st.write(
                                        "**What is Welch's method?**"
                                    )
                                    st.write(
                                        explanation["what"]
                                    )
                                    st.write(
                                        "**For two groups:**"
                                    )
                                    st.write(
                                        explanation["welch_t"]
                                    )
                                    st.write(
                                        "**For three or more groups:**"
                                    )
                                    st.write(
                                        explanation["welch_anova"]
                                    )
                                    st.write(
                                        "**Why use it?**"
                                    )
                                    st.write(
                                        explanation["why"]
                                    )
                                    st.warning(
                                        explanation["important"]
                                    )

                        # ========================================================
                        # KRUSKAL-WALLIS
                        # ========================================================

                        else:

                            result = kruskal_analysis(
                                df[outcome],
                                df[group]
                            )

                            if result is None:

                                st.error(
                                    "At least three independent groups are required."
                                )

                            else:

                                st.success(
                                    "Kruskal-Wallis H test completed."
                                )

                                col1, col2 = st.columns(2)

                                with col1:

                                    st.metric(
                                        "H statistic",
                                        f"{result['statistic']:.3f}"
                                    )

                                with col2:

                                    st.metric(
                                        "p-value",
                                        f"{result['p_value']:.4f}"
                                    )

                                decision = p_value_decision(
                                    result["p_value"]
                                )

                                if decision["significant"]:

                                    st.success(
                                        decision["decision"]
                                    )

                                    st.write(
                                        "The result indicates that at least one "
                                    "group distribution differs from another."
                                    )

                                    st.subheader(
                                        "🔍 Dunn's Post-Hoc Test"
                                    )

                                    correction = st.selectbox(
                                        "Multiple-comparison correction",
                                        [
                                            "bonferroni",
                                            "holm",
                                            "fdr_bh"
                                        ],
                                        key="dunn_correction"
                                    )

                                    if st.button(
                                        "Run Dunn's Post-Hoc",
                                        key="run_dunn"
                                    ):

                                        dunn = dunn_posthoc(
                                            df,
                                            outcome,
                                            group,
                                            correction
                                        )

                                        st.dataframe(
                                            dunn,
                                            use_container_width=True
                                        )

                                        st.write(
                                            "The table contains adjusted pairwise "
                                        "p-values. Small adjusted p-values indicate "
                                        "evidence of a difference between the "
                                        "corresponding groups."
                                        )

                                else:

                                    st.info(
                                        decision["explanation"]
                                    )

                                with st.expander(
                                    "📚 Why use Kruskal-Wallis instead of One-Way ANOVA?"
                                ):

                                    explanation = explain_kruskal()

                                    st.write(
                                        explanation["what"]
                                    )

                                    st.write(
                                        explanation["why"]
                                    )

                                    st.warning(
                                        explanation["important"]
                                    )


            # ============================================================
            # PAIRED DATA
            # ============================================================

            elif analysis_design == (
                "Compare paired measurements"
            ):

                st.subheader("🔗 Paired Measurements")

                col1, col2 = st.columns(2)

                with col1:

                    before = st.selectbox(
                        "Measurement 1",
                        df.columns,
                        key="paired_before"
                    )

                with col2:

                    after = st.selectbox(
                        "Measurement 2",
                        df.columns,
                        key="paired_after"
                    )

                if st.button(
                    "▶ Run Paired t-test",
                    key="run_paired"
                ):

                    if before == after:

                        st.warning(
                            "Please select two different measurements."
                        )

                    else:

                        result = paired_t_test(
                            df[before],
                            df[after]
                        )

                        st.metric(
                            "t-statistic",
                            f"{result['statistic']:.4f}"
                        )

                        st.metric(
                            "p-value",
                            f"{result['p_value']:.4f}"
                        )

                        st.metric(
                            "Mean difference",
                            f"{result['mean_difference']:.4f}"
                        )

                        st.info(
                            interpret_p_value(
                                result["p_value"]
                            )
                        )


            # ============================================================
            # TWO-WAY ANOVA
            # ============================================================

            elif analysis_design == (
                "Two-Way ANOVA"
            ):

                st.subheader("🧩 Two-Way ANOVA")

                st.write(
                    "Two-way ANOVA evaluates the effects of two categorical "
                "factors on one numerical outcome, including their "
                "interaction."
                )

                outcome = st.selectbox(
                    "Numerical outcome",
                    df.columns,
                    key="two_way_outcome"
                )

                factor_a = st.selectbox(
                    "Factor 1",
                    df.columns,
                    key="two_way_factor1"
                )

                factor_b = st.selectbox(
                    "Factor 2",
                    df.columns,
                    key="two_way_factor2"
                )

                if st.button(
                    "▶ Run Two-Way ANOVA",
                    key="run_two_way"
                ):

                    if len({
                        outcome,
                        factor_a,
                        factor_b
                    }) < 3:

                        st.warning(
                            "Outcome, Factor 1 and Factor 2 must all be different."
                        )

                    else:

                        result = two_way_anova(
                            df,
                            outcome,
                            factor_a,
                            factor_b
                        )

                        if result is None:

                            st.error(
                                "Two-way ANOVA could not be performed."
                            )

                        elif "error" in result:

                            st.error(
                                result["error"]
                            )

                        else:

                            st.success(
                                "Two-way ANOVA completed."
                            )

                            table = result["anova_table"]

                            st.dataframe(
                                table,
                                use_container_width=True
                            )

                            st.info(
                                "The C(Factor1):C(Factor2) row represents "
                            "the interaction between the two factors."
                            )

                            st.subheader(
                                "🧠 Statistical Interpretation"
                            )

                            # ------------------------------------------------
                            # Find effects
                            # ------------------------------------------------

                            effect_sizes = (
                                partial_eta_squared(
                                    table
                                )
                            )

                            interaction_name = (
                                f'C(Q("{factor_a}")):'
                            f'C(Q("{factor_b}"))'
                            )

                            factor_a_name = (
                                f'C(Q("{factor_a}"))'
                            )

                            factor_b_name = (
                                f'C(Q("{factor_b}"))'
                            )

                            # ------------------------------------------------
                            # FACTOR A
                            # ------------------------------------------------

                            if factor_a_name in table.index:

                                p_a = table.loc[
                                    factor_a_name,
                                    "PR(>F)"
                                ]

                                st.markdown(
                                    f"### 1️⃣ Effect of {factor_a}"
                                )

                                st.write(
                                    main_effect_interpretation(
                                        p_a,
                                        factor_a
                                    )
                                )

                                eta_a = effect_sizes.get(
                                    factor_a_name
                                )

                                if eta_a is not None:

                                    st.write(
                                        interpret_partial_eta_squared(
                                            eta_a
                                        )
                                    )

                            # ------------------------------------------------
                            # FACTOR B
                            # ------------------------------------------------

                            if factor_b_name in table.index:

                                p_b = table.loc[
                                    factor_b_name,
                                    "PR(>F)"
                                ]

                                st.markdown(
                                    f"### 2️⃣ Effect of {factor_b}"
                                )

                                st.write(
                                    main_effect_interpretation(
                                        p_b,
                                        factor_b
                                    )
                                )

                                eta_b = effect_sizes.get(
                                    factor_b_name
                                )

                                if eta_b is not None:

                                    st.write(
                                        interpret_partial_eta_squared(
                                            eta_b
                                        )
                                    )

                            # ------------------------------------------------
                            # INTERACTION
                            # ------------------------------------------------

                            if interaction_name in table.index:

                                p_interaction = table.loc[
                                    interaction_name,
                                    "PR(>F)"
                                ]

                                st.markdown(
                                    "### 3️⃣ Interaction"
                                )

                                st.write(
                                    interaction_interpretation(
                                        p_interaction,
                                        factor_a,
                                        factor_b
                                    )
                                )

                                eta_interaction = (
                                    effect_sizes.get(
                                        interaction_name
                                    )
                                )

                                if eta_interaction is not None:

                                    st.write(
                                        interpret_partial_eta_squared(
                                            eta_interaction
                                        )
                                    )

                                if p_interaction < 0.05:

                                    st.warning(
                                        "⚠️ Significant interaction detected. "
                                    "The effect of one factor depends on the level of "
                                    "the other factor. Interpret the main effects "
                                    "cautiously and examine the interaction plot and "
                                    "simple effects before drawing substantive conclusions."
                                    )

                            # ------------------------------------------------
                            # GROUP COMBINATION SUMMARY
                            # ------------------------------------------------

                            st.subheader(
                                "📋 Group Combination Summary"
                            )

                            means = cell_means(
                                df,
                                outcome,
                                factor_a,
                                factor_b
                            )

                            st.dataframe(
                                means,
                                use_container_width=True
                            )

                            # ------------------------------------------------
                            # INTERACTION PLOT
                            # ------------------------------------------------

                            st.subheader(
                                "📈 Interaction Plot"
                            )

                            import plotly.express as px

                            fig = px.line(
                                means,
                                x=factor_a,
                                y="Mean",
                                color=factor_b,
                                markers=True,
                                title=(
                                    f"Interaction between "
                                f"{factor_a} and {factor_b}"
                                )
                            )

                            fig.update_layout(
                                xaxis_title=factor_a,
                                yaxis_title=f"Mean {outcome}",
                                legend_title=factor_b
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            # ------------------------------------------------
                            # EXPLANATION
                            # ------------------------------------------------

                            with st.expander(
                                "📚 What does Two-Way ANOVA actually test?"
                            ):
                                st.write(
                                    f"Two-Way ANOVA examines whether "
                                f"{factor_a} is associated with differences "
                                f"in {outcome}, whether {factor_b} is "
                                f"associated with differences in {outcome}, "
                                f"and whether the effect of one factor "
                                f"depends on the level of the other factor."
                                )
                                st.write(
                                    "The interaction term is especially important. "
                                "A significant interaction means that the "
                                "effect of one factor is not constant across "
                                "the levels of the other factor."
                                )
                                st.info(
                                    "Remember: statistical significance does "
                                "not automatically imply practical importance. "
                                "Statify therefore reports effect sizes "
                                "alongside p-values."
                                )


            # ============================================================
            # CATEGORICAL ASSOCIATION
            # ============================================================

            elif analysis_design == (
                "Association between categorical variables"
            ):

                st.subheader("🔗 Categorical Association")

                col1, col2 = st.columns(2)

                with col1:

                    variable1 = st.selectbox(
                        "Categorical variable 1",
                        df.columns,
                        key="cat_variable1"
                    )

                with col2:

                    variable2 = st.selectbox(
                        "Categorical variable 2",
                        df.columns,
                        key="cat_variable2"
                    )

                method = st.selectbox(
                    "Statistical method",
                    [
                        "Chi-square test",
                        "Fisher's exact test"
                    ],
                    key="cat_method"
                )

                if st.button(
                    "▶ Run Categorical Analysis",
                    key="run_categorical"
                ):

                    if variable1 == variable2:

                        st.warning(
                            "Please select two different variables."
                        )

                    elif method == "Chi-square test":

                        result = chi_square(
                            df[variable1],
                            df[variable2]
                        )

                        st.dataframe(
                            result["table"],
                            use_container_width=True
                        )

                        st.metric(
                            "Chi-square",
                            f"{result['statistic']:.4f}"
                        )

                        st.metric(
                            "p-value",
                            f"{result['p_value']:.4f}"
                        )

                        st.info(
                            interpret_p_value(
                                result["p_value"]
                            )
                        )

                    else:

                        result = fisher_exact(
                            df[variable1],
                            df[variable2]
                        )

                        if result is None:

                            st.error(
                                "Fisher's exact test currently requires "
                            "a 2 × 2 contingency table."
                            )

                        else:

                            st.dataframe(
                                result["table"],
                                use_container_width=True
                            )

                            st.metric(
                                "Odds ratio",
                                f"{result['odds_ratio']:.4f}"
                            )

                            st.metric(
                                "p-value",
                                f"{result['p_value']:.4f}"
                            )

                            st.info(
                                interpret_p_value(
                                    result["p_value"]
                                )
                            )

            # ============================================================
            # NON-PARAMETRIC ANALYSIS GUIDE
            # ============================================================
            st.divider()
            st.header("📉 Non-Parametric Analysis Guide")
            st.write(
                "Statify explains why a rank-based method may be appropriate "
            "instead of simply giving the test name."
            )
            nonparametric_test = st.selectbox(
                "Choose a non-parametric method",
                [
                    "Mann-Whitney U",
                    "Wilcoxon signed-rank",
                    "Kruskal-Wallis"
                ],
                key="nonparametric_guide"
            )
            if nonparametric_test == "Mann-Whitney U":
                explanation = explain_mann_whitney()
                st.subheader("What is it?")
                st.write(
                    explanation["what"]
                )
                st.subheader("Why might Statify recommend it?")
                st.write(
                    explanation["why"]
                )
                st.subheader("⚠️ Important interpretation")
                st.warning(
                    explanation["not_what"]
                )
                st.subheader("How should the result be interpreted?")
                st.write(
                    explanation["interpretation"]
                )
            elif nonparametric_test == "Wilcoxon signed-rank":
                explanation = explain_wilcoxon()
                st.subheader("What is it?")
                st.write(
                    explanation["what"]
                )
                st.subheader("Why might Statify recommend it?")
                st.write(
                    explanation["why"]
                )
                st.subheader("How should the result be interpreted?")
                st.write(
                    explanation["interpretation"]
                )
            else:
                explanation = explain_kruskal()
                st.subheader("What is it?")
                st.write(
                    explanation["what"]
                )
                st.subheader("Why might Statify recommend it?")
                st.write(
                    explanation["why"]
                )
                st.subheader("How should the result be interpreted?")
                st.write(
                    explanation["interpretation"]
                )
                st.subheader("Important")
                st.warning(
                    explanation["important"]
                )
            st.info(
                nonparametric_warning()
            )


            # ============================================================

        elif page == "📝 Research Intelligence":
            # DATA STORY
            # ============================================================
            st.divider()
            st.header("📖 Data Story")
            st.write(
                "Statify automatically scans the dataset for important "
            "patterns, distributions, missing-data issues and "
            "potentially interesting findings."
            )
            if st.button(
                "🔎 Generate Data Findings",
                key="generate_data_findings"
            ):
                numeric_columns = df.select_dtypes(
                    include="number"
                ).columns.tolist()

                categorical_columns = df.select_dtypes(
                    exclude="number"
                ).columns.tolist()

                findings = generate_findings(
                    df,
                    numeric_columns,
                    categorical_columns
                )
                if not findings:
                    st.info(
                        "Statify did not identify major automatic findings "
                    "from the available diagnostics."
                    )
                else:
                    st.subheader(
                        "💡 Key Findings"
                    )
                    for finding in findings:
                        if finding["type"] == "Missing Data":
                            st.warning(
                                f"**{finding['type']} — "
                            f"{finding['variable']}**\n\n"
                            f"{finding['finding']}"
                            )
                        elif finding["type"] == "Distribution":
                            st.info(
                                f"**{finding['type']} — "
                            f"{finding['variable']}**\n\n"
                            f"{finding['finding']}"
                            )
                        else:
                            st.write(
                                f"**{finding['type']} — "
                            f"{finding['variable']}**"
                            )
                            st.write(
                                finding["finding"]
                            )

                st.subheader(
                    "📊 Numerical Overview"
                )

                summary = dataset_numerical_summary(
                    df,
                    numeric_columns
                )

                st.dataframe(
                    summary,
                    use_container_width=True
                )

                st.subheader(
                    "📈 Distribution Patterns"
                )
                skewness = detect_skewness(
                    df,
                    numeric_columns
                )
                st.dataframe(
                    skewness,
                    use_container_width=True
                )

                st.subheader(
                    "🕳️ Missing Data Overview"
                )
                missing = detect_missing_patterns(
                    df
                )
                st.dataframe(
                    missing,
                    use_container_width=True
                )

                st.subheader(
                    "🎓 Researcher's Perspective"
                )
                st.write(
                    "The findings above are exploratory. They are intended "
                "to help identify patterns that may deserve further "
                "statistical investigation. They should not be treated "
                "as proof of causation."
                )
                st.info(
                    "💡 A useful research workflow is: "
                "identify a pattern → formulate a research question → "
                "select an appropriate statistical method → test the "
                "evidence → interpret the result in context."
                )
            # ============================================================
            # RELATIONSHIP DISCOVERY
            # ============================================================

            st.divider()

            st.header("🔗 Relationship Discovery")

            st.write(
                "Statify automatically examines relationships between "
            "numerical variables and identifies potentially important "
            "associations."
            )

            correlation_method = st.selectbox(
                "Correlation method",
                [
                    "Pearson",
                    "Spearman"
                ],
                key="story_correlation_method"
            )

            method = correlation_method.lower()

            if st.button(
                "🔎 Discover Relationships",
                key="discover_relationships"
            ):

                correlations = correlation_analysis(
                    df,
                    numeric_columns,
                    method=method
                )

                correlations = interpret_correlations(
                    correlations
                )

                if correlations.empty:

                    st.info(
                        "Not enough numerical data to identify relationships."
                    )

                else:

                    st.subheader(
                        "📊 Relationship Results"
                    )

                    display_columns = [
                        "Variable 1",
                        "Variable 2",
                        "Correlation",
                        "p-value",
                        "N",
                        "Strength",
                        "Direction",
                        "Significant"
                    ]

                    st.dataframe(
                        correlations[display_columns],
                        use_container_width=True
                    )

                    st.subheader(
                        "⭐ Strongest Relationships"
                    )

                    strongest = strongest_relationships(
                        correlations
                    )

                    st.dataframe(
                        strongest[display_columns],
                        use_container_width=True
                    )
                    st.subheader(
                        "🧠 Relationship Insights"
                    )

                    significant = correlations[
                        correlations["Significant"]
                    ]

                    if significant.empty:

                        st.info(
                            "No statistically significant correlations "
                        "were detected at α = 0.05."
                        )

                    else:

                        for _, row in significant.head(5).iterrows():

                            r = row["Correlation"]

                            direction = row["Direction"]
                            strength = row["Strength"]

                            st.write(
                                f"**{row['Variable 1']} ↔ "
                            f"{row['Variable 2']}**"
                            )

                            st.write(
                                f"A {strength.lower()} {direction.lower()} "
                            f"association was detected "
                            f"(r = {r:.3f}, "
                            f"p = {row['p-value']:.4f})."
                            )

                            st.caption(
                                "Association does not establish causation."
                            )
                    st.subheader(
                        "📈 Visualize an Important Relationship"
                    )

                    if not strongest.empty:

                        selected_relationship = st.selectbox(
                            "Choose relationship to visualize",
                            [
                                f"{row['Variable 1']} ↔ {row['Variable 2']}"
                                for _, row in strongest.iterrows()
                            ],
                            key="relationship_plot_selection"
                        )

                        selected_row = strongest.iloc[
                            [
                                f"{row['Variable 1']} ↔ {row['Variable 2']}"
                                for _, row in strongest.iterrows()
                            ].index(selected_relationship)
                        ]

                        x_var = selected_row["Variable 1"]
                        y_var = selected_row["Variable 2"]

                        plot_data = df[
                            [x_var, y_var]
                        ].dropna()

                        import plotly.express as px

                        fig = px.scatter(
                            plot_data,
                            x=x_var,
                            y=y_var,
                            trendline="ols",
                            title=(
                                f"{x_var} vs {y_var}"
                            )
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )
            # ============================================================
            # STATIFY DATA STORY
            # ============================================================

            st.divider()

            st.header("📖 Statify Data Story")

            st.write(
                "This section brings together important statistical "
            "patterns detected across the dataset."
            )

            if st.button(
                "✨ Build Data Story",
                key="build_data_story"
            ):

                findings = generate_findings(
                    df,
                    numeric_columns,
                    categorical_columns
                )

                correlations = correlation_analysis(
                    df,
                    numeric_columns,
                    method="pearson"
                )

                correlations = interpret_correlations(
                    correlations
                )

                st.subheader(
                    "1️⃣ Dataset Overview"
                )

                st.write(
                    f"The dataset contains **{len(df):,} observations** "
                f"and **{len(df.columns)} variables**."
                )

                st.subheader(
                    "2️⃣ Important Data Patterns"
                )

                if findings:

                    for finding in findings[:8]:

                        st.write(
                            f"• {finding['finding']}"
                        )

                else:

                    st.write(
                        "No major automatic patterns were identified."
                    )

                st.subheader(
                    "3️⃣ Important Relationships"
                )

                if not correlations.empty:

                    strongest = strongest_relationships(
                        correlations,
                        top_n=5
                    )

                    for _, row in strongest.iterrows():

                        st.write(
                            f"• **{row['Variable 1']}** and "
                        f"**{row['Variable 2']}**: "
                        f"r = {row['Correlation']:.3f}, "
                        f"{row['Strength'].lower()} "
                        f"{row['Direction'].lower()} association."
                        )

                else:

                    st.write(
                        "No relationships could be assessed."
                    )

                st.subheader(
                    "4️⃣ Researcher's Perspective"
                )

                st.info(
                    """
                These findings are exploratory. They identify patterns
                that may deserve further investigation. A detected
                association does not prove causation, and statistical
                significance should be considered alongside effect size,
                study design and substantive context.
                """
                )
            # ============================================================
            # PATTERN INTELLIGENCE
            # ============================================================

            st.divider()

            st.subheader("📈 Ordered Pattern Analysis")

            st.write(
                "Statify examines variability, extreme values and "
            "ordered observations for potentially important patterns."
            )

            if numeric_columns:

                pattern_results = generate_pattern_summary(
                    df,
                    numeric_columns
                )

                if pattern_results:

                    st.subheader(
                        "📊 Variability Patterns"
                    )

                    for pattern in pattern_results:

                        st.write(
                            f"• {pattern['pattern']}"
                        )

                    st.subheader(
                        "📌 Extreme Values"
                    )

                    extremes = detect_extremes(
                        df,
                        numeric_columns
                    )

                    st.dataframe(
                        extremes,
                        use_container_width=True
                    )
            st.subheader(
                "📈 Ordered Pattern Analysis"
            )

            trend_order = st.selectbox(
                "Select ordering variable",
                df.columns,
                key="trend_order"
            )

            trend_value = st.selectbox(
                "Select variable to analyse",
                numeric_columns,
                key="trend_value"
            )

            if trend_value is None:

                st.info(
                    "This dataset has no numeric variables to analyse."
                )

            elif trend_order == trend_value:

                st.info(
                    "Choose a different variable for ordering than the "
                    "one being analysed."
                )

            elif st.button(
                "Analyse Trend",
                key="analyse_trend"
            ):

                trend = detect_trend(
                    df,
                    trend_value,
                    trend_order
                )

                if trend is None:

                    st.warning(
                        "There are not enough valid observations "
                    "to estimate a trend."
                    )

                else:

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Direction",
                            trend["direction"]
                        )

                    with col2:

                        st.metric(
                            "Slope",
                            f"{trend['slope']:.3f}"
                        )

                    with col3:

                        if pd.isna(
                            trend["percentage_change"]
                        ):

                            change = "N/A"

                        else:

                            change = (
                                f"{trend['percentage_change']:.2f}%"
                            )

                        st.metric(
                            "Overall Change",
                            change
                        )
                    import plotly.express as px

                    plot_data = df[
                        [trend_order, trend_value]
                    ].copy()

                    plot_data[trend_value] = pd.to_numeric(
                        plot_data[trend_value],
                        errors="coerce"
                    )

                    plot_data = plot_data.dropna()

                    plot_data = plot_data.sort_values(
                        trend_order
                    )

                    fig = px.line(
                        plot_data,
                        x=trend_order,
                        y=trend_value,
                        markers=True,
                        title=(
                            f"{trend_value} over {trend_order}"
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.info(
                        f"Statify detected an "
                    f"**{trend['direction'].lower()}** pattern "
                    f"in {trend_value} across the selected ordering "
                    f"variable."
                    )
            # ============================================================
            # RESEARCH INTELLIGENCE DASHBOARD
            # ============================================================

            st.divider()

            st.header("🧠 Research Intelligence Dashboard")

            st.write(
                "A consolidated view of the most important patterns "
            "identified in your dataset."
            )

            if st.button(
                "🚀 Generate Research Intelligence",
                key="research_intelligence"
            ):

                findings = generate_findings(
                    df,
                    numeric_columns,
                    categorical_columns
                )

                correlations = correlation_analysis(
                    df,
                    numeric_columns,
                    method="pearson"
                )

                correlations = interpret_correlations(
                    correlations
                )

                st.subheader(
                    "📋 Dataset Profile"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Observations",
                        f"{len(df):,}"
                    )

                with col2:

                    st.metric(
                        "Variables",
                        len(df.columns)
                    )

                with col3:

                    missing_total = (
                        df.isna().sum().sum()
                    )

                    st.metric(
                        "Missing Values",
                        int(missing_total)
                    )


                # ========================================================
                # KEY FINDINGS
                # ========================================================

                st.subheader(
                    "💡 Key Findings"
                )

                if findings:

                    for finding in findings[:10]:

                        st.write(
                            f"• {finding['finding']}"
                        )

                else:

                    st.write(
                        "No major automatic findings were detected."
                    )


                # ========================================================
                # RELATIONSHIPS
                # ========================================================

                st.subheader(
                    "🔗 Strongest Relationships"
                )

                if not correlations.empty:

                    strongest = strongest_relationships(
                        correlations,
                        top_n=5
                    )

                    for _, row in strongest.iterrows():

                        significance = (
                            "statistically significant"
                            if row["p-value"] < 0.05
                            else "not statistically significant"
                        )

                        st.write(
                            f"• **{row['Variable 1']}** and "
                        f"**{row['Variable 2']}** show a "
                        f"{row['Strength'].lower()} "
                        f"{row['Direction'].lower()} association "
                        f"(r = {row['Correlation']:.3f}, "
                        f"p = {row['p-value']:.4f}; "
                        f"{significance})."
                        )

                else:

                    st.write(
                        "No numerical relationships could be assessed."
                    )


                # ========================================================
                # RESEARCH GUIDANCE
                # ========================================================

                st.subheader(
                    "🎓 Research Guidance"
                )

                st.info(
                    """
                Use these findings as a starting point for further
                investigation. Descriptive patterns and associations
                do not by themselves establish causation.

                Before drawing conclusions, consider the research
                question, study design, sampling method, assumptions,
                effect size and substantive context.
                """
                )
            # ============================================================

        elif page == "📊 Performance Intelligence":

            st.header("📊 Performance Intelligence")

            selected_domain = st.session_state.get(
                "statify_domain",
                "🔬 General / Other"
            )

            st.write(
                f"Analyzing performance for **{selected_domain}**. Statify will "
            "identify the performance-related variables for that field and "
            "summarize the strongest performers in your dataset. "
            "Switch fields anytime with the **🌐 Your Field** picker in the sidebar."
            )

            top_n = st.slider(
                "Number of top performers to display",
                min_value=5,
                max_value=20,
                value=10,
                step=5,
                key="performance_top_n"
            )

            if st.button(
                "🚀 Generate Performance Intelligence",
                key="generate_performance_intelligence"
            ):

                try:

                    profile, sections, findings = generate_performance_report(
                        df,
                        selected_domain,
                        top_n
                    )

                    st.session_state.performance_profile = profile
                    st.session_state.performance_sections = sections
                    st.session_state.performance_findings = findings

                except Exception as error:

                    st.error(
                        f"Performance analysis could not be completed: {error}"
                    )

            if "performance_profile" in st.session_state:

                profile = st.session_state.performance_profile
                sections = st.session_state.performance_sections
                findings = st.session_state.performance_findings

                # --------------------------------------------------------
                # VARIABLES UNDERSTOOD
                # --------------------------------------------------------

                st.subheader("🧠 Variables Used")

                if profile:

                    profile_table = pd.DataFrame(
                        [
                            {
                                "Performance Role": role.replace("_", " ").title(),
                                "Dataset Variable": column
                            }
                            for role, column in profile.items()
                        ]
                    )

                    st.dataframe(
                        profile_table,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.warning(
                        "No performance-related variables were confidently detected."
                    )

                # --------------------------------------------------------
                # AUTOMATIC FINDINGS
                # --------------------------------------------------------

                st.subheader("💡 Automatic Findings")

                for finding in findings:

                    st.write(f"• {finding}")

                # --------------------------------------------------------
                # PERFORMANCE TABLES
                # --------------------------------------------------------

                if sections:

                    st.subheader("🏆 Performance Rankings")

                    for section_title, result_table in sections:

                        st.markdown(f"### {section_title}")

                        st.dataframe(
                            result_table,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Optional visual ranking
                        if (
                            not result_table.empty
                            and len(result_table.columns) >= 2
                            and pd.api.types.is_numeric_dtype(
                                result_table.iloc[:, 1]
                            )
                        ):

                            chart_data = result_table.copy()

                            st.bar_chart(
                                chart_data.set_index(
                                    chart_data.columns[0]
                                )[chart_data.columns[1]]
                            )

                else:

                    st.info(
                        "No automatic performance ranking could be generated "
                    "from the available variables."
                    )

                st.caption(
                    "Performance results are descriptive summaries of the uploaded "
                "data. They do not by themselves establish causation or explain "
                "why an entity performed differently."
                )

        elif page == "📄 Export":
            st.header("📄 Export")
            st.write(
                "Download the current cleaned working dataset and its reproducible "
            "cleaning history."
            )

            cleaned_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Cleaned CSV",
                data=cleaned_csv,
                file_name="statify_cleaned_data.csv",
                mime="text/csv",
                key="export_cleaned_csv"
            )

            history_df = history_table(
                st.session_state.get("cleaning_history", [])
            )

            if history_df.empty:
                st.info("No cleaning transformations have been recorded yet.")
            else:
                st.subheader("Cleaning History")
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )
                history_csv = history_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Cleaning Log",
                    data=history_csv,
                    file_name="statify_cleaning_log.csv",
                    mime="text/csv",
                    key="export_cleaning_log"
                )
    except Exception as page_error:
        import traceback as _tb
        _tb.print_exc()  # full details go to the server log ("Manage app")
        st.error(
            f"Something on the **{page}** page could not process this dataset. "
            "Your data is unchanged and the other pages still work. "
            "Try a different variable selection, or clean the data first."
        )
        with st.expander("Technical details"):
            st.code(f"{type(page_error).__name__}: {page_error}")
