"""
app.py
------
Streamlit dashboard: Factory-to-Customer Shipping Route Efficiency Analysis
for Nassau Candy Distributor.

Design system (kept intentionally light on custom CSS -- sidebar and base
colors use Streamlit's native theme engine via .streamlit/config.toml, which
renders reliably across versions instead of fighting Streamlit's internal DOM):
    Navy  #14213D  - brand / sidebar / structure
    Teal  #1F8A85  - "efficient" / good performance
    Coral #D64550  - "bottleneck" / delay / poor performance
    Gold  #E8A33D  - highlight / accent
"""

import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px

from data_processing import (
    load_and_clean_data,
    compute_kpis,
    get_route_stats,
    get_state_stats,
    get_region_stats,
    get_ship_mode_stats,
    get_bottleneck_states,
    generate_key_insights,
    get_monthly_trend,
    get_outlier_orders,
    get_route_summary,
    generate_word_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Page config + logo (native Streamlit APIs -- no fragile CSS needed)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy | Shipping Route Efficiency",
    page_icon=os.path.join(BASE_DIR, "assets", "logo_icon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo(
    os.path.join(BASE_DIR, "assets", "logo_sidebar.png"),
    icon_image=os.path.join(BASE_DIR, "assets", "logo_icon.png"),
    size="large",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK, INK_SOFT = "#14213D", "#5B6B85"
TEAL, CORAL, GOLD, SLATE, BORDER = "#1F8A85", "#D64550", "#E8A33D", "#8B98AE", "#E3E6EC"
DISCRETE_PALETTE = [INK, TEAL, GOLD, CORAL, SLATE]
CONTINUOUS_SCALE = [[0.0, TEAL], [0.55, "#F2E9D8"], [1.0, CORAL]]

# ---------------------------------------------------------------------------
# Minimal, low-risk custom styling (only for elements Streamlit has no
# native equivalent for: KPI cards, section eyebrow labels, insight rows)
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@600;700&display=swap');
h1, h2, h3, .section-title, .topbar-title {{ font-family: 'Bricolage Grotesque', sans-serif; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}

.topbar {{ padding: 2px 0 14px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 18px; }}
.topbar-crumb {{ font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: {TEAL}; font-weight: 600; margin-bottom: 4px; }}
.topbar-title {{ font-size: 27px; font-weight: 700; color: {INK}; }}
.topbar-sub {{ font-size: 14px; color: {INK_SOFT}; margin-top: 2px; }}

.section-eyebrow {{ font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: {TEAL}; font-weight: 700; margin-bottom: 1px; }}
.section-title {{ font-size: 18px; font-weight: 700; color: {INK}; }}
.section-caption {{ font-size: 13px; color: {INK_SOFT}; margin-top: 1px; margin-bottom: 4px; }}

.kpi-card {{ background: white; border-radius: 12px; padding: 16px 18px 14px 18px; border-top: 3px solid var(--accent, {TEAL}); box-shadow: 0 1px 3px rgba(20,33,61,0.08); height: 100%; }}
.kpi-icon {{ font-size: 17px; margin-bottom: 6px; opacity: .85; }}
.kpi-label {{ font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: {INK_SOFT}; font-weight: 600; margin-bottom: 3px; }}
.kpi-value {{ font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: {INK}; }}

.insight-row {{ display: flex; align-items: flex-start; gap: 12px; padding: 10px 2px; font-size: 14.5px; line-height: 1.5; color: {INK}; }}
.insight-icon {{ font-size: 16px; flex-shrink: 0; margin-top: 1px; }}

.step-num {{ display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:50%; background:{INK}; color:white; font-weight:700; font-size:13px; font-family:'JetBrains Mono',monospace; flex-shrink:0; }}
.step-title {{ font-weight: 700; color: {INK}; font-size: 15px; }}
</style>
""", unsafe_allow_html=True)


def metric_card(icon, label, value, accent=TEAL):
    st.markdown(f'<div class="kpi-card" style="--accent:{accent}"><div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
                unsafe_allow_html=True)


def section_header(eyebrow, title, caption=None):
    cap = f'<div class="section-caption">{caption}</div>' if caption else ""
    st.markdown(f'<div class="section-eyebrow">{eyebrow}</div><div class="section-title">{title}</div>{cap}',
                unsafe_allow_html=True)


def step_row(num, icon, title, body):
    c1, c2 = st.columns([0.06, 0.94])
    with c1:
        st.markdown(f'<div class="step-num">{num}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="step-title">{icon} {title}</div>', unsafe_allow_html=True)
        st.markdown(body)


def style_fig(fig, height=380, showlegend=True):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=6, r=6, t=10, b=6), height=height, showlegend=showlegend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
    return fig


@st.cache_data
def get_data():
    return load_and_clean_data()


df_raw = get_data()

# ---------------------------------------------------------------------------
# Sidebar — global filters (colors come from [theme.sidebar] in config.toml)
# ---------------------------------------------------------------------------
st.sidebar.caption("FACTORY-TO-CUSTOMER ROUTE ANALYSIS")
st.sidebar.divider()
st.sidebar.markdown("**Filters**")

min_date, max_date = df_raw["Order Date"].min(), df_raw["Order Date"].max()
date_range = st.sidebar.date_input("Order Date range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)

regions = sorted(df_raw["Region"].unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

available_states = sorted(df_raw[df_raw["Region"].isin(selected_regions)]["State/Province"].unique())
selected_states = st.sidebar.multiselect("State / Province", available_states, default=available_states)

ship_modes = sorted(df_raw["Ship Mode"].unique())
selected_modes = st.sidebar.multiselect("Ship Mode", ship_modes, default=ship_modes)

lt_min, lt_max = int(df_raw["Lead_Time"].min()), int(df_raw["Lead_Time"].max())
default_threshold = int(df_raw["Lead_Time"].median())
delay_threshold = st.sidebar.slider("Delay threshold (days)", min_value=lt_min, max_value=lt_max,
                                     value=default_threshold,
                                     help="Shipments with Lead Time above this value count as 'delayed'.")

st.sidebar.divider()
with st.sidebar.expander("ℹ️ About the data"):
    st.caption("Lead Time = Ship Date − Order Date. See the **Methodology** "
               "tab for full cleaning steps and a data quality note.")

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = min_date, max_date

df = df_raw[
    (df_raw["Order Date"] >= start_date) & (df_raw["Order Date"] <= end_date)
    & (df_raw["Region"].isin(selected_regions))
    & (df_raw["State/Province"].isin(selected_states))
    & (df_raw["Ship Mode"].isin(selected_modes))
]

# ---------------------------------------------------------------------------
# Top identity bar
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="topbar">
    <div class="topbar-crumb">Nassau Candy · Supply Chain Analytics</div>
    <div class="topbar-title">Factory-to-Customer Shipping Route Efficiency</div>
    <div class="topbar-sub">Where are shipments slow, and why — by route, region, state, and ship mode.</div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.error("No shipments match the current filters. Widen the date range or re-select "
              "more regions / states / ship modes in the sidebar.")
    st.stop()

kpis = compute_kpis(df, delay_threshold)
route_stats = get_route_stats(df)
state_stats = get_state_stats(df)
region_stats = get_region_stats(df)
mode_stats = get_ship_mode_stats(df, delay_threshold)
bottleneck_states = get_bottleneck_states(state_stats)

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Routes", "Geography", "Ship Modes", "Compare", "Drill-Down", "Methodology"]
)

# ===== TAB 0: Executive Summary ============================================
with tab0:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("📦", "Total Shipments", f"{kpis['total_orders']:,}", INK)
    with c2: metric_card("⏱️", "Avg Lead Time", f"{kpis['avg_lead_time']} d", TEAL)
    with c3: metric_card("⚠️", "Delayed Shipments", f"{kpis['delayed_orders']:,}", GOLD)
    with c4: metric_card("📉", "Delay Rate", f"{kpis['delay_percentage']}%", CORAL)
    with c5: metric_card("🗺️", "Active Routes", f"{kpis['total_routes']}", SLATE)

    st.write("")
    with st.container(border=True):
        section_header("At a glance", "Key Insights", "Computed live from the current filter selection.")
        insights = generate_key_insights(df, route_stats, state_stats, region_stats, mode_stats, delay_threshold)
        for icon, text in zip(["🐌", "📍", "🚩", "🚢", "⏱️"], insights):
            html_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
            st.markdown(f'<div class="insight-row"><span class="insight-icon">{icon}</span><span>{html_text}</span></div>',
                        unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        section_header("Trend", "Lead Time Over Time",
                        "Monthly average — watch for whether performance is improving or worsening.")
        monthly_trend = get_monthly_trend(df)
        if len(monthly_trend) < 2:
            st.info("Not enough months in the current filter selection to show a trend.")
        else:
            fig_trend = px.line(monthly_trend, x="Month", y="Avg_Lead_Time", markers=True,
                                 labels={"Avg_Lead_Time": "Avg Lead Time (days)"})
            fig_trend.update_traces(line_color=TEAL, marker=dict(size=7, color=INK))
            st.plotly_chart(style_fig(fig_trend, height=300, showlegend=False), width="stretch")

    st.write("")
    with st.expander("📖 How to read this dashboard"):
        st.markdown("""
- **Routes** — every Factory → State route ranked fastest to slowest, with a 0–100 Efficiency Score.
- **Geography** — *where* delays cluster: a state map, a regional ranking, and a list of high-volume + slow "congestion-prone" states.
- **Ship Modes** — compares delivery methods on speed, delay rate, and cost.
- **Drill-Down** — pick one state to see its individual orders and shipment timeline.
- **Sidebar filters** apply to every tab at once.
        """)

    st.write("")
    section_header("Export", "Download this analysis")
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.download_button("⬇ Filtered shipment data (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                            file_name="nassau_candy_filtered_shipments.csv", mime="text/csv", width="stretch")
    with exp2:
        st.download_button("⬇ Route performance report (CSV)", data=route_stats.to_csv(index=False).encode("utf-8"),
                            file_name="nassau_candy_route_performance.csv", mime="text/csv", width="stretch")
    with exp3:
        word_bytes = generate_word_report(df, kpis, insights, route_stats, region_stats, mode_stats, delay_threshold)
        st.download_button("⬇ Executive summary report (Word)", data=word_bytes,
                            file_name="nassau_candy_executive_report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            width="stretch")

# ===== TAB 1: Route Efficiency ==============================================
with tab1:
    with st.container(border=True):
        section_header("Bottlenecks", "Top 15 Slowest Routes", "Longer bar = bigger bottleneck.")
        worst = route_stats.sort_values("Avg_Lead_Time", ascending=False).head(15)
        fig_bar = px.bar(worst, x="Avg_Lead_Time", y="Route", orientation="h",
                          color="Avg_Lead_Time", color_continuous_scale=CONTINUOUS_SCALE,
                          labels={"Avg_Lead_Time": "Avg Lead Time (days)"})
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig_bar, height=460, showlegend=False), width="stretch")

    st.write("")
    with st.container(border=True):
        section_header("Leaderboard", "Route Performance Ranking",
                        "Efficiency Score is normalized 0–100 across routes in view; higher is better.")
        lcol, rcol = st.columns(2)
        cfg = {
            "Total_Shipments": st.column_config.NumberColumn("Shipments", format="%d"),
            "Avg_Lead_Time": st.column_config.NumberColumn("Avg Lead Time", format="%.1f d"),
            "Efficiency_Score": st.column_config.ProgressColumn("Efficiency", min_value=0, max_value=100, format="%.0f"),
        }
        with lcol:
            st.markdown("**:green[● Top 10 Most Efficient]**")
            st.dataframe(route_stats.sort_values("Efficiency_Score", ascending=False).head(10)
                         [["Route", "Total_Shipments", "Avg_Lead_Time", "Efficiency_Score"]],
                         hide_index=True, width="stretch", column_config=cfg)
        with rcol:
            st.markdown("**:red[● Bottom 10 Least Efficient]**")
            st.dataframe(route_stats.sort_values("Efficiency_Score", ascending=True).head(10)
                         [["Route", "Total_Shipments", "Avg_Lead_Time", "Efficiency_Score"]],
                         hide_index=True, width="stretch", column_config=cfg)

    st.write("")
    with st.container(border=True):
        section_header("Anomalies", "Outlier Shipments",
                        "Individual orders far slower than normal for their OWN route "
                        "(more than 2 standard deviations above that route's average) — "
                        "worth investigating case-by-case rather than as a route-wide trend.")
        outliers = get_outlier_orders(df)
        if outliers.empty:
            st.success("No statistical outliers detected in the current filter selection.")
        else:
            st.dataframe(
                outliers, hide_index=True, width="stretch",
                column_config={
                    "Lead_Time": st.column_config.NumberColumn("Lead Time", format="%.0f d"),
                    "Route_Avg": st.column_config.NumberColumn("Route Avg", format="%.1f d"),
                    "Deviation": st.column_config.NumberColumn("Above Route Avg", format="%+.1f d"),
                },
            )

# ===== TAB 2: Geographic Analysis ===========================================
with tab2:
    with st.container(border=True):
        section_header("Map", "Average Lead Time by State (US)", "Coral = slower, teal = faster than average.")
        if state_stats.empty:
            st.info("No US shipments in the current filter selection.")
        else:
            fig_map = px.choropleth(state_stats, locations="State_Abbr", locationmode="USA-states",
                                     color="Avg_Lead_Time", scope="usa", color_continuous_scale=CONTINUOUS_SCALE,
                                     hover_name="State/Province",
                                     hover_data={"State_Abbr": False, "Total_Shipments": True},
                                     labels={"Avg_Lead_Time": "Avg Lead Time (days)"})
            fig_map.update_layout(font=dict(family="Inter, sans-serif", color=INK, size=13),
                                   paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=6, r=6, t=10, b=6), height=430)
            st.plotly_chart(fig_map, width="stretch")

    st.write("")
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        with st.container(border=True):
            section_header("Regions", "Regional Bottleneck Ranking", "Label shows shipment volume.")
            fig_region = px.bar(region_stats, x="Avg_Lead_Time", y="Region", orientation="h",
                                 color="Avg_Lead_Time", color_continuous_scale=CONTINUOUS_SCALE,
                                 text="Total_Shipments", labels={"Avg_Lead_Time": "Avg Lead Time (days)"})
            fig_region.update_traces(texttemplate="%{text} shpts", textposition="outside")
            fig_region.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_region, height=320, showlegend=False), width="stretch")
    with col_b:
        with st.container(border=True):
            section_header("Flagged", "Congestion-Prone States", "Above-median volume AND lead time.")
            if bottleneck_states.empty:
                st.info("No states currently flagged under this filter.")
            else:
                st.dataframe(bottleneck_states[["State/Province", "Total_Shipments", "Avg_Lead_Time"]],
                             hide_index=True, width="stretch", height=320,
                             column_config={"Total_Shipments": st.column_config.NumberColumn("Shipments", format="%d"),
                                            "Avg_Lead_Time": st.column_config.NumberColumn("Avg Lead Time", format="%.1f d")})

# ===== TAB 3: Ship Mode Comparison ==========================================
with tab3:
    with st.container(border=True):
        section_header("Distribution", "Lead Time by Ship Mode", "Box = middle 50% of shipments; line = median.")
        fig_box = px.box(df, x="Ship Mode", y="Lead_Time", color="Ship Mode", points=False,
                          color_discrete_sequence=DISCRETE_PALETTE, labels={"Lead_Time": "Lead Time (days)"})
        st.plotly_chart(style_fig(fig_box, height=360, showlegend=False), width="stretch")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section_header("Reliability", "Delay Rate by Ship Mode", f"Share of shipments over {delay_threshold} days.")
            fig_delay = px.bar(mode_stats, x="Ship Mode", y="Delay_Pct", color="Ship Mode",
                                color_discrete_sequence=DISCRETE_PALETTE, labels={"Delay_Pct": "Delay Rate (%)"})
            st.plotly_chart(style_fig(fig_delay, height=320, showlegend=False), width="stretch")
    with col2:
        with st.container(border=True):
            section_header("Tradeoff", "Cost vs. Lead Time", "Bubble size = shipment volume.")
            fig_scatter = px.scatter(mode_stats, x="Avg_Lead_Time", y="Avg_Cost", size="Total_Shipments",
                                      color="Ship Mode", text="Ship Mode", color_discrete_sequence=DISCRETE_PALETTE,
                                      labels={"Avg_Lead_Time": "Avg Lead Time (days)", "Avg_Cost": "Avg Cost ($)"})
            fig_scatter.update_traces(textposition="top center")
            st.plotly_chart(style_fig(fig_scatter, height=320, showlegend=False), width="stretch")

    st.write("")
    with st.container(border=True):
        section_header("Summary", "Ship Mode Performance Table")
        st.dataframe(mode_stats, hide_index=True, width="stretch", column_config={
            "Total_Shipments": st.column_config.NumberColumn("Shipments", format="%d"),
            "Avg_Lead_Time": st.column_config.NumberColumn("Avg Lead Time", format="%.1f d"),
            "Delay_Pct": st.column_config.ProgressColumn("Delay Rate", min_value=0, max_value=100, format="%.1f%%"),
            "Avg_Cost": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
            "Avg_Sales": st.column_config.NumberColumn("Avg Sales", format="$%.2f"),
        })

# ===== TAB 4: Route Comparison ==============================================
with tab4:
    with st.container(border=True):
        section_header("Head-to-Head", "Compare Two Routes",
                        "Pick any two Factory → State routes to compare side by side.")
        all_routes = sorted(route_stats["Route"].unique())
        c1, c2 = st.columns(2)
        with c1:
            route_a = st.selectbox("Route A", all_routes, index=0, key="route_a")
        with c2:
            default_b_idx = 1 if len(all_routes) > 1 else 0
            route_b = st.selectbox("Route B", all_routes, index=default_b_idx, key="route_b")

    st.write("")
    summary_a = get_route_summary(df, route_stats, route_a)
    summary_b = get_route_summary(df, route_stats, route_b)

    col_a, col_b = st.columns(2)
    for col, summary, accent in [(col_a, summary_a, TEAL), (col_b, summary_b, CORAL)]:
        with col:
            with st.container(border=True):
                st.markdown(f"**{summary['Route']}**")
                m1, m2 = st.columns(2)
                with m1:
                    metric_card("📦", "Shipments", f"{summary['Total_Shipments']:,}", accent)
                    metric_card("🏆", "Efficiency Score", f"{summary['Efficiency_Score']:.0f}", accent)
                with m2:
                    metric_card("⏱️", "Avg Lead Time", f"{summary['Avg_Lead_Time']:.1f} d", accent)
                    metric_card("📊", "Std Dev", f"{summary['Lead_Time_StdDev']:.1f} d", accent)

    st.write("")
    with st.container(border=True):
        section_header("Distribution", "Lead Time Spread — Both Routes",
                        "Box shows the middle 50% of shipments for each route.")
        compare_df = df[df["Route"].isin([route_a, route_b])]
        fig_cmp = px.box(compare_df, x="Route", y="Lead_Time", color="Route", points="all",
                          color_discrete_sequence=[TEAL, CORAL], labels={"Lead_Time": "Lead Time (days)"})
        st.plotly_chart(style_fig(fig_cmp, height=380, showlegend=False), width="stretch")

    if summary_a["Avg_Lead_Time"] > 0 and summary_b["Avg_Lead_Time"] > 0:
        faster = summary_a if summary_a["Avg_Lead_Time"] < summary_b["Avg_Lead_Time"] else summary_b
        slower = summary_b if faster is summary_a else summary_a
        diff = slower["Avg_Lead_Time"] - faster["Avg_Lead_Time"]
        pct = round((diff / slower["Avg_Lead_Time"]) * 100, 1) if slower["Avg_Lead_Time"] else 0
        st.info(f"**{faster['Route']}** is faster by **{diff:.1f} days** on average "
                f"({pct}% quicker) than **{slower['Route']}**.")

# ===== TAB 5: Route Drill-Down ================================================
with tab5:
    with st.container(border=True):
        section_header("Explore", "State-Level Performance", "Pick a state to see its factories and orders.")
        drill_state = st.selectbox("State", sorted(df["State/Province"].unique()), label_visibility="collapsed")
        state_df = df[df["State/Province"] == drill_state]
        d1, d2, d3 = st.columns(3)
        with d1: metric_card("📦", "Shipments", f"{len(state_df):,}", INK)
        with d2: metric_card("⏱️", "Avg Lead Time", f"{round(state_df['Lead_Time'].mean(), 1)} d", TEAL)
        with d3: metric_card("📉", "Delay Rate", f"{round((state_df['Lead_Time'] > delay_threshold).mean() * 100, 1)}%", CORAL)

    st.write("")
    col_l, col_r = st.columns([1, 1.6])
    with col_l:
        with st.container(border=True):
            section_header("Breakdown", f"Factories Serving {drill_state}")
            factory_breakdown = (state_df.groupby("Factory", as_index=False)["Lead_Time"].mean().round(2)
                                 .sort_values("Lead_Time"))
            st.dataframe(factory_breakdown, hide_index=True, width="stretch", height=280,
                         column_config={"Lead_Time": st.column_config.NumberColumn("Avg Lead Time", format="%.1f d")})
    with col_r:
        with st.container(border=True):
            section_header("Timeline", "Order-Level Shipment Timeline", "First 30 orders, Order Date to Ship Date.")
            sample = state_df.sort_values("Order Date").head(30).copy()
            sample["Order_Label"] = sample["Order ID"].astype(str) + " · " + sample["Product Name"]
            if not sample.empty:
                fig_timeline = px.timeline(sample, x_start="Order Date", x_end="Ship Date", y="Order_Label",
                                            color="Ship Mode", color_discrete_sequence=DISCRETE_PALETTE,
                                            labels={"Order_Label": "Order"})
                fig_timeline.update_yaxes(autorange="reversed")
                st.plotly_chart(style_fig(fig_timeline, height=280), width="stretch")
            else:
                st.info("No orders to display for this state with the current filters.")

    st.write("")
    with st.expander("View raw order rows for this state"):
        st.dataframe(state_df[["Order ID", "Order Date", "Ship Date", "Ship Mode", "Product Name", "Factory", "Lead_Time"]]
                     .sort_values("Order Date"), hide_index=True, width="stretch")

# ===== TAB 6: Methodology (visual step flow, not a text wall) ===============
with tab6:
    with st.container(border=True):
        section_header("Process", "Analytical Methodology",
                        "How raw order data becomes the metrics shown in this dashboard.")
        st.write("")
        step_row("1", "🧹", "Data Cleaning & Validation",
                 "Text fields are trimmed of extra whitespace. `Order Date` / `Ship Date` are parsed as "
                 "`dd-mm-yyyy`; rows with unparseable or missing dates are dropped. Rows with a negative "
                 "Lead Time are removed as invalid.")
        st.divider()
        step_row("2", "🏗️", "Feature Engineering",
                 "**Lead Time** = `Ship Date − Order Date`. **Factory** is derived from `Product Name` via "
                 "a fixed mapping. **Route** = `Factory → State`. Shipments are grouped by Ship Mode.")
        st.divider()
        step_row("3", "🔗", "Route Aggregation",
                 "Each route is aggregated for total shipments, average lead time, and lead-time "
                 "variability (standard deviation).")
        st.divider()
        step_row("4", "🏆", "Efficiency Benchmarking",
                 "Routes are ranked fastest to slowest. **Efficiency Score** = min-max normalized "
                 "average lead time, inverted and scaled 0–100 (higher = better, relative to routes in view).")
        st.divider()
        step_row("5", "🚩", "Geographic Bottleneck Analysis",
                 "A state is flagged **congestion-prone** when both its average lead time AND shipment "
                 "volume are above the median across all states.")
        st.divider()
        step_row("6", "🚢", "Ship Mode Performance",
                 "Ship modes are compared on lead time, delay rate, and a descriptive cost-time tradeoff "
                 "(average cost vs. average lead time per mode).")

    st.write("")
    with st.container(border=True):
        section_header("Definitions", "Key Performance Indicators")
        st.dataframe(pd.DataFrame({
            "KPI": ["Shipping Lead Time", "Average Lead Time", "Route Volume", "Delay Frequency", "Route Efficiency Score"],
            "Definition": [
                "Ship Date − Order Date, per shipment",
                "Mean shipping duration per route",
                "Number of orders per route",
                "% of shipments exceeding the delay threshold (sidebar slider)",
                "Normalized 0–100 lead-time performance vs. all other routes in view",
            ],
        }), hide_index=True, width="stretch")

    st.write("")
    with st.container(border=True):
        section_header("Transparency", "Data Quality Summary",
                        "Exact record counts from the cleaning pipeline that produced this dashboard.")
        dq1, dq2, dq3, dq4 = st.columns(4)
        with dq1: metric_card("📄", "Raw Rows Loaded", f"{df_raw.attrs.get('rows_initial', 0):,}", SLATE)
        with dq2: metric_card("🧹", "Dropped (Missing Dates)", f"{df_raw.attrs.get('rows_dropped_missing_dates', 0):,}", GOLD)
        with dq3: metric_card("🚫", "Dropped (Invalid Lead Time)", f"{df_raw.attrs.get('rows_dropped_negative_leadtime', 0):,}", CORAL)
        with dq4: metric_card("✅", "Clean Rows Used", f"{df_raw.attrs.get('rows_final', 0):,}", TEAL)

    st.write("")
    with st.container(border=True):
        section_header("Caveat", "Data Quality Note")
        st.warning(
            f"In this dataset, `Ship Date` values fall several years after `Order Date` (overall average "
            f"Lead Time ≈ **{int(df_raw['Lead_Time'].mean()):,} days**), and barely varies by Ship Mode — "
            f"Same Day shipments show a similar average lead time to Standard Class. This is consistent "
            f"with a data-generation artifact rather than real-world shipping performance. All KPIs use "
            f"the literal `Ship Date − Order Date` definition as specified in the project requirements; "
            f"treat them as **relative comparisons between routes/states/modes**, not calendar-accurate "
            f"day counts."
        )
