"""
app.py
------
Streamlit dashboard: Factory-to-Customer Shipping Route Efficiency Analysis
for Nassau Candy Distributor.
"""

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
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy | Shipping Route Efficiency",
    page_icon="🚚",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_and_clean_data()


df_raw = get_data()

# ---------------------------------------------------------------------------
# Sidebar — global filters (User Capabilities)
# ---------------------------------------------------------------------------
st.sidebar.title("🚚 Nassau Candy")
st.sidebar.caption("Factory-to-Customer Shipping Route Efficiency Analysis")
st.sidebar.divider()

# Date range filter
min_date, max_date = df_raw["Order Date"].min(), df_raw["Order Date"].max()
date_range = st.sidebar.date_input(
    "Order Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Region / State selector
regions = sorted(df_raw["Region"].unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

available_states = sorted(df_raw[df_raw["Region"].isin(selected_regions)]["State/Province"].unique())
selected_states = st.sidebar.multiselect("State / Province", available_states, default=available_states)

# Ship Mode filter
ship_modes = sorted(df_raw["Ship Mode"].unique())
selected_modes = st.sidebar.multiselect("Ship Mode", ship_modes, default=ship_modes)

# Lead-time threshold slider (drives Delay Frequency KPI everywhere)
lt_min, lt_max = int(df_raw["Lead_Time"].min()), int(df_raw["Lead_Time"].max())
default_threshold = int(df_raw["Lead_Time"].median())
delay_threshold = st.sidebar.slider(
    "Delay threshold (days)",
    min_value=lt_min, max_value=lt_max, value=default_threshold,
    help="Shipments with Lead Time above this value count as 'delayed'.",
)

st.sidebar.divider()
st.sidebar.caption(
    f"⚠️ Data note: Ship Date values in this dataset run several years past "
    f"Order Date (avg lead time ≈ {int(df_raw['Lead_Time'].mean())} days), and "
    f"barely vary by Ship Mode. This looks like a data-generation artifact in "
    f"the source file, not real-world shipping. Metrics below still use the "
    f"literal Ship Date − Order Date definition, so treat them as relative "
    f"(route vs. route) rather than as calendar-accurate day counts."
)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = min_date, max_date

df = df_raw[
    (df_raw["Order Date"] >= start_date)
    & (df_raw["Order Date"] <= end_date)
    & (df_raw["Region"].isin(selected_regions))
    & (df_raw["State/Province"].isin(selected_states))
    & (df_raw["Ship Mode"].isin(selected_modes))
]

# ---------------------------------------------------------------------------
# Header + KPI cards
# ---------------------------------------------------------------------------
st.title("📦 Factory-to-Customer Shipping Route Efficiency")

if df.empty:
    st.warning("No shipments match the selected filters. Try widening your filter selection.")
    st.stop()

kpis = compute_kpis(df, delay_threshold)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Shipments", f"{kpis['total_orders']:,}")
c2.metric("Avg Lead Time", f"{kpis['avg_lead_time']} days")
c3.metric("Delayed Shipments", f"{kpis['delayed_orders']:,}")
c4.metric("Delay Rate", f"{kpis['delay_percentage']}%")
c5.metric("Active Routes", f"{kpis['total_routes']}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs — Dashboard Modules
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Route Efficiency Overview",
    "🗺️ Geographic Shipping Map",
    "🚢 Ship Mode Comparison",
    "🔍 Route Drill-Down",
])

# ===== TAB 1: Route Efficiency Overview ====================================
with tab1:
    route_stats = get_route_stats(df)

    st.subheader("Top 15 Bottleneck Routes (highest avg lead time)")
    worst = route_stats.sort_values("Avg_Lead_Time", ascending=False).head(15)
    fig_bar = px.bar(
        worst, x="Avg_Lead_Time", y="Route", orientation="h",
        color="Avg_Lead_Time", color_continuous_scale="Reds",
        labels={"Avg_Lead_Time": "Avg Lead Time (days)"},
    )
    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Route Performance Leaderboard")
    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**🟢 Top 10 Most Efficient Routes**")
        top10 = route_stats.sort_values("Efficiency_Score", ascending=False).head(10)
        st.dataframe(
            top10[["Route", "Total_Shipments", "Avg_Lead_Time", "Efficiency_Score"]],
            hide_index=True, use_container_width=True,
        )
    with rcol:
        st.markdown("**🔴 Bottom 10 Least Efficient Routes**")
        bottom10 = route_stats.sort_values("Efficiency_Score", ascending=True).head(10)
        st.dataframe(
            bottom10[["Route", "Total_Shipments", "Avg_Lead_Time", "Efficiency_Score"]],
            hide_index=True, use_container_width=True,
        )

# ===== TAB 2: Geographic Shipping Map =======================================
with tab2:
    state_stats = get_state_stats(df)

    st.subheader("US Heatmap — Average Lead Time by State")
    if state_stats.empty:
        st.info("No US shipments in the current filter selection.")
    else:
        fig_map = px.choropleth(
            state_stats, locations="State_Abbr", locationmode="USA-states",
            color="Avg_Lead_Time", scope="usa",
            color_continuous_scale="RdYlGn_r",
            hover_name="State/Province",
            hover_data={"State_Abbr": False, "Total_Shipments": True},
            labels={"Avg_Lead_Time": "Avg Lead Time (days)"},
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Regional Bottleneck Analysis")
    region_stats = get_region_stats(df)
    fig_region = px.bar(
        region_stats, x="Region", y="Avg_Lead_Time", color="Avg_Lead_Time",
        color_continuous_scale="Oranges", text="Total_Shipments",
        labels={"Avg_Lead_Time": "Avg Lead Time (days)"},
    )
    fig_region.update_traces(texttemplate="%{text} shipments", textposition="outside")
    fig_region.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_region, use_container_width=True)

# ===== TAB 3: Ship Mode Comparison ==========================================
with tab3:
    mode_stats = get_ship_mode_stats(df, delay_threshold)

    st.subheader("Lead Time Distribution by Ship Mode")
    fig_box = px.box(
        df, x="Ship Mode", y="Lead_Time", color="Ship Mode",
        points=False, labels={"Lead_Time": "Lead Time (days)"},
    )
    fig_box.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Delay Rate by Ship Mode")
        fig_delay = px.bar(
            mode_stats, x="Ship Mode", y="Delay_Pct", color="Delay_Pct",
            color_continuous_scale="Reds", labels={"Delay_Pct": "Delay Rate (%)"},
        )
        fig_delay.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_delay, use_container_width=True)
    with col2:
        st.subheader("Cost vs. Lead Time Tradeoff")
        fig_scatter = px.scatter(
            mode_stats, x="Avg_Lead_Time", y="Avg_Cost", size="Total_Shipments",
            color="Ship Mode", text="Ship Mode",
            labels={"Avg_Lead_Time": "Avg Lead Time (days)", "Avg_Cost": "Avg Cost ($)"},
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Ship Mode Summary Table")
    st.dataframe(mode_stats, hide_index=True, use_container_width=True)

# ===== TAB 4: Route Drill-Down ===============================================
with tab4:
    st.subheader("State-Level Performance Insights")

    drill_state = st.selectbox("Choose a state to drill into", sorted(df["State/Province"].unique()))
    state_df = df[df["State/Province"] == drill_state]

    d1, d2, d3 = st.columns(3)
    d1.metric("Shipments", f"{len(state_df):,}")
    d2.metric("Avg Lead Time", f"{round(state_df['Lead_Time'].mean(), 1)} days")
    d3.metric(
        "Delay Rate",
        f"{round((state_df['Lead_Time'] > delay_threshold).mean() * 100, 1)}%",
    )

    st.markdown(f"**Avg lead time by factory shipping into {drill_state}:**")
    factory_breakdown = (
        state_df.groupby("Factory", as_index=False)["Lead_Time"].mean().round(2)
        .sort_values("Lead_Time")
    )
    st.dataframe(factory_breakdown, hide_index=True, use_container_width=True)

    st.subheader("Order-Level Shipment Timeline")
    sample = state_df.sort_values("Order Date").head(30).copy()
    sample["Order_Label"] = sample["Order ID"].astype(str) + " · " + sample["Product Name"]
    if not sample.empty:
        fig_timeline = px.timeline(
            sample, x_start="Order Date", x_end="Ship Date", y="Order_Label",
            color="Ship Mode",
            labels={"Order_Label": "Order"},
        )
        fig_timeline.update_yaxes(autorange="reversed")
        fig_timeline.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=600)
        st.plotly_chart(fig_timeline, use_container_width=True)
        st.caption("Showing the first 30 orders (by Order Date) for the selected state.")
    else:
        st.info("No orders to display for this state with the current filters.")

    with st.expander("View raw order rows for this state"):
        st.dataframe(
            state_df[["Order ID", "Order Date", "Ship Date", "Ship Mode",
                      "Product Name", "Factory", "Lead_Time"]].sort_values("Order Date"),
            hide_index=True, use_container_width=True,
        )
