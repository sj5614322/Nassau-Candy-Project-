"""
data_processing.py
-------------------
Data cleaning, feature engineering, and aggregation logic for the
Nassau Candy Factory-to-Customer Shipping Route Efficiency dashboard.

Pipeline:
    1. Load raw CSV
    2. Clean & validate (dates, geography, duplicates)
    3. Feature engineering (Factory mapping, real Lead Time, Route)
    4. Aggregation helpers for the Streamlit app (route stats, state stats, KPIs)
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Nassau Candy Distributor.csv")

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

# Product -> Factory mapping (verified against the actual Product Name values
# present in the dataset -- all 15 SKUs map cleanly, no unmapped products).
FACTORY_MAPPING = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Kazookles": "The Other Factory",
}

# Full US state name -> 2-letter code, needed for the Plotly choropleth
# (locationmode="USA-states" only accepts postal abbreviations).
US_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_and_clean_data(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and run the full cleaning + feature-engineering pipeline."""

    df = pd.read_csv(data_path)
    rows_initial = len(df)

    # --- 1. Standardize text / geography fields -----------------------------
    text_cols = ["Product Name", "Ship Mode", "Country/Region", "City",
                 "State/Province", "Division", "Region"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # --- 2. Parse dates (dd-mm-yyyy as produced by this dataset) -----------
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")

    # Drop rows where either date failed to parse (missing/garbled shipment records)
    before = len(df)
    df = df.dropna(subset=["Order Date", "Ship Date", "State/Province"])
    dropped_missing = before - len(df)

    # --- 3. Feature engineering: real Lead Time -----------------------------
    df["Lead_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # Remove invalid / negative lead times (a shipment can't ship before it's ordered)
    before = len(df)
    df = df[df["Lead_Time"] >= 0]
    dropped_negative = before - len(df)

    # --- 4. Factory mapping ---------------------------------------------
    df["Factory"] = df["Product Name"].map(FACTORY_MAPPING)
    unmapped_mask = df["Factory"].isna()
    if unmapped_mask.any():
        df.loc[unmapped_mask, "Factory"] = "Unmapped Product"

    # --- 5. Route definitions (Factory -> State, Factory -> Region) --------
    df["Route"] = df["Factory"] + " \u2192 " + df["State/Province"]
    df["Route_Region"] = df["Factory"] + " \u2192 " + df["Region"]

    # --- 6. US state abbreviation for choropleth map ------------------------
    df["State_Abbr"] = df["State/Province"].map(US_STATE_ABBR)

    # Attach cleaning metadata as DataFrame attrs (visible for debugging/QA)
    df.attrs["rows_initial"] = rows_initial
    df.attrs["rows_dropped_missing_dates"] = dropped_missing
    df.attrs["rows_dropped_negative_leadtime"] = dropped_negative
    df.attrs["rows_final"] = len(df)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation helpers used by the Streamlit app
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, delay_threshold: float) -> dict:
    """Top-level KPI numbers for the overview cards."""
    total_orders = len(df)
    if total_orders == 0:
        return {
            "total_orders": 0, "avg_lead_time": 0, "delayed_orders": 0,
            "delay_percentage": 0, "total_routes": 0,
        }

    avg_lead_time = round(df["Lead_Time"].mean(), 2)
    delayed_orders = int((df["Lead_Time"] > delay_threshold).sum())
    delay_percentage = round((delayed_orders / total_orders) * 100, 1)
    total_routes = df["Route"].nunique()

    return {
        "total_orders": total_orders,
        "avg_lead_time": avg_lead_time,
        "delayed_orders": delayed_orders,
        "delay_percentage": delay_percentage,
        "total_routes": total_routes,
    }


def get_route_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-route aggregation: volume, avg lead time, variability, efficiency score."""
    if df.empty:
        return pd.DataFrame(columns=["Route", "Factory", "State/Province",
                                      "Total_Shipments", "Avg_Lead_Time",
                                      "Lead_Time_StdDev", "Efficiency_Score"])

    stats = (
        df.groupby(["Route", "Factory", "State/Province"], as_index=False)
        .agg(
            Total_Shipments=("Order ID", "count"),
            Avg_Lead_Time=("Lead_Time", "mean"),
            Lead_Time_StdDev=("Lead_Time", "std"),
        )
    )
    stats["Lead_Time_StdDev"] = stats["Lead_Time_StdDev"].fillna(0).round(2)
    stats["Avg_Lead_Time"] = stats["Avg_Lead_Time"].round(2)

    # Efficiency Score: min-max normalize avg lead time -> 0-100, inverted
    # so a LOWER lead time gives a HIGHER (better) score.
    lo, hi = stats["Avg_Lead_Time"].min(), stats["Avg_Lead_Time"].max()
    if hi > lo:
        stats["Efficiency_Score"] = round(100 * (1 - (stats["Avg_Lead_Time"] - lo) / (hi - lo)), 1)
    else:
        stats["Efficiency_Score"] = 100.0

    return stats.sort_values("Avg_Lead_Time")


def get_state_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-state aggregation for the geographic choropleth (US only)."""
    us_df = df[df["Country/Region"] == "United States"].dropna(subset=["State_Abbr"])
    if us_df.empty:
        return pd.DataFrame(columns=["State/Province", "State_Abbr",
                                      "Avg_Lead_Time", "Total_Shipments"])

    stats = (
        us_df.groupby(["State/Province", "State_Abbr"], as_index=False)
        .agg(Avg_Lead_Time=("Lead_Time", "mean"), Total_Shipments=("Order ID", "count"))
    )
    stats["Avg_Lead_Time"] = stats["Avg_Lead_Time"].round(2)
    return stats


def get_region_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-region aggregation for the bottleneck bar chart."""
    if df.empty:
        return pd.DataFrame(columns=["Region", "Avg_Lead_Time", "Total_Shipments"])
    stats = (
        df.groupby("Region", as_index=False)
        .agg(Avg_Lead_Time=("Lead_Time", "mean"), Total_Shipments=("Order ID", "count"))
    )
    stats["Avg_Lead_Time"] = stats["Avg_Lead_Time"].round(2)
    return stats.sort_values("Avg_Lead_Time", ascending=False)


def get_ship_mode_stats(df: pd.DataFrame, delay_threshold: float) -> pd.DataFrame:
    """Per-ship-mode aggregation: lead time, delay %, and cost-time tradeoff."""
    if df.empty:
        return pd.DataFrame(columns=["Ship Mode", "Total_Shipments", "Avg_Lead_Time",
                                      "Delay_Pct", "Avg_Cost", "Avg_Sales"])

    grp = df.groupby("Ship Mode")
    stats = grp.agg(
        Total_Shipments=("Order ID", "count"),
        Avg_Lead_Time=("Lead_Time", "mean"),
        Avg_Cost=("Cost", "mean"),
        Avg_Sales=("Sales", "mean"),
    ).reset_index()

    delay_pct = grp.apply(
        lambda g: round((g["Lead_Time"] > delay_threshold).mean() * 100, 1),
        include_groups=False,
    ).reset_index(name="Delay_Pct")

    stats = stats.merge(delay_pct, on="Ship Mode")
    stats["Avg_Lead_Time"] = stats["Avg_Lead_Time"].round(2)
    stats["Avg_Cost"] = stats["Avg_Cost"].round(2)
    stats["Avg_Sales"] = stats["Avg_Sales"].round(2)
    return stats.sort_values("Avg_Lead_Time")


def get_bottleneck_states(state_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Flags 'congestion-prone' states: those with BOTH above-median lead time
    AND above-median shipment volume -- i.e. high traffic that is also slow.
    """
    if state_stats.empty:
        return state_stats

    lt_median = state_stats["Avg_Lead_Time"].median()
    vol_median = state_stats["Total_Shipments"].median()
    flagged = state_stats[
        (state_stats["Avg_Lead_Time"] > lt_median)
        & (state_stats["Total_Shipments"] > vol_median)
    ].copy()
    return flagged.sort_values("Avg_Lead_Time", ascending=False)


def generate_key_insights(df: pd.DataFrame, route_stats: pd.DataFrame,
                           state_stats: pd.DataFrame, region_stats: pd.DataFrame,
                           mode_stats: pd.DataFrame, delay_threshold: float) -> list:
    """
    Produces a short list of plain-English, data-driven insight strings for the
    Executive Summary tab. Every number here is computed live from the current
    (filtered) data -- nothing is hardcoded.
    """
    insights = []
    if df.empty or route_stats.empty:
        return ["No data available for the current filter selection."]

    # 1. Fastest vs slowest route
    fastest = route_stats.loc[route_stats["Avg_Lead_Time"].idxmin()]
    slowest = route_stats.loc[route_stats["Avg_Lead_Time"].idxmax()]
    if fastest["Avg_Lead_Time"] > 0:
        multiple = round(slowest["Avg_Lead_Time"] / fastest["Avg_Lead_Time"], 1)
        insights.append(
            f"**{slowest['Route']}** is the slowest route at "
            f"**{slowest['Avg_Lead_Time']} days** on average -- "
            f"**{multiple}x slower** than the fastest route, "
            f"**{fastest['Route']}** ({fastest['Avg_Lead_Time']} days)."
        )

    # 2. Regional bottleneck
    if not region_stats.empty:
        worst_region = region_stats.iloc[0]
        insights.append(
            f"The **{worst_region['Region']}** region has the highest average "
            f"lead time (**{worst_region['Avg_Lead_Time']} days** across "
            f"{int(worst_region['Total_Shipments'])} shipments), making it the "
            f"top regional bottleneck."
        )

    # 3. Congestion-prone states (high volume + high lead time)
    bottleneck_states = get_bottleneck_states(state_stats)
    if not bottleneck_states.empty:
        top_state = bottleneck_states.iloc[0]
        insights.append(
            f"**{len(bottleneck_states)} state(s)** show both above-average "
            f"shipment volume AND above-average lead time -- flagged as "
            f"congestion-prone. The most severe is "
            f"**{top_state['State/Province']}** "
            f"({top_state['Avg_Lead_Time']} days, {int(top_state['Total_Shipments'])} shipments)."
        )

    # 4. Ship mode performance spread
    if not mode_stats.empty and len(mode_stats) > 1:
        best_mode = mode_stats.iloc[0]
        worst_mode = mode_stats.iloc[-1]
        insights.append(
            f"Among ship modes, **{best_mode['Ship Mode']}** performs best "
            f"({best_mode['Avg_Lead_Time']} days avg, {best_mode['Delay_Pct']}% delayed), "
            f"while **{worst_mode['Ship Mode']}** lags furthest behind "
            f"({worst_mode['Avg_Lead_Time']} days avg, {worst_mode['Delay_Pct']}% delayed)."
        )

    # 5. Overall delay rate
    total = len(df)
    delayed = int((df["Lead_Time"] > delay_threshold).sum())
    pct = round((delayed / total) * 100, 1) if total else 0
    insights.append(
        f"**{pct}%** of all shipments ({delayed:,} of {total:,}) exceed the "
        f"current delay threshold of **{delay_threshold} days**, across "
        f"**{route_stats.shape[0]} distinct factory-to-state routes**."
    )

    return insights


def get_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly aggregation of Lead Time and volume, based on Order Date.
    Powers the 'is performance improving or worsening over time' trend chart.
    """
    if df.empty:
        return pd.DataFrame(columns=["Month", "Avg_Lead_Time", "Total_Shipments"])

    tmp = df.copy()
    tmp["Month"] = tmp["Order Date"].dt.to_period("M").dt.to_timestamp()
    trend = (
        tmp.groupby("Month", as_index=False)
        .agg(Avg_Lead_Time=("Lead_Time", "mean"), Total_Shipments=("Order ID", "count"))
        .sort_values("Month")
    )
    trend["Avg_Lead_Time"] = trend["Avg_Lead_Time"].round(2)
    return trend


def get_outlier_orders(df: pd.DataFrame, std_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Flags individual orders whose Lead Time is a statistical outlier RELATIVE TO
    THEIR OWN ROUTE (mean + std_multiplier * std-dev for that specific route) --
    i.e. shipments that are unusually slow even compared to their own route's
    normal behavior, not just slow in absolute terms.
    """
    if df.empty:
        return pd.DataFrame(columns=["Order ID", "Route", "Order Date", "Ship Date",
                                      "Lead_Time", "Route_Avg", "Deviation"])

    route_agg = df.groupby("Route")["Lead_Time"].agg(["mean", "std"]).reset_index()
    route_agg.columns = ["Route", "Route_Mean", "Route_Std"]
    route_agg["Route_Std"] = route_agg["Route_Std"].fillna(0)

    merged = df.merge(route_agg, on="Route", how="left")
    merged["Threshold"] = merged["Route_Mean"] + std_multiplier * merged["Route_Std"]
    outliers = merged[
        (merged["Route_Std"] > 0) & (merged["Lead_Time"] > merged["Threshold"])
    ].copy()

    outliers["Deviation"] = (outliers["Lead_Time"] - outliers["Route_Mean"]).round(1)
    outliers = outliers.rename(columns={"Route_Mean": "Route_Avg"})
    outliers["Route_Avg"] = outliers["Route_Avg"].round(1)

    return outliers[["Order ID", "Route", "Order Date", "Ship Date", "Lead_Time",
                      "Route_Avg", "Deviation"]].sort_values("Deviation", ascending=False)


def get_route_summary(df: pd.DataFrame, route_stats: pd.DataFrame, route_name: str) -> dict:
    """Single-route summary dict, used by the Route Comparison tool."""
    row = route_stats[route_stats["Route"] == route_name]
    route_df = df[df["Route"] == route_name]
    if row.empty or route_df.empty:
        return {"Route": route_name, "Total_Shipments": 0, "Avg_Lead_Time": 0,
                "Lead_Time_StdDev": 0, "Efficiency_Score": 0, "Fastest_Order": 0, "Slowest_Order": 0}
    r = row.iloc[0]
    return {
        "Route": route_name,
        "Total_Shipments": int(r["Total_Shipments"]),
        "Avg_Lead_Time": float(r["Avg_Lead_Time"]),
        "Lead_Time_StdDev": float(r["Lead_Time_StdDev"]),
        "Efficiency_Score": float(r["Efficiency_Score"]),
        "Fastest_Order": float(route_df["Lead_Time"].min()),
        "Slowest_Order": float(route_df["Lead_Time"].max()),
    }


def generate_word_report(df, kpis, insights, route_stats, region_stats, mode_stats, delay_threshold):
    """
    Builds a formatted Word (.docx) executive report summarizing the current
    (filtered) analysis, using python-docx. Returns raw bytes ready for a
    Streamlit download_button.
    """
    import re
    import io
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    NAVY = RGBColor(0x14, 0x21, 0x3D)
    TEAL = RGBColor(0x1F, 0x8A, 0x85)

    doc = Document()

    # Title
    title = doc.add_heading("Factory-to-Customer Shipping Route Efficiency", level=0)
    title.runs[0].font.color.rgb = NAVY
    sub = doc.add_paragraph("Nassau Candy Distributor \u00b7 Supply Chain Analytics Report")
    sub.runs[0].font.size = Pt(12)
    sub.runs[0].font.color.rgb = TEAL
    doc.add_paragraph(f"Generated from {kpis['total_orders']:,} shipments matching the current dashboard filters.")

    # KPI table
    doc.add_heading("Key Metrics", level=1)
    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, label in enumerate(["Total Shipments", "Avg Lead Time", "Delayed Shipments", "Delay Rate", "Active Routes"]):
        hdr[i].text = label
    vals = table.add_row().cells
    vals[0].text = f"{kpis['total_orders']:,}"
    vals[1].text = f"{kpis['avg_lead_time']} days"
    vals[2].text = f"{kpis['delayed_orders']:,}"
    vals[3].text = f"{kpis['delay_percentage']}%"
    vals[4].text = f"{kpis['total_routes']}"

    # Key insights
    doc.add_heading("Key Insights", level=1)
    for insight in insights:
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", insight)  # strip markdown bold for plain Word text
        p = doc.add_paragraph(clean, style="List Bullet")

    # Top / bottom routes
    doc.add_heading("Route Performance", level=1)
    doc.add_paragraph("Top 10 Most Efficient Routes:", style="Intense Quote")
    t1 = doc.add_table(rows=1, cols=4)
    t1.style = "Light List Accent 1"
    for i, h in enumerate(["Route", "Shipments", "Avg Lead Time", "Efficiency Score"]):
        t1.rows[0].cells[i].text = h
    for _, r in route_stats.sort_values("Efficiency_Score", ascending=False).head(10).iterrows():
        row_cells = t1.add_row().cells
        row_cells[0].text = str(r["Route"])
        row_cells[1].text = str(int(r["Total_Shipments"]))
        row_cells[2].text = f"{r['Avg_Lead_Time']:.1f} d"
        row_cells[3].text = f"{r['Efficiency_Score']:.0f}"

    doc.add_paragraph("")
    doc.add_paragraph("Bottom 10 Least Efficient Routes:", style="Intense Quote")
    t2 = doc.add_table(rows=1, cols=4)
    t2.style = "Light List Accent 2"
    for i, h in enumerate(["Route", "Shipments", "Avg Lead Time", "Efficiency Score"]):
        t2.rows[0].cells[i].text = h
    for _, r in route_stats.sort_values("Efficiency_Score", ascending=True).head(10).iterrows():
        row_cells = t2.add_row().cells
        row_cells[0].text = str(r["Route"])
        row_cells[1].text = str(int(r["Total_Shipments"]))
        row_cells[2].text = f"{r['Avg_Lead_Time']:.1f} d"
        row_cells[3].text = f"{r['Efficiency_Score']:.0f}"

    # Regional summary
    doc.add_heading("Regional Bottleneck Ranking", level=1)
    t3 = doc.add_table(rows=1, cols=3)
    t3.style = "Light Grid Accent 3"
    for i, h in enumerate(["Region", "Avg Lead Time", "Total Shipments"]):
        t3.rows[0].cells[i].text = h
    for _, r in region_stats.iterrows():
        row_cells = t3.add_row().cells
        row_cells[0].text = str(r["Region"])
        row_cells[1].text = f"{r['Avg_Lead_Time']:.1f} d"
        row_cells[2].text = str(int(r["Total_Shipments"]))

    # Ship mode summary
    doc.add_heading("Ship Mode Performance", level=1)
    t4 = doc.add_table(rows=1, cols=4)
    t4.style = "Light Grid Accent 4"
    for i, h in enumerate(["Ship Mode", "Avg Lead Time", "Delay Rate", "Shipments"]):
        t4.rows[0].cells[i].text = h
    for _, r in mode_stats.iterrows():
        row_cells = t4.add_row().cells
        row_cells[0].text = str(r["Ship Mode"])
        row_cells[1].text = f"{r['Avg_Lead_Time']:.1f} d"
        row_cells[2].text = f"{r['Delay_Pct']:.1f}%"
        row_cells[3].text = str(int(r["Total_Shipments"]))

    # Methodology note
    doc.add_heading("Methodology & Data Quality Note", level=1)
    doc.add_paragraph(
        "Lead Time is computed as Ship Date minus Order Date. Routes are defined as "
        "Factory \u2192 State/Province. The Efficiency Score is a min-max normalized, "
        "inverted average lead time scaled 0-100 (higher is better), computed relative "
        "to all routes in the current filter selection."
    )
    doc.add_paragraph(
        f"Data quality note: in this dataset, Ship Date values fall several years after "
        f"Order Date and barely vary by Ship Mode, consistent with a data-generation "
        f"artifact in the source file. All figures in this report should be read as "
        f"relative comparisons between routes, states, and ship modes rather than as "
        f"calendar-accurate day counts. Current delay threshold used: {delay_threshold} days."
    )

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
