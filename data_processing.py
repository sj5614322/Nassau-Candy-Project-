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
