
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sales_data.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_sales_data():
    df = pd.read_csv(DATA_FILE)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


# --------------------------------------------------
# FORMATTING
# --------------------------------------------------

def format_inr(value, position=None):
    """Format values as Indian rupees in lakhs."""
    return f"₹{value / 100000:.1f}L"


def create_bar_chart(result, title, xlabel, filename):
    """Create and save a reusable sales bar chart."""

    fig, ax = plt.subplots(figsize=(10, 6))

    result.plot(kind="bar", ax=ax)

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Amount (INR)", fontsize=11)

    ax.yaxis.set_major_formatter(FuncFormatter(format_inr))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()

    output_file = OUTPUT_DIR / filename

    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_file)


# --------------------------------------------------
# 1. SALES BY REGION
# --------------------------------------------------

def plot_sales_by_region():
    df = load_sales_data()

    result = (
        df.groupby("region")["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    return create_bar_chart(
        result,
        "Sales by Region",
        "Region",
        "sales_by_region.png",
    )


# --------------------------------------------------
# 2. MONTHLY SALES TREND
# --------------------------------------------------

def plot_monthly_sales():
    df = load_sales_data()

    df["month"] = df["order_date"].dt.to_period("M")

    result = (
        df.groupby("month")["sales"]
        .sum()
        .sort_index()
    )

    result.index = result.index.astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))

    result.plot(
        kind="line",
        marker="o",
        linewidth=2,
        markersize=5,
        ax=ax,
    )

    ax.set_title(
        "Monthly Sales Trend (2024–2025)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Total Sales", fontsize=11)

    ax.yaxis.set_major_formatter(FuncFormatter(format_inr))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "monthly_sales.png"

    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_file)


# --------------------------------------------------
# 3. SALES BY PRODUCT CATEGORY
# --------------------------------------------------

def plot_sales_by_category():
    df = load_sales_data()

    result = (
        df.groupby("product_category")["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    return create_bar_chart(
        result,
        "Sales by Product Category",
        "Product Category",
        "sales_by_category.png",
    )


# --------------------------------------------------
# 4. SALES BY CUSTOMER SEGMENT
# --------------------------------------------------

def plot_sales_by_customer_segment():
    df = load_sales_data()

    result = (
        df.groupby("customer_segment")["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    return create_bar_chart(
        result,
        "Sales by Customer Segment",
        "Customer Segment",
        "sales_by_customer_segment.png",
    )


# --------------------------------------------------
# 5. PROFIT BY REGION
# --------------------------------------------------

def plot_profit_by_region():
    df = load_sales_data()

    result = (
        df.groupby("region")["profit"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    result.plot(kind="bar", ax=ax)

    ax.set_title(
        "Profit by Region",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Region", fontsize=11)
    ax.set_ylabel("Total Profit", fontsize=11)

    ax.yaxis.set_major_formatter(FuncFormatter(format_inr))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "profit_by_region.png"

    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_file)


# --------------------------------------------------
# TEST VISUALIZATION AGENT
# --------------------------------------------------

if __name__ == "__main__":

    print("Visualization Agent")
    print("-------------------")

    charts = {
        "Sales by Region": plot_sales_by_region(),
        "Monthly Sales": plot_monthly_sales(),
        "Sales by Category": plot_sales_by_category(),
        "Sales by Customer Segment": plot_sales_by_customer_segment(),
        "Profit by Region": plot_profit_by_region(),
    }

    for chart_name, chart_path in charts.items():
        print(f"{chart_name}: {chart_path}")
