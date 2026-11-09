import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Dataset path
# ---------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_sales_data():
    """Load the sales dataset."""

    df = pd.read_csv(DATA_FILE)

    df["order_date"] = pd.to_datetime(df["order_date"])

    return df


# ---------------------------------------------------------
# Basic dataset summary
# ---------------------------------------------------------

def get_dataset_summary():
    """Return basic information about the sales dataset."""

    df = load_sales_data()

    return {
        "total_orders": len(df),
        "total_customers": df["customer_id"].nunique(),
        "total_sales": round(df["sales"].sum(), 2),
        "total_profit": round(df["profit"].sum(), 2),
        "average_order_value": round(df["sales"].mean(), 2),
    }


# ---------------------------------------------------------
# Sales by region
# ---------------------------------------------------------

def sales_by_region():
    """Calculate sales and profit by region."""

    df = load_sales_data()

    result = (
        df.groupby("region")
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            orders=("order_id", "count")
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )

    return result


# ---------------------------------------------------------
# Sales by product category
# ---------------------------------------------------------

def sales_by_category():
    """Calculate sales and profit by product category."""

    df = load_sales_data()

    result = (
        df.groupby("product_category")
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            orders=("order_id", "count")
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )

    return result

def sales_by_customer_segment():
    df = load_sales_data()
    result = (
        df.groupby("customer_segment")
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            orders=("order_id", "count")
        )
        .reset_index()
        .sort_values("total_profit", ascending=False)
    )
    return result


# ---------------------------------------------------------
# Monthly sales
# ---------------------------------------------------------

def monthly_sales():
    """Calculate monthly sales and profit."""

    df = load_sales_data()

    df["month"] = df["order_date"].dt.to_period("M").astype(str)

    result = (
        df.groupby("month")
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            orders=("order_id", "count")
        )
        .reset_index()
        .sort_values("month")
    )

    return result


# ---------------------------------------------------------
# Top products
# ---------------------------------------------------------

def top_products(n=5):
    """Return top products by sales."""

    df = load_sales_data()

    result = (
        df.groupby("product")
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            orders=("order_id", "count")
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(n)
    )

    return result


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nDataset Summary:")
    print(get_dataset_summary())

    print("\nSales by Region:")
    print(sales_by_region())

    print("\nSales by Category:")
    print(sales_by_category())

    print("\nTop Products:")
    print(top_products())

    print("\nMonthly Sales:")
    print(monthly_sales().head())

    print("\nSales by Customer Segment:")
    print(sales_by_customer_segment())