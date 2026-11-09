import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

np.random.seed(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "sales_data.csv"


# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------

NUM_RECORDS = 5000

regions = [
    "North",
    "South",
    "East",
    "West"
]

customer_segments = [
    "Consumer",
    "Corporate",
    "Small Business"
]

product_categories = [
    "Technology",
    "Furniture",
    "Office Supplies"
]

products = {
    "Technology": [
        "Laptop",
        "Monitor",
        "Keyboard",
        "Mouse",
        "Headphones"
    ],
    "Furniture": [
        "Office Chair",
        "Desk",
        "Bookshelf",
        "Conference Table"
    ],
    "Office Supplies": [
        "Notebook",
        "Printer Paper",
        "Pens",
        "Stapler",
        "File Folder"
    ]
}


# ---------------------------------------------------------
# Generate basic order information
# ---------------------------------------------------------

order_ids = np.arange(10001, 10001 + NUM_RECORDS)

dates = pd.date_range(
    start="2024-01-01",
    end="2025-12-31",
    periods=NUM_RECORDS
)

customer_ids = np.random.randint(
    1001,
    1501,
    NUM_RECORDS
)

regions_data = np.random.choice(
    regions,
    NUM_RECORDS
)

segments_data = np.random.choice(
    customer_segments,
    NUM_RECORDS,
    p=[0.50, 0.30, 0.20]
)

categories_data = np.random.choice(
    product_categories,
    NUM_RECORDS
)


# ---------------------------------------------------------
# Select products based on category
# ---------------------------------------------------------

products_data = [
    np.random.choice(products[category])
    for category in categories_data
]


# ---------------------------------------------------------
# Generate transaction metrics
# ---------------------------------------------------------

quantity = np.random.randint(
    1,
    10,
    NUM_RECORDS
)

unit_price = np.round(
    np.random.uniform(10, 1000, NUM_RECORDS),
    2
)

discount = np.round(
    np.random.uniform(0, 0.30, NUM_RECORDS),
    2
)


# ---------------------------------------------------------
# Calculate sales and profit
# ---------------------------------------------------------

sales = np.round(
    quantity * unit_price * (1 - discount),
    2
)

cost_percentage = np.random.uniform(
    0.55,
    0.85,
    NUM_RECORDS
)

cost = sales * cost_percentage

profit = np.round(
    sales - cost,
    2
)


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame({
    "order_id": order_ids,
    "order_date": dates,
    "customer_id": customer_ids,
    "customer_segment": segments_data,
    "region": regions_data,
    "product_category": categories_data,
    "product": products_data,
    "quantity": quantity,
    "unit_price": unit_price,
    "discount": discount,
    "sales": sales,
    "profit": profit
})


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Display information
# ---------------------------------------------------------

print("Dataset created successfully.")
print(f"Records: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nDataset preview:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())