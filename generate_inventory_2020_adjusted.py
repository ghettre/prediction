from pathlib import Path
import zlib

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "inventory_2020_sales_stock.xlsx"
OUTPUT_PATH = BASE_DIR / "inventory_2020_sales_stock_adjusted.xlsx"


def load_product_settings() -> pd.DataFrame:
    settings = pd.read_excel(SOURCE_PATH, sheet_name="Product_Settings")
    required = {"Product", "Delivery_Days", "Initial_Stock", "Shipment_Qty", "Base_Demand"}
    missing = required.difference(settings.columns)
    if missing:
        raise ValueError(f"Product_Settings is missing columns: {', '.join(sorted(missing))}")
    return settings


def demand_multiplier(current_date: pd.Timestamp) -> float:
    multiplier = 1.0
    if current_date.dayofweek in (5, 6):
        multiplier *= 1.08
    if current_date.day <= 5:
        multiplier *= 1.12
    return multiplier


def daily_demand(rng: np.random.Generator, base_demand: float, current_date: pd.Timestamp) -> int:
    uplift = demand_multiplier(current_date)
    noise = rng.normal(0, base_demand * 0.08)
    demand = base_demand * uplift + noise
    return max(0, int(round(demand)))


def build_product_rows(product_cfg: pd.Series, dates: pd.DatetimeIndex) -> list[dict]:
    seed = zlib.crc32(str(product_cfg["Product"]).encode("utf-8"))
    rng = np.random.default_rng(seed)

    delivery_days = int(product_cfg["Delivery_Days"])
    stock = float(product_cfg["Initial_Stock"])
    shipment_qty = float(product_cfg["Shipment_Qty"])
    base_demand = float(product_cfg["Base_Demand"])
    rows: list[dict] = []

    for idx, current_date in enumerate(dates):
        shipment_received = 0.0
        if idx > 0 and idx % delivery_days == 0:
            stock += shipment_qty
            shipment_received = shipment_qty

        sold_qty = 0.0
        demand_qty = daily_demand(rng, base_demand, current_date)
        if stock > 0:
            sold_qty = float(min(stock, demand_qty))
            stock = max(0.0, stock - sold_qty)

        rows.append(
            {
                "Date": current_date.date(),
                "Product": product_cfg["Product"],
                "SALES JOUR": int(round(sold_qty)),
                "STOCK FIN JOUR": int(round(stock)),
                "Shipment_Received": int(round(shipment_received)),
                "Delivery_Days": delivery_days,
                "Stock_Status": "Out of stock" if stock == 0 else "Available",
            }
        )

    return rows


def build_workbook() -> Path:
    settings = load_product_settings()
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    rows = []
    for _, product_cfg in settings.iterrows():
        rows.extend(build_product_rows(product_cfg, dates))

    daily_df = pd.DataFrame(rows).sort_values(["Date", "Product"]).reset_index(drop=True)
    summary_df = (
        daily_df.groupby("Product", as_index=False)
        .agg(
            Total_Sales=("SALES JOUR", "sum"),
            Final_Stock=("STOCK FIN JOUR", "last"),
            Delivery_Days=("Delivery_Days", "first"),
        )
        .sort_values("Product")
    )

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        daily_df.to_excel(writer, sheet_name="Daily_2020", index=False)
        settings.to_excel(writer, sheet_name="Product_Settings", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    return OUTPUT_PATH


if __name__ == "__main__":
    print(build_workbook())
