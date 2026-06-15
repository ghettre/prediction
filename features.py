import pandas as pd

def encode_stock_status(value: str) -> int:
    status = str(value).strip().lower()
    return 1 if status == "available" else 0


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])

    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["day_of_month"] = out["Date"].dt.day
    out["day_of_week"] = out["Date"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_payday_period"] = (out["day_of_month"] <= 5).astype(int)
    out["quarter"] = ((out["month"] - 1) // 3 + 1).astype(int)

    return out