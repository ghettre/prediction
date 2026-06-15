from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from features import add_time_features, encode_stock_status

BASE_DIR = Path(__file__).resolve().parent
MAPPING_PATH = BASE_DIR / "product_mapping.csv"
MODEL_PATH = BASE_DIR / "xgboost_forecasting_model.pkl"

def load_dotenv():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with open(env_path, encoding='utf-8') as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_dotenv()

import streamlit as st

def get_supabase_credentials():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
    
    try:
        if not url and "SUPABASE_URL" in st.secrets:
            url = st.secrets["SUPABASE_URL"]
        if not key:
            if "SUPABASE_SERVICE_ROLE_KEY" in st.secrets:
                key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
            elif "SUPABASE_ANON_KEY" in st.secrets:
                key = st.secrets["SUPABASE_ANON_KEY"]
    except Exception:
        pass
        
    return url.rstrip("/"), key

SUPABASE_URL, SUPABASE_KEY = get_supabase_credentials()

SOURCE_TABLES = ["headphones", "laptops", "phones", "tablets"]

FEATURES = [
    "Product_ID",
    "STOCK FIN JOUR",
    "Shipment_Received",
    "Delivery_Days",
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "is_payday_period",
    "is_weekend",
    "quarter",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_30",
    "Stock_Status_Encoded",
]

COLUMN_ALIASES = {
    "date": "Date",
    "product": "Product",
    "sales_jour": "SALES JOUR",
    "stock_fin_jour": "STOCK FIN JOUR",
    "shipment_received": "Shipment_Received",
    "delivery_days": "Delivery_Days",
    "stock_status": "Stock_Status",
}

NUMERIC_COLUMNS = [
    "SALES JOUR",
    "STOCK FIN JOUR",
    "Shipment_Received",
    "Delivery_Days",
]


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def get_headers() -> dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in your environment.")
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {
        source: target
        for source, target in COLUMN_ALIASES.items()
        if source in renamed.columns and target not in renamed.columns
    }
    if rename_map:
        renamed = renamed.rename(columns=rename_map)
    return renamed


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = normalize_columns(df)

    for column in ["Date", "Product", "Stock_Status"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip()

    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned["Product"] = cleaned["Product"].replace({"nan": "", "None": ""}).str.strip()
    cleaned["Stock_Status"] = cleaned["Stock_Status"].replace({"nan": "", "None": ""}).str.strip()
    cleaned["Stock_Status"] = cleaned["Stock_Status"].replace("", "Available")

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0)

    cleaned = cleaned.dropna(subset=["Date"]).copy()
    cleaned = cleaned[cleaned["Product"].astype(str).str.strip() != ""].copy()
    cleaned = cleaned.drop_duplicates(subset=["Product", "Date"], keep="last").copy()
    cleaned = cleaned.sort_values(["Product", "Date"]).reset_index(drop=True)
    return cleaned


def fetch_all_rows(table: str) -> list[dict]:
    offset = 0
    limit = 1000
    rows: list[dict] = []

    while True:
        query = f"?select=*&order=date.asc,id.asc&limit={limit}&offset={offset}"
        url = f"{SUPABASE_URL}/rest/v1/{table}{query}"
        req = urllib_request.Request(url, headers=get_headers(), method="GET")

        try:
            with urllib_request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8").strip()
        except urllib_error.HTTPError as exc:
            # If table doesn't exist, ignore
            if exc.code in [400, 404]:
                break
            raise RuntimeError(f"Supabase GET {table} failed ({exc.code}): {exc.read().decode('utf-8').strip()}") from exc

        batch = json.loads(raw) if raw else []
        if not batch:
            break

        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    return rows

def sync_to_zz():
    """
    تزامن بيانات الجداول الأربعة مع الجدول الرئيسي zz.
    - سطر جديد  → INSERT
    - سطر مُعدَّل في الجدول الأصلي → PATCH (تحديث) في zz
    - سطر غير مُعدَّل → لا شيء
    """
    from urllib.parse import quote

    COMPARE_FIELDS = [
        "sales_jour", "stock_fin_jour",
        "shipment_received", "delivery_days", "stock_status"
    ]

    # 1️⃣ اقرأ كل سجلات zz وضعها في قاموس (date, product) → row
    zz_rows = fetch_all_rows("zz")
    zz_map: dict[tuple, dict] = {}
    for row in zz_rows:
        date = str(row.get("date", "")).strip()
        product = str(row.get("product", "")).strip()
        if date and product:
            zz_map[(date, product)] = row

    rows_to_insert: list[dict] = []
    rows_to_patch: list[tuple[str, str, dict]] = []  # (date, product, fields)

    # 2️⃣ جلب البيانات من الجداول الأربعة ومقارنتها مع zz
    for table in SOURCE_TABLES:
        for row in fetch_all_rows(table):
            date = str(row.get("date", "")).strip()
            product = str(row.get("product", "")).strip()
            if not date or not product:
                continue

            key = (date, product)
            row_clean = row.copy()
            row_clean.pop("id", None)

            if key not in zz_map:
                # سطر جديد كليّاً → إدراج
                rows_to_insert.append(row_clean)
                zz_map[key] = row_clean  # تجنب تكرار الإدراج
            else:
                # سطر موجود → تحقق من التعديلات
                existing = zz_map[key]
                changed_fields = {
                    f: row.get(f)
                    for f in COMPARE_FIELDS
                    if str(row.get(f, "")).strip() != str(existing.get(f, "")).strip()
                }
                if changed_fields:
                    rows_to_patch.append((date, product, changed_fields))
                    zz_map[key] = row_clean  # تحديث الكاش المحلي

    # 3️⃣ إدراج السجلات الجديدة (دفعات)
    if rows_to_insert:
        for i in range(0, len(rows_to_insert), 500):
            batch = rows_to_insert[i:i + 500]
            url = f"{SUPABASE_URL}/rest/v1/zz"
            headers = get_headers()
            headers["Prefer"] = "return=minimal"
            data = json.dumps(batch).encode("utf-8")
            req = urllib_request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib_request.urlopen(req, timeout=60):
                    pass
            except urllib_error.HTTPError as exc:
                raise RuntimeError(f"Insert to zz failed: {exc.read().decode()}") from exc

    # 4️⃣ تحديث السجلات المُعدَّلة (كل سطر بطلب PATCH منفصل)
    for date, product, fields in rows_to_patch:
        url = (
            f"{SUPABASE_URL}/rest/v1/zz"
            f"?date=eq.{quote(date)}&product=eq.{quote(product)}"
        )
        headers = get_headers()
        headers["Prefer"] = "return=minimal"
        data = json.dumps(fields).encode("utf-8")
        req = urllib_request.Request(url, data=data, headers=headers, method="PATCH")
        try:
            with urllib_request.urlopen(req, timeout=60):
                pass
        except urllib_error.HTTPError as exc:
            raise RuntimeError(f"Patch in zz failed: {exc.read().decode()}") from exc


def load_data() -> pd.DataFrame:
    sync_to_zz()
    all_rows = fetch_all_rows("zz")
    df = pd.DataFrame(all_rows)
    return clean_rows(df)




def load_product_mapping(path: Path | str = MAPPING_PATH) -> dict[str, int]:
    mapping = pd.read_csv(path)
    if not {"Product", "Product_ID"}.issubset(mapping.columns):
        raise ValueError("يجب أن يحتوي product_mapping.csv على العمودين Product و Product_ID")
    return dict(zip(mapping["Product"], mapping["Product_ID"]))


def build_training_frame(
    df: pd.DataFrame | None = None,
    product_to_id: dict[str, int] | None = None,
) -> pd.DataFrame:
    data = load_data() if df is None else clean_rows(df.copy())
    if data.empty:
        raise ValueError("لا توجد بيانات للتدريب")
        
    mapping = load_product_mapping() if product_to_id is None else product_to_id

    required_columns = {
        "Date",
        "Product",
        "SALES JOUR",
        "STOCK FIN JOUR",
        "Shipment_Received",
        "Delivery_Days",
        "Stock_Status",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError("الأعمدة التالية مفقودة: " + ", ".join(sorted(missing_columns)))

    missing_products = sorted(set(data["Product"].dropna().unique()) - set(mapping))
    if missing_products:
        raise ValueError("المنتجات التالية لا تحتوي على Product_ID: " + ", ".join(missing_products))

    data["Product_ID"] = data["Product"].map(mapping)
    data = data.sort_values(["Product", "Date"]).copy()
    data = add_time_features(data)
    data["Stock_Status_Encoded"] = data["Stock_Status"].apply(encode_stock_status)
    data["sales_lag_7"] = data.groupby("Product")["SALES JOUR"].shift(7)
    data["sales_lag_14"] = data.groupby("Product")["SALES JOUR"].shift(14)
    data["sales_lag_30"] = data.groupby("Product")["SALES JOUR"].shift(30)
    data = data.dropna(subset=FEATURES + ["SALES JOUR"]).copy()
    return data


def train_model_from_sql(
    df: pd.DataFrame | None = None,
    product_to_id: dict[str, int] | None = None,
) -> tuple[XGBRegressor, dict[str, float]]:
    data = build_training_frame(df, product_to_id)
    X = data[FEATURES]
    y = data["SALES JOUR"].astype(float)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X, y)

    predictions = model.predict(X)
    target_mean = float(np.mean(y))
    total_variance = float(np.sum((y - target_mean) ** 2))
    metrics = {
        "mae": float(np.mean(np.abs(y - predictions))),
        "rmse": float(np.sqrt(np.mean((y - predictions) ** 2))),
        "r2": float(1.0 - np.sum((y - predictions) ** 2) / total_variance) if total_variance > 0 else 0.0,
        "rows": float(len(data)),
    }
    return model, metrics


def save_trained_model(model: XGBRegressor, path: Path | str = MODEL_PATH) -> None:
    joblib.dump(model, path)


def main() -> None:
    model, metrics = train_model_from_sql()
    save_trained_model(model)

    print("Model trained from Supabase successfully.")
    print(f"Rows used: {int(metrics['rows'])}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Saved model to: {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
