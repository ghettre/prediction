import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import train2
from features import encode_stock_status

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="لوحة المخزون", layout="wide", initial_sidebar_state="expanded")

# Load external CSS if exists
css_path = Path(__file__).resolve().parent / "style.css"
if css_path.exists():
    with open(css_path, encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_forecasting_model.pkl"
MAPPING_PATH = BASE_DIR / "product_mapping.csv"

FEATURES = train2.FEATURES

# -----------------------
# FUNCTIONS
# -----------------------
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_app_data():
    # Uses train2's load_data to fetch from the 4 tables in Supabase directly
    df = train2.load_data()
    return df

@st.cache_resource
def load_model(mtime):
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_mapping():
    return train2.load_product_mapping(MAPPING_PATH)

def load_runtime_state():
    df = load_app_data()
    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    
    if not MODEL_PATH.exists():
        model = None
    else:
        model = load_model(model_mtime)
        
    product_to_id = load_mapping()
    return df, model, product_to_id

# -----------------------
# LOAD
# -----------------------
try:
    df, model, product_to_id = load_runtime_state()
except Exception as e:
    st.error(f"خطأ في الاتصال بقاعدة البيانات: {e}")
    st.stop()

# -----------------------
# DASHBOARD TITLE
# -----------------------
st.title("📊 لوحة المخزون الإدارية")
st.markdown("<hr style='margin-top: 0; margin-bottom: 2rem;'>", unsafe_allow_html=True)

# -----------------------
# SIDEBAR CONFIGURATION
# -----------------------
st.sidebar.title("⚙️ الإعدادات والتحكم")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# 1️⃣ PRODUCT SELECTION
if df.empty:
    st.warning("لا توجد بيانات في قاعدة البيانات. قم بإدخال بيانات أولاً.")
    products = []
else:
    products = sorted(df["Product"].dropna().unique())

if not products:
    st.stop()

product = st.sidebar.selectbox("📦 اختر المنتج للتنبؤ", products)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# 2️⃣ DATA ACTIONS
st.sidebar.subheader("🔄 إدارة البيانات والنموذج")
st.sidebar.caption("يتم قراءة البيانات مباشرة من جداول Supabase المتعددة.")

if st.sidebar.button("🔄 تحديث البيانات وإعادة تدريب المودل"):
    with st.spinner("جاري جلب البيانات من الجداول الأربعة وتدريب المودل..."):
        try:
            # Force cache clear to fetch new data
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # Train model using direct Supabase connection from train2.py
            new_model, metrics = train2.train_model_from_sql()
            train2.save_trained_model(new_model, MODEL_PATH)
            
            # Reload state
            df, model, product_to_id = load_runtime_state()
            st.sidebar.success(f"تم التدريب بنجاح! الأسطر: {int(metrics['rows'])} | R2: {metrics['r2']:.2f}")
        except Exception as error:
            st.sidebar.error(f"فشل التحديث: {error}")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

if model is None:
    st.warning("النموذج غير مدرب بعد. الرجاء الضغط على تحديث البيانات وإعادة تدريب المودل.")
    st.stop()

product_df = df[df["Product"] == product].sort_values("Date").copy()

if product_df.empty:
    st.error("لا توجد بيانات لهذا المنتج.")
    st.stop()

product_id = product_to_id.get(product, 0)
current_stock = float(product_df["STOCK FIN JOUR"].iloc[-1])
sales_series = product_df["SALES JOUR"].reset_index(drop=True)

# -----------------------
# SAFE LAG
# -----------------------
def safe_lag(series, lag):
    return float(series.iloc[-lag]) if len(series) >= lag else float(series.iloc[-1])

# -----------------------
# FORECAST FUNCTION
# -----------------------
def calculate_safety_days(product_df: pd.DataFrame, lead_time: int) -> tuple[int, dict[str, float]]:
    sales = product_df["SALES JOUR"].astype(float)
    dates = pd.to_datetime(product_df["Date"])

    window = min(len(product_df), 60)
    recent_sales = sales.tail(window)
    recent_dates = dates.tail(window)

    mean_sales = float(recent_sales.mean()) if len(recent_sales) else 0.0
    std_sales = float(recent_sales.std(ddof=0)) if len(recent_sales) > 1 else 0.0
    volatility = std_sales / mean_sales if mean_sales > 0 else 0.0

    last_14 = recent_sales.tail(min(14, len(recent_sales)))
    prev_14 = recent_sales.iloc[:-len(last_14)].tail(len(last_14)) if len(recent_sales) > len(last_14) else last_14
    recent_trend = float(last_14.mean()) / max(float(prev_14.mean()), 1.0) if len(last_14) else 1.0

    seasonal_table = pd.DataFrame(
        {
            "weekday": recent_dates.dt.dayofweek,
            "day": recent_dates.dt.day,
            "sales": recent_sales.values,
        }
    )
    weekday_spread = seasonal_table.groupby("weekday")["sales"].mean().std(ddof=0)
    weekday_spread = float(weekday_spread) / mean_sales if mean_sales > 0 else 0.0

    payday_mask = seasonal_table["day"] <= 5
    payday_avg = float(seasonal_table.loc[payday_mask, "sales"].mean()) if payday_mask.any() else mean_sales
    non_payday_avg = float(seasonal_table.loc[~payday_mask, "sales"].mean()) if (~payday_mask).any() else mean_sales
    payday_effect = abs(payday_avg - non_payday_avg) / mean_sales if mean_sales > 0 else 0.0

    out_of_stock_rate = (
        product_df["Stock_Status"].astype(str).str.strip().str.lower().eq("out of stock").tail(window).mean()
    )

    safety_multiplier = (
        0.75
        + (1.20 * volatility)
        + (0.60 * weekday_spread)
        + (0.35 * max(0.0, recent_trend - 1.0))
        + (0.35 * float(out_of_stock_rate))
        + (0.20 * payday_effect)
    )

    max_days = max(lead_time * 3, 7)
    safety_days = int(round(lead_time * safety_multiplier))
    safety_days = max(1, min(safety_days, max_days))

    diagnostics = {
        "mean_sales": mean_sales,
        "volatility": volatility,
        "recent_trend": recent_trend,
        "weekday_spread": weekday_spread,
        "payday_effect": payday_effect,
        "out_of_stock_rate": float(out_of_stock_rate),
        "window_days": float(window),
    }
    return safety_days, diagnostics

def build_history_frame(product_df: pd.DataFrame) -> pd.DataFrame:
    history = product_df[["Date", "SALES JOUR"]].copy()
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.rename(columns={"SALES JOUR": "Actual_Sales"})
    return history


def make_chart(history, forecast_dates, preds, realized_sales, out_day=None):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=("📈 اتجاه المبيعات والتوقع", "📊 مقارنة التوقع والواقع"),
        vertical_spacing=0.15
    )

    history_tail = history.tail(60)
    
    fig.add_trace(go.Scatter(
        x=history_tail["Date"], y=history_tail["Actual_Sales"],
        name="المبيعات الفعلية", line=dict(color="#334155", width=2.5),
        mode='lines+markers'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=preds,
        name="التوقع", line=dict(color="#2563eb", width=2.5),
        mode='lines+markers'
    ), row=1, col=1)

    if out_day is not None:
        fig.add_vline(x=out_day, line_dash="dot", line_color="#dc2626", annotation_text="نفاد المخزون")
        fig.add_vrect(x0=out_day, x1=forecast_dates[-1], fillcolor="#dc2626", opacity=0.05, line_width=0)

    fig.add_trace(go.Scatter(
        x=forecast_dates, y=preds,
        name="المتوقع", line=dict(color="#0f766e", width=2.5),
        mode='lines+markers'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=realized_sales,
        name="الفعلي", line=dict(color="#ef4444", width=2.5),
        mode='lines+markers'
    ), row=2, col=1)

    fig.update_layout(
        height=900,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def format_date(value: pd.Timestamp) -> str:
    return f"{value.day}/{value.month}/{value.year}"

def forecast(df_product, product_id, model, forecast_dates):
    preds = []
    realized_sales = []
    stock_history = []
    out_day = None

    last_row = df_product.iloc[-1].copy()
    current_stock = float(df_product["STOCK FIN JOUR"].iloc[-1])

    for d in forecast_dates:
        last_row["Date"] = d
        last_row["year"] = d.year
        last_row["month"] = d.month
        last_row["day_of_month"] = d.day
        last_row["day_of_week"] = d.dayofweek
        last_row["is_weekend"] = 1 if d.dayofweek in [5, 6] else 0
        last_row["is_payday_period"] = 1 if d.day <= 5 else 0
        last_row["quarter"] = (d.month - 1) // 3 + 1

        last_row["Product_ID"] = product_id
        last_row["Stock_Status_Encoded"] = encode_stock_status(
            df_product["Stock_Status"].iloc[-1]
        )

        last_row["sales_lag_7"] = safe_lag(sales_series, 7)
        last_row["sales_lag_14"] = safe_lag(sales_series, 14)
        last_row["sales_lag_30"] = safe_lag(sales_series, 30)

        X = pd.DataFrame([[last_row[f] for f in FEATURES]], columns=FEATURES)

        pred = max(0, float(model.predict(X)[0]))
        preds.append(pred)

        executed_sale = 0.0 if current_stock <= 0 else min(pred, current_stock)
        realized_sales.append(executed_sale)
        current_stock -= executed_sale
        stock_history.append(current_stock)
        
        if current_stock <= 0 and out_day is None:
            out_day = d

    return preds, realized_sales, stock_history, out_day

# -----------------------
# RUN FORECAST
# -----------------------
last_history_date = pd.to_datetime(product_df["Date"]).max()
forecast_days = st.sidebar.slider("عدد أيام التنبؤ", min_value=1, max_value=30, value=15)
start_date = pd.Timestamp(last_history_date) + pd.Timedelta(days=1)
forecast_dates = pd.date_range(start_date, periods=forecast_days, freq="D")

preds, realized_sales, stock_history, out_day = forecast(product_df, product_id, model, forecast_dates)

lead_time = max(1, int(product_df["Delivery_Days"].mode().iloc[0]))
history_df = build_history_frame(product_df)

forecast_df = pd.DataFrame(
    {
        "Date": forecast_dates,
        "Predicted_Sales": preds,
        "Realized_Sales": realized_sales,
        "Stock_Level": stock_history,
        "Status": [
            "نفاد المخزون" if value <= 0 else "مبيعات ممكنة"
            for value in stock_history
        ],
    }
)

# -----------------------
# OUTPUT
# -----------------------
st.subheader("📊 ملخص المنتج")

metric_cols = st.columns(4)
metric_cols[0].metric("اسم المنتج", product)
metric_cols[1].metric("المخزون الحالي", f"{int(current_stock)}")
metric_cols[2].metric("متوسط المبيعات المتوقعة", f"{float(np.mean(preds)):.1f}")
metric_cols[3].metric("زمن التوريد", f"{lead_time} يوم")

fig = make_chart(history_df, forecast_dates, preds, realized_sales, out_day)
st.plotly_chart(fig, use_container_width=True)
st.subheader("📋 جدول التوقعات")

def highlight_stockout(row: pd.Series):
    if row["Status"] == "نفاد المخزون":
        return ["background-color: #fee2e2; color: #991b1b;"] * len(row)
    return [""] * len(row)

st.dataframe(
    forecast_df.style.apply(highlight_stockout, axis=1).format(
        {
            "Predicted_Sales": "{:.2f}",
            "Realized_Sales": "{:.2f}",
            "Stock_Level": "{:.2f}",
        }
    ),
    width="stretch",
    height=420,
)

# -----------------------
# DECISION LOGIC
# -----------------------
safety_days, safety_info = calculate_safety_days(product_df, lead_time)
coverage_days = lead_time + safety_days
required_qty = float(np.mean(preds)) * coverage_days
order_qty = max(0.0, required_qty - current_stock)

st.subheader("📦 قرار إعادة الطلب")

if out_day:
    reorder_day = out_day - pd.Timedelta(days=lead_time)

    st.error(
        f"""
🚨 تاريخ نفاد المخزون المتوقع: {format_date(out_day)}

⏳ زمن التوريد: {lead_time} يوم
📅 آخر يوم آمن للطلب: {format_date(reorder_day)}
📦 كمية الطلب المقترحة: {int(round(order_qty))}
"""
    )

    if pd.Timestamp.today().normalize() >= reorder_day.normalize():
        st.error("⚠️ يجب الطلب الآن")
    else:
        st.warning("⏳ جهّز أمر الشراء قريبًا")
else:
    st.success(
        f"""
✅ المخزون آمن

📦 كمية المخزون الاحتياطي المقترحة: {int(round(order_qty))}
"""
    )

with st.expander("تفاصيل تحليل الطلب"):
    detail_cols = st.columns(3)
    detail_cols[0].metric("التذبذب", f"{safety_info['volatility']:.2f}")
    detail_cols[1].metric("الموسمية الأسبوعية", f"{safety_info['weekday_spread']:.2f}")
    detail_cols[2].metric("تأثير فترات الراتب", f"{safety_info['payday_effect']:.2f}")
    st.write(
        {
            "نافذة التحليل بالأيام": int(safety_info["window_days"]),
            "اتجاه آخر 14 يوم": round(safety_info["recent_trend"], 2),
            "نسبة أيام نفاد المخزون": round(safety_info["out_of_stock_rate"], 2),
        }
    )
