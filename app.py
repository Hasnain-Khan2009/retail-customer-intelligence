"""
Customer Spend, Product & Sales Analytics - Streamlit Dashboard (app.py)
=========================================================================
An End-to-End Exploratory and Advanced Analytics Study of Retail Transactions
Using Pareto (80/20), RFM, Cohort Retention, Market Basket and K-Means Techniques.

Run with:   streamlit run app.py
Requires:   streamlit, pandas, numpy, matplotlib, seaborn, scikit-learn

Data loading priority (per file):  sidebar upload  ->  local disk (outputs/tables/)
                                   ->  automatic synthetic data (RANDOM_STATE = 42)
"""

# ----------------------------------------------------------------------------
# 0. IMPORTS
# ----------------------------------------------------------------------------
import io
import itertools
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless backend - required for Streamlit servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# 1. GLOBAL CONSTANTS, PALETTES & PAGE CONFIG
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
CURRENCY_SYMBOL = "₹"
PROJECT_TITLE = (
    "Customer Spend, Product & Sales Analytics: An End-to-End Exploratory and Advanced "
    "Analytics Study of Retail Transactions Using Pareto (80/20), RFM, Cohort Retention, "
    "Market Basket and K-Means Techniques"
)

PALETTE = {
    "primary": "#1F4E79",
    "secondary": "#2E86AB",
    "accent": "#F18F01",
    "positive": "#2A9D8F",
    "negative": "#E63946",
    "neutral": "#6C757D",
}
SEGMENT_COLORS = {"Low": "#F4A261", "Medium": "#2A9D8F", "High": "#1F4E79"}
RP_COLORS = {"1-Time Buyers": "#F28E2B", "Repeat Buyers": "#4E79A7"}
SEGMENT_ORDER = ["Low", "Medium", "High"]
BUYER_ORDER = ["1-Time Buyers", "Repeat Buyers"]

TX_COLUMNS = [
    "transaction_id", "customer_id", "transaction_date", "product_name",
    "category", "quantity", "unit_price", "total_amount",
]
CUSTOMER_BASE_COLUMNS = [
    "customer_id", "total_spend", "num_orders", "avg_order_value",
    "first_purchase_date", "last_purchase_date", "recency_days",
]
LOG_COLUMNS = [
    "step_no", "cleaning_stage", "issue_addressed", "action_taken",
    "rows_before", "rows_after", "rows_removed", "pct_removed",
]
CLUSTER_FEATURES = ["total_spend", "num_orders", "avg_order_value", "recency_days"]

DISK_PATHS = {
    "df_clean": os.path.join("outputs", "tables", "cleaned_transactions.csv"),
    "customer_df": os.path.join("outputs", "tables", "customer_analytics_master.csv"),
    "cleaning_log": os.path.join("outputs", "tables", "data_cleaning_log.csv"),
}

# Season mapping used for Indian retail seasonality (Section 11)
SEASON_MAP = {
    12: "Winter (Dec-Feb)", 1: "Winter (Dec-Feb)", 2: "Winter (Dec-Feb)",
    3: "Summer (Mar-May)", 4: "Summer (Mar-May)", 5: "Summer (Mar-May)",
    6: "Monsoon (Jun-Sep)", 7: "Monsoon (Jun-Sep)", 8: "Monsoon (Jun-Sep)", 9: "Monsoon (Jun-Sep)",
    10: "Festive (Oct-Nov)", 11: "Festive (Oct-Nov)",
}
SEASON_ORDER = ["Winter (Dec-Feb)", "Summer (Mar-May)", "Monsoon (Jun-Sep)", "Festive (Oct-Nov)"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(
    page_title="Retail Customer & Sales Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 100,
})

st.markdown(
    """
    <style>
    .interp {background:#EEF5FA;border-left:5px solid #2E86AB;padding:10px 14px;
             border-radius:6px;margin:8px 0 20px 0;font-size:0.95rem;}
    .badge {display:inline-block;padding:4px 12px;border-radius:14px;color:#fff;
            font-weight:600;font-size:0.85rem;margin:2px 6px 2px 0;}
    .member-tag {color:#6C757D;font-size:0.85rem;margin-top:-8px;margin-bottom:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 2. SMALL UI / FORMAT HELPERS
# ----------------------------------------------------------------------------
def money(x, decimals=0):
    """Format a number as Indian-rupee currency string."""
    if x is None or pd.isna(x):
        return "-"
    return f"{CURRENCY_SYMBOL}{x:,.{decimals}f}"


def compact_money(x):
    """Format large rupee values using lakh (L) / crore (Cr) notation."""
    if x is None or pd.isna(x):
        return "-"
    if abs(x) >= 1e7:
        return f"{CURRENCY_SYMBOL}{x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"{CURRENCY_SYMBOL}{x / 1e5:,.2f} L"
    return f"{CURRENCY_SYMBOL}{x:,.0f}"


def pct(x, decimals=1):
    """Format a percentage value (already on 0-100 scale)."""
    if x is None or pd.isna(x):
        return "-"
    return f"{x:.{decimals}f}%"


def interp(text):
    """Render the data-driven interpretation block shown under tables and charts."""
    st.markdown(f'<div class="interp">💡 <b>Interpretation:</b> {text}</div>', unsafe_allow_html=True)


def badge(text, color):
    """Return HTML for a coloured badge."""
    return f'<span class="badge" style="background:{color}">{text}</span>'


def section_header(title, member):
    """Section sub-header with the responsible team member."""
    st.subheader(title)
    st.markdown(f'<div class="member-tag">👤 {member}</div>', unsafe_allow_html=True)


def show_fig(fig):
    """Render a Matplotlib figure and release its memory."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_table(df):
    """Render a DataFrame full-width without the index (version-tolerant)."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:  # older Streamlit versions
        st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# 3. DATA PREPARATION FUNCTIONS
# ----------------------------------------------------------------------------
def normalize_transactions(df):
    """Coerce a transactions DataFrame to the mandatory df_clean schema contract."""
    missing = [c for c in TX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df[TX_COLUMNS].copy()
    out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce")
    out = out.dropna(subset=["transaction_id", "customer_id", "transaction_date",
                             "product_name", "category", "quantity", "unit_price"])
    out = out[(out["quantity"] > 0) & (out["unit_price"] > 0)]
    out["transaction_id"] = out["transaction_id"].astype(str)
    out["customer_id"] = out["customer_id"].astype(str)
    out["product_name"] = out["product_name"].astype(str)
    out["category"] = out["category"].astype(str)
    out["quantity"] = out["quantity"].astype("int64")
    out["unit_price"] = out["unit_price"].astype("float64")
    out["total_amount"] = (out["quantity"] * out["unit_price"]).round(2).astype("float64")
    return out.reset_index(drop=True)


def build_customer_df(tx):
    """Aggregate transactions to the customer-level base table (schema contract)."""
    snapshot_date = tx["transaction_date"].max() + pd.Timedelta(days=1)
    cust = tx.groupby("customer_id").agg(
        total_spend=("total_amount", "sum"),
        num_orders=("transaction_id", "nunique"),
        first_purchase_date=("transaction_date", "min"),
        last_purchase_date=("transaction_date", "max"),
    ).reset_index()
    cust["total_spend"] = cust["total_spend"].round(2).astype("float64")
    cust["num_orders"] = cust["num_orders"].astype("int64")
    cust["avg_order_value"] = (cust["total_spend"] / cust["num_orders"]).round(2).astype("float64")
    cust["recency_days"] = (snapshot_date - cust["last_purchase_date"]).dt.days.astype("int64")
    return cust[CUSTOMER_BASE_COLUMNS]


def assign_spend_segment(total_spend):
    """Split customers into Low / Medium / High tertiles with pd.qcut."""
    try:
        return pd.qcut(total_spend.rank(method="first"), 3, labels=SEGMENT_ORDER).astype(str)
    except ValueError:  # fewer than 3 customers
        return pd.Series("Medium", index=total_spend.index)


def assign_buyer_type(num_orders):
    """1-Time Buyers (1 order) vs Repeat Buyers (2+ orders)."""
    return np.where(num_orders >= 2, "Repeat Buyers", "1-Time Buyers")


def _quintile_score(series, ascending):
    """Return 1-5 quantile scores (5 = best). Uses rank(method='first') to avoid tie errors."""
    n = int(series.notna().sum())
    bins = int(min(5, max(n, 1)))
    if bins < 2:
        return pd.Series(3, index=series.index)
    ranks = series.rank(method="first", ascending=ascending)
    return (pd.qcut(ranks, bins, labels=False).astype(int) + 1)


def compute_rfm(cdf):
    """Compute R/F/M scores and named RFM segments for a customer table."""
    out = cdf.copy()
    out["R_score"] = _quintile_score(out["recency_days"], ascending=False)  # recent = high
    out["F_score"] = pd.cut(out["num_orders"], [0, 1, 2, 3, 5, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
    out["M_score"] = _quintile_score(out["total_spend"], ascending=True)
    out["RFM_score"] = out["R_score"].astype(str) + out["F_score"].astype(str) + out["M_score"].astype(str)
    fm = (out["F_score"] + out["M_score"]) / 2
    r = out["R_score"]
    conditions = [
        (r >= 4) & (fm >= 4),
        (r >= 3) & (fm >= 3),
        (r >= 4) & (fm < 3),
        (r <= 2) & (fm >= 3),
        (r <= 2) & (fm < 3),
    ]
    names = ["Champions", "Loyal Customers", "New / Promising", "At Risk", "Hibernating"]
    out["rfm_segment"] = np.select(conditions, names, default="Need Attention")
    return out


def fit_kmeans(cdf, k):
    """StandardScaler + KMeans(n_clusters=k, n_init=10, random_state=42). Returns (labels, X, model)."""
    X = StandardScaler().fit_transform(cdf[CLUSTER_FEATURES].astype(float))
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    return model.labels_, X, model


def ensure_derived(cdf):
    """Compute the derived columns (spend_segment, buyer_type, rfm_segment, cluster) if missing."""
    cdf = cdf.copy()
    seg_ok = ("spend_segment" in cdf.columns and cdf["spend_segment"].notna().all()
              and set(cdf["spend_segment"].unique()) <= set(SEGMENT_ORDER))
    if not seg_ok:
        cdf["spend_segment"] = assign_spend_segment(cdf["total_spend"])
    buyer_ok = ("buyer_type" in cdf.columns and cdf["buyer_type"].notna().all()
                and set(cdf["buyer_type"].unique()) <= set(BUYER_ORDER))
    if not buyer_ok:
        cdf["buyer_type"] = assign_buyer_type(cdf["num_orders"])
    if "rfm_segment" not in cdf.columns or cdf["rfm_segment"].isna().any():
        cdf["rfm_segment"] = compute_rfm(cdf)["rfm_segment"].values
    if "cluster" not in cdf.columns or cdf["cluster"].isna().any():
        if len(cdf) >= 5:
            cdf["cluster"] = fit_kmeans(cdf, 4)[0].astype("int64")
        else:
            cdf["cluster"] = 0
    return cdf


ensure_derived_cached = st.cache_data(show_spinner=False)(ensure_derived)


def run_cleaning_pipeline(raw):
    """Apply the 5-step cleaning pipeline. Returns (df_clean, cleaning_log, df_returns)."""
    log_rows = []
    df = raw.copy()

    def add_step(no, stage, issue, action, before, after):
        removed = before - after
        log_rows.append({
            "step_no": no, "cleaning_stage": stage, "issue_addressed": issue,
            "action_taken": action, "rows_before": int(before), "rows_after": int(after),
            "rows_removed": int(removed),
            "pct_removed": round(removed / before * 100, 2) if before else 0.0,
        })

    # Step 1 - missing customer identifiers
    before = len(df)
    df = df[df["customer_id"].notna() & (df["customer_id"].astype(str).str.strip() != "")]
    add_step(1, "Identity validation", "Missing customer_id",
             "Dropped rows with no customer identifier", before, len(df))

    # Step 2 - missing dates
    before = len(df)
    df = df[df["transaction_date"].notna()]
    add_step(2, "Temporal validation", "Missing transaction_date",
             "Dropped rows with no transaction date", before, len(df))

    # Step 3 - exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    add_step(3, "Duplicate removal", "Exact duplicate rows",
             "Kept first occurrence of every exact duplicate", before, len(df))

    # Step 4 - returns / cancellations (stored separately as _df_returns)
    before = len(df)
    returns_mask = (df["quantity"] <= 0) | (df["total_amount"] < 0)
    df_returns = df[returns_mask].copy()
    df = df[~returns_mask]
    add_step(4, "Returns separation", "Negative quantity / amount (returns, cancellations)",
             "Moved to _df_returns and excluded from sales analytics", before, len(df))

    # Step 5 - price validation and arithmetic reconciliation
    before = len(df)
    df = df[df["unit_price"].notna() & (df["unit_price"] > 0)]
    add_step(5, "Price & arithmetic validation",
             "Non-positive unit_price; total_amount != quantity x unit_price",
             "Dropped invalid prices; recomputed total_amount = round(quantity x unit_price, 2)",
             before, len(df))

    df_clean = normalize_transactions(df)
    log = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    return df_clean, log, df_returns.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. SYNTHETIC DATA GENERATOR (auto-fallback, RANDOM_STATE = 42)
# ----------------------------------------------------------------------------
CATALOG = {
    "Electronics": {"Wireless Earbuds": 2499, "Smart Watch": 4999, "Power Bank": 1299,
                    "Bluetooth Speaker": 1999, "USB-C Cable": 399},
    "Fashion": {"Cotton T-Shirt": 599, "Denim Jeans": 1499, "Running Shoes": 2999,
                "Leather Wallet": 799, "Sunglasses": 999},
    "Home & Kitchen": {"Non-stick Pan": 1199, "Water Bottle": 449, "Bedsheet Set": 1799,
                       "LED Desk Lamp": 899, "Storage Box": 549},
    "Groceries": {"Basmati Rice 5kg": 649, "Cooking Oil 1L": 189, "Masala Tea 250g": 229,
                  "Dry Fruits Pack": 799, "Packaged Snacks": 99},
    "Beauty & Personal Care": {"Face Wash": 249, "Shampoo": 349, "Perfume": 1299,
                               "Moisturizer": 449, "Hair Dryer": 1599},
    "Books & Stationery": {"Notebook Set": 299, "Fountain Pen": 499, "Novel": 399,
                           "Planner Diary": 349, "Geometry Box": 199},
    "Sports & Fitness": {"Yoga Mat": 899, "Dumbbell Set": 1999, "Cricket Bat": 1799,
                         "Resistance Bands": 499, "Skipping Rope": 249},
}
# Products that are frequently bought together (gives the market-basket tab real signal)
BUNDLES = [
    ("Wireless Earbuds", "Power Bank"), ("Cotton T-Shirt", "Denim Jeans"),
    ("Shampoo", "Face Wash"), ("Notebook Set", "Fountain Pen"),
    ("Yoga Mat", "Resistance Bands"), ("Non-stick Pan", "Storage Box"),
    ("Basmati Rice 5kg", "Cooking Oil 1L"),
]


@st.cache_data(show_spinner="Generating synthetic retail data (RANDOM_STATE = 42)...")
def generate_synthetic(n_customers=1500):
    """Create df_raw (with realistic defects), df_clean, customer_df, cleaning_log, df_returns."""
    rng = np.random.default_rng(RANDOM_STATE)
    prod_cat = {p: c for c, items in CATALOG.items() for p in items}
    prod_price = {p: v for items in CATALOG.values() for p, v in items.items()}
    cats = list(CATALOG)
    cat_products = {c: list(items) for c, items in CATALOG.items()}

    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    weights = np.ones(len(days))
    months = days.month.values
    weights[np.isin(months, [10, 11])] *= 1.7   # festive season
    weights[months == 12] *= 1.3
    weights[days.day_of_week >= 5] *= 1.25  # weekends
    first_pool = np.arange(0, 540)               # acquisition window: first 18 months
    first_w = weights[first_pool] / weights[first_pool].sum()

    rows = []
    order_no = 0
    for i in range(n_customers):
        cid = f"CUST{i + 1:05d}"
        scale = rng.lognormal(0, 0.8)
        n_orders = 1 if rng.random() < 0.45 else int(min(1 + rng.geometric(0.3), 30))
        fav = rng.choice(cats, size=2, replace=False)
        offset = int(rng.choice(first_pool, p=first_w))
        for o in range(n_orders):
            if o > 0:
                offset += int(rng.exponential(55)) + 1
            if offset >= len(days):
                break
            order_no += 1
            tid = f"TXN{order_no:06d}"
            n_lines = int(min(1 + rng.poisson(1.1), 5))
            prods = []
            for _ in range(n_lines):
                c = rng.choice(fav) if rng.random() < 0.7 else rng.choice(cats)
                prods.append(str(rng.choice(cat_products[str(c)])))
            if rng.random() < 0.22:
                prods.extend(BUNDLES[int(rng.integers(len(BUNDLES)))])
            for p in dict.fromkeys(prods):  # unique products per order
                qty = int(min(1 + rng.poisson(0.5 * scale), 8))
                price = round(prod_price[p] * rng.uniform(0.95, 1.05), 2)
                rows.append((tid, cid, days[offset], p, prod_cat[p], qty, price, round(qty * price, 2)))

    base = pd.DataFrame(rows, columns=TX_COLUMNS)
    base["quantity"] = base["quantity"].astype("int64")

    # ---- inject realistic data-quality defects to build df_raw ----
    ret = base.sample(frac=0.02, random_state=RANDOM_STATE).copy()
    ret["transaction_id"] = [f"RTN{i + 1:06d}" for i in range(len(ret))]
    ret["quantity"] = -ret["quantity"]
    ret["total_amount"] = -ret["total_amount"]
    raw = pd.concat([base, ret], ignore_index=True)
    raw["customer_id"] = raw["customer_id"].astype(object)

    n = len(raw)
    idx = np.arange(n)
    miss_id = rng.choice(idx, size=int(0.02 * n), replace=False)
    raw.loc[miss_id, "customer_id"] = np.nan
    miss_dt = rng.choice(idx, size=int(0.015 * n), replace=False)
    raw.loc[miss_dt, "transaction_date"] = pd.NaT
    bad_price = rng.choice(idx, size=int(0.003 * n), replace=False)
    raw.loc[bad_price, "unit_price"] = 0.0
    mismatch = rng.choice(idx, size=int(0.01 * n), replace=False)
    raw.loc[mismatch, "total_amount"] = (raw.loc[mismatch, "total_amount"] * 1.1).round(2)

    eligible = raw[raw["customer_id"].notna() & raw["transaction_date"].notna()]
    dupes = eligible.sample(frac=0.015, random_state=RANDOM_STATE)
    raw = pd.concat([raw, dupes], ignore_index=True)
    df_raw = raw.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # ---- clean it with the documented 5-step pipeline ----
    df_clean, cleaning_log, df_returns = run_cleaning_pipeline(df_raw)
    customer_df = ensure_derived(build_customer_df(df_clean))
    return {
        "df_raw": df_raw, "df_clean": df_clean, "customer_df": customer_df,
        "cleaning_log": cleaning_log, "df_returns": df_returns,
    }


# ----------------------------------------------------------------------------
# 5. DATA LOADING WITH AUTO-FALLBACK
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(raw_bytes):
    """Parse uploaded CSV bytes (cached so reruns stay fast)."""
    return pd.read_csv(io.BytesIO(raw_bytes))


def read_source(uploaded, path):
    """Return (DataFrame|None, origin) trying upload first, then disk."""
    if uploaded is not None:
        try:
            return read_csv_bytes(uploaded.getvalue()), "uploaded"
        except Exception:
            return None, "unreadable upload"
    if os.path.exists(path):
        try:
            return pd.read_csv(path), "disk"
        except Exception:
            return None, "unreadable file"
    return None, "not found"


def _valid_log(log):
    return log is not None and all(c in log.columns for c in LOG_COLUMNS)


def _valid_customer(cdf):
    return cdf is not None and all(c in cdf.columns for c in CUSTOMER_BASE_COLUMNS)


def assemble_data(up_clean, up_cust, up_log):
    """Load df_clean / customer_df / cleaning_log; fall back to synthetic data when needed."""
    raw_clean, src_clean = read_source(up_clean, DISK_PATHS["df_clean"])
    raw_cust, src_cust = read_source(up_cust, DISK_PATHS["customer_df"])
    raw_log, src_log = read_source(up_log, DISK_PATHS["cleaning_log"])
    notes = []

    df_clean = None
    if raw_clean is not None:
        try:
            df_clean = normalize_transactions(raw_clean)
            if df_clean.empty:
                raise ValueError("no valid rows after validation")
        except Exception as exc:
            notes.append(f"cleaned_transactions.csv ({src_clean}) rejected: {exc}")
            df_clean = None

    if df_clean is None:
        # ----- fully synthetic fallback -----
        syn = generate_synthetic()
        log = syn["cleaning_log"]
        if raw_log is not None and _valid_log(raw_log):
            log = raw_log[LOG_COLUMNS].copy()
            notes.append("Cleaning log taken from provided file.")
        if raw_cust is not None:
            notes.append("customer_analytics_master.csv ignored because no transactions file was available.")
        return {
            "df_raw": syn["df_raw"], "df_clean": syn["df_clean"],
            "customer_df": syn["customer_df"], "cleaning_log": log,
            "df_returns": syn["df_returns"], "is_synthetic": True, "raw_available": True,
            "source": "Synthetic data (RANDOM_STATE = 42)", "notes": notes,
        }

    # ----- real transactions available -----
    if _valid_customer(raw_cust):
        cust = raw_cust.copy()
        for col in ["first_purchase_date", "last_purchase_date"]:
            cust[col] = pd.to_datetime(cust[col], errors="coerce")
        cust["customer_id"] = cust["customer_id"].astype(str)
        if not set(df_clean["customer_id"].unique()) <= set(cust["customer_id"]):
            notes.append("customer_analytics_master.csv did not cover all customers - rebuilt from transactions.")
            cust = build_customer_df(df_clean)
    else:
        if raw_cust is not None:
            notes.append("customer_analytics_master.csv missing required columns - rebuilt from transactions.")
        cust = build_customer_df(df_clean)
    customer_df = ensure_derived_cached(cust)

    if raw_log is not None and _valid_log(raw_log):
        log = raw_log[LOG_COLUMNS].copy()
        for c in ["step_no", "rows_before", "rows_after", "rows_removed", "pct_removed"]:
            log[c] = pd.to_numeric(log[c], errors="coerce")
        log_provided = True
    else:
        n = len(df_clean)
        stages = ["Identity validation", "Temporal validation", "Duplicate removal",
                  "Returns separation", "Price & arithmetic validation"]
        log = pd.DataFrame({
            "step_no": range(1, 6), "cleaning_stage": stages,
            "issue_addressed": "n/a (log file not provided)",
            "action_taken": "Raw data unavailable - cleaned file loaded as-is",
            "rows_before": n, "rows_after": n, "rows_removed": 0, "pct_removed": 0.0,
        })
        log_provided = False
        notes.append("data_cleaning_log.csv not provided - placeholder log shown.")

    return {
        "df_raw": df_clean.copy(), "df_clean": df_clean, "customer_df": customer_df,
        "cleaning_log": log, "df_returns": pd.DataFrame(columns=TX_COLUMNS),
        "is_synthetic": False, "raw_available": False, "log_provided": log_provided,
        "source": f"Loaded files (transactions: {src_clean})", "notes": notes,
    }


# ----------------------------------------------------------------------------
# 6. ANALYTICS HELPERS (shared by several tabs)
# ----------------------------------------------------------------------------
def seg_summary(cdf):
    """Spend-segment summary table (Sections 7-8)."""
    g = cdf.groupby("spend_segment").agg(
        customer_count=("customer_id", "count"), total_revenue=("total_spend", "sum"),
        total_orders=("num_orders", "sum"), mean_spend=("total_spend", "mean"),
        median_spend=("total_spend", "median"), avg_orders=("num_orders", "mean"),
        mean_aov=("avg_order_value", "mean"), mean_recency_days=("recency_days", "mean"),
    ).reindex(SEGMENT_ORDER)
    for c in ["customer_count", "total_revenue", "total_orders"]:
        g[c] = g[c].fillna(0)
    out = pd.DataFrame(index=g.index)
    out["customer_count"] = g["customer_count"].astype(int)
    out["customer_share_pct"] = g["customer_count"] / max(g["customer_count"].sum(), 1) * 100
    out["total_revenue"] = g["total_revenue"]
    out["revenue_share_pct"] = g["total_revenue"] / max(g["total_revenue"].sum(), 1e-9) * 100
    out["total_orders"] = g["total_orders"].astype(int)
    out["order_share_pct"] = g["total_orders"] / max(g["total_orders"].sum(), 1) * 100
    for c in ["mean_spend", "median_spend", "avg_orders", "mean_aov", "mean_recency_days"]:
        out[c] = g[c]
    out["concentration_index"] = np.where(out["customer_share_pct"] > 0,
                                          out["revenue_share_pct"] / out["customer_share_pct"].replace(0, np.nan), np.nan)
    return out


def buyer_summary(cdf):
    """Table 10.1 - 1-Time vs Repeat buyers."""
    g = cdf.groupby("buyer_type").agg(
        customer_count=("customer_id", "count"), total_orders=("num_orders", "sum"),
        total_revenue=("total_spend", "sum"),
    ).reindex(BUYER_ORDER)
    g = g.fillna(0)
    out = pd.DataFrame(index=g.index)
    out["customer_count"] = g["customer_count"].astype(int)
    out["customer_share_pct"] = g["customer_count"] / max(g["customer_count"].sum(), 1) * 100
    out["total_orders"] = g["total_orders"].astype(int)
    out["orders_per_customer"] = g["total_orders"] / g["customer_count"].replace(0, np.nan)
    out["total_revenue"] = g["total_revenue"]
    out["revenue_share_pct"] = g["total_revenue"] / max(g["total_revenue"].sum(), 1e-9) * 100
    out["aov"] = g["total_revenue"] / g["total_orders"].replace(0, np.nan)
    out["spend_per_customer"] = g["total_revenue"] / g["customer_count"].replace(0, np.nan)
    return out


def pareto_table(cdf):
    """Cumulative customer % vs cumulative revenue % (customers ranked by spend)."""
    s = cdf["total_spend"].sort_values(ascending=False).reset_index(drop=True)
    n = len(s)
    return pd.DataFrame({
        "cust_pct": np.arange(1, n + 1) / n * 100,
        "rev_pct": s.cumsum().values / s.sum() * 100,
    })


def pareto_cut(par, threshold=80.0):
    """Percentage of customers needed to reach `threshold` % of revenue."""
    pos = int(np.argmax(par["rev_pct"].values >= threshold))
    return float(par["cust_pct"].iloc[pos])


def pareto_revenue_at(par, cust_pct):
    """Revenue share generated by the top `cust_pct` % of customers."""
    pos = int(np.argmax(par["cust_pct"].values >= cust_pct))
    return float(par["rev_pct"].iloc[pos])


def cohort_matrix(fdf, fcust, max_offset=12):
    """Monthly cohort retention % (rows = acquisition month, cols = months since first purchase)."""
    first = fcust.set_index("customer_id")["first_purchase_date"]
    t = fdf[["customer_id", "transaction_date"]].copy()
    t["cohort"] = t["customer_id"].map(first).dt.to_period("M")
    t["order_m"] = t["transaction_date"].dt.to_period("M")
    t = t.dropna(subset=["cohort"])
    t["offset"] = ((t["order_m"].dt.year - t["cohort"].dt.year) * 12
                   + (t["order_m"].dt.month - t["cohort"].dt.month))
    t = t[(t["offset"] >= 0) & (t["offset"] <= max_offset)]
    active = t.groupby(["cohort", "offset"])["customer_id"].nunique().unstack(fill_value=0)
    active = active.reindex(columns=range(0, max_offset + 1), fill_value=0)
    sizes = fcust.assign(cohort=fcust["first_purchase_date"].dt.to_period("M")).groupby("cohort").size()
    sizes = sizes.reindex(active.index)
    ret = active.div(sizes, axis=0) * 100
    last = fdf["transaction_date"].max().to_period("M")
    max_obs = np.array([(last.year - c.year) * 12 + (last.month - c.month) for c in ret.index])
    unobserved = np.arange(max_offset + 1)[None, :] > max_obs[:, None]
    ret = ret.where(~unobserved)
    ret.index = [str(c) for c in ret.index]
    return ret, sizes.values


def market_basket(items_df, min_support):
    """Frequent product pairs using itertools/Counter. Returns (rules_df, n_baskets, n_multi)."""
    baskets = items_df.groupby("basket_id")["product_name"].agg(lambda s: sorted(set(s)))
    n = len(baskets)
    item_cnt, pair_cnt = Counter(), Counter()
    multi = 0
    for b in baskets:
        item_cnt.update(b)
        if len(b) >= 2:
            multi += 1
            pair_cnt.update(itertools.combinations(b, 2))
    rows = []
    for (a, b), c in pair_cnt.items():
        support = c / n
        if support < min_support:
            continue
        rows.append({
            "product_a": a, "product_b": b, "pair_count": c, "support": support,
            "confidence_a_to_b": c / item_cnt[a], "confidence_b_to_a": c / item_cnt[b],
            "lift": support / ((item_cnt[a] / n) * (item_cnt[b] / n)),
        })
    cols = ["product_a", "product_b", "pair_count", "support", "confidence_a_to_b", "confidence_b_to_a", "lift"]
    rules = pd.DataFrame(rows, columns=cols)
    if not rules.empty:
        rules = rules.sort_values(["lift", "support"], ascending=False).reset_index(drop=True)
    return rules, n, multi


market_basket_cached = st.cache_data(show_spinner=False)(market_basket)


@st.cache_data(show_spinner=False)
def kmeans_diagnostics(features_df, k_max):
    """WCSS (inertia) and silhouette score for k = 1..k_max on scaled features."""
    X = StandardScaler().fit_transform(features_df.values.astype(float))
    rows = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = np.nan
        if k >= 2:
            sil = silhouette_score(X, km.labels_, sample_size=min(len(X), 3000), random_state=RANDOM_STATE)
        rows.append((k, km.inertia_, sil))
    return pd.DataFrame(rows, columns=["k", "wcss", "silhouette"])


def horizontal_boxplot(ax, data, **kwargs):
    """Horizontal boxplot that works on old (vert=) and new (orientation=) Matplotlib."""
    try:
        return ax.boxplot(data, orientation="horizontal", **kwargs)  # Matplotlib >= 3.10
    except (TypeError, ValueError):
        return ax.boxplot(data, vert=False, **kwargs)  # older Matplotlib


def grouped_boxplot(ax, df, value_col, title, ylabel):
    """Log-scale boxplot by buyer type using only plain Matplotlib (version-safe)."""
    present = [b for b in BUYER_ORDER if (df["buyer_type"] == b).any()]
    data = [df.loc[df["buyer_type"] == b, value_col].values for b in present]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="black", linewidth=1.5))
    for patch, b in zip(bp["boxes"], present):
        patch.set_facecolor(RP_COLORS[b])
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(present)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)


# ----------------------------------------------------------------------------
# 7. TAB 1 - OVERVIEW, TEAM & DATA DICTIONARY
# ----------------------------------------------------------------------------
TEAM = [
    (1, "Hasnain Raza Khan", "Sections 1, 2, 17"),
    (2, "Aditi Sharma", "Section 3, Appendix A"),
    (3, "Sonam", "Section 4"),
    (4, "Samiya Khan", "Section 5"),
    (5, "Aditya Singh", "Section 6"),
    (6, "Saif Khan", "Section 7"),
    (7, "Divyanshi", "Section 8"),
    (8, "Abhi Rastogi", "Section 9"),
    (9, "Sanjay Rathour", "Section 10"),
    (10, "Dev Rastogi", "Section 11"),
    (11, "Vidhan Agarwal", "Sections 12.1, 12.2"),
    (12, "Anshu", "Sections 12.3, 12.4"),
    (13, "Rudra Pratap Singh", "Sections 12.5, 13, 14"),
    (14, "Ayan Khan", "Sections 15, 16, Appendix B"),
]


def render_tab1(fdf, fcust, B):
    st.header("Project Overview, Team Allocation & Data Dictionary")
    st.caption("Sections 1-3 & Appendix A")

    revenue = fdf["total_amount"].sum()
    orders = fdf["transaction_id"].nunique()
    customers = fdf["customer_id"].nunique()
    aov = revenue / orders if orders else np.nan
    repeaters = int((fcust["num_orders"] >= 2).sum())
    rpr = repeaters / len(fcust) * 100 if len(fcust) else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", compact_money(revenue))
    c2.metric("Total Orders", f"{orders:,}")
    c3.metric("Total Customers", f"{customers:,}")
    c4.metric("Average Order Value (AOV)", money(aov))
    c5.metric("Repeat Purchase Rate (RPR)", pct(rpr))
    interp(f"The current view contains {orders:,} orders from {customers:,} customers worth {money(revenue)}, "
           f"giving an AOV of {money(aov)}. {repeaters:,} customers ({pct(rpr)}) have purchased at least twice, "
           f"which is the repeat purchase rate used throughout the study.")

    # ---- Section 1 & 2 - Introduction, objectives, methodology ----
    section_header("Sections 1-2: Introduction, Objectives & Methodology", "Member 1 - Hasnain Raza Khan")
    st.markdown(
        f"""
**Business problem.** Retailers earn most of their revenue from a small share of customers and a small set of products.
This study uses retail transaction data to understand *who* spends, *what* they buy, *when* they buy and *which customers
deserve targeted action*.

**Objectives**
1. Audit and clean the raw transactions, documenting every cleaning step (Sections 4-5).
2. Profile customers and segment them by spend (Sections 6-8).
3. Explore products, repeat purchase behaviour and seasonality (Sections 9-11).
4. Apply advanced techniques - **Pareto 80/20, RFM, Cohort Retention, Market Basket Analysis and K-Means** (Section 12).
5. Convert findings into business recommendations (Sections 13-16).

**Methodology.** Data cleaning -> customer aggregation -> descriptive statistics -> segmentation -> advanced analytics
-> interpretation. All random operations use `RANDOM_STATE = {RANDOM_STATE}` so results are reproducible.
Currency is expressed in Indian rupees ({CURRENCY_SYMBOL}).
        """
    )

    # ---- Team allocation ----
    section_header("Team Allocation & Credit Distribution", "Members 1 & 2")
    team_df = pd.DataFrame(TEAM, columns=["member_no", "member_name", "sections_owned"])
    team_df["credit_allocation"] = f"{100 / 14:.2f}%"
    show_table(team_df)
    interp(f"All {len(team_df)} members receive an equal credit allocation of {100 / 14:.2f}% "
           f"(total {100 / 14 * 14:.0f}%). Section 10 (repeat purchase) and Section 12 (advanced analytics) "
           f"are the heaviest technical workloads and are split across members 9, 11, 12 and 13.")

    # ---- Section 3 - Data dictionary ----
    section_header("Section 3 & Appendix A: Data Dictionary", "Member 2 - Aditi Sharma")
    dd = [
        ("df_clean", "transaction_id", "string", "Unique order / basket identifier (one order can have several item lines)"),
        ("df_clean", "customer_id", "string", "Unique customer identifier"),
        ("df_clean", "transaction_date", "datetime64", "Date on which the transaction occurred"),
        ("df_clean", "product_name", "string", "Name of the product purchased"),
        ("df_clean", "category", "string", "Product category"),
        ("df_clean", "quantity", "int64 (> 0)", "Units purchased in the line"),
        ("df_clean", "unit_price", "float64 (> 0)", f"Price per unit in {CURRENCY_SYMBOL}"),
        ("df_clean", "total_amount", "float64", "round(quantity x unit_price, 2)"),
        ("customer_df", "customer_id", "string (unique)", "Customer key"),
        ("customer_df", "total_spend", "float64", "Lifetime spend of the customer"),
        ("customer_df", "num_orders", "int64", "Distinct transaction_id count"),
        ("customer_df", "avg_order_value", "float64", "total_spend / num_orders"),
        ("customer_df", "first_purchase_date", "datetime64", "Date of first purchase (cohort anchor)"),
        ("customer_df", "last_purchase_date", "datetime64", "Date of most recent purchase"),
        ("customer_df", "recency_days", "int64", "(snapshot_date - last_purchase_date) in days; snapshot = max date + 1 day"),
        ("customer_df", "spend_segment", "category (derived)", "Low / Medium / High tertile of total_spend (pd.qcut)"),
        ("customer_df", "buyer_type", "string (derived)", "1-Time Buyers if num_orders == 1, else Repeat Buyers"),
        ("customer_df", "rfm_segment", "string (derived)", "Named RFM group (Champions, Loyal, At Risk ...)"),
        ("customer_df", "cluster", "int64 (derived)", "K-Means cluster id (k = 4 default)"),
    ]
    for col in LOG_COLUMNS:
        dd.append(("cleaning_log", col, "int/float/string", "Audit-trail field describing one cleaning step"))
    dd_df = pd.DataFrame(dd, columns=["table", "column", "expected_type", "description"])
    frames = {"df_clean": B["df_clean"], "customer_df": B["customer_df"], "cleaning_log": B["cleaning_log"]}
    dd_df["observed_dtype"] = [
        str(frames[t][c].dtype) if c in frames[t].columns else "n/a" for t, c in zip(dd_df["table"], dd_df["column"])
    ]
    show_table(dd_df)
    interp(f"The dictionary documents {len(dd_df)} fields across three tables: df_clean "
           f"({len(B['df_clean']):,} rows), customer_df ({len(B['customer_df']):,} rows) and cleaning_log "
           f"({len(B['cleaning_log'])} rows). The observed_dtype column confirms that the loaded data follows the schema contract.")

    st.markdown("**Raw vs. clean data preview**")
    left, right = st.columns(2)
    with left:
        st.markdown("*df_raw (first 10 rows)*")
        show_table(B["df_raw"].head(10))
    with right:
        st.markdown("*df_clean (first 10 rows)*")
        show_table(B["df_clean"].head(10))
    retained = len(B["df_clean"]) / len(B["df_raw"]) * 100 if len(B["df_raw"]) else np.nan
    if B["raw_available"]:
        interp(f"The raw table has {len(B['df_raw']):,} rows versus {len(B['df_clean']):,} after cleaning, so "
               f"{pct(retained)} of the raw records were retained. Raw rows can show missing IDs, missing dates, "
               f"duplicates and negative quantities that the clean table no longer contains.")
    else:
        interp(f"Raw data is not available in the loaded files, so the preview shows the cleaned table on both sides "
               f"({len(B['df_clean']):,} rows). Upload or generate raw data to see the cleaning impact.")


# ----------------------------------------------------------------------------
# 8. TAB 2 - DATA QUALITY AUDIT & CLEANING PIPELINE
# ----------------------------------------------------------------------------
def reconciliation(df, name):
    """Check quantity x unit_price == total_amount within a 0.01 tolerance."""
    d = df.dropna(subset=["quantity", "unit_price", "total_amount"])
    diff = (d["quantity"] * d["unit_price"] - d["total_amount"]).abs()
    passed = int((diff <= 0.01).sum())
    tested = len(d)
    return {
        "dataset": name, "rows_tested": tested, "rows_passed": passed, "rows_failed": tested - passed,
        "pass_rate_pct": round(passed / tested * 100, 2) if tested else np.nan,
        "max_abs_difference": round(float(diff.max()), 2) if tested else np.nan,
    }


def render_tab2(B):
    st.header("Data Quality Audit & Cleaning Pipeline")
    st.caption("Sections 4-5 (audit uses the full raw/cleaned data and is not affected by sidebar filters)")
    raw, log, rets = B["df_raw"], B["cleaning_log"].sort_values("step_no"), B["df_returns"]

    # ---- Section 4 ----
    section_header("Section 4: Data Quality Audit", "Member 3 - Sonam")
    if B["raw_available"]:
        dq = {
            "Missing IDs": int((raw["customer_id"].isna() | (raw["customer_id"].astype(str).str.strip() == "")).sum()),
            "Missing Dates": int(raw["transaction_date"].isna().sum()),
            "Exact Duplicates": int(raw.duplicated().sum()),
            "Negative Returns": int((pd.to_numeric(raw["quantity"], errors="coerce") < 0).sum()),
        }
        basis = f"{len(raw):,} raw rows"
    else:
        steps = log["rows_removed"].tolist()
        steps = (steps + [0, 0, 0, 0])[:4]
        dq = dict(zip(["Missing IDs", "Missing Dates", "Exact Duplicates", "Negative Returns"], [int(x) for x in steps]))
        basis = "rows removed per step in the supplied cleaning log (raw file not provided)"

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [PALETTE["negative"], PALETTE["accent"], PALETTE["secondary"], PALETTE["neutral"]]
    bars = ax.bar(list(dq.keys()), list(dq.values()), color=colors)
    for b_, v in zip(bars, dq.values()):
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), f"{v:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Affected rows")
    ax.set_title("Four headline data-quality issues")
    show_fig(fig)
    worst = max(dq, key=dq.get)
    interp(f"Based on {basis}, the largest issue is '{worst}' with {dq[worst]:,} rows. Together the four headline issues "
           f"affect {sum(dq.values()):,} rows (some rows can have more than one issue), which is why each one is handled in its own cleaning step.")

    recon = [reconciliation(raw, "df_raw" if B["raw_available"] else "df_raw (= df_clean)"),
             reconciliation(B["df_clean"], "df_clean")]
    recon_df = pd.DataFrame(recon)
    st.markdown("**Arithmetic reconciliation check** - `quantity x unit_price == total_amount` (tolerance 0.01)")
    show_table(recon_df)
    interp(f"{recon[0]['dataset']} passes the reconciliation for {pct(recon[0]['pass_rate_pct'], 2)} of "
           f"{recon[0]['rows_tested']:,} rows ({recon[0]['rows_failed']:,} failures, max gap {money(recon[0]['max_abs_difference'], 2)}). "
           f"After cleaning, df_clean passes {pct(recon[1]['pass_rate_pct'], 2)} of {recon[1]['rows_tested']:,} rows, "
           f"confirming total_amount is fully consistent.")

    # ---- Section 5 ----
    section_header("Section 5: Cleaning Pipeline", "Member 4 - Samiya Khan")
    st.markdown("**5-step cleaning log**")
    show_table(log)
    total_removed = int(log["rows_removed"].sum())
    start_rows = int(log["rows_before"].iloc[0]) if len(log) else 0
    end_rows = int(log["rows_after"].iloc[-1]) if len(log) else 0
    biggest = log.loc[log["rows_removed"].idxmax()] if len(log) and log["rows_removed"].max() > 0 else None
    interp(f"The pipeline starts with {start_rows:,} rows and ends with {end_rows:,}, removing {total_removed:,} rows "
           f"({pct(total_removed / start_rows * 100 if start_rows else 0, 2)}). "
           + (f"The biggest single reduction is step {int(biggest['step_no'])} ({biggest['cleaning_stage']}) with "
              f"{int(biggest['rows_removed']):,} rows." if biggest is not None else "No rows were removed by any step."))

    st.markdown("**Returns & cancellations summary (`_df_returns`)**")
    if len(rets):
        r_val = rets["total_amount"].abs().sum()
        ret_summary = pd.DataFrame({
            "metric": ["Return lines", "Units returned", f"Value returned ({CURRENCY_SYMBOL})",
                       "Distinct customers", "Return lines as % of raw rows"],
            "value": [f"{len(rets):,}", f"{int(rets['quantity'].abs().sum()):,}", money(r_val),
                      f"{rets['customer_id'].nunique():,}", pct(len(rets) / len(raw) * 100, 2)],
        })
        by_cat = (rets.assign(value=rets["total_amount"].abs())
                  .groupby("category").agg(return_lines=("value", "size"), value_returned=("value", "sum"))
                  .sort_values("value_returned", ascending=False).reset_index())
        a, b = st.columns(2)
        with a:
            show_table(ret_summary)
        with b:
            show_table(by_cat.round(2))
        top_cat = by_cat.iloc[0]
        interp(f"{len(rets):,} return/cancellation lines worth {money(r_val)} were separated from sales analytics "
               f"({pct(len(rets) / len(raw) * 100, 2)} of raw rows). {top_cat['category']} has the highest returned value "
               f"({money(top_cat['value_returned'])}), so it is the first category to review for quality or fit issues.")
    else:
        st.info("No returns/cancellations are available in the loaded data (raw file not provided or none present).")

    st.markdown("**Data cleaning funnel**")
    if len(log):
        stages = ["Raw input"] + [f"S{int(r.step_no)}: {r.cleaning_stage}" for r in log.itertuples()]
        widths = [float(log["rows_before"].iloc[0])] + log["rows_after"].astype(float).tolist()
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 4.8))
        widest = max(widths) if max(widths) > 0 else 1
        ypos = np.arange(len(stages))
        axl.barh(ypos, widths, left=[(widest - w) / 2 for w in widths],
                 color=sns.light_palette(PALETTE["primary"], len(stages) + 2)[2:][::-1])
        for y, w in zip(ypos, widths):
            axl.text(widest / 2, y, f"{int(w):,}", ha="center", va="center", color="black", fontweight="bold", fontsize=9)
        axl.set_yticks(ypos)
        axl.set_yticklabels(stages)
        axl.invert_yaxis()
        axl.set_xticks([])
        axl.set_title("Panel 1: Rows remaining after each step (funnel)")
        steps_lbl = [f"S{int(s)}" for s in log["step_no"]]
        retained_ = log["rows_after"].astype(float).values
        removed_ = log["rows_removed"].astype(float).values
        axr.bar(steps_lbl, retained_, color=PALETTE["positive"], label="Retained")
        axr.bar(steps_lbl, removed_, bottom=retained_, color=PALETTE["negative"], label="Removed")
        for i, (rem, p_) in enumerate(zip(removed_, log["pct_removed"].astype(float).values)):
            axr.text(i, retained_[i] + removed_[i], f"-{int(rem):,}\n({p_:.2f}%)", ha="center", va="bottom", fontsize=8)
        axr.set_ylim(0, (retained_ + removed_).max() * 1.15 if len(retained_) else 1)
        axr.set_ylabel("Rows")
        axr.set_title("Panel 2: Retained vs. removed rows per step")
        axr.legend(loc="lower right")
        show_fig(fig)
        interp(f"Retention stays above {pct(end_rows / start_rows * 100 if start_rows else 0, 1)} of the original rows at every stage, "
               f"with {end_rows:,} rows surviving all five steps. The red segments in Panel 2 are small relative to the retained base, "
               f"showing that cleaning improves data quality without discarding a large part of the dataset.")


# ----------------------------------------------------------------------------
# 9. TAB 3 - CUSTOMER SUMMARY & SPEND SEGMENTATION
# ----------------------------------------------------------------------------
def render_tab3(fdf, fcust):
    st.header("Customer Summary & Spend Segmentation")
    st.caption("Sections 6-8")

    # ---- Section 6 ----
    section_header("Section 6: Descriptive Statistics & Distributions", "Member 5 - Aditya Singh")
    cols = ["total_spend", "num_orders", "avg_order_value", "recency_days"]
    desc = pd.DataFrame({
        "variable": cols,
        "mean": [fcust[c].mean() for c in cols],
        "sd": [fcust[c].std() for c in cols],
        "min": [fcust[c].min() for c in cols],
        "25%": [fcust[c].quantile(0.25) for c in cols],
        "median": [fcust[c].median() for c in cols],
        "75%": [fcust[c].quantile(0.75) for c in cols],
        "max": [fcust[c].max() for c in cols],
        "skewness": [fcust[c].skew() for c in cols],
    }).round(2)
    show_table(desc)
    sp = desc.set_index("variable")
    interp(f"Average customer spend is {money(sp.loc['total_spend', 'mean'])} but the median is only "
           f"{money(sp.loc['total_spend', 'median'])} with skewness {sp.loc['total_spend', 'skewness']:.2f}, so a few big spenders pull the mean up. "
           f"Customers place {sp.loc['num_orders', 'mean']:.2f} orders on average and last bought "
           f"{sp.loc['recency_days', 'median']:.0f} days ago at the median.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    panels = [("total_spend", PALETTE["primary"], f"Total spend ({CURRENCY_SYMBOL})"),
              ("num_orders", PALETTE["accent"], "Number of orders"),
              ("recency_days", PALETTE["positive"], "Recency (days)")]
    for j, (col, color, label) in enumerate(panels):
        ax_h, ax_b = axes[0, j], axes[1, j]
        bins = int(min(30, max(fcust[col].nunique(), 2)))
        ax_h.hist(fcust[col], bins=bins, color=color, edgecolor="white", alpha=0.9)
        ax_h.axvline(fcust[col].mean(), color=PALETTE["negative"], linestyle="--", label="Mean")
        ax_h.axvline(fcust[col].median(), color="black", linestyle=":", label="Median")
        ax_h.set_title(f"Distribution of {label}")
        ax_h.set_ylabel("Customers")
        ax_h.legend(fontsize=8)
        horizontal_boxplot(ax_b, fcust[col], patch_artist=True, widths=0.6,
                           boxprops=dict(facecolor=color, alpha=0.6), medianprops=dict(color="black"))
        ax_b.set_yticks([])
        ax_b.set_xlabel(label)
    show_fig(fig)
    q3 = fcust["total_spend"].quantile(0.75)
    iqr = q3 - fcust["total_spend"].quantile(0.25)
    outliers = int((fcust["total_spend"] > q3 + 1.5 * iqr).sum())
    interp(f"The spend histogram is right-skewed, and {outliers:,} customers ({pct(outliers / len(fcust) * 100)}) lie beyond the boxplot whisker "
           f"({money(q3 + 1.5 * iqr)}). Order counts cluster at low values while recency spreads from {fcust['recency_days'].min()} "
           f"to {fcust['recency_days'].max()} days, showing both very active and long-lapsed customers.")

    # ---- Sections 7 & 8 ----
    section_header("Sections 7-8: Spend Segmentation (Low / Medium / High)", "Members 6 & 7 - Saif Khan & Divyanshi")
    seg = seg_summary(fcust)
    seg_display = seg.reset_index().rename(columns={"spend_segment": "spend_segment"}).round(2)
    show_table(seg_display)
    hi = seg.loc["High"]
    lo = seg.loc["Low"]
    interp(f"High-spend customers are {pct(hi['customer_share_pct'])} of the base but generate {pct(hi['revenue_share_pct'])} of revenue, "
           f"while Low-spend customers are {pct(lo['customer_share_pct'])} of customers and only {pct(lo['revenue_share_pct'])} of revenue. "
           f"The mean spend of a High customer ({money(hi['mean_spend'])}) is {hi['mean_spend'] / lo['mean_spend'] if lo['mean_spend'] else float('nan'):.1f}x that of a Low customer.")

    # Customer share vs revenue share + Concentration Index badges
    badges = ""
    for s in SEGMENT_ORDER:
        ci = seg.loc[s, "concentration_index"]
        badges += badge(f"{s}: CI {ci:.2f}x" if pd.notna(ci) else f"{s}: CI n/a", SEGMENT_COLORS[s])
    st.markdown("**Concentration Index (revenue share ÷ customer share):** " + badges, unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(SEGMENT_ORDER))
    w = 0.38
    ax.bar(x - w / 2, seg["customer_share_pct"], w, label="Customer share %", color=PALETTE["secondary"])
    ax.bar(x + w / 2, seg["revenue_share_pct"], w, label="Revenue share %", color=PALETTE["accent"])
    for i, s in enumerate(SEGMENT_ORDER):
        top = max(seg.loc[s, "customer_share_pct"], seg.loc[s, "revenue_share_pct"])
        ci = seg.loc[s, "concentration_index"]
        ax.text(i, top + 3, f"CI {ci:.2f}x" if pd.notna(ci) else "CI n/a", ha="center", fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor=SEGMENT_COLORS[s], edgecolor="none"))
    ax.set_xticks(x)
    ax.set_xticklabels(SEGMENT_ORDER)
    ax.set_ylim(0, max(seg["revenue_share_pct"].max(), seg["customer_share_pct"].max()) * 1.2 + 5)
    ax.set_ylabel("Share (%)")
    ax.set_title("Customer share vs. revenue share by spend segment")
    ax.legend()

    ax = axes[1]
    metrics = pd.DataFrame({"Customers": seg["customer_share_pct"], "Revenue": seg["revenue_share_pct"],
                            "Orders": seg["order_share_pct"]}).T
    left = np.zeros(len(metrics))
    for s in SEGMENT_ORDER:
        vals = metrics[s].values
        ax.barh(metrics.index, vals, left=left, color=SEGMENT_COLORS[s], label=s)
        for i, v in enumerate(vals):
            if v > 4:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share (%)")
    ax.set_title("100% stacked view: customers, revenue and orders")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    show_fig(fig)
    interp(f"The High segment has a Concentration Index of {hi['concentration_index']:.2f}x, meaning each High customer contributes "
           f"{hi['concentration_index']:.2f} times its fair share of revenue, whereas the Low segment sits at {lo['concentration_index']:.2f}x. "
           f"High customers also account for {pct(hi['order_share_pct'])} of orders, so their value comes from both frequency and basket size.")

    # Category mix breakdown across spend segments
    mix = fdf.pivot_table(index="category", columns="spend_segment", values="total_amount", aggfunc="sum", fill_value=0)
    mix = mix.reindex(columns=SEGMENT_ORDER, fill_value=0)
    mix_pct = mix.div(mix.sum(axis=0).replace(0, np.nan), axis=0) * 100
    mix_pct = mix_pct.fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    palette = sns.color_palette("tab10", n_colors=len(mix_pct))
    left = np.zeros(len(SEGMENT_ORDER))
    for color, (cat, row) in zip(palette, mix_pct.iterrows()):
        ax.barh(SEGMENT_ORDER, row.values, left=left, label=cat, color=color)
        for i, v in enumerate(row.values):
            if v > 6:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
        left += row.values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of segment revenue (%)")
    ax.set_title("Category mix across spend segments")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    show_fig(fig)
    show_table(mix_pct.round(2).reset_index())
    top_by_seg = {s: (mix_pct[s].idxmax(), mix_pct[s].max()) for s in SEGMENT_ORDER}
    interp(f"{top_by_seg['High'][0]} is the largest category for High-spend customers ({pct(top_by_seg['High'][1])} of their revenue), "
           f"while {top_by_seg['Low'][0]} leads for Low-spend customers ({pct(top_by_seg['Low'][1])}). "
           f"Differences in category mix show which categories to emphasise when upgrading customers between segments.")


# ----------------------------------------------------------------------------
# 10. TAB 4 - PRODUCT, REPEAT PURCHASE & SEASONALITY EDA
# ----------------------------------------------------------------------------
def render_tab4(fdf, fcust):
    st.header("Product, Repeat Purchase & Seasonality EDA")
    st.caption("Sections 9-11")

    # ---- Section 9 ----
    section_header("Section 9: Product & Category Performance", "Member 8 - Abhi Rastogi")
    prod = (fdf.groupby(["product_name", "category"]).agg(
        revenue=("total_amount", "sum"), units_sold=("quantity", "sum"), orders=("transaction_id", "nunique"))
        .reset_index().sort_values("revenue", ascending=False))
    prod["revenue_share_pct"] = prod["revenue"] / prod["revenue"].sum() * 100
    top10 = prod.head(10)
    cat = fdf.groupby("category")["total_amount"].sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].barh(top10["product_name"][::-1], top10["revenue"][::-1], color=PALETTE["secondary"])
    for y, v in enumerate(top10["revenue"][::-1]):
        axes[0].text(v, y, f" {compact_money(v)}", va="center", fontsize=8)
    axes[0].set_xlabel(f"Revenue ({CURRENCY_SYMBOL})")
    axes[0].set_title("Top 10 products by revenue")
    axes[0].set_xlim(0, top10["revenue"].max() * 1.18)
    axes[1].pie(cat.values, labels=cat.index, autopct="%1.1f%%", startangle=90, pctdistance=0.8,
                colors=sns.color_palette("tab10", len(cat)), wedgeprops=dict(width=0.45, edgecolor="white"),
                textprops={"fontsize": 8})
    axes[1].set_title("Category revenue share")
    show_fig(fig)
    show_table(top10.round(2))
    top_p = top10.iloc[0]
    interp(f"{top_p['product_name']} is the top product with {money(top_p['revenue'])} ({pct(top_p['revenue_share_pct'])} of revenue). "
           f"{cat.index[0]} is the leading category at {pct(cat.iloc[0] / cat.sum() * 100)} of revenue, and the top 10 products together "
           f"contribute {pct(top10['revenue'].sum() / prod['revenue'].sum() * 100)} of sales.")

    # ---- Section 10 ----
    section_header("Section 10: Repeat Purchase Analysis", "Member 9 - Sanjay Rathour")
    bs = buyer_summary(fcust)
    one, rep = bs.loc["1-Time Buyers"], bs.loc["Repeat Buyers"]

    st.markdown("**Table 10.1 - 1-Time Buyers vs. Repeat Buyers**")
    show_table(bs.reset_index().rename(columns={"buyer_type": "buyer_type"}).round(2))
    interp(f"Repeat buyers are {pct(rep['customer_share_pct'])} of customers but deliver {pct(rep['revenue_share_pct'])} of revenue. "
           f"They place {rep['orders_per_customer']:.2f} orders each versus {one['orders_per_customer']:.2f} for 1-time buyers, "
           f"and spend {money(rep['spend_per_customer'])} per customer compared with {money(one['spend_per_customer'])}.")

    # Figure 10.1
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axes[0]
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, bs["customer_share_pct"], w, label="Customer share %", color=PALETTE["secondary"])
    ax.bar(x + w / 2, bs["revenue_share_pct"], w, label="Revenue share %", color=PALETTE["accent"])
    for i, b_ in enumerate(BUYER_ORDER):
        gap = bs.loc[b_, "revenue_share_pct"] - bs.loc[b_, "customer_share_pct"]
        top = max(bs.loc[b_, "revenue_share_pct"], bs.loc[b_, "customer_share_pct"])
        ax.text(i, top + 3, f"{gap:+.1f} pp", ha="center", fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["positive"] if gap >= 0 else PALETTE["negative"], edgecolor="none"))
    ax.set_xticks(x)
    ax.set_xticklabels(BUYER_ORDER)
    ax.set_ylim(0, max(bs["revenue_share_pct"].max(), bs["customer_share_pct"].max()) * 1.2 + 5)
    ax.set_ylabel("Share (%)")
    ax.set_title("Figure 10.1: Customer vs. revenue share")
    ax.legend()
    for ax, col, ttl in [(axes[1], "customer_count", "Customers"), (axes[2], "total_revenue", "Revenue")]:
        vals = bs[col].values.astype(float)
        if vals.sum() > 0:
            ax.pie(vals, labels=BUYER_ORDER, autopct="%1.1f%%", startangle=90,
                   colors=[RP_COLORS[b_] for b_ in BUYER_ORDER], wedgeprops=dict(width=0.45, edgecolor="white"))
        ax.set_title(f"{ttl} split")
    show_fig(fig)
    gap_r = rep["revenue_share_pct"] - rep["customer_share_pct"]
    interp(f"Repeat buyers over-index on revenue by {gap_r:+.1f} percentage points (revenue share minus customer share), while 1-time buyers "
           f"under-index by {one['revenue_share_pct'] - one['customer_share_pct']:+.1f} pp. The donuts show the same imbalance: "
           f"{pct(rep['customer_share_pct'])} of customers versus {pct(rep['revenue_share_pct'])} of revenue.")

    # Figure 10.2
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    grouped_boxplot(axes[0], fcust, "total_spend", "Figure 10.2a: Spend distribution (log)", f"Total spend ({CURRENCY_SYMBOL}, log)")
    grouped_boxplot(axes[1], fcust, "avg_order_value", "Figure 10.2b: AOV distribution (log)", f"AOV ({CURRENCY_SYMBOL}, log)")
    ax = axes[2]
    xb = np.arange(2)
    ax.bar(xb - 0.18, bs["spend_per_customer"].fillna(0), 0.36, label="Avg spend / customer", color=PALETTE["primary"])
    ax.bar(xb + 0.18, bs["aov"].fillna(0), 0.36, label="Avg order value", color=PALETTE["positive"])
    for i, b_ in enumerate(BUYER_ORDER):
        ax.text(i - 0.18, bs.loc[b_, "spend_per_customer"] if pd.notna(bs.loc[b_, "spend_per_customer"]) else 0,
                compact_money(bs.loc[b_, "spend_per_customer"]), ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.18, bs.loc[b_, "aov"] if pd.notna(bs.loc[b_, "aov"]) else 0,
                compact_money(bs.loc[b_, "aov"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xb)
    ax.set_xticklabels(BUYER_ORDER)
    ax.set_title("Figure 10.2c: Average spend and AOV")
    ax.legend(fontsize=8)
    show_fig(fig)
    med = fcust.groupby("buyer_type")["total_spend"].median()
    interp(f"The median customer spend is {money(med.get('Repeat Buyers', np.nan))} for repeat buyers versus {money(med.get('1-Time Buyers', np.nan))} for 1-time buyers "
           f"(log-scale boxplots). AOV is {money(rep['aov'])} for repeat buyers and {money(one['aov'])} for 1-time buyers, "
           f"so the main driver of the spend gap is the number of orders ({rep['orders_per_customer']:.2f} vs. {one['orders_per_customer']:.2f}), not only basket value.")

    # Table 10.2 & Figure 10.3
    st.markdown("**Table 10.2 - Repeat-buyer frequency breakdown**")
    rb = fcust[fcust["buyer_type"] == "Repeat Buyers"].copy()
    freq_order = ["2 orders", "3 orders", "4+ orders"]
    if len(rb):
        rb["frequency_band"] = pd.cut(rb["num_orders"], [0, 2, 3, np.inf], labels=freq_order)
        freq = rb.groupby("frequency_band", observed=False).agg(
            customers=("customer_id", "count"), revenue=("total_spend", "sum"), avg_spend=("total_spend", "mean")
        ).reindex(freq_order).fillna(0)
        freq["headcount_share_pct"] = freq["customers"] / max(freq["customers"].sum(), 1) * 100
        freq["revenue_share_pct"] = freq["revenue"] / max(freq["revenue"].sum(), 1e-9) * 100
        show_table(freq.reset_index().round(2))
        fig, ax = plt.subplots(figsize=(9, 4.2))
        xf = np.arange(len(freq_order))
        ax.bar(xf - 0.18, freq["headcount_share_pct"], 0.36, label="Headcount share % (of repeat buyers)", color=PALETTE["primary"])
        ax.bar(xf + 0.18, freq["revenue_share_pct"], 0.36, label="Revenue share % (of repeat revenue)", color=PALETTE["accent"])
        for i in range(len(freq_order)):
            ax.text(i - 0.18, freq["headcount_share_pct"].iloc[i], f"{freq['headcount_share_pct'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=8)
            ax.text(i + 0.18, freq["revenue_share_pct"].iloc[i], f"{freq['revenue_share_pct'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xf)
        ax.set_xticklabels(freq_order)
        ax.set_ylabel("Share (%)")
        ax.set_title("Figure 10.3: Headcount vs. revenue share by order frequency")
        ax.legend(fontsize=8)
        show_fig(fig)
        heavy = freq.loc["4+ orders"]
        interp(f"Customers with 4+ orders are {pct(heavy['headcount_share_pct'])} of repeat buyers but contribute {pct(heavy['revenue_share_pct'])} of repeat revenue, "
               f"averaging {money(heavy['avg_spend'])} each. Revenue therefore concentrates in the most frequent repeaters, "
               f"which makes moving 2-order customers to a 3rd and 4th order the highest-leverage goal.")
    else:
        st.info("No repeat buyers in the current filter selection.")

    # Table 10.3 & Figure 10.4
    st.markdown("**Table 10.3 - Category preference (Preference Index = Repeat revenue % ÷ 1-Time revenue %)**")
    cat_pref = fdf.pivot_table(index="category", columns="buyer_type", values="total_amount", aggfunc="sum", fill_value=0)
    cat_pref = cat_pref.reindex(columns=BUYER_ORDER, fill_value=0)
    if cat_pref["Repeat Buyers"].sum() > 0 and cat_pref["1-Time Buyers"].sum() > 0:
        cp = pd.DataFrame({
            "repeat_revenue": cat_pref["Repeat Buyers"], "onetime_revenue": cat_pref["1-Time Buyers"],
            "repeat_revenue_pct": cat_pref["Repeat Buyers"] / cat_pref["Repeat Buyers"].sum() * 100,
            "onetime_revenue_pct": cat_pref["1-Time Buyers"] / cat_pref["1-Time Buyers"].sum() * 100,
        })
        cp["preference_index"] = cp["repeat_revenue_pct"] / cp["onetime_revenue_pct"].replace(0, np.nan)
        cp = cp.sort_values("preference_index", ascending=False)
        show_table(cp.reset_index().round(2))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        pi_vals = cp["preference_index"].fillna(0)
        ax.barh(cp.index[::-1], pi_vals[::-1], color=[PALETTE["positive"] if v >= 1 else PALETTE["negative"] for v in pi_vals[::-1]])
        ax.axvline(1, color="black", linestyle="--")
        for y, v in enumerate(pi_vals[::-1]):
            ax.text(v, y, f" {v:.2f}", va="center", fontsize=8)
        ax.set_xlabel("Preference Index (1.0 = equal preference)")
        ax.set_title("Figure 10.4: Category preference of repeat vs. 1-time buyers")
        show_fig(fig)
        best, worst = cp["preference_index"].idxmax(), cp["preference_index"].idxmin()
        interp(f"{best} is the category most favoured by repeat buyers (index {cp.loc[best, 'preference_index']:.2f}), and "
               f"{worst} is the most 1-time-oriented (index {cp.loc[worst, 'preference_index']:.2f}). "
               f"Categories with an index above 1 are natural candidates for retention and replenishment campaigns.")
    else:
        cp = None
        st.info("Category preference needs both 1-Time and Repeat buyers in the current selection.")

    # Table 10.4 - basket size
    st.markdown("**Table 10.4 - Basket size comparison & AOV bridge**")
    order_level = fdf.groupby(["transaction_id", "buyer_type"]).agg(
        items=("quantity", "sum"), lines=("product_name", "count"), value=("total_amount", "sum")).reset_index()
    rows = []
    for b_ in BUYER_ORDER:
        o = order_level[order_level["buyer_type"] == b_]
        if len(o):
            rows.append({"buyer_type": b_, "avg_items_per_order": o["items"].mean(), "median_items_per_order": o["items"].median(),
                         "item_lines_per_order": o["lines"].mean(), "value_per_item": o["value"].sum() / o["items"].sum(),
                         "aov": o["value"].mean()})
    basket = pd.DataFrame(rows)
    show_table(basket.round(2))
    if len(basket) == 2:
        b1 = basket[basket["buyer_type"] == "1-Time Buyers"].iloc[0]
        b2 = basket[basket["buyer_type"] == "Repeat Buyers"].iloc[0]
        d_aov = b2["aov"] - b1["aov"]
        size_eff = (b2["avg_items_per_order"] - b1["avg_items_per_order"]) * b1["value_per_item"]
        price_eff = (b2["value_per_item"] - b1["value_per_item"]) * b2["avg_items_per_order"]
        bridge = pd.DataFrame({
            "bridge_component": ["1-Time Buyers AOV", "Basket-size effect (more items per order)",
                                 "Price-mix effect (value per item)", "Repeat Buyers AOV"],
            f"amount_{CURRENCY_SYMBOL}": [b1["aov"], size_eff, price_eff, b2["aov"]],
        }).round(2)
        show_table(bridge)
        interp(f"Repeat buyers' AOV is {money(b2['aov'])} versus {money(b1['aov'])} (a gap of {money(d_aov)}). Of that gap, "
               f"{money(size_eff)} comes from larger baskets ({b2['avg_items_per_order']:.2f} vs. {b1['avg_items_per_order']:.2f} items per order) "
               f"and {money(price_eff)} from a different value per item ({money(b2['value_per_item'])} vs. {money(b1['value_per_item'])}).")
    else:
        d_aov = np.nan
        interp("Basket comparison needs both buyer groups in the current selection.")

    # Key findings and recommendations
    st.markdown("**Section 10 - Key Findings**")
    findings = [
        f"Repeat buyers are {pct(rep['customer_share_pct'])} of customers and generate {pct(rep['revenue_share_pct'])} of revenue.",
        f"A repeat buyer places {rep['orders_per_customer']:.2f} orders on average versus {one['orders_per_customer']:.2f} for a 1-time buyer.",
        f"Spend per repeat customer is {money(rep['spend_per_customer'])}, i.e. "
        f"{(rep['spend_per_customer'] / one['spend_per_customer']) if one['spend_per_customer'] else float('nan'):.1f}x the 1-time buyer figure.",
    ]
    if cp is not None:
        findings.append(f"{cp['preference_index'].idxmax()} shows the strongest repeat-buyer preference (index {cp['preference_index'].max():.2f}).")
    st.markdown("\n".join(f"- {f}" for f in findings))

    st.markdown("**Section 10 - Actionable Recommendations**")
    recs = pd.DataFrame([
        ("Post-purchase journey", "Automated thank-you, usage tips and review request within 7 days of the first order",
         f"{pct(one['customer_share_pct'])} of customers never return"),
        ("2nd-order incentive", "Time-limited coupon or free delivery on the next order",
         f"Converting a 1-time buyer adds about {money(rep['spend_per_customer'] - one['spend_per_customer'])} in lifetime spend"),
        ("Cross-sell", "Recommend complementary products from the strongest repeat-preferred categories",
         "See Preference Index (Table 10.3) and market-basket pairs (Section 12.4)"),
        ("Replenishment reminders", "Reminder emails/SMS timed to the typical repurchase gap for consumable categories",
         f"Repeat buyers order {rep['orders_per_customer']:.2f}x on average"),
        ("Loyalty tiers", "Reward customers with 4+ orders via points, early access and tier benefits",
         "Highest-frequency band holds a disproportionate revenue share (Table 10.2)"),
    ], columns=["recommendation", "action", "evidence"])
    show_table(recs)

    # ---- Section 11 ----
    section_header("Section 11: Monthly, Weekday & Seasonal Trends", "Member 10 - Dev Rastogi")
    monthly = fdf.groupby(pd.Grouper(key="transaction_date", freq="MS")).agg(
        revenue=("total_amount", "sum"), orders=("transaction_id", "nunique")).reset_index()
    monthly["mom_growth_pct"] = monthly["revenue"].pct_change() * 100
    tmp = fdf.assign(weekday=fdf["transaction_date"].dt.day_name(),
                     season=fdf["transaction_date"].dt.month.map(SEASON_MAP),
                     ym=fdf["transaction_date"].dt.to_period("M"))
    wk = tmp.groupby("weekday").agg(revenue=("total_amount", "sum"), orders=("transaction_id", "nunique")).reindex(WEEKDAY_ORDER).fillna(0)
    wk["revenue_share_pct"] = wk["revenue"] / wk["revenue"].sum() * 100
    sea = tmp.groupby("season").agg(revenue=("total_amount", "sum"), months=("ym", "nunique")).reindex(SEASON_ORDER).fillna(0)
    sea["avg_monthly_revenue"] = sea["revenue"] / sea["months"].replace(0, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.8, 1, 1]})
    ax = axes[0]
    ax.plot(monthly["transaction_date"], monthly["revenue"], marker="o", color=PALETTE["primary"], label="Revenue")
    ax.set_ylabel(f"Revenue ({CURRENCY_SYMBOL})")
    ax2 = ax.twinx()
    ax2.bar(monthly["transaction_date"], monthly["orders"], width=20, alpha=0.25, color=PALETTE["accent"], label="Orders")
    ax2.set_ylabel("Orders")
    ax2.grid(False)
    ax.set_title("Monthly revenue and orders")
    ax.tick_params(axis="x", rotation=45)
    axes[1].bar([d[:3] for d in WEEKDAY_ORDER], wk["revenue"], color=PALETTE["secondary"])
    axes[1].set_title("Revenue by weekday")
    axes[1].set_ylabel(f"Revenue ({CURRENCY_SYMBOL})")
    axes[2].bar([s.split(" ")[0] for s in SEASON_ORDER], sea["avg_monthly_revenue"].fillna(0),
                color=[PALETTE["secondary"], PALETTE["accent"], PALETTE["positive"], PALETTE["negative"]])
    axes[2].set_title("Avg monthly revenue by season")
    axes[2].set_ylabel(f"Revenue ({CURRENCY_SYMBOL})")
    show_fig(fig)
    show_table(monthly.assign(transaction_date=monthly["transaction_date"].dt.strftime("%Y-%m")).round(2)
               .rename(columns={"transaction_date": "month"}))
    pk = monthly.loc[monthly["revenue"].idxmax()]
    best_day = wk["revenue"].idxmax()
    best_season = sea["avg_monthly_revenue"].idxmax()
    interp(f"Revenue peaks in {pk['transaction_date']:%B %Y} at {money(pk['revenue'])} from {int(pk['orders']):,} orders. "
           f"{best_day} is the strongest weekday ({pct(wk.loc[best_day, 'revenue_share_pct'])} of revenue) and {best_season} is the highest-earning season "
           f"at {money(sea.loc[best_season, 'avg_monthly_revenue'])} per month on average. Campaign calendars and stock planning should be timed around these peaks.")


# ----------------------------------------------------------------------------
# 11. TAB 5 - PARETO, RFM, COHORT & MARKET BASKET
# ----------------------------------------------------------------------------
RFM_ACTIONS = {
    "Champions": "Reward, early access, referral programme",
    "Loyal Customers": "Upsell premium items, loyalty tiers",
    "New / Promising": "Onboarding journey, 2nd-order incentive",
    "Need Attention": "Limited-time offers, personalised recommendations",
    "At Risk": "Win-back campaign, reactivation discount",
    "Hibernating": "Low-cost reactivation or sunset from campaigns",
}


def render_tab5(fdf, fcust):
    st.header("Advanced Analytics - Pareto, RFM, Cohort & Market Basket")
    st.caption("Sections 12.1-12.4")

    # ---- 12.1 Pareto ----
    section_header("Section 12.1: Pareto 80/20 Analysis", "Member 11 - Vidhan Agarwal")
    par = pareto_table(fcust)
    cut80 = pareto_cut(par, 80)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(par["cust_pct"], par["rev_pct"], color=PALETTE["primary"], linewidth=2.5, label="Cumulative revenue share")
    ax.fill_between(par["cust_pct"], par["rev_pct"], color=PALETTE["secondary"], alpha=0.15)
    ax.plot([0, 100], [0, 100], linestyle="--", color=PALETTE["neutral"], label="Perfect equality")
    ax.axhline(80, color=PALETTE["negative"], linestyle=":")
    ax.axvline(cut80, color=PALETTE["negative"], linestyle=":")
    ax.scatter([cut80], [80], color=PALETTE["negative"], zorder=5)
    ax.annotate(f"{cut80:.1f}% of customers\n= 80% of revenue", xy=(cut80, 80), xytext=(cut80 + 10, 55),
                arrowprops=dict(arrowstyle="->", color=PALETTE["negative"]), fontweight="bold")
    ax.set_xlabel("Cumulative % of customers (ranked by spend)")
    ax.set_ylabel("Cumulative % of revenue")
    ax.set_title("Pareto curve - customer revenue concentration")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right")
    show_fig(fig)
    levels = [1, 5, 10, 20, 50]
    ptab = pd.DataFrame({"top_customers_pct": levels,
                         "revenue_share_pct": [pareto_revenue_at(par, l) for l in levels]}).round(2)
    show_table(ptab)
    interp(f"The top 20% of customers generate {pct(pareto_revenue_at(par, 20))} of revenue, and {cut80:.1f}% of customers are needed to reach 80% of revenue "
           f"({'close to' if abs(cut80 - 20) < 8 else 'different from'} the classic 80/20 rule). The top 10% alone contribute {pct(pareto_revenue_at(par, 10))}, "
           f"so retention of this small group protects a very large share of sales.")

    # ---- 12.2 RFM ----
    section_header("Section 12.2: RFM Segmentation", "Member 11 - Vidhan Agarwal")
    rfm = compute_rfm(fcust)
    seg = rfm.groupby("rfm_segment").agg(
        customers=("customer_id", "count"), avg_recency_days=("recency_days", "mean"), avg_frequency=("num_orders", "mean"),
        avg_monetary=("total_spend", "mean"), total_revenue=("total_spend", "sum")).reset_index()
    seg["customer_share_pct"] = seg["customers"] / seg["customers"].sum() * 100
    seg["revenue_share_pct"] = seg["total_revenue"] / seg["total_revenue"].sum() * 100
    seg["recommended_action"] = seg["rfm_segment"].map(RFM_ACTIONS)
    seg = seg.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    show_table(seg.round(2))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    pal = sns.color_palette("Set2", len(seg))
    axes[0].barh(seg["rfm_segment"][::-1], seg["customers"][::-1], color=pal[::-1])
    axes[0].set_title("Customers per RFM segment")
    axes[0].set_xlabel("Customers")
    axes[1].barh(seg["rfm_segment"][::-1], seg["revenue_share_pct"][::-1], color=pal[::-1])
    axes[1].set_title("Revenue share per RFM segment (%)")
    axes[1].set_xlabel("Revenue share (%)")
    show_fig(fig)
    top = seg.iloc[0]
    interp(f"{top['rfm_segment']} is the most valuable group, with {int(top['customers']):,} customers ({pct(top['customer_share_pct'])}) producing "
           f"{pct(top['revenue_share_pct'])} of revenue. R, F and M scores are computed on the current filtered view (quintiles for R and M, order-count bands for F), "
           f"and each segment is paired with a recommended action in the table.")

    # ---- 12.3 Cohort retention ----
    section_header("Section 12.3: Monthly Cohort Retention", "Member 12 - Anshu")
    ret, sizes = cohort_matrix(fdf, fcust, max_offset=12)
    if len(ret) >= 1:
        fig_h = max(4, 0.38 * len(ret) + 1.5)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        sns.heatmap(ret, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100, linewidths=0.4,
                    annot_kws={"size": 7}, cbar_kws={"label": "Retention (%)"}, ax=ax)
        ax.set_xlabel("Months since first purchase")
        ax.set_ylabel("Acquisition cohort (first_purchase_date month)")
        ax.set_title("Cohort retention heatmap (% of cohort active)")
        show_fig(fig)
        m1 = ret[1].mean() if 1 in ret.columns else np.nan
        m3 = ret[3].mean() if 3 in ret.columns else np.nan
        m6 = ret[6].mean() if 6 in ret.columns else np.nan
        biggest = pd.Series(sizes, index=ret.index).idxmax()
        interp(f"On average {pct(m1)} of a cohort buys again in month 1, {pct(m3)} in month 3 and {pct(m6)} in month 6. "
               f"The largest cohort is {biggest} with {int(pd.Series(sizes, index=ret.index).max()):,} customers. "
               f"Retention drops sharply after the first purchase month, so early engagement has the largest impact.")
    else:
        st.info("Not enough data to build cohorts for the current selection.")

    # ---- 12.4 Market basket ----
    section_header("Section 12.4: Market Basket Analysis", "Member 12 - Anshu")
    c1, c2 = st.columns(2)
    min_sup_pct = c1.slider("Minimum support (%)", 0.1, 5.0, 0.5, 0.1)
    top_n = c2.slider("Number of pairs to show", 5, 30, 15)
    items = fdf[["transaction_id", "customer_id", "product_name"]].copy()
    items["basket_id"] = items["transaction_id"]
    multi_orders = (items.groupby("basket_id")["product_name"].nunique() >= 2).sum()
    level = "transaction (order) baskets"
    if multi_orders < 20:
        items["basket_id"] = items["customer_id"]
        level = "customer-level baskets (too few multi-item orders)"
    rules, n_b, n_multi = market_basket_cached(items[["basket_id", "product_name"]], min_sup_pct / 100)
    st.caption(f"Basket definition: {level}. Baskets analysed: {n_b:,}; baskets with 2+ distinct products: {n_multi:,}. "
               f"Computed with itertools.combinations + collections.Counter (no external library needed).")
    if rules.empty:
        st.info("No product pairs meet the minimum support. Lower the support slider.")
    else:
        shown = rules.head(top_n)
        disp = shown.copy()
        disp["support_pct"] = disp["support"] * 100
        disp["confidence_a_to_b_pct"] = disp["confidence_a_to_b"] * 100
        disp["confidence_b_to_a_pct"] = disp["confidence_b_to_a"] * 100
        show_table(disp[["product_a", "product_b", "pair_count", "support_pct", "confidence_a_to_b_pct",
                         "confidence_b_to_a_pct", "lift"]].round(3))
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.38 * len(shown) + 1.5)))
        labels = (shown["product_a"] + "  +  " + shown["product_b"])[::-1]
        ax.barh(labels, shown["lift"][::-1], color=sns.color_palette("viridis", len(shown))[::-1])
        ax.axvline(1, color="black", linestyle="--", linewidth=1)
        for y, (l_, s_) in enumerate(zip(shown["lift"][::-1], shown["support"][::-1])):
            ax.text(l_, y, f" {l_:.2f} (sup {s_ * 100:.1f}%)", va="center", fontsize=8)
        ax.set_xlim(0, shown["lift"].max() * 1.25)
        ax.set_xlabel("Lift (1.0 = independent)")
        ax.set_title("Top product pairs by lift")
        show_fig(fig)
        best = rules.iloc[0]
        interp(f"The strongest association is {best['product_a']} + {best['product_b']}: they appear together in {best['support'] * 100:.2f}% of baskets "
               f"with lift {best['lift']:.2f}, i.e. {best['lift']:.1f}x more often than chance. Confidence shows that "
               f"{best['confidence_a_to_b'] * 100:.1f}% of {best['product_a']} baskets also contain {best['product_b']}, making these pairs good bundle and cross-sell candidates.")


# ----------------------------------------------------------------------------
# 12. TAB 6 - K-MEANS CLUSTERING
# ----------------------------------------------------------------------------
def name_clusters(prof):
    """Assign business-friendly names to clusters based on spend rank and recency."""
    order = prof.sort_values("mean_spend", ascending=False).index.tolist()
    med_rec = prof["mean_recency_days"].median()
    names = {}
    for rank, cl in enumerate(order):
        rec = prof.loc[cl, "mean_recency_days"]
        if rank == 0:
            n = "High-Value Loyalists"
        elif rank == len(order) - 1:
            n = "Dormant Low-Spenders" if rec >= med_rec else "Occasional Low-Spenders"
        else:
            n = "At-Risk Mid-Spenders" if rec >= med_rec else "Growing Regulars"
        names[cl] = f"C{cl} - {n}"
    return names


def render_tab6(fcust):
    st.header("Machine Learning - K-Means Customer Clustering")
    st.caption("Section 12.5 - Member 13: Rudra Pratap Singh")
    section_header("Section 12.5: Feature Scaling, K-Means, Elbow, Silhouette & PCA", "Member 13 - Rudra Pratap Singh")
    st.markdown(f"Features: `{', '.join(CLUSTER_FEATURES)}` -> `StandardScaler` -> `KMeans(n_clusters=k, n_init=10, random_state={RANDOM_STATE})`.")

    n = len(fcust)
    k_max = int(min(10, n - 1))
    if k_max < 3:
        st.warning("At least 4 customers are needed for K-Means. Please widen the sidebar filters.")
        return
    k = st.slider("Select number of clusters (k)", 2, k_max, min(4, k_max))

    diag = kmeans_diagnostics(fcust[CLUSTER_FEATURES], k_max)
    best_k = int(diag.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).iloc[0]["k"])
    sil_k = float(diag.loc[diag["k"] == k, "silhouette"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(diag["k"], diag["wcss"], marker="o", color=PALETTE["primary"])
    axes[0].axvline(k, color=PALETTE["accent"], linestyle="--", label=f"Selected k = {k}")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("WCSS (inertia)")
    axes[0].set_title("Elbow curve")
    axes[0].legend()
    sil_df = diag.dropna(subset=["silhouette"])
    axes[1].plot(sil_df["k"], sil_df["silhouette"], marker="s", color=PALETTE["positive"])
    axes[1].axvline(k, color=PALETTE["accent"], linestyle="--", label=f"Selected k = {k}")
    axes[1].scatter([best_k], [sil_df.loc[sil_df["k"] == best_k, "silhouette"].iloc[0]], color=PALETTE["negative"], s=90, zorder=5, label=f"Best k = {best_k}")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette evaluation")
    axes[1].legend()
    show_fig(fig)
    quality = "strong" if sil_k > 0.5 else "reasonable" if sil_k > 0.25 else "weak"
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected k", k)
    c2.metric("Silhouette score (k)", f"{sil_k:.3f}")
    c3.metric("Best k by silhouette", best_k)
    interp(f"The silhouette score at k = {k} is {sil_k:.3f}, indicating {quality} cluster separation, while the highest score ({sil_df['silhouette'].max():.3f}) occurs at k = {best_k}. "
           f"WCSS falls from {diag['wcss'].iloc[0]:,.0f} at k = 1 to {diag.loc[diag['k'] == k, 'wcss'].iloc[0]:,.0f} at k = {k}; the elbow shows where extra clusters bring diminishing returns.")

    labels, X, model = fit_kmeans(fcust, k)
    work = fcust.copy()
    work["cluster_k"] = labels
    prof = work.groupby("cluster_k").agg(
        customers=("customer_id", "count"), mean_spend=("total_spend", "mean"), mean_orders=("num_orders", "mean"),
        mean_aov=("avg_order_value", "mean"), mean_recency_days=("recency_days", "mean"), total_revenue=("total_spend", "sum"))
    prof["customer_share_pct"] = prof["customers"] / prof["customers"].sum() * 100
    prof["revenue_share_pct"] = prof["total_revenue"] / prof["total_revenue"].sum() * 100
    names = name_clusters(prof)
    prof.insert(0, "cluster_name", [names[c] for c in prof.index])
    work["cluster_name"] = work["cluster_k"].map(names)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    centers = pca.transform(model.cluster_centers_)
    plot_idx = np.arange(n)
    if n > 5000:
        plot_idx = np.random.default_rng(RANDOM_STATE).choice(n, 5000, replace=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("tab10", k)
    for cl in range(k):
        mask = labels[plot_idx] == cl
        ax.scatter(coords[plot_idx][mask, 0], coords[plot_idx][mask, 1], s=18, alpha=0.65, color=colors[cl], label=names[cl])
        ax.scatter(centers[cl, 0], centers[cl, 1], marker="X", s=220, color=colors[cl], edgecolor="black", linewidth=1.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(f"PCA 2D cluster scatter (k = {k}, X = centroid)")
    ax.legend(fontsize=8, loc="best")
    show_fig(fig)
    interp(f"The first two principal components explain {pca.explained_variance_ratio_.sum() * 100:.1f}% of the variance, so the 2D map is a faithful view of the clusters. "
           f"The largest cluster ({names[prof['customers'].idxmax()]}) holds {pct(prof['customer_share_pct'].max())} of customers.")

    st.markdown("**Cluster profile**")
    show_table(prof.reset_index().rename(columns={"cluster_k": "cluster"}).round(2))
    top_cl = prof["mean_spend"].idxmax()
    interp(f"{names[top_cl]} contains {pct(prof.loc[top_cl, 'customer_share_pct'])} of customers but {pct(prof.loc[top_cl, 'revenue_share_pct'])} of revenue, "
           f"with mean spend of {money(prof.loc[top_cl, 'mean_spend'])} and {prof.loc[top_cl, 'mean_orders']:.1f} orders. "
           f"Cluster names are assigned automatically from spend rank and recency to make the profile easier to act on.")


# ----------------------------------------------------------------------------
# 13. TAB 7 - FINDINGS, RECOMMENDATIONS, EXPORTS & GLOSSARY
# ----------------------------------------------------------------------------
GLOSSARY = [
    ("Pareto Principle (80/20)", "Observation that roughly 80% of outcomes come from 20% of causes, e.g. revenue from customers."),
    ("Cumulative Revenue Share", "Running total of revenue as customers are ranked from highest to lowest spend."),
    ("RFM", "Segmentation technique scoring customers on Recency, Frequency and Monetary value."),
    ("Recency", "Days since a customer's last purchase relative to the snapshot date."),
    ("Frequency", "Number of distinct orders placed by a customer."),
    ("Monetary Value", "Total amount a customer has spent."),
    ("Average Order Value (AOV)", "Total revenue divided by number of orders."),
    ("Repeat Purchase Rate (RPR)", "Share of customers who placed two or more orders."),
    ("Customer Lifetime Value (CLV)", "Expected total value a customer generates over the relationship."),
    ("Cohort", "Group of customers sharing the same acquisition month."),
    ("Retention Rate", "Percentage of a cohort that purchases again in a later period."),
    ("Churn", "Loss of customers who stop purchasing."),
    ("Market Basket Analysis", "Technique that finds products frequently bought together."),
    ("Support", "Proportion of baskets that contain an item or item pair."),
    ("Confidence", "Probability that item B is bought given that item A is in the basket."),
    ("Lift", "Support of the pair divided by the product of individual supports; > 1 means positive association."),
    ("K-Means", "Unsupervised algorithm partitioning observations into k clusters by minimising within-cluster variance."),
    ("WCSS", "Within-cluster sum of squares (inertia), used in the elbow method."),
    ("Elbow Method", "Choosing k where adding clusters stops reducing WCSS substantially."),
    ("Silhouette Score", "Measure (-1 to 1) of how well each point fits its cluster versus others."),
    ("StandardScaler", "Scaling of features to zero mean and unit variance before distance-based modelling."),
    ("PCA", "Principal Component Analysis - dimensionality reduction used here to plot clusters in 2D."),
    ("Concentration Index", "Revenue share divided by customer share of a segment; > 1 means over-contribution."),
    ("Preference Index", "Repeat-buyer revenue % divided by 1-time-buyer revenue % for a category."),
    ("Quantile Binning (qcut)", "Splitting a variable into equal-sized groups, used for spend tertiles and RFM scores."),
    ("Skewness", "Statistical measure of distribution asymmetry; positive values indicate a long right tail."),
]

REFERENCES = [
    "Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. In *Proceedings of the 20th International Conference on Very Large Data Bases* (pp. 487-499). Morgan Kaufmann.",
    "Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). \"Counting your customers\" the easy way: An alternative to the Pareto/NBD model. *Marketing Science, 24*(2), 275-284. https://doi.org/10.1287/mksc.1040.0098",
    "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357-362. https://doi.org/10.1038/s41586-020-2649-2",
    "Hughes, A. M. (1994). *Strategic database marketing*. Probus Publishing.",
    "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90-95. https://doi.org/10.1109/MCSE.2007.55",
    "Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A, 374*(2065), Article 20150202. https://doi.org/10.1098/rsta.2015.0202",
    "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281-297). University of California Press.",
    "McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (pp. 56-61). https://doi.org/10.25080/Majora-92bf1922-00a",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.",
    "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53-65. https://doi.org/10.1016/0377-0427(87)90125-7",
    "Thorndike, R. L. (1953). Who belongs in the family? *Psychometrika, 18*(4), 267-276. https://doi.org/10.1007/BF02289263",
    "Waskom, M. L. (2021). seaborn: Statistical data visualization. *Journal of Open Source Software, 6*(60), Article 3021. https://doi.org/10.21105/joss.03021",
]


def csv_bytes(df):
    """Encode a DataFrame as UTF-8 CSV bytes for download buttons."""
    return df.to_csv(index=False).encode("utf-8")


def render_tab7(fdf, fcust, B):
    st.header("Findings, Recommendations, Exports & Glossary")
    st.caption("Sections 13-17 & Appendix B")

    # ----- compute data-driven facts -----
    revenue = fdf["total_amount"].sum()
    par = pareto_table(fcust)
    cut80 = pareto_cut(par, 80)
    seg = seg_summary(fcust)
    bs = buyer_summary(fcust)
    rfm = compute_rfm(fcust)
    rfm_rev = rfm.groupby("rfm_segment")["total_spend"].sum().sort_values(ascending=False)
    rfm_top = rfm_rev.index[0]
    rfm_cnt = int((rfm["rfm_segment"] == rfm_top).sum())
    cat_rev = fdf.groupby("category")["total_amount"].sum().sort_values(ascending=False)
    monthly = fdf.groupby(pd.Grouper(key="transaction_date", freq="MS"))["total_amount"].sum()
    peak_month = monthly.idxmax()
    ret, _ = cohort_matrix(fdf, fcust, 12)
    m1 = ret[1].mean() if (len(ret) and 1 in ret.columns) else np.nan
    items = fdf[["transaction_id", "product_name"]].rename(columns={"transaction_id": "basket_id"})
    rules, _, _ = market_basket_cached(items, 0.005)
    clust = fcust.groupby("cluster").agg(customers=("customer_id", "count"), rev=("total_spend", "sum"), spend=("total_spend", "mean"))
    top_cl = clust["spend"].idxmax()
    top_cl_rev = clust.loc[top_cl, "rev"] / clust["rev"].sum() * 100
    top_cl_cust = clust.loc[top_cl, "customers"] / clust["customers"].sum() * 100
    log = B["cleaning_log"]
    retained_pct = (log["rows_after"].iloc[-1] / log["rows_before"].iloc[0] * 100) if len(log) and log["rows_before"].iloc[0] else np.nan

    pair_txt = (f"{rules.iloc[0]['product_a']} + {rules.iloc[0]['product_b']} (lift {rules.iloc[0]['lift']:.2f})"
                if not rules.empty else "no pair above 0.5% support")

    # ----- Sections 13 & 14 -----
    section_header("Sections 13-14: Key Findings & Business Recommendations", "Member 13 - Rudra Pratap Singh")
    findings = pd.DataFrame([
        ("12.1", "Revenue concentration", f"{cut80:.1f}% of customers generate 80% of revenue; top 10% contribute {pct(pareto_revenue_at(par, 10))}."),
        ("7-8", "Spend segments", f"High-spend customers are {pct(seg.loc['High', 'customer_share_pct'])} of the base and {pct(seg.loc['High', 'revenue_share_pct'])} of revenue "
                                  f"(Concentration Index {seg.loc['High', 'concentration_index']:.2f}x)."),
        ("10", "Repeat purchase", f"Repeat buyers are {pct(bs.loc['Repeat Buyers', 'customer_share_pct'])} of customers and {pct(bs.loc['Repeat Buyers', 'revenue_share_pct'])} of revenue."),
        ("9", "Category mix", f"{cat_rev.index[0]} leads with {pct(cat_rev.iloc[0] / cat_rev.sum() * 100)} of revenue."),
        ("11", "Seasonality", f"Peak month is {peak_month:%B %Y} ({money(monthly.max())})."),
        ("12.2", "RFM", f"{rfm_top} ({rfm_cnt:,} customers) is the top-earning RFM group with {pct(rfm_rev.iloc[0] / rfm_rev.sum() * 100)} of revenue."),
        ("12.3", "Cohort retention", f"Average month-1 retention is {pct(m1)}."),
        ("12.4", "Market basket", f"Strongest pair: {pair_txt}."),
        ("12.5", "K-Means clusters", f"Highest-spend cluster (C{top_cl}) is {pct(top_cl_cust)} of customers and {pct(top_cl_rev)} of revenue."),
        ("4-5", "Data quality", f"{pct(retained_pct)} of raw rows survived the 5-step cleaning pipeline."),
    ], columns=["section", "area", "finding"])
    show_table(findings)
    interp(f"Across the ten analyses, revenue of {money(revenue)} is strongly concentrated: {cut80:.1f}% of customers produce 80% of sales and the {rfm_top} RFM group leads revenue. "
           f"Retention after the first purchase is the main leak, with only {pct(m1)} of a cohort returning in month 1.")

    recs = pd.DataFrame([
        ("High", "Protect and reward top customers", f"Launch a loyalty tier for the {rfm_top} and High-spend groups.", f"Top 10% = {pct(pareto_revenue_at(par, 10))} of revenue", "Retention of top decile"),
        ("High", "Drive the 2nd order", "Post-purchase journey plus a time-limited 2nd-order incentive.", f"{pct(bs.loc['1-Time Buyers', 'customer_share_pct'])} of customers are 1-time buyers", "Repeat Purchase Rate"),
        ("High", "Improve early retention", "Onboarding emails and reminders in days 7-30 after first purchase.", f"Month-1 retention {pct(m1)}", "Month-1 / month-3 retention"),
        ("Medium", "Bundle and cross-sell", f"Create bundles from top pairs, e.g. {pair_txt}.", "Lift above 1 shows genuine association", "Items per order / AOV"),
        ("Medium", "Season-aware campaigns", f"Concentrate promotions and stock before {peak_month:%B}.", f"Peak month revenue {money(monthly.max())}", "Revenue vs. forecast"),
        ("Medium", "Win-back at-risk customers", "Reactivation offers for At Risk / Hibernating RFM groups.", "RFM segment table (Section 12.2)", "Reactivation rate"),
        ("Low", "Category focus", f"Invest in {cat_rev.index[0]} ranging and promote repeat-preferred categories.", f"{pct(cat_rev.iloc[0] / cat_rev.sum() * 100)} of revenue", "Category growth"),
    ], columns=["priority", "recommendation", "action", "evidence", "kpi"])
    show_table(recs)
    interp(f"Seven recommendations are ranked by expected impact; the three High-priority actions target the biggest gaps found in the data "
           f"({pct(bs.loc['1-Time Buyers', 'customer_share_pct'])} 1-time buyers, {pct(m1)} month-1 retention, and revenue concentration in the top {cut80:.0f}% of customers).")

    # ----- Sections 15 & 16.1 -----
    section_header("Sections 15 & 16.1: Conclusions, Limitations & Future Scope", "Member 14 - Ayan Khan")
    st.markdown(
        f"""
**Section 15 - Conclusions**
- Revenue is highly concentrated: **{cut80:.1f}%** of customers drive 80% of sales, confirming the Pareto pattern.
- Repeat buyers are the profit engine: **{pct(bs.loc['Repeat Buyers', 'revenue_share_pct'])}** of revenue from **{pct(bs.loc['Repeat Buyers', 'customer_share_pct'])}** of customers.
- RFM, cohort and K-Means analyses agree on who the valuable and at-risk customers are, so targeted campaigns are justified.
- Market-basket results show product pairs with lift above 1 that can be turned into bundles.
        """
    )
    st.warning(
        "**Limitations & disclaimer.** "
        + ("The dashboard is currently running on **synthetic data** generated with RANDOM_STATE = 42; all numbers illustrate the methodology and must not be read as real business results. "
           if B["is_synthetic"] else "Findings depend on the supplied dataset; the data may be a sample or contain synthetic elements. ")
        + "Other limitations: no cost/margin data (revenue is not profit), no customer demographics, a limited observation window "
          "(recent cohorts have little retention history), and K-Means assumes roughly spherical clusters."
    )
    roadmap = pd.DataFrame([
        ("Predictive CLV", "BG/NBD + Gamma-Gamma models", "Forecast future purchases and spend per customer", "Prioritise acquisition and retention budgets"),
        ("Churn prediction", "Gradient boosting / logistic regression on RFM features", "Predict which customers will lapse", "Trigger win-back before customers leave"),
        ("Real-time recommendation engine", "Association rules + collaborative filtering", "Show relevant products at checkout and in email", "Higher items per order and AOV"),
        ("A/B testing framework", "Randomised experiments with significance testing", "Measure real uplift from incentives and bundles", "Evidence-based campaign decisions"),
    ], columns=["initiative", "method", "purpose", "expected_benefit"])
    st.markdown("**Section 16.1 - Future scope roadmap**")
    show_table(roadmap)

    # ----- Section 16.2 Export -----
    section_header("Section 16.2: Data Export & Manifest", "Member 14 - Ayan Khan")
    scope = st.radio("Export scope", ["Full dataset", "Current filtered view"], horizontal=True)
    if scope == "Full dataset":
        ex_clean, ex_cust = B["df_clean"], B["customer_df"]
    else:
        ex_clean, ex_cust = fdf[TX_COLUMNS], fcust
    ex_log = B["cleaning_log"]
    files = [
        ("cleaned_transactions.csv", ex_clean, "Validated transaction lines (df_clean)"),
        ("customer_analytics_master.csv", ex_cust, "Customer master with spend segment, buyer type, RFM segment and cluster"),
        ("data_cleaning_log.csv", ex_log, "5-step cleaning audit trail"),
    ]
    cols = st.columns(3)
    for col, (fname, df, _) in zip(cols, files):
        col.download_button(f"⬇️ {fname}", data=csv_bytes(df), file_name=fname, mime="text/csv", key=f"dl_{fname}")
    manifest = pd.DataFrame([{"file_name": f, "description": d, "rows": len(df), "columns": df.shape[1],
                              "size_kb": round(len(csv_bytes(df)) / 1024, 1), "scope": scope} for f, df, d in files])
    show_table(manifest)
    interp(f"The export set contains {len(ex_clean):,} transaction lines, {len(ex_cust):,} customer records and {len(ex_log)} cleaning-log steps "
           f"({manifest['size_kb'].sum():,.1f} KB in total, scope: {scope.lower()}). Files use the same column names as the schema contract so they can be reloaded from outputs/tables/ without changes.")

    # ----- Appendix B & Section 17 -----
    section_header("Appendix B: Glossary of Technical Terms", "Member 14 - Ayan Khan")
    gl = pd.DataFrame(GLOSSARY, columns=["term", "definition"])
    gl.insert(0, "no", range(1, len(gl) + 1))
    show_table(gl)
    interp(f"The glossary defines {len(gl)} terms used across the study, from the Pareto Principle to Skewness, so readers can follow every metric in the dashboard.")

    section_header("Section 17: References (APA 7th edition)", "Member 1 - Hasnain Raza Khan")
    for ref in REFERENCES:
        st.markdown(f"- {ref}")


# ----------------------------------------------------------------------------
# 14. MAIN APPLICATION
# ----------------------------------------------------------------------------
def parse_date_selection(sel, min_d, max_d):
    """Normalise Streamlit's date_input output to a (start, end) pair."""
    if isinstance(sel, (tuple, list)):
        if len(sel) == 2:
            return sel[0], sel[1]
        if len(sel) == 1:
            return sel[0], sel[0]
        return min_d, max_d
    return sel, sel


def main():
    st.title("🛒 Customer Spend, Product & Sales Analytics")
    st.caption(PROJECT_TITLE)

    # ---------- Sidebar: data sources ----------
    st.sidebar.header("📂 Data Sources")
    st.sidebar.caption("Optional: upload files. Otherwise local files in outputs/tables/ are used, "
                       "and synthetic data is generated automatically if none exist.")
    up_clean = st.sidebar.file_uploader("cleaned_transactions.csv", type="csv", key="up_clean")
    up_cust = st.sidebar.file_uploader("customer_analytics_master.csv", type="csv", key="up_cust")
    up_log = st.sidebar.file_uploader("data_cleaning_log.csv", type="csv", key="up_log")

    B = assemble_data(up_clean, up_cust, up_log)
    st.sidebar.success(f"Data source: {B['source']}")
    for note in B["notes"]:
        st.sidebar.warning(note)

    df_clean, customer_df = B["df_clean"], B["customer_df"]

    # ---------- Sidebar: global filters ----------
    st.sidebar.header("🔎 Global Filters")
    min_d = df_clean["transaction_date"].min().date()
    max_d = df_clean["transaction_date"].max().date()
    date_sel = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    start, end = parse_date_selection(date_sel, min_d, max_d)
    all_cats = sorted(df_clean["category"].unique())
    cat_sel = st.sidebar.multiselect("Category", all_cats, default=all_cats)
    seg_sel = st.sidebar.multiselect("Spend segment", SEGMENT_ORDER, default=SEGMENT_ORDER)
    buyer_sel = st.sidebar.multiselect("Buyer type", BUYER_ORDER, default=BUYER_ORDER)
    st.sidebar.caption("Spend segment, buyer type, RFM and cluster labels are assigned once on the full dataset; "
                       "all metrics and charts are recomputed on the filtered transactions.")

    # ---------- Apply filters ----------
    labels = customer_df[["customer_id", "spend_segment", "buyer_type"]]
    keep_ids = labels.loc[labels["spend_segment"].isin(seg_sel) & labels["buyer_type"].isin(buyer_sel), "customer_id"]
    mask = (
        (df_clean["transaction_date"] >= pd.Timestamp(start))
        & (df_clean["transaction_date"] < pd.Timestamp(end) + pd.Timedelta(days=1))
        & df_clean["category"].isin(cat_sel)
        & df_clean["customer_id"].isin(keep_ids)
    )
    fdf = df_clean.loc[mask].merge(labels, on="customer_id", how="left")
    if fdf.empty:
        st.warning("No transactions match the current filters. Please widen the sidebar selections.")
        st.stop()

    filters_default = (start <= min_d and end >= max_d and set(cat_sel) == set(all_cats))
    if filters_default:
        fcust = customer_df[customer_df["customer_id"].isin(fdf["customer_id"].unique())].reset_index(drop=True)
    else:
        label_cols = customer_df[["customer_id", "spend_segment", "buyer_type", "rfm_segment", "cluster"]]
        fcust = build_customer_df(fdf).merge(label_cols, on="customer_id", how="left").reset_index(drop=True)

    st.sidebar.metric("Transactions in view", f"{len(fdf):,}")
    st.sidebar.metric("Customers in view", f"{len(fcust):,}")

    # ---------- Tabs ----------
    tabs = st.tabs([
        "1 · Overview & Data Dictionary",
        "2 · Data Quality & Cleaning",
        "3 · Customer Summary & Segmentation",
        "4 · Product, Repeat Purchase & Seasonality",
        "5 · Pareto, RFM, Cohort & Basket",
        "6 · K-Means Clustering",
        "7 · Findings, Exports & Glossary",
    ])
    with tabs[0]:
        render_tab1(fdf, fcust, B)
    with tabs[1]:
        render_tab2(B)
    with tabs[2]:
        render_tab3(fdf, fcust)
    with tabs[3]:
        render_tab4(fdf, fcust)
    with tabs[4]:
        render_tab5(fdf, fcust)
    with tabs[5]:
        render_tab6(fcust)
    with tabs[6]:
        render_tab7(fdf, fcust, B)

    st.markdown("---")
    st.caption("Customer Spend, Product & Sales Analytics · 14-member college project · RANDOM_STATE = 42")


if __name__ == "__main__":
    main()
