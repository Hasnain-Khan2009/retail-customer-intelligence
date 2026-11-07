#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================================================
 CUSTOMER SPEND, PRODUCT & SALES ANALYTICS  -  interactive Streamlit dashboard  (single file: app.py)
=====================================================================================================
 An End-to-End Exploratory and Advanced Analytics Study of Retail Transactions Using
 Pareto (80/20), RFM, Cohort Retention, Market Basket and K-Means Techniques.

 14-member college project  |  Sections 1-17 + Appendices A-B merged into one runnable dashboard.

 RUN
     pip install streamlit pandas numpy matplotlib seaborn scikit-learn
     streamlit run app.py

 DATA (nothing has to be prepared by hand)
     1. The app first looks for the project's exported tables on disk
            outputs/tables/cleaned_transactions.csv          -> df_clean
            outputs/tables/customer_analytics_master.csv     -> customer_df
            outputs/tables/data_cleaning_log.csv             -> cleaning_log
     2. The same three files can be uploaded in the sidebar (an upload overrides the disk copy).
     3. If nothing is found, the project's data generator (Appendix A, RANDOM_STATE = 42) creates a
        realistic synthetic df_raw, which is audited, cleaned and aggregated exactly as in Sections 4-6.

 HOW THE SIDEBAR FILTERS WORK
     Date range and Category define the period / assortment that is analysed.  Inside that scope the
     customer table and the Low / Medium / High and 1-Time / Repeat labels are (re)computed with the
     notebook logic; the Spend-segment and Buyer-type filters then keep only the chosen labelled
     customers.  With the default filters every number equals the notebooks' output.  Tab 2 (raw-to-clean
     pipeline) follows the Date and Category filters too (tick-box, on by default); Spend-segment and
     Buyer-type are customer labels and cannot be applied to raw rows.

 TEAM CREDITS (equal college credit, 100% / 14 = 7.14% each)
     M1 Hasnain Raza Khan (Sec 1, 2, 17)      M2 Aditi Sharma (Sec 3, App A)    M3 Sonam (Sec 4)
     M4 Samiya Khan (Sec 5)                   M5 Aditya Singh (Sec 6)           M6 Saif Khan (Sec 7)
     M7 Divyanshi (Sec 8)                     M8 Abhi Rastogi (Sec 9)           M9 Sanjay Rathour (Sec 10)
     M10 Dev Rastogi (Sec 11)                 M11 Vidhan Agarwal (Sec 12.1, 12.2)
     M12 Anshu (Sec 12.3, 12.4)               M13 Rudra Pratap Singh (Sec 12.5, 13, 14)
     M14 Ayan Khan (Sec 15, 16, App B)
=====================================================================================================
"""

# ---------------------------------------------------------------------------------------------------
# 0.  DEPENDENCY CHECK  (a missing library produces a friendly message instead of a traceback)
# ---------------------------------------------------------------------------------------------------
import importlib.util as _ilu
import sys

try:
    import streamlit as st
except ImportError:                                                     # pragma: no cover
    sys.exit("Streamlit is not installed.  Run:  pip install streamlit pandas numpy matplotlib "
             "seaborn scikit-learn   and start the dashboard with:  streamlit run app.py")

_REQUIRED_LIBS = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
                  "seaborn": "seaborn", "sklearn": "scikit-learn"}
_MISSING_LIBS = [pip for mod, pip in _REQUIRED_LIBS.items() if _ilu.find_spec(mod) is None]
if _MISSING_LIBS:                                                       # pragma: no cover
    st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
    st.error("Missing Python libraries: " + ", ".join(_MISSING_LIBS))
    st.code("pip install " + " ".join(_MISSING_LIBS), language="bash")
    st.stop()

# ---------------------------------------------------------------------------------------------------
# 1.  IMPORTS
# ---------------------------------------------------------------------------------------------------
import hashlib
import inspect
import io
import re
import traceback
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")                                                   # headless backend (no GUI window)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore", category=FutureWarning)                 # library noise (pandas / seaborn)
warnings.simplefilter("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------------------------------
# 2.  GLOBAL PROJECT CONSTANTS  (Section 2 of the master notebook)
# ---------------------------------------------------------------------------------------------------
PROJECT_TITLE = "Customer Spend, Product & Sales Analytics"
PROJECT_SUBTITLE = ("An End-to-End Exploratory and Advanced Analytics Study of Retail Transactions Using "
                    "Pareto (80/20), RFM, Cohort Retention, Market Basket and K-Means Techniques")
RANDOM_STATE = 42                                    # ONE seed for generator, K-Means, sampling
CURRENCY_SYMBOL = "\u20b9"                           # Rupee sign
KMEANS_PARAMS = dict(n_init=10, random_state=RANDOM_STATE)   # usage: KMeans(n_clusters=k, **KMEANS_PARAMS)

# Main project palette (PALETTE) and the colour dictionaries used by the sections
PALETTE = {"primary": "#1F4E79", "secondary": "#2E86AB", "accent": "#F18F01",
           "positive": "#2A9D8F", "negative": "#E63946", "neutral": "#6C757D"}
PALETTE_LIST = list(PALETTE.values())
SEGMENT_COLORS = {"Low": "#F4A261", "Medium": "#2A9D8F", "High": "#1F4E79"}     # one colour = one segment
SEGMENT_ORDER = ["Low", "Medium", "High"]
RP_ONE, RP_REP = "1-Time Buyers", "Repeat Buyers"                               # repeat-purchase groups
RP_GROUPS = [RP_ONE, RP_REP]
RP_COLORS = {RP_ONE: "#F28E2B", RP_REP: "#4E79A7"}
RFM_ORDER = ["Champions", "Loyal Customers", "Potential Loyalists", "New / Promising", "At Risk",
             "Hibernating / Lost"]
CLUSTER_PALETTE = ["#1F4E79", "#2A9D8F", "#F4A261", "#E76F51", "#8E6BBF", "#6C757D", "#E9C46A", "#264653"]

# Schema contract (Section 2.11): column names that every analysis relies on
CLEAN_COLS = ["transaction_id", "customer_id", "transaction_date", "product_name",
              "category", "quantity", "unit_price", "total_amount"]
CUSTOMER_COLS = ["customer_id", "total_spend", "num_orders", "avg_order_value",
                 "first_purchase_date", "last_purchase_date", "recency_days"]
LOG_COLS = ["step_no", "cleaning_stage", "issue_addressed", "action_taken",
            "rows_before", "rows_after", "rows_removed", "pct_removed"]
SCHEMA_CONTRACT = {
    "df_clean": {"created_in": "Section 5 (Member 4)",
                 "required": {"transaction_id": "id", "customer_id": "id", "transaction_date": "datetime",
                              "product_name": "text", "category": "text", "quantity": "numeric",
                              "unit_price": "numeric", "total_amount": "numeric"},
                 "no_null": ["transaction_id", "customer_id", "transaction_date"]},
    "customer_df": {"created_in": "Section 6 (Member 5)",
                    "required": {"customer_id": "id", "total_spend": "numeric", "num_orders": "numeric",
                                 "avg_order_value": "numeric", "first_purchase_date": "datetime",
                                 "last_purchase_date": "datetime", "recency_days": "numeric"},
                    "no_null": ["customer_id", "total_spend", "num_orders"], "unique": ["customer_id"],
                    "later_columns": {"spend_segment": "Section 7 (Member 6)",
                                      "rfm_segment": "Section 12.2 (Member 11)",
                                      "cluster": "Section 12.5 (Member 13)"}},
}
# Typical alternative spellings (from pre_merge_sanity_check.py) - used to rescue uploaded files
COLUMN_ALIASES = {
    "transaction_id": ["order_id", "orderid", "invoice_id", "invoice_no", "invoiceno", "txn_id", "transactionid"],
    "customer_id": ["customerid", "cust_id", "custid", "customer", "client_id", "user_id"],
    "transaction_date": ["date", "order_date", "orderdate", "invoice_date", "invoicedate", "purchase_date"],
    "product_name": ["product", "item", "item_name", "product_title", "description"],
    "category": ["product_category", "category_name", "department"],
    "quantity": ["qty", "units", "quantity_sold"],
    "unit_price": ["price", "unitprice", "item_price"],
    "total_amount": ["revenue", "sales", "amount", "total", "total_price", "line_total", "sales_amount"],
}

# Where the project's exported tables are searched (relative to the working folder and to app.py)
DISK_FILES = {"df_clean": "cleaned_transactions.csv",
              "customer_df": "customer_analytics_master.csv",
              "cleaning_log": "data_cleaning_log.csv"}

# ---------------------------------------------------------------------------------------------------
# 3.  GLOBAL CHART STYLE  (Section 2.9 - same look for all 14 members' charts)
# ---------------------------------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({
    "figure.figsize": (11, 6), "figure.dpi": 100, "savefig.bbox": "tight",
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 10, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.30,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.frameon": False, "legend.fontsize": 9.5,
    "font.family": "DejaVu Sans",
})
CHART_DPI = 110                                                         # resolution of the PNGs shown in the app
currency_formatter = mticker.FuncFormatter(lambda x, pos=None: f"{CURRENCY_SYMBOL}{x:,.0f}")
thousands_formatter = mticker.FuncFormatter(lambda x, pos=None: f"{x:,.0f}")


def pct_axis(v, pos=None):
    """Tick label for values stored on a 0-100 scale."""
    return f"{v:.0f}%"


# ---------------------------------------------------------------------------------------------------
# 4.  SMALL FORMATTING HELPERS
# ---------------------------------------------------------------------------------------------------
def is_num(x):
    """True for a usable (non-missing) number."""
    try:
        return x is not None and not pd.isna(x) and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def money(v, dec=0):
    """'₹12,345' (NaN-safe)."""
    if not is_num(v):
        return "n/a"
    return f"{'-' if v < 0 else ''}{CURRENCY_SYMBOL}{abs(v):,.{dec}f}"


def money_compact(v):
    """Short currency label for annotations: ₹2.4L / ₹1.2Cr (rupees) or K / M / B (other currencies)."""
    if not is_num(v):
        return "n/a"
    units = ((1e7, "Cr"), (1e5, "L"), (1e3, "K")) if CURRENCY_SYMBOL == "\u20b9" else \
            ((1e9, "B"), (1e6, "M"), (1e3, "K"))
    for size, suffix in units:
        if abs(v) >= size:
            return f"{CURRENCY_SYMBOL}{v / size:.1f}{suffix}"
    return f"{CURRENCY_SYMBOL}{v:,.0f}"


def pct(v, dec=1):
    """'12.3%' for a value already on a 0-100 scale."""
    return "n/a" if not is_num(v) else f"{v:.{dec}f}%"


def frac(v, dec=1):
    """'12.3%' for a 0-1 fraction."""
    return "n/a" if not is_num(v) else f"{100 * v:.{dec}f}%"


def num(v, dec=0):
    """'12,345' (NaN-safe)."""
    return "n/a" if not is_num(v) else f"{v:,.{dec}f}"


def signed_pct(v, dec=1):
    return "n/a" if not is_num(v) else f"{v:+.{dec}f}%"


def plural(n, word, plural_word=None):
    """'1 customer' / '2 customers'."""
    n = int(n)
    return f"{n:,} {word if n == 1 else (plural_word or word + 's')}"


def short(text, n=26):
    """Shorten a label for chart axes."""
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "\u2026"


def esc(text):
    """Escape characters that would be read as Markdown / LaTeX in dynamic text (product names ...)."""
    text = str(text)
    for ch in ("\\", "*", "_", "`", "$", "|", "<", ">", "~"):
        text = text.replace(ch, "\\" + ch)
    return text


def mpl_text(text):
    """Escape '$' so matplotlib does not switch to math-text."""
    return str(text).replace("$", r"\$")


def text_color_for(bg):
    """Dark or white label colour - whichever stays readable on a bar of colour `bg`."""
    try:
        r, g, b = mcolors.to_rgb(bg)
        return "#1B1B1B" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"
    except Exception:
        return "white"


def skew_label(s):
    """Plain-language label for a skewness value."""
    if not is_num(s):
        return "undefined (no variation)"
    if s >= 1:
        return "strongly right-skewed"
    if s >= 0.5:
        return "moderately right-skewed"
    if s <= -1:
        return "strongly left-skewed"
    if s <= -0.5:
        return "moderately left-skewed"
    return "approximately symmetric"


def safe_div(a, b, default=np.nan):
    """a / b that never raises and never returns inf."""
    try:
        if b is None or b == 0 or pd.isna(b) or pd.isna(a):
            return default
        return a / b
    except Exception:
        return default


# ---------------------------------------------------------------------------------------------------
# 5.  STREAMLIT UI HELPERS  (tables, charts, interpretation blocks)
# ---------------------------------------------------------------------------------------------------
def interpret(*sentences, title="Interpretation"):
    """Data-driven 2-3 sentence block shown under every major table / chart (numbers come from the data)."""
    text = " ".join(str(s).strip() for s in sentences if s and str(s).strip())
    st.info(f"**{title}.** " + text.replace("$", r"\$"), icon="\U0001F4A1")


def caption_md(text):
    """Bold caption above a table or figure."""
    st.markdown(f"**{text}**".replace("$", r"\$"))


def show_df(df, caption=None, hide_index=False, height=None):
    """Display a DataFrame (optionally with a bold caption)."""
    if caption:
        caption_md(caption)
    kwargs = {"hide_index": hide_index}
    if height:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def md_table(df, index=False):
    """Render a (text-heavy) DataFrame as a wrapped Markdown table; cells may use **bold** / `code`."""
    frame = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "|" + "|".join(":--" for _ in frame.columns) + "|"
    rows = []
    for rec in frame.itertuples(index=False):
        cells = []
        for v in rec:
            cell = "" if (isinstance(v, float) and pd.isna(v)) else str(v)
            cells.append(cell.replace("|", "\\|").replace("\n", " ").replace("$", r"\$"))
        rows.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join([head, sep] + rows))


def fig_to_png(fig, dpi=CHART_DPI):
    """Render a matplotlib figure to PNG bytes and release its memory."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def show_png(png):
    """Show PNG bytes full-width (works on old and new Streamlit versions)."""
    if png is None:
        return
    try:
        st.image(png, width="stretch")
    except Exception:
        try:
            st.image(png, use_container_width=True)
        except Exception:
            st.image(png)


def chart(fn):
    """Decorator: cache the PNG bytes returned by a plotting function (keyed by its arguments)."""
    try:
        return st.cache_data(show_spinner=False, max_entries=160)(fn)
    except Exception:                                                   # pragma: no cover - very old Streamlit
        return fn


def run_safely(fn, *args, **kwargs):
    """Run one dashboard block; a failure shows a readable warning instead of crashing the whole app."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                                            # noqa: BLE001
        st.warning(f"This block could not be rendered for the current selection "
                   f"({type(exc).__name__}: {exc}).")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return None


def section_header(title, owner=None, level=3):
    """Section title with the member credit line."""
    st.markdown(f"{'#' * level} {title}")
    if owner:
        st.caption(owner)


def kpi_card(col, label, value, help_text=None):
    """One metric card."""
    with col:
        st.metric(label, value, help=help_text)

# ===================================================================================================
# 6.  STATIC PROJECT CONTENT  (text and reference tables taken from the team's notebooks)
# ===================================================================================================
# ---- 6.1  Team allocation and credit table (Section 1.1) ----------------------------------------
TEAM_ROWS = [
    # (member no, name, part, notebook section(s), phase, paper share %, dashboard tab)
    (1, "Hasnain Raza Khan", 1, "Sections 1, 2, 17", "A", 7, "Tab 1, Tab 7"),
    (2, "Aditi Sharma", 2, "Section 3, Appendix A", "A", 7, "Tab 1"),
    (3, "Sonam", 3, "Section 4", "A", 7, "Tab 2"),
    (4, "Samiya Khan", 4, "Section 5", "A", 8, "Tab 2"),
    (5, "Aditya Singh", 5, "Section 6", "A", 7, "Tab 3"),
    (6, "Saif Khan", 6, "Section 7", "B", 7, "Tab 3"),
    (7, "Divyanshi", 7, "Section 8", "B", 7, "Tab 3"),
    (8, "Abhi Rastogi", 8, "Section 9", "B", 8, "Tab 4"),
    (9, "Sanjay Rathour", 9, "Section 10", "B", 7, "Tab 4"),
    (10, "Dev Rastogi", 10, "Section 11", "B", 7, "Tab 4"),
    (11, "Vidhan Agarwal", 11, "Sections 12.1, 12.2", "C", 7, "Tab 5"),
    (12, "Anshu", 12, "Sections 12.3, 12.4", "C", 7, "Tab 5"),
    (13, "Rudra Pratap Singh", 13, "Sections 12.5, 13, 14", "C", 7, "Tab 6, Tab 7"),
    (14, "Ayan Khan", 14, "Sections 15, 16, Appendix B", "C", 7, "Tab 7"),
]
CREDIT_PER_MEMBER = 100.0 / 14                                           # 7.142857...%
PHASE_NAMES = {
    "A": "Data Setup, Cleaning and Final Merging",
    "B": "Customer Spend, Product and Sales Trends EDA",
    "C": "Advanced Analytics, Machine Learning and Conclusion",
}
PHASE_SECTIONS = {"A": "1, 2, 3, 4, 5, 6, 17 + Appendix A", "B": "7, 8, 9, 10, 11",
                  "C": "12.1-12.5, 13, 14, 15, 16 + Appendix B"}

# ---- 6.2  Business problem and questions (Section 1.3) ------------------------------------------
BUSINESS_PROBLEM = (
    "The business holds a large log of customer transactions but has no consolidated, evidence-based picture "
    "of how its revenue is created. It cannot yet say which customers generate the revenue and how concentrated "
    "that revenue is, which products and categories carry the business, whether customers come back, when demand "
    "peaks, or whether natural groups of customers exist that deserve different treatment. Without these answers "
    "marketing spend is spread evenly, high-value customers are neither identified nor protected, slow-moving "
    "products are over-stocked and cross-selling opportunities are missed.")
BUSINESS_AIM = (
    "To transform a raw and imperfect transaction dataset into a validated analytical base (`df_clean` and "
    "`customer_df`) and to use it to answer the business questions below with reproducible tables, charts and "
    "statistical evidence, ending in prioritised, actionable recommendations.")
BQ_TABLE = [
    ("BQ1", "Is the data complete, consistent and reliable enough to analyse?", "Data audit, cleaning and cleaning log", "3-5", "2, 3, 4"),
    ("BQ2", "What does each customer look like in terms of spend, frequency and recency?", "Customer-level aggregation", "6", "5"),
    ("BQ3", "How do Low, Medium and High spenders differ, and how much revenue does each contribute?", "Threshold-based spend segmentation; revenue-share comparison", "7-8", "6, 7"),
    ("BQ4", "Which products and categories drive sales, and how is the category mix composed?", "Product and category EDA", "9", "8"),
    ("BQ5", "How do one-time buyers differ from repeat buyers?", "Repeat-purchase analysis", "10", "9"),
    ("BQ6", "When do customers buy: which month, weekday and season?", "Time-based EDA", "11", "10"),
    ("BQ7", "Is revenue concentrated in a few customers, and who are the best and the at-risk customers?", "Pareto (80/20) analysis; RFM segmentation", "12.1-12.2", "11"),
    ("BQ8", "Do customers stay, and which products are bought together?", "Cohort retention; market basket analysis", "12.3-12.4", "12"),
    ("BQ9", "Which natural behavioural groups exist, and what should the business do about them?", "K-Means clustering; key findings; recommendations", "12.5, 13, 14", "13"),
    ("BQ10", "What are the overall conclusions, limitations and next steps?", "Synthesis, export, glossary", "15, 16, Appendix B", "14"),
]
OBJECTIVES = {
    "Phase A - Data Cleaning and Setup (Members 1-5; Sections 1-6, 17, Appendix A)": [
        "**A1.** Define the project scope, objectives, workflow and team conventions, and establish one reproducible environment (Sections 1-2).",
        "**A2.** Document the dataset with a data dictionary and generate the raw transaction data reproducibly with the generator in Appendix A (Section 3).",
        "**A3.** Inspect the raw data and audit its quality: missing records, schema consistency and data types (Section 4).",
        "**A4.** Clean the data (duplicates, missing IDs and dates, returns), record every action in a cleaning log and deliver `df_clean` (Section 5).",
        "**A5.** Build the customer-level summary table `customer_df` as the single analytical base for all later sections (Section 6).",
        "**A6.** Compile the academic and technical references and integrate all 14 parts into one error-free deliverable (Section 17).",
    ],
    "Phase B - Spend, Product, Repeat Purchase and Seasonality EDA (Members 6-10; Sections 7-11)": [
        "**B1.** Segment customers into Low, Medium and High spend groups using transparent, justified thresholds (Section 7).",
        "**B2.** Compare the segments on customer count, average spend and revenue share, with a table and a chart (Section 8).",
        "**B3.** Identify the top-selling products and categories and describe the category mix (Section 9).",
        "**B4.** Compare the shopping behaviour of one-time and repeat buyers (Section 10).",
        "**B5.** Reveal monthly, weekday and seasonal sales patterns and state the business insight behind each (Section 11).",
    ],
    "Phase C - Pareto (80/20), RFM, Cohort, Market Basket and K-Means (Members 11-14; Sections 12-16, Appendix B)": [
        "**C1.** Test the Pareto (80/20) hypothesis: what share of customers generates 80% of revenue (Section 12.1).",
        "**C2.** Score customers on Recency, Frequency and Monetary value and name actionable RFM segments (Section 12.2).",
        "**C3.** Measure customer retention by acquisition cohort over time (Section 12.3).",
        "**C4.** Discover products that are bought together using support, confidence and lift (Section 12.4).",
        "**C5.** Cluster customers with K-Means: scale features, choose k with the elbow and silhouette methods and profile each cluster (Section 12.5).",
        "**C6.** Consolidate the evidence into key findings and prioritised business recommendations (Sections 13-14).",
        "**C7.** Conclude, state limitations and future scope, export results and provide a glossary (Sections 15-16, Appendix B).",
    ],
}
TECHNIQUE_TABLE = [
    ("Data audit and cleaning", "Is the data complete, valid and unique?", "Missing %, duplicate count, invalid values, cleaning log", "4-5"),
    ("Customer aggregation", "What does each customer look like?", "Total spend, number of orders, average order value, recency", "6"),
    ("Spend segmentation", "Who are the Low, Medium and High spenders?", "Thresholds on total spend (tertiles)", "7-8"),
    ("Product and category EDA", "What sells?", "Revenue and units by product and category; share of total", "9"),
    ("Repeat-purchase analysis", "Who comes back?", "Orders per customer; one-time vs repeat buyers", "10"),
    ("Time-based EDA", "When do customers buy?", "Revenue by month, weekday and season", "11"),
    ("Pareto analysis", "How concentrated is revenue?", "Cumulative revenue share vs cumulative customer share", "12.1"),
    ("RFM segmentation", "Who is valuable or at risk?", "Recency, Frequency and Monetary scores", "12.2"),
    ("Cohort retention", "Do customers stay?", "Retention rate by acquisition cohort and period", "12.3"),
    ("Market basket analysis", "What is bought together?", "Support, confidence and lift", "12.4"),
    ("K-Means clustering", "Which natural groups exist?", "Within-cluster sum of squares, silhouette coefficient", "12.5"),
]
KEY_FORMULAS = [
    ("Pareto cumulative revenue share (Section 12.1)",
     r"C_k = \frac{\sum_{j=1}^{k} r_{(j)}}{\sum_{j=1}^{N} r_{(j)}}, \qquad r_{(1)} \ge r_{(2)} \ge \dots \ge r_{(N)}"),
    ("RFM measures (Section 12.2)",
     r"R_i = t_{\mathrm{ref}} - t_i^{\mathrm{last}}, \qquad F_i = n_i, \qquad M_i = \sum_{t=1}^{n_i} s_{i,t}"),
    ("Cohort retention (Section 12.3)",
     r"\mathrm{Retention}_{c,k} = \frac{\text{customers of cohort } c \text{ who buy in period } k}{n_c}"),
    ("Association-rule measures (Section 12.4)",
     r"\mathrm{support}(A)=\frac{|\{T:A\subseteq T\}|}{|\mathcal{T}|},\quad "
     r"\mathrm{confidence}(A\Rightarrow B)=\frac{\mathrm{support}(A\cup B)}{\mathrm{support}(A)},\quad "
     r"\mathrm{lift}(A\Rightarrow B)=\frac{\mathrm{confidence}(A\Rightarrow B)}{\mathrm{support}(B)}"),
    ("K-Means objective and silhouette (Section 12.5)",
     r"J = \sum_{j=1}^{K} \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^{2}, \qquad s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}"),
]
WORKFLOW_DIAGRAM = """\
PHASE A : DATA SETUP, CLEANING & FINAL MERGING                       (Members 1-5)
  [Sec 1]   Introduction, Objectives & Project Workflow               Member 1
  [Sec 2]   Libraries, Environment & Global Configuration             Member 1
      |
  [Sec 3]   Dataset Description & Data Dictionary  (App. A -> df_raw) Member 2
      |   df_raw
  [Sec 4]   Data Inspection & Quality Audit                           Member 3
      |
  [Sec 5]   Data Cleaning & Cleaning Log  ->  df_clean                Member 4
      |   df_clean   (mandatory name)
  [Sec 6]   Customer-Level Summary Table  ->  customer_df             Member 5
      |   customer_df (mandatory name)
PHASE B : CUSTOMER SPEND, PRODUCT & SALES TRENDS EDA                  (Members 6-10)
  [Sec 7]   Low / Medium / High Spend Segmentation                    Member 6
  [Sec 8]   Segment Comparison & Revenue Contribution                 Member 7
  [Sec 9]   Product, Category & Category-Mix Analysis                 Member 8
  [Sec 10]  Repeat Purchase Analysis (1-time vs repeat)               Member 9
  [Sec 11]  Monthly, Weekday & Seasonality Trends                     Member 10
      |   segments + EDA insights
PHASE C : ADVANCED ANALYTICS, MACHINE LEARNING & CONCLUSION          (Members 11-14)
  [Sec 12.1] Pareto Analysis (80/20 Rule)       [Sec 12.2] RFM Segmentation        Member 11
  [Sec 12.3] Cohort Retention Analysis          [Sec 12.4] Market Basket Analysis  Member 12
  [Sec 12.5] K-Means Customer Clustering        [Sec 13/14] Findings & Recommendations  Member 13
  [Sec 15]   Conclusion & Limitations           [Sec 16]  Future Scope & Export    Member 14
  [Sec 17]   References (Member 1)              [App. B]  Glossary (Member 14)

 SHARED OBJECTS : df_raw (Sec 3) -> df_clean (Sec 5) -> customer_df (Sec 6) -> Sec 7 ... Sec 16
 RULE           : every section reads the objects above it and never overwrites them.
"""
WORKFLOW_STEPS = [
    (1, "Project framing", "1", "Define the problem, objectives, workflow, conventions and credit table", "1"),
    (2, "Environment set-up", "2", "Import libraries, fix the seed, set display and chart style, define helpers", "1"),
    (3, "Data understanding and generation", "3, App. A", "Describe the dataset, write the data dictionary, run the data generator", "2"),
    (4, "Inspection and quality audit", "4", "Check shape, types, missing records and schema; report issues without changing data", "3"),
    (5, "Data cleaning", "5", "Remove duplicates, resolve missing IDs / dates, treat returns, log every action", "4"),
    (6, "Customer summarisation", "6", "Aggregate spend, orders, average order value, first and last purchase, recency", "5"),
    (7, "Spend segmentation and comparison", "7-8", "Create Low / Medium / High groups; compare revenue contribution", "6, 7"),
    (8, "Product, repeat and seasonality EDA", "9-11", "Top products and categories, one-time vs repeat buyers, monthly / weekday / seasonal trends", "8, 9, 10"),
    (9, "Advanced analytics", "12.1-12.5", "Pareto 80/20, RFM, cohort retention, market basket, K-Means", "11, 12, 13"),
    (10, "Synthesis", "13-14", "Combine evidence into key findings and prioritised actions", "13"),
    (11, "Closure", "15-16, App. B", "Conclusion, limitations, future scope, export of results, glossary", "14"),
    (12, "Referencing and integration", "17 + merge", "Compile references, merge all parts in order, run checkpoints", "1"),
]

# ---- 6.3  Dataset description and data dictionary (Section 3) -----------------------------------
DATASET_PROPERTIES = [
    ("Raw rows in df_raw", "about 6,700 (about 6,400 valid lines plus 310 intentional impurity rows)"),
    ("Customers", "1,000 (CUST0001 to CUST1000)"),
    ("Transaction IDs", "about 2,700 (genuine sales baskets plus return, guest and undated orders)"),
    ("Products and categories", "50 products in 5 categories"),
    ("Period", "2 years, 2024-01-01 to 2025-12-31"),
    ("Columns", "8 (see the data dictionary)"),
    ("Grain of the table", "One row = one product line inside a transaction (basket of 1 to 4 different products)"),
]
DATA_DICTIONARY = [
    ("transaction_id", "object (text)", "Basket or invoice number. One ID belongs to exactly one customer and one date and holds 1 to 4 different products (one row per product).", "TXN100001", "Basket key: counts orders, average order value and purchase frequency, and provides the item sets for Market Basket Analysis (Section 12.4)."),
    ("customer_id", "object (text)", "Identifier of the customer who placed the order (1,000 distinct customers). Missing for guest checkouts in the raw data.", "CUST0001", "Join key for customer_df; basis of spend segments, Pareto, RFM, cohorts and K-Means."),
    ("transaction_date", "datetime64[ns]", "Calendar date of the transaction (daily precision, 2024-01-01 to 2025-12-31). Missing in some raw rows.", "2024-10-27", "Time axis for monthly, weekday and seasonal trends; recency in RFM; cohort month."),
    ("product_name", "object (text)", "Name of the product sold in the line (50 distinct products).", "Wireless Earbuds", "Product ranking, category mix and the items analysed in baskets."),
    ("category", "object (text)", "Product category: Electronics, Apparel, Home & Kitchen, Beauty or Groceries.", "Electronics", "Category revenue mix, seasonality by category and customer category preferences."),
    ("quantity", "int64", "Units bought in the line. A negative value marks a return or cancellation.", "2", "Volume driver; negative values identify returns during cleaning."),
    ("unit_price", "float64", "Selling price per unit in INR (list price; some festive-sale lines carry a 10% discount).", "2499.0", "Pricing and revenue driver."),
    ("total_amount", "float64", "Line revenue in INR, equal to quantity x unit_price. Negative for returns.", "4998.0", "Monetary value: customer spend, spend segments, Pareto share and RFM Monetary."),
]
BUSINESS_PATTERNS = [
    ("Pareto (80/20) skew", "Customer value is heavy-tailed (four value tiers). Bigger customers order more often, fill larger baskets, buy more units and more premium items, so roughly the top 20% of customers produce about 80% of revenue.", "Sections 7, 8 and 12.1"),
    ("Repeat purchases", "Customers stay active for a limited lifetime and re-order within it; each customer has favourite products that are bought again. About 45% of customers buy only once.", "Sections 10, 12.2 and 12.3"),
    ("Festive and seasonal peaks", "Demand rises around Diwali and Dussehra, Holi, Eid, Republic Day and Independence Day sales, and Christmas to New Year. Weekends are busier and some products are seasonal (for example Winter Jacket and Sunscreen SPF 50).", "Sections 9 and 11"),
    ("Multi-item baskets", "Each transaction holds 1 to 4 different products. Natural bundles (Smartphone with Phone Case and Screen Protector, Shampoo with Conditioner, Tea Powder with Sugar) create strong association rules.", "Section 12.4"),
    ("Customers join over time", "Part of the customer base is active from the start; the rest makes a first purchase at different points of the two years.", "Section 12.3"),
]
DQ_PROBLEMS = [
    ("Exact duplicate rows", 120, "All 8 columns are identical to an existing row", "Section 4", "Section 5"),
    ("Missing customer_id", 60, "Guest-checkout lines with customer_id = NaN", "Section 4", "Section 5"),
    ("Missing transaction_date", 40, "Lines with transaction_date = NaT", "Section 4", "Section 5"),
    ("Returns and cancellations", 90, "Negative quantity and negative total_amount", "Section 4", "Section 5"),
]
GENERATOR_PARAMS = [
    ("Random seed", "RANDOM_STATE (value 42), passed as `seed`; the function uses a local numpy random generator"),
    ("Customers", "1,000 (CUST0001 to CUST1000)"),
    ("Genuine sales transactions", "about 2,500 (target 2,500; the exact count varies slightly with the seed)"),
    ("Basket size", "1 to 4 different products per transaction"),
    ("Products", "50 products in 5 categories (Electronics, Apparel, Home & Kitchen, Beauty, Groceries)"),
    ("Period", "2024-01-01 to 2025-12-31 (731 calendar days, including the leap day of 29 February 2024)"),
    ("Customer value tiers", "4 (low, medium, high and elite = top 5%), driving order frequency, basket size, units per line and premium purchases"),
    ("Existing customers at the start", "40% of customers make their first purchase within the first 90 days"),
    ("Intentional impurities", "310 extra rows: 120 exact duplicates, 60 missing customer_id, 40 missing transaction_date, 90 returns"),
    ("Output", "One DataFrame with 8 columns: transaction_id, customer_id, transaction_date, product_name, category, quantity, unit_price, total_amount"),
]

# ---- 6.4  Section 5: planned cleaning pipeline (Table 5.1) --------------------------------------
PLANNED_PIPELINE = [
    (1, "Duplicate removal", "`drop_duplicates()` on all columns", "Repeated rows would double-count revenue"),
    (2, "Missing ID / date removal", "Drop rows with a null `transaction_id`, `customer_id` or `transaction_date`", "A sale that cannot be tied to a customer or a day cannot be analysed"),
    (3, "Date parsing and text trimming", "`pd.to_datetime(errors=\"coerce\")`, drop invalid dates, `.str.strip()`", "Gives a real time axis and prevents look-alike IDs such as \"C01\" and \"C01 \""),
    (4, "Returns and cancellations", "Quantity, unit price or total amount `<= 0` moved to `_df_returns`", "Keeps `df_clean` strictly positive"),
    (5, "Amount verification", "`total_amount = (quantity * unit_price).round(2)` and `reset_index(drop=True)`", "Makes revenue reproducible and the index contiguous"),
]

# ---- 6.5  Section 12.2: RFM rules and suggested actions -----------------------------------------
RFM_RULES_TABLE = [
    ("Champions", "R >= 4 and F >= 4 and M >= 4", "Bought recently, buy often and spend the most"),
    ("Loyal Customers", "R >= 3 and F >= 4 (not Champions)", "Frequent buyers who are either slightly less recent or spend less than the top tier"),
    ("Potential Loyalists", "R >= 3 and F = 3", "Recent customers with moderate frequency"),
    ("New / Promising", "R >= 3 and F <= 2", "Recent customers with few orders so far"),
    ("At Risk", "R <= 2 and F >= 3", "Used to buy regularly but have gone quiet"),
    ("Hibernating / Lost", "R <= 2 and F <= 2", "Inactive for a long time with low engagement"),
]
RFM_ACTIONS = [
    ("Champions", "Recent, frequent, high spend", "Reward and retain: early access, loyalty perks, referral requests"),
    ("Loyal Customers", "Frequent and reasonably recent", "Cross-sell and upsell, loyalty programme, nudge towards a higher basket value"),
    ("Potential Loyalists", "Recent, moderate frequency", "Build the habit: second and third purchase offers, bundles, personalised recommendations"),
    ("New / Promising", "Recent, few orders", "Onboard well: welcome series and an incentive for the first repeat purchase"),
    ("At Risk", "Previously active, now quiet", "Win-back campaign: personalised reminders, time-limited offers, feedback survey"),
    ("Hibernating / Lost", "Long inactive, low engagement", "Low-cost reactivation only; limit paid marketing and study why they left"),
]

# ---- 6.6  Section 10: recommendations and limitations -------------------------------------------
RP_RECOMMENDATIONS = [
    (1, "**Post-purchase journey (first 30 days)**", "Automated thank-you -> usage tips -> review request -> \"you may also like\" message by e-mail / SMS / WhatsApp", "Findings 1, 4", "% of 1-time buyers placing a 2nd order within 30 / 60 / 90 days"),
    (2, "**Second-order incentive**", "Time-limited coupon, free shipping or bonus points valid only on the *next* order and above a minimum basket value (protects margin)", "Findings 1, 6", "2nd-order conversion rate; AOV of 2nd orders"),
    (3, "**Cross-sell from repeat-leaning categories**", "Recommend the categories that are over-represented among repeat buyers in the order-confirmation e-mail and in on-site \"bought together\" blocks", "Finding 5", "Share of first orders that contain a repeat-leaning category"),
    (4, "**Build a bigger first basket**", "Bundles, add-on suggestions and a free-shipping threshold set slightly above today's AOV", "Findings 3, 6", "Items per order and AOV of first orders"),
    (5, "**Replenishment reminders**", "For frequently re-bought categories, send reminders at the typical re-buy interval (estimate it with the cohort analysis of Section 12.3)", "Findings 4, 5", "Repeat rate inside the category; reminder-to-order rate"),
    (6, "**Tiered loyalty program**", "Points and tier benefits that reward the 2nd -> 3rd -> 4th order; VIP perks for customers with 4+ orders", "Findings 2, 4", "Movement of customers from 2 to 3 and from 3 to 4+ orders"),
    (7, "**Win-back and feedback loop**", "Personal win-back offer for 1-time buyers who stay inactive beyond the usual re-buy window, plus a one-question survey on why they did not return", "Finding 1", "Re-activation rate; survey reasons (price, delivery, product, service)"),
]
RP_LIMITATIONS = [
    "**Observation window:** a customer who joined late in the data period has had little time to buy again, so the 1-time group can include customers who may still return. Time-based retention is analysed in Section 12.3 (cohorts).",
    "**Association, not causation:** category and basket differences describe what repeat buyers *do*; they do not prove that buying a category *causes* repeat purchases. Test the recommendations (A/B tests) before scaling them.",
    "**Definition of an order:** one unique order ID; item-lines with the same ID form one order. Basket size counts units (quantity); AOV uses line revenue.",
    "**Cleaned data:** results inherit the cleaning decisions of Section 5 (duplicates, missing IDs / dates, returns). `df_clean` was not altered here.",
    "**Simulated data:** if the dataset comes from the project's data generator (Appendix A), the patterns reflect the generator's assumptions and may differ from a real retailer.",
    "**No cost data:** the dataset holds no marketing or margin figures, so the ROI of the recommendations cannot be estimated.",
]
KMEANS_METHOD = [
    ("Features", "`recency_days`, `num_orders`, `total_spend`, `avg_order_value`", "How recently, how often, how much and how big each customer buys"),
    ("Skew treatment", "`np.log1p()` on a **copy** of the features", "Spend, frequency and AOV are right-skewed; K-Means minimises squared distances, so a few very large customers would otherwise dominate the centroids"),
    ("Scaling", "`StandardScaler()`", "Days, order counts and currency amounts are put on one scale (mean 0, SD 1)"),
    ("Initialisation", "k-means++ with `n_init=10`, `random_state=42`", "Spreads the starting centroids and keeps the best of 10 runs, so results are stable and reproducible"),
    ("Choosing k", "Inertia (elbow), silhouette and Davies-Bouldin index for k = 2 ... 8", "Three independent views of cluster quality"),
]
KMEANS_ASSUMPTIONS = [
    "K-Means assumes compact, roughly spherical groups of similar size; elongated or very unequal groups can be split imperfectly (MacQueen, 1967; scikit-learn Developers, n.d.).",
    "Results depend on the feature choice and on the log + z-score transformation; adding or removing a feature (for example product mix) can change the clusters.",
    "Silhouette and Davies-Bouldin values for customer-behaviour data are usually moderate; clusters are *useful segments*, not natural, perfectly separated species.",
    "`n_init=10` and k-means++ initialisation (Arthur & Vassilvitskii, 2007) reduce, but do not eliminate, sensitivity to the random seed; `random_state=42` makes the run reproducible.",
    "Clusters describe the customer base at the snapshot date; they should be refreshed regularly because recency changes every day.",
]

# ---- 6.7  Sections 13-14: business-question map, finding topics, scenario assumptions ------------
BQ_CONFIG = {
    "BQ1": ("Is the transaction data reliable enough for decisions, and what had to be cleaned?", "Sections 4-5"),
    "BQ2": ("What does a typical customer look like (spend, orders, order value, recency)?", "Section 6"),
    "BQ3": ("How do Low / Medium / High spend segments differ and how much revenue does each contribute?", "Sections 7-8"),
    "BQ4": ("Which products and categories drive sales?", "Section 9"),
    "BQ5": ("How do one-time buyers differ from repeat buyers?", "Section 10"),
    "BQ6": ("When do sales peak - by month, season and weekday?", "Section 11"),
    "BQ7": ("How concentrated is revenue, and which RFM segments matter most?", "Sections 12.1-12.2"),
    "BQ8": ("How well are new customers retained, and which products sell together?", "Sections 12.3-12.4"),
    "BQ9": ("Which distinct behavioural customer groups exist, and where is the value?", "Section 12.5"),
}
FINDING_TOPIC = {"F01": "Data quality", "F02": "Customer value skew", "F03": "Spend-segment concentration",
                 "F04": "Product and category mix", "F05": "Repeat vs one-time buyers", "F06": "Monthly seasonality",
                 "F07": "Weekday pattern", "F08": "Revenue concentration (Pareto)", "F09": "RFM segments",
                 "F10": "Cohort retention", "F11": "Market-basket associations", "F12": "K-Means value cluster",
                 "F13": "K-Means inactive cluster"}
DEFAULT_SCENARIO = {                       # every revenue effect in Section 14 is an ILLUSTRATIVE scenario, not a forecast
    "vip_upgrade_rate": 0.10, "category_lift": 0.05, "second_order_rate": 0.05, "peak_month_lift": 0.05,
    "top_customer_attrition": 0.05, "reactivation_rate": 0.10, "retention_lift_pp": 0.05,
    "attach_rate_lift": 0.10, "champion_frequency_lift": 0.05,
}
ASSUMPTION_NOTES = {
    "vip_upgrade_rate": ("R03", "Share of next-lower-tier customers lifted to the top-tier average spend"),
    "category_lift": ("R04", "Extra revenue in the leading category (or hero product)"),
    "second_order_rate": ("R05", "Share of one-time buyers converted into repeat buyers"),
    "peak_month_lift": ("R06", "Extra revenue captured in the strongest month"),
    "top_customer_attrition": ("R08", "Share of top-20% revenue lost without a retention programme"),
    "reactivation_rate": ("R09, R13", "Share of inactive customers re-activated with one more order"),
    "retention_lift_pp": ("R10", "Lift in month-1 retention (absolute, as a fraction: 0.05 = 5 percentage points)"),
    "attach_rate_lift": ("R11", "Relative lift in the cross-sell attach rate"),
    "champion_frequency_lift": ("R12", "Extra purchase frequency in the value cluster"),
}
PHASES = {1: ("Phase 1 - Quick wins and foundations", "Days 0-30"),
          2: ("Phase 2 - Core retention and loyalty programmes", "Days 30-90"),
          3: ("Phase 3 - Growth and optimisation", "Days 90-180")}
PHASE_LOGIC = [
    ("Phase 1 - Quick wins and foundations", "Days 0-30", "Low-effort actions: data-quality gate, value-tier reporting, weekday timing, bundle prompts"),
    ("Phase 2 - Core retention and loyalty programmes", "Days 30-90", "Programmes that protect or recover the largest revenue pools: loyalty tiers, second-order journey, VIP desk, onboarding, win-back"),
    ("Phase 3 - Growth and optimisation", "Days 90-180", "Range, seasonal and advocacy actions that build on the cleaner data and customer map from Phases 1-2"),
]
PRIORITY_MEANING = [
    ("High", "Protects or recovers a large revenue pool, or closes the biggest leak in the customer lifecycle"),
    ("Medium", "A scalable growth lever with a clearly defined target group"),
    ("Low", "An enabler or optimisation that supports the other actions"),
]
RISKS = [
    "**Scenarios are illustrative, not forecasts.** They multiply an assumed rate by a measured base; real response rates must be learned from tests.",
    "**Impacts overlap and must not be added up.** R03, R08 and R12 address partly the same high-value customers, and R09 and R13 partly the same inactive customers.",
    "**Association is not causation.** Segment and cluster differences describe *who* buys, not *why*; each programme should be proven with a holdout (A/B) group before it is scaled.",
    "**Customer groups move.** Recency changes every day, so RFM segments and K-Means clusters should be refreshed regularly (monthly is a sensible start).",
]

# ---- 6.8  Sections 15-16: conclusion, limitations and future scope ------------------------------
CONCLUSION_TEXT = (
    "This study built an end-to-end, reproducible analytics workflow for **Customer Spend, Product & Sales "
    "Analytics**, moving from raw transaction records to behaviour-based customer segments. Its central "
    "conclusion is that every analytical layer is only as reliable as the one beneath it: trustworthy data "
    "cleaning makes meaningful descriptive analysis possible, and descriptive analysis in turn motivates and "
    "cross-checks the advanced behavioural models.")
CONCLUSION_PHASES = [
    ("A: Data foundation & cleaning", "Sections 3-6", "A documented dataset and data dictionary, a data-quality audit, and a fully logged cleaning step (duplicates, missing IDs/dates, returns) produced two reconciled analysis tables: `df_clean` (transaction level) and `customer_df` (customer level). Every later figure is traceable to a recorded cleaning decision."),
    ("B: Spend, product, repeat-purchase & seasonality EDA", "Sections 7-11", "Revenue was examined through three lenses: customer value (Low/Medium/High spend tiers), product (top items, categories, category mix) and time (monthly, weekday and seasonal patterns), with one-time buyers contrasted against repeat buyers. Together they show where revenue originates and how it varies."),
    ("C: Advanced analytics", "Sections 12.1-12.5", "Pareto (80/20) analysis quantified revenue concentration; RFM scoring ranked customers by recency, frequency and monetary value; cohort retention tracked how customer groups persist over time; market-basket rules (support, confidence, lift) exposed cross-sell affinities; and K-Means clustering on standardised features produced data-driven segments, validated with WCSS and the silhouette score."),
]
CONCLUSION_SYNTHESIS = (
    "Read together, the three phases describe customers along complementary dimensions: *how much* they spend "
    "(spend tiers, Pareto), *how recently and how often* they buy (RFM, repeat purchase, cohorts), *what* they buy "
    "together (market basket) and *which natural groups* exist (K-Means). The rule-based segmentations (spend tiers, "
    "RFM) and the unsupervised K-Means solution act as cross-checks on one another: agreement raises confidence in a "
    "segment, while disagreement flags customers that deserve closer inspection. For the business, this yields a "
    "clear decision framework: protect and grow high-value customers, re-activate lapsing ones, convert one-time "
    "buyers into repeat buyers, and align promotions and inventory with seasonality and cross-sell patterns.")
SYNTHETIC_DISCLAIMER = (
    "The dataset used in this study is **synthetic/simulated**: it was generated programmatically (see Appendix A). "
    "Its findings therefore demonstrate the analytical methodology and **do not represent the confidential data of "
    "any real company**. They should not be generalised to real markets without validation on real transactional data.")
LIMITATIONS = [
    "**Simulated data-generating process.** Spend distributions, seasonality, repeat-purchase behaviour and basket composition reflect the assumptions built into the data generator, not observed customer behaviour. Patterns such as revenue concentration or strong association rules may be weaker, stronger or absent in real data.",
    "**Finite observation window.** Seasonality, cohort retention and recency are measured only over the span covered by the data. Recent cohorts have had less time to make repeat purchases (right-censoring), and within a short window seasonal effects cannot be cleanly separated from long-term trend.",
    "**Rule-based cleaning decisions.** Removing duplicates, records with missing IDs/dates, and returns (documented in the Data Cleaning Log) was necessary for consistency, but it means that return behaviour and revenue net of returns are not analysed, and that records without a valid customer ID or date are absent from customer-level metrics.",
    "**Analyst-defined thresholds.** The Low/Medium/High spend cut-offs and the RFM score boundaries are heuristic choices; different thresholds would change segment sizes and revenue shares.",
    "**K-Means assumptions.** The algorithm requires *k* to be chosen in advance, assumes compact, roughly spherical clusters, and is sensitive to feature scaling, outliers and initialisation. Standardisation, WCSS/elbow and silhouette analysis, and a fixed random seed reduce this sensitivity but do not eliminate it, and no ground truth exists against which the clusters can be verified.",
    "**Association, not causation.** Market-basket metrics (support, confidence, lift) describe co-occurrence, and the rules obtained depend on the minimum-support threshold chosen; they do not prove that one product causes the purchase of another.",
    "**Descriptive scope.** The study is retrospective and descriptive: no predictive model was validated on hold-out data, and customer value is measured through revenue-based metrics (Monetary, total spend, AOV). Profitability, costs, discounts and customer demographics are not modelled, and each customer ID is assumed to represent one customer.",
]
FUTURE_SCOPE = [
    {"title": "1. Predictive Customer Lifetime Value (CLV) with BG/NBD and Gamma-Gamma models", "horizon": "near term",
     "objective": "Convert the backward-looking RFM view into the expected future value of each customer over a chosen horizon (for example 6 or 12 months).",
     "method": "The BG/NBD model estimates the expected number of future purchases and the probability that a customer is still active, using frequency, recency and customer age; the Gamma-Gamma model estimates the expected value per order; CLV = expected purchases x expected order value (optionally discounted).",
     "reuses": "The customer-level frequency, recency and monetary features from Sections 6 and 12.2.",
     "validation": "Calibration/holdout split (fit on an early period, compare predicted with actual purchases in the later period using MAE or RMSE) and a check of the Gamma-Gamma assumption that frequency and order value are only weakly correlated.",
     "use": "Rank customers by expected CLV to prioritise retention budgets."},
    {"title": "2. Supervised Churn Prediction", "horizon": "near to mid term",
     "objective": "Estimate each customer's probability of becoming inactive.",
     "method": "Define churn operationally (for example, no purchase within a fixed window after a snapshot date); build features from a past window (RFM, AOV, category diversity, gaps between purchases) and the label from the following window to avoid data leakage; start with logistic regression as a baseline and compare with tree-based ensembles (Random Forest, Gradient Boosting); handle class imbalance with class weights.",
     "reuses": "RFM scores, cohort index and the spend segments of Sections 7, 12.2 and 12.3.",
     "validation": "Time-based train/test split; ROC-AUC, PR-AUC, precision and recall at top-k, and lift for campaign targeting; probability calibration and feature-importance (for example SHAP) analysis.",
     "use": "Trigger win-back actions for high-risk, high-value customers."},
    {"title": "3. Real-time Recommendation Engine", "horizon": "mid term",
     "objective": "Move from segment-level cross-sell rules to personalised, low-latency product recommendations.",
     "method": "Start from the association rules of Section 12.4; add item-item collaborative filtering and matrix factorisation (for example ALS), with category-level popularity as a cold-start fallback; serve top-N recommendations through an API backed by a cache or feature store that is updated by a stream of live purchase events.",
     "reuses": "Support, confidence and lift rules from Section 12.4 and the category mix from Section 9.",
     "validation": "Offline precision@k, recall@k, NDCG and catalogue coverage, followed by online testing.",
     "use": "Raise basket size and repeat-purchase rate."},
    {"title": "4. A/B Testing & Experimentation Framework", "horizon": "continuous",
     "objective": "Verify that insights actually change customer behaviour instead of relying on correlation alone.",
     "method": "Randomise customers into control and treatment groups (for example a targeted offer for one spend or RFM segment); pre-define the hypothesis, the primary metric (conversion, AOV, revenue per customer, repeat rate) and guardrail metrics (return rate, margin); size the test with a power analysis (significance level, power, minimum detectable effect); run it over whole weekly cycles to absorb weekday seasonality; analyse with an appropriate test (two-proportion z-test, t-test or a non-parametric alternative) and correct for multiple comparisons.",
     "reuses": "The recommendations of Section 14 and the KPIs of the roadmap.",
     "validation": "Pre-registered hypothesis, power analysis and guardrail metrics.",
     "use": "Base rollout decisions on measured incremental lift."},
]
FUTURE_ENABLERS = ("Validate the whole pipeline on real transactional data, keep the Schema Contract and Data Cleaning Log as "
                   "automated data-quality gates, schedule the pipeline, and monitor model drift after deployment.")

# ---- 6.9  Appendix B: glossary (26 terms) -------------------------------------------------------
GLOSSARY = [
    ("Synthetic Dataset", "Programmatically simulated transaction data that mimics the structure of real retail sales. Results obtained from it demonstrate the analytical methodology and do not describe any real company.", "Sections 3, 15.2; Appendix A"),
    ("Schema Contract", "Formal specification of the columns, data types and validity rules (for example non-null keys, positive quantity and price) that a table must satisfy before analysis. Logic: for every column, `observed dtype = expected dtype` and the validity rule holds; violations are reported during inspection.", "Sections 3-4"),
    ("Data Cleaning Log", "Audit trail of every cleaning action (duplicates removed, missing IDs/dates handled, returns excluded) with row counts before and after each step. Logic: `rows_after = rows_before - rows_removed`, and the final count must reconcile with `df_clean`.", "Section 5"),
    ("Average Order Value (AOV)", "Average revenue per order, computed per customer or per segment: `AOV = total revenue / number of orders`.", "Sections 6-10"),
    ("Spend Segmentation", "Grouping of customers into Low, Medium and High tiers by total spend so that behaviour and revenue contribution can be compared. Logic: `Low if spend <= T1; Medium if T1 < spend <= T2; High if spend > T2`, with thresholds T1 and T2 justified in Section 7.", "Sections 7-8"),
    ("Repeat Purchase Rate", "Share of customers who bought more than once: `repeat rate = customers with 2 or more orders / all customers x 100`.", "Section 10"),
    ("Seasonality", "Recurring, calendar-linked pattern in sales (month, weekday, festive period). Logic: `seasonal index = average sales of a period / average sales across all periods x 100`; values above 100 mark above-average periods.", "Section 11"),
    ("Pareto 80/20 Rule", "Principle that a small share of customers (about 20%) typically generates a large share of revenue (about 80%). Logic: sort customers by spend in descending order, compute the cumulative revenue share and find the customer percentage at which it reaches 80%.", "Section 12.1"),
    ("Recency (R)", "Days since a customer's most recent purchase, measured at a fixed snapshot date: `R = snapshot date - last purchase date`. Lower values mean more recently active customers.", "Section 12.2"),
    ("Frequency (F)", "Number of purchase occasions of a customer in the observation window: `F = number of distinct orders per customer`. Higher values mean more loyal customers.", "Section 12.2"),
    ("Monetary (M)", "Total value spent by a customer in the observation window: `M = sum of transaction revenue per customer`. Higher values mean more valuable customers.", "Section 12.2"),
    ("Cohort Retention", "Percentage of customers acquired in the same period (cohort, for example first-purchase month) who are active again in later periods: `Retention(cohort, k) = active customers in period k / cohort size x 100`.", "Section 12.3"),
    ("Market Basket Analysis", "Technique that finds products frequently bought together through association rules of the form `A -> B` (customers who buy A also tend to buy B).", "Section 12.4"),
    ("Support", "Share of baskets that contain an itemset: `Support(A) = baskets containing A / total baskets`. For a rule, `Support(A -> B)` counts baskets containing both A and B.", "Section 12.4"),
    ("Confidence", "Likelihood of buying B given that A is bought: `Confidence(A -> B) = Support(A and B) / Support(A)`.", "Section 12.4"),
    ("Lift", "Strength of an association relative to independence: `Lift(A -> B) = Confidence(A -> B) / Support(B)`. A lift above 1 means positive association, equal to 1 means independence, below 1 means negative association.", "Section 12.4"),
    ("K-Means Clustering", "Unsupervised algorithm that partitions customers into `k` clusters by repeatedly assigning each customer to the nearest centroid and recomputing the centroids, so that the within-cluster sum of squares (WCSS) is minimised.", "Section 12.5"),
    ("WCSS / Inertia", "Within-Cluster Sum of Squares: `WCSS = sum of (distance from each point to its own cluster centroid)^2`. It always decreases as `k` grows, so the elbow of the WCSS-versus-k curve is used to choose `k`.", "Section 12.5"),
    ("Silhouette Score", "Cluster-quality measure per point, `s = (b - a) / max(a, b)`, where `a` is the mean distance to points in its own cluster and `b` is the mean distance to points in the nearest other cluster. It ranges from -1 to +1; the average over all points is reported, and higher values mean better-separated clusters.", "Section 12.5"),
    ("StandardScaler", "Feature scaling to zero mean and unit variance, `z = (x - mean) / standard deviation`, so that large-scale features (for example Monetary) do not dominate distance-based methods such as K-Means and PCA.", "Section 12.5"),
    ("PCA (Principal Component Analysis)", "Linear dimensionality-reduction technique that projects standardised features onto orthogonal directions (principal components) of maximum variance; used here to visualise multi-dimensional clusters in 2-D. `Explained variance ratio of a component = its variance / total variance`.", "Section 12.5"),
    ("Customer Lifetime Value (CLV)", "Expected value a customer generates over the whole relationship: `CLV = expected number of future purchases x expected value per purchase` (optionally discounted).", "Section 16.1"),
    ("BG/NBD and Gamma-Gamma Models", "Probabilistic models used together for predictive CLV: BG/NBD predicts future purchase counts and the probability that a customer is still active from frequency, recency and customer age; Gamma-Gamma predicts the average order value.", "Section 16.1"),
    ("Churn", "Customer attrition, i.e., a customer stops purchasing. An operational definition (for example, no purchase within N days of the snapshot date) must be fixed before modelling.", "Section 16.1"),
    ("A/B Testing", "Randomised controlled experiment that compares a control group (A) with a treatment group (B) on a pre-defined success metric, using a statistical hypothesis test to judge whether the observed difference is real.", "Section 16.1"),
    ("Export Manifest", "Verification table printed after the CSV export, listing each exported file with its name, relative path, row count and size in KB.", "Section 16.2"),
]

# ---- 6.10  Section 17: references (APA 7th edition) ---------------------------------------------
REFERENCES = [
    "Agrawal, R., Imieli\u0144ski, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. In *Proceedings of the 1993 ACM SIGMOD International Conference on Management of Data* (pp. 207-216). Association for Computing Machinery. https://doi.org/10.1145/170035.170072",
    "Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms* (pp. 1027-1035). Society for Industrial and Applied Mathematics.",
    "Fader, P. S., & Hardie, B. G. S. (2007). How to project customer retention. *Journal of Interactive Marketing, 21*(1), 76-90. https://doi.org/10.1002/dir.20074",
    "Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). RFM and CLV: Using iso-value curves for customer base analysis. *Journal of Marketing Research, 42*(4), 415-430. https://doi.org/10.1509/jmkr.2005.42.4.415",
    "Hughes, A. M. (1994). *Strategic database marketing: The masterplan for starting and managing a profitable, customer-based marketing program*. Probus Publishing.",
    "Koch, R. (1998). *The 80/20 principle: The secret of achieving more with less*. Currency/Doubleday.",
    "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In L. M. Le Cam & J. Neyman (Eds.), *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281-297). University of California Press.",
    "Matplotlib Development Team. (n.d.). *Matplotlib documentation*. Retrieved October 31, 2026, from https://matplotlib.org/stable/",
    "pandas Development Team. (n.d.). *pandas documentation*. Retrieved October 31, 2026, from https://pandas.pydata.org/docs/",
    "Rahm, E., & Do, H. H. (2000). Data cleaning: Problems and current approaches. *Bulletin of the IEEE Computer Society Technical Committee on Data Engineering, 23*(4), 3-13.",
    "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53-65. https://doi.org/10.1016/0377-0427(87)90125-7",
    "scikit-learn Developers. (n.d.). *sklearn.cluster.KMeans* [Documentation]. Retrieved October 31, 2026, from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
    "Tukey, J. W. (1977). *Exploratory data analysis*. Addison-Wesley.",
    "Waskom, M. L. (n.d.). *seaborn: Statistical data visualization* [Documentation]. Retrieved October 31, 2026, from https://seaborn.pydata.org/",
    "Wickham, H. (2014). Tidy data. *Journal of Statistical Software, 59*(10), 1-23. https://doi.org/10.18637/jss.v059.i10",
]
REFERENCE_USE = [
    ("Exploratory data analysis", "Tukey (1977)", "Philosophy of examining the data before modelling", "1, 4, 7-11"),
    ("Data cleaning and tidy data", "Rahm & Do (2000); Wickham (2014)", "Classes of data errors; tidy table structure", "4-6"),
    ("Pareto principle (80/20 rule)", "Koch (1998)", "Concentration of revenue in a few customers", "7-8, 12.1"),
    ("RFM segmentation", "Hughes (1994); Fader et al. (2005)", "Definition of RFM; link between RFM and customer lifetime value", "12.2"),
    ("Cohort retention", "Fader & Hardie (2007)", "Retention by customer group over time", "12.3"),
    ("Market basket analysis", "Agrawal et al. (1993)", "Association rules; support, confidence, lift", "12.4"),
    ("K-Means clustering", "MacQueen (1967); Arthur & Vassilvitskii (2007); Rousseeuw (1987); scikit-learn Developers (n.d.)", "Algorithm, initialisation, cluster validation, implementation", "12.5"),
    ("Python libraries", "pandas Development Team (n.d.); Matplotlib Development Team (n.d.); Waskom (n.d.); scikit-learn Developers (n.d.)", "Official documentation of the functions used", "2-16"),
]
CITATION_GUIDE = [
    "**In the text:** one author (Hughes, 1994); two authors (Fader & Hardie, 2007); three or more authors (Fader et al., 2005); documentation without a date (pandas Development Team, n.d.).",
    "**A source that is not listed:** send its full details to Member 1, who adds it here so that the list stays consistent and alphabetical.",
    "**Wording:** paraphrase and cite; do not copy sentences from the sources.",
]


# ===================================================================================================
# 7.  DATA LAYER  -  Appendix A / Section 3: synthetic data generator  (Member 2 - Aditi Sharma)
# ===================================================================================================
# The function below is the team's generator, reproduced unchanged.  It is used (a) as the automatic
# fallback when no CSV files are found / uploaded and (b) as the "reference raw data" for Sections 4-5.

def generate_retail_transactions(seed=RANDOM_STATE):
    """Build the synthetic RAW retail-transactions table (df_raw).

    Grain   : one row = one product line inside a transaction (basket).
    Columns : transaction_id, customer_id, transaction_date, product_name,
              category, quantity, unit_price, total_amount
    Design  : 1,000 customers | 2024-01-01 to 2025-12-31 | 1-4 distinct products per
              transaction | Pareto-skewed spend | festive seasonality | repeat buyers.
    Impurities (duplicates, missing IDs / dates, returns) are APPENDED as extra rows,
    so removing them later leaves a fully clean and reconciled dataset.
    A local random generator is used, so no global random state is touched.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    N_CUST, TARGET_TXN = 1000, 2500
    START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
    N_DUP, N_NO_CUST, N_NO_DATE, N_RETURN = 120, 60, 40, 90
    SIGMA, LIFE_MEAN = 1.00, 700.0                     # spread of customer value / average active lifetime (days)
    TIER_RATE = [1.60, 2.20, 4.00, 9.00]               # relative repeat-order rate: low | medium | high | elite customers
    P_EXTRA = [0.28, 0.45, 0.65, 0.85]                 # chance of each extra basket slot, by customer tier
    QTY_BOOST = [1.00, 1.30, 1.80, 2.50]               # bigger customers buy more units per line
    P_COMPANION = 0.80                                 # chance an extra slot is filled by a bundled companion
    OLD_SHARE = 0.40                                   # share of customers already active when the data starts
    COLS = ["transaction_id", "customer_id", "transaction_date", "product_name",
            "category", "quantity", "unit_price", "total_amount"]

    # ------------------------------------------------------------------
    # 1. Product catalogue: (name, category, list price in Rs, popularity, max units per line)
    # ------------------------------------------------------------------
    catalog = [
        ("Smartphone 5G 128GB",    "Electronics",    18999, 1.00, 1),
        ("Laptop 15.6 inch",       "Electronics",    54999, 0.35, 1),
        ("LED Smart TV 43 inch",   "Electronics",    27999, 0.30, 1),
        ("Wireless Earbuds",       "Electronics",     2499, 1.50, 2),
        ("Smart Watch",            "Electronics",     3999, 0.90, 1),
        ("Bluetooth Speaker",      "Electronics",     1999, 1.10, 2),
        ("Power Bank 10000mAh",    "Electronics",     1299, 1.30, 2),
        ("Fast Charger 33W",       "Electronics",      799, 1.20, 2),
        ("Phone Case",             "Electronics",      399, 1.50, 2),
        ("Screen Protector",       "Electronics",      249, 1.50, 2),
        ("Wireless Mouse",         "Electronics",      699, 1.00, 2),
        ("Laptop Bag",             "Electronics",     1299, 0.60, 1),
        ("Men's Cotton T-Shirt",   "Apparel",          599, 1.80, 3),
        ("Slim Fit Jeans",         "Apparel",         1799, 1.30, 2),
        ("Formal Shirt",           "Apparel",         1299, 1.00, 2),
        ("Women's Kurti",          "Apparel",          999, 1.50, 3),
        ("Running Shoes",          "Apparel",         2999, 0.90, 2),
        ("Sports Socks 3-Pack",    "Apparel",          349, 1.20, 3),
        ("Winter Jacket",          "Apparel",         3499, 0.60, 1),
        ("Leather Belt",           "Apparel",          499, 0.80, 2),
        ("Silk Saree",             "Apparel",         4499, 0.50, 1),
        ("Designer Lehenga",       "Apparel",        12999, 0.25, 1),
        ("Non-Stick Frying Pan",   "Home & Kitchen",   899, 1.20, 2),
        ("Spatula & Ladle Set",    "Home & Kitchen",   349, 1.00, 2),
        ("Pressure Cooker 5L",     "Home & Kitchen",  1899, 0.80, 1),
        ("Mixer Grinder",          "Home & Kitchen",  3299, 0.60, 1),
        ("Air Fryer",              "Home & Kitchen",  5999, 0.40, 1),
        ("Bed Sheet Set",          "Home & Kitchen",  1299, 0.90, 2),
        ("Cotton Bath Towel",      "Home & Kitchen",   499, 1.00, 3),
        ("Storage Container Set",  "Home & Kitchen",   799, 1.10, 2),
        ("Microwave Oven 25L",     "Home & Kitchen",  8999, 0.35, 1),
        ("Washing Machine 7kg",    "Home & Kitchen", 24999, 0.22, 1),
        ("Face Wash",              "Beauty",           249, 1.50, 3),
        ("Moisturizer",            "Beauty",           449, 1.20, 2),
        ("Sunscreen SPF 50",       "Beauty",           399, 1.00, 2),
        ("Lipstick",               "Beauty",           499, 1.10, 3),
        ("Shampoo",                "Beauty",           349, 1.50, 3),
        ("Conditioner",            "Beauty",           349, 0.90, 2),
        ("Hair Oil",               "Beauty",           199, 1.10, 3),
        ("Face Serum",             "Beauty",           799, 0.60, 2),
        ("Perfume Gift Set",       "Beauty",          3499, 0.45, 2),
        ("Basmati Rice 5kg",       "Groceries",        549, 1.40, 3),
        ("Whole Wheat Atta 10kg",  "Groceries",        449, 1.40, 3),
        ("Toor Dal 1kg",           "Groceries",        169, 1.20, 4),
        ("Sunflower Oil 1L",       "Groceries",        149, 1.20, 4),
        ("Tea Powder 500g",        "Groceries",        249, 1.20, 4),
        ("Sugar 1kg",              "Groceries",         49, 1.10, 5),
        ("Biscuits Family Pack",   "Groceries",         99, 1.50, 5),
        ("Instant Noodles 5-Pack", "Groceries",         85, 1.20, 5),
        ("Dry Fruits Gift Box",    "Groceries",       1999, 0.50, 2),
    ]
    names = np.array([c[0] for c in catalog])
    cats = np.array([c[1] for c in catalog])
    price = np.array([c[2] for c in catalog], dtype=float)
    pop = np.array([c[3] for c in catalog], dtype=float)
    max_q = np.array([c[4] for c in catalog])
    n_prod = len(catalog)
    cat_list = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Groceries"]
    cat_idx = np.array([cat_list.index(c) for c in cats])
    qty_lam = np.array([0.15, 0.45, 0.30, 0.40, 0.90])          # extra units per line, by category
    p_tier = np.where(price >= 10000, 2, np.where(price >= 1500, 1, 0))   # 0 budget | 1 mid | 2 premium
    tier_w = np.array([[1.00, 0.40, 0.02],                       # low-value customers rarely buy premium
                       [1.00, 0.85, 0.12],
                       [0.85, 1.20, 1.00],
                       [0.70, 1.20, 2.00]])                      # elite customers: mid and premium items

    # Bundles that create strong association rules: product -> [(companion, weight), ...]
    bundles = {
        "Smartphone 5G 128GB":   [("Phone Case", .55), ("Screen Protector", .50), ("Fast Charger 33W", .30), ("Wireless Earbuds", .25)],
        "Phone Case":            [("Screen Protector", .60)],
        "Screen Protector":      [("Phone Case", .60)],
        "Power Bank 10000mAh":   [("Fast Charger 33W", .30)],
        "Laptop 15.6 inch":      [("Laptop Bag", .50), ("Wireless Mouse", .55)],
        "Wireless Mouse":        [("Laptop Bag", .20)],
        "Slim Fit Jeans":        [("Men's Cotton T-Shirt", .50), ("Leather Belt", .35)],
        "Formal Shirt":          [("Leather Belt", .40)],
        "Running Shoes":         [("Sports Socks 3-Pack", .60)],
        "Winter Jacket":         [("Slim Fit Jeans", .25)],
        "Non-Stick Frying Pan":  [("Spatula & Ladle Set", .60)],
        "Pressure Cooker 5L":    [("Storage Container Set", .30)],
        "Mixer Grinder":         [("Storage Container Set", .25)],
        "Bed Sheet Set":         [("Cotton Bath Towel", .35)],
        "Microwave Oven 25L":    [("Storage Container Set", .25)],
        "Shampoo":               [("Conditioner", .60), ("Hair Oil", .30)],
        "Conditioner":           [("Shampoo", .30)],
        "Face Wash":             [("Moisturizer", .50), ("Sunscreen SPF 50", .35)],
        "Moisturizer":           [("Sunscreen SPF 50", .40), ("Face Serum", .25)],
        "Basmati Rice 5kg":      [("Toor Dal 1kg", .50), ("Sunflower Oil 1L", .35)],
        "Whole Wheat Atta 10kg": [("Toor Dal 1kg", .35), ("Sunflower Oil 1L", .35)],
        "Tea Powder 500g":       [("Sugar 1kg", .55), ("Biscuits Family Pack", .45)],
        "Biscuits Family Pack":  [("Instant Noodles 5-Pack", .25)],
    }
    pos = {n: i for i, n in enumerate(names)}
    comp = {pos[a]: [(pos[b], w) for b, w in lst] for a, lst in bundles.items()}

    # ------------------------------------------------------------------
    # 2. Calendar: weekday effect + monthly effect + festive peaks (+ sale days)
    # ------------------------------------------------------------------
    days = pd.date_range(START, END, freq="D")
    n_days = len(days)
    month0 = np.asarray(days.month) - 1
    month_mult = np.array([1.05, 0.85, 1.00, 0.95, 0.90, 0.85, 0.80, 1.00, 0.95, 1.15, 1.10, 1.20])
    dow_mult = np.array([0.90, 0.90, 0.95, 1.00, 1.10, 1.30, 1.35])          # Mon ... Sun
    festivals = [("2024-01-26", 3, 0.35), ("2024-03-23", 5, 0.35), ("2024-04-09", 4, 0.25),
                 ("2024-08-14", 5, 0.35), ("2024-10-10", 5, 0.45), ("2024-10-27", 8, 1.00),
                 ("2024-12-26", 6, 0.55), ("2025-01-26", 3, 0.35), ("2025-03-12", 5, 0.35),
                 ("2025-03-29", 4, 0.25), ("2025-08-08", 6, 0.35), ("2025-09-30", 5, 0.45),
                 ("2025-10-16", 8, 1.00), ("2025-12-26", 6, 0.55)]       # (peak day, width, uplift)
    fest = np.zeros(n_days)
    for peak, width, uplift in festivals:
        gap = np.asarray((days - pd.Timestamp(peak)).days, dtype=float)
        fest += uplift * np.exp(-0.5 * (gap / width) ** 2)
    day_w = month_mult[month0] * dow_mult[np.asarray(days.dayofweek)] * (1.0 + fest)
    day_w = day_w / day_w.sum()
    sale = fest > 0.25                                                   # festive-sale days (discounts possible)

    cat_boost = {"Electronics": {0: 1.2, 7: 1.2, 9: 1.5, 10: 1.6, 11: 1.2},
                 "Apparel": {0: 1.1, 2: 1.3, 9: 1.5, 10: 1.5, 11: 1.4},
                 "Home & Kitchen": {0: 1.1, 9: 1.5, 10: 1.6},
                 "Beauty": {0: 1.2, 1: 1.2, 7: 1.2, 10: 1.3, 11: 1.4},
                 "Groceries": {2: 1.2, 3: 1.2, 9: 1.4, 10: 1.5}}
    prod_boost = {"Designer Lehenga": {0: 2.0, 1: 2.0, 9: 2.0, 10: 2.5, 11: 3.0},
                  "Silk Saree": {9: 1.8, 10: 2.2, 11: 1.5},
                  "Dry Fruits Gift Box": {9: 3.0, 10: 4.0, 11: 1.5},
                  "Perfume Gift Set": {1: 2.0, 10: 2.0, 11: 2.5},
                  "Winter Jacket": {0: 4.0, 1: 3.0, 2: 1.0, 3: 0.3, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2, 8: 0.3, 9: 1.2, 10: 3.0, 11: 4.0},
                  "Sunscreen SPF 50": {0: 0.5, 2: 1.8, 3: 2.2, 4: 2.5, 5: 2.2, 6: 1.5, 10: 0.6, 11: 0.5}}
    season = np.ones((n_prod, 12))                                       # product x month demand multiplier
    for c, mult in cat_boost.items():
        for m, f in mult.items():
            season[cats == c, m] *= f
    for p, mult in prod_boost.items():
        for m, f in mult.items():
            season[names == p, m] *= f

    # ------------------------------------------------------------------
    # 3. Customers: heavy-tailed value score (Pareto), preferences, lifetime, repeat pattern
    # ------------------------------------------------------------------
    cust_ids = [f"CUST{i:04d}" for i in range(1, N_CUST + 1)]
    z = rng.lognormal(0.0, SIGMA, N_CUST)
    z = z / z.mean()                                                     # customer value score
    tier = np.digitize(z, np.quantile(z, [0.50, 0.80, 0.95]))            # 0 low | 1 medium | 2 high | 3 elite
    pref = rng.dirichlet([1.5, 1.5, 1.2, 1.0, 1.8], size=N_CUST)         # category preference per customer
    W = pop * pref[:, cat_idx] * tier_w[tier][:, p_tier]                 # customer x product base weights
    fav = [rng.choice(n_prod, 3, replace=False, p=W[i] / W[i].sum()) for i in range(N_CUST)]

    n_old = int(OLD_SHARE * N_CUST)                                      # existing base buys early in the window
    early = np.arange(90)
    acq = np.concatenate([rng.choice(early, n_old, p=day_w[early] / day_w[early].sum()),
                          rng.choice(n_days, N_CUST - n_old, p=day_w)])   # day of first purchase
    rng.shuffle(acq)
    life = rng.exponential(LIFE_MEAN, N_CUST) * z ** 0.35                # active lifetime after first purchase
    last = np.minimum(acq + life.astype(int), n_days - 1)                # last possible purchase day
    lam = np.array(TIER_RATE)[tier] * z ** 0.15 * (last - acq) / 365.0
    lam = lam * (TARGET_TXN - N_CUST) / lam.sum()                        # expected repeat orders per customer
    n_txn = 1 + rng.poisson(lam)                                         # 1 = one-time buyer

    ev_cust, ev_day = [], []
    for i in range(N_CUST):
        d_i = [int(acq[i])]
        window = np.arange(acq[i] + 1, last[i] + 1)
        m = min(int(n_txn[i]) - 1, len(window))
        if m > 0:
            pw = day_w[window]
            d_i += [int(x) for x in rng.choice(window, size=m, replace=False, p=pw / pw.sum())]
        ev_cust += [i] * len(d_i)
        ev_day += d_i
    ev_cust, ev_day = np.array(ev_cust), np.array(ev_day)
    order = np.lexsort((rng.random(len(ev_day)), ev_day))                # chronological, random tie-break
    ev_cust, ev_day = ev_cust[order], ev_day[order]

    # ------------------------------------------------------------------
    # 4. Baskets: 1-4 distinct products per transaction, bundles + favourites
    # ------------------------------------------------------------------
    id_box = [100001]

    def new_ids(n):
        ids = [f"TXN{x}" for x in range(id_box[0], id_box[0] + n)]
        id_box[0] += n
        return ids

    def make_line(p, d, boost=1.0):
        qty = int(min(max_q[p], 1 + rng.poisson(qty_lam[cat_idx[p]] * boost)))
        disc = 0.90 if (sale[d] and rng.random() < 0.5) else 1.0
        unit = float(np.round(price[p] * disc))
        return names[p], cats[p], qty, unit

    rows = []
    for i, d in zip(ev_cust, ev_day):
        w = W[i] * season[:, month0[d]]
        w[fav[i]] *= 3.0                                                 # repeat purchase of favourites
        k = 1 + rng.binomial(3, P_EXTRA[tier[i]])                        # basket size 1-4
        basket = [int(rng.choice(n_prod, p=w / w.sum()))]
        while len(basket) < k:
            cand = {}
            for b in basket:
                for q, wt in comp.get(b, []):
                    if q not in basket:
                        cand[q] = max(cand.get(q, 0.0), wt)
            if cand and rng.random() < P_COMPANION:
                qs = list(cand)
                wt = np.array([cand[q] for q in qs])
                basket.append(qs[int(rng.choice(len(qs), p=wt / wt.sum()))])
            else:
                w2 = w.copy()
                w2[basket] = 0.0
                basket.append(int(rng.choice(n_prod, p=w2 / w2.sum())))
        tid = new_ids(1)[0]
        for p in basket:
            nm, ct, qty, unit = make_line(p, d, QTY_BOOST[tier[i]])
            rows.append((tid, cust_ids[i], days[d], nm, ct, qty, unit, qty * unit, "clean"))
    _base = pd.DataFrame(rows, columns=COLS + ["_kind"])                 # 100% valid, positive rows
    n_base = len(_base)

    # ------------------------------------------------------------------
    # 5. Intentional impurities, APPENDED as extra rows
    # ------------------------------------------------------------------
    _dup = _base.iloc[rng.choice(n_base, N_DUP, replace=False)].copy()   # exact duplicate rows
    _dup["_kind"] = "dup"

    def extra_orders(n, kind, customers):
        d_x = rng.choice(n_days, n, p=day_w)
        p_x = rng.choice(n_prod, n, p=pop / pop.sum())
        lines = [make_line(int(p), int(d)) for p, d in zip(p_x, d_x)]
        return pd.DataFrame({"transaction_id": new_ids(n), "customer_id": customers,
                             "transaction_date": days[d_x],
                             "product_name": [x[0] for x in lines], "category": [x[1] for x in lines],
                             "quantity": [x[2] for x in lines], "unit_price": [x[3] for x in lines],
                             "total_amount": [x[2] * x[3] for x in lines], "_kind": kind})

    _guest = extra_orders(N_NO_CUST, "guest", cust_ids[0])               # customer_id blanked later
    _nodate = extra_orders(N_NO_DATE, "nodate", [cust_ids[j] for j in rng.choice(N_CUST, N_NO_DATE)])

    _ret = _base.iloc[rng.choice(n_base, N_RETURN, replace=False)].copy()   # returns / cancellations
    q0 = _ret["quantity"].to_numpy()
    r_qty = np.where(rng.random(N_RETURN) < 0.70, q0, rng.integers(1, q0 + 1))
    _ret["transaction_id"] = new_ids(N_RETURN)
    delay = pd.to_timedelta(rng.integers(2, 31, N_RETURN), unit="D").to_numpy()   # return happens 2-30 days later
    _ret["transaction_date"] = (_ret["transaction_date"] + delay).clip(upper=END)
    _ret["quantity"] = -r_qty
    _ret["total_amount"] = _ret["quantity"] * _ret["unit_price"]
    _ret["_kind"] = "return"

    _raw = pd.concat([_base, _dup, _guest, _nodate, _ret], ignore_index=True)
    _raw = _raw.sort_values(["transaction_date", "transaction_id"], kind="mergesort").reset_index(drop=True)
    _raw.loc[_raw["_kind"] == "guest", "customer_id"] = np.nan           # missing customer_id
    _raw.loc[_raw["_kind"] == "nodate", "transaction_date"] = pd.NaT     # missing transaction_date
    _raw = _raw[COLS].copy()
    _raw["transaction_date"] = _raw["transaction_date"].astype("datetime64[ns]")
    _raw["quantity"] = _raw["quantity"].astype("int64")
    _raw["unit_price"] = _raw["unit_price"].astype("float64")
    _raw["total_amount"] = _raw["total_amount"].astype("float64")
    return _raw


# ===================================================================================================
# 8.  DATA LAYER  -  Section 5: cleaning pipeline and cleaning log  (Member 4 - Samiya Khan)
# ===================================================================================================
ID_COLS = ["transaction_id", "customer_id"]
TEXT_COLS = ["product_name", "category"]
NULL_TOKENS = {"", "nan", "none", "null", "nat", "n/a", "na"}          # blank / placeholder text


def is_blank(series):
    """True where a value is missing: NaN / None / NaT, or an empty or 'null'-like text token."""
    mask = series.isna()
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        mask = mask | series.astype(str).str.strip().str.lower().isin(NULL_TOKENS)
    return mask


def trim_text(series):
    """Trim leading / trailing spaces; missing values stay missing (they never become the text 'nan')."""
    if pd.api.types.is_float_dtype(series):               # IDs stored as floats, e.g. 1001.0 -> "1001"
        series = series.map(lambda v: str(int(v)) if pd.notna(v) and float(v).is_integer() else v)
    obj = series.astype(object)
    return obj.where(obj.isna(), obj.astype(str).str.strip())


def raw_issue_profile(df_raw):
    """Table 5.2 - data-quality profile of the untouched df_raw (every issue counted independently)."""
    n_raw = len(df_raw)
    qty = pd.to_numeric(df_raw["quantity"], errors="coerce")
    prc = pd.to_numeric(df_raw["unit_price"], errors="coerce")
    amt = pd.to_numeric(df_raw["total_amount"], errors="coerce")
    date_blank = is_blank(df_raw["transaction_date"])
    date_parsed = pd.to_datetime(df_raw["transaction_date"], errors="coerce")
    padded_any = np.zeros(n_raw, dtype=bool)                # rows with stray spaces in any ID / text column
    for col in ID_COLS + TEXT_COLS:
        txt = df_raw[col].astype(object)
        padded_any |= (txt.notna() & (txt.astype(str) != txt.astype(str).str.strip())).to_numpy()
    issues = [
        ("Exact duplicate rows", "all columns", int(df_raw.duplicated().sum()), 1),
        ("Missing transaction_id", "transaction_id", int(is_blank(df_raw["transaction_id"]).sum()), 2),
        ("Missing customer_id", "customer_id", int(is_blank(df_raw["customer_id"]).sum()), 2),
        ("Missing transaction_date", "transaction_date", int(date_blank.sum()), 2),
        ("Unparseable transaction_date", "transaction_date", int((date_parsed.isna() & ~date_blank).sum()), 3),
        ("Leading/trailing spaces in text", "IDs, product_name, category", int(padded_any.sum()), 3),
        ("Negative quantity (returns)", "quantity", int((qty < 0).sum()), 4),
        ("Zero quantity (cancellations)", "quantity", int((qty == 0).sum()), 4),
        ("Zero or negative unit_price", "unit_price", int((prc <= 0).sum()), 4),
        ("Zero or negative total_amount", "total_amount", int((amt <= 0).sum()), 4),
        ("total_amount differs from round(quantity * unit_price, 2)", "total_amount",
         int(((amt - (qty * prc).round(2)).abs() > 0.005).sum()), 5),
    ]
    profile = pd.DataFrame(issues, columns=["issue", "columns", "rows_affected", "handled_in_step"])
    profile.insert(3, "pct_of_raw_rows", (100 * profile["rows_affected"] / max(n_raw, 1)).round(2))
    return profile


def clean_raw_data(df_raw):
    """Section 5 pipeline: five logged steps  ->  (df_clean, cleaning_log, df_returns).

    df_raw is never modified; every step works on a copy and writes rows_before / rows_after to the log.
    """
    log_rows = []

    def log_step(step_no, stage, issue, action, before, after):
        removed = int(before) - int(after)
        pct_removed = round(100.0 * removed / before, 2) if before else 0.0
        log_rows.append({"step_no": int(step_no), "cleaning_stage": stage, "issue_addressed": issue,
                         "action_taken": action, "rows_before": int(before), "rows_after": int(after),
                         "rows_removed": removed, "pct_removed": pct_removed})

    work = df_raw.copy().reset_index(drop=True)

    # ---- Step 1: remove exact duplicate rows ----
    before = len(work)
    work = work.drop_duplicates()
    log_step(1, "Duplicate removal",
             "Exact duplicate rows (all columns identical) would double-count revenue",
             "drop_duplicates(): kept the first occurrence of every repeated row", before, len(work))

    # ---- Step 2: remove rows with a missing transaction_id, customer_id or transaction_date ----
    before = len(work)
    missing_key = (is_blank(work["transaction_id"]) | is_blank(work["customer_id"])
                   | is_blank(work["transaction_date"]))
    work = work.loc[~missing_key].copy()
    log_step(2, "Missing ID / date removal",
             "Rows without transaction_id, customer_id or transaction_date cannot be linked to a customer or day",
             "Dropped rows where any of the three key fields is NaN / None / blank (identifiers are never imputed)",
             before, len(work))

    # ---- Step 3: parse dates (errors="coerce"), drop invalid dates, trim text columns ----
    before = len(work)
    parsed = pd.to_datetime(work["transaction_date"], errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):                       # keep naive timestamps
        parsed = parsed.dt.tz_localize(None)
    work["transaction_date"] = parsed.astype("datetime64[ns]")
    n_bad_dates = int(work["transaction_date"].isna().sum())
    work = work.loc[work["transaction_date"].notna()].copy()
    n_trimmed = 0
    for col in ID_COLS + TEXT_COLS:                                         # .str.strip() on every ID / text column
        trimmed = trim_text(work[col])
        n_trimmed += int((trimmed.notna() & (trimmed.astype(str) != work[col].astype(str))).sum())
        work[col] = trimmed
    n_filled = 0                                                            # blank labels are kept as sales
    for col, label in (("product_name", "Unknown Product"), ("category", "Uncategorized")):
        blank = is_blank(work[col])
        n_filled += int(blank.sum())
        work.loc[blank, col] = label
    log_step(3, "Date parsing and text trimming",
             "Dates stored as text (some invalid) and stray leading/trailing spaces in ID and text columns",
             f"pd.to_datetime(errors='coerce') -> datetime64[ns], dropped {n_bad_dates} invalid dates; "
             f".str.strip() on ID/text columns ({n_trimmed} cells changed)"
             + (f"; {n_filled} blank labels set to 'Unknown'" if n_filled else ""),
             before, len(work))

    # ---- Step 4: move returns / cancellations (<= 0 values) out of the analysis table ----
    before = len(work)
    qty = pd.to_numeric(work["quantity"], errors="coerce")
    prc = pd.to_numeric(work["unit_price"], errors="coerce")
    amt = pd.to_numeric(work["total_amount"], errors="coerce")
    valid_sale = (qty > 0) & (prc > 0) & ((qty * prc).round(2) > 0) & ~(amt <= 0)
    reason = np.select([qty < 0, qty == 0, prc <= 0, amt <= 0],
                       ["Return (negative quantity)", "Cancellation (zero quantity)",
                        "Zero or negative unit price", "Zero or negative total amount"],
                       default="Non-numeric or zero-rounded value")
    df_returns = work.loc[~valid_sale].copy()                               # saved separately, NOT deleted
    df_returns["exclusion_reason"] = reason[(~valid_sale).to_numpy()]
    work = work.loc[valid_sale].assign(quantity=qty[valid_sale], unit_price=prc[valid_sale],
                                       total_amount=amt[valid_sale])
    log_step(4, "Returns and cancellations",
             "Zero or negative quantity, unit_price or total_amount (returns, cancellations) would distort spend and revenue",
             f"Moved {len(df_returns)} rows to _df_returns (with exclusion_reason); "
             f"df_clean keeps strictly positive sales only", before, len(work))

    # ---- Step 5: verify total_amount, enforce dtypes, reset the index ----
    before = len(work)
    df_clean = work[CLEAN_COLS].copy()
    for col in ID_COLS + TEXT_COLS:                                         # plain strings (no missing values left)
        df_clean[col] = df_clean[col].astype(str)
    df_clean["transaction_date"] = df_clean["transaction_date"].astype("datetime64[ns]")
    if (df_clean["quantity"] % 1 == 0).all():                               # whole-number quantities -> int64
        df_clean["quantity"] = df_clean["quantity"].astype("int64")
    df_clean["unit_price"] = df_clean["unit_price"].astype("float64")
    old_total = df_clean["total_amount"].astype("float64")
    df_clean["total_amount"] = (df_clean["quantity"] * df_clean["unit_price"]).round(2)
    gap = (old_total - df_clean["total_amount"]).abs()
    n_corrected = int(((gap > 0.005) | old_total.isna()).sum())            # wrong or empty totals
    n_rounded = int((old_total != df_clean["total_amount"]).sum()) - n_corrected   # float residue only
    df_clean = df_clean.reset_index(drop=True)
    log_step(5, "Amount verification and index reset",
             "total_amount must equal quantity x unit_price rounded to 2 decimals so that revenue reconciles downstream",
             f"Recomputed total_amount = (quantity * unit_price).round(2): {n_corrected} inconsistent or empty values "
             f"corrected, {n_rounded} float-rounding residues cleaned; dtypes enforced; reset_index(drop=True)",
             before, len(df_clean))

    cleaning_log = pd.DataFrame(log_rows, columns=LOG_COLS)
    return df_clean, cleaning_log, df_returns.reset_index(drop=True)


def returns_summary(df_returns):
    """Table 5.4 - what Step 4 set aside (returns, cancellations ...)."""
    if df_returns is None or len(df_returns) == 0:
        return pd.DataFrame(columns=["exclusion_reason", "rows", "pct_of_excluded_rows", "net_quantity", "net_amount"])
    view = pd.DataFrame({
        "exclusion_reason": df_returns["exclusion_reason"].to_numpy(),
        "quantity": pd.to_numeric(df_returns["quantity"], errors="coerce").to_numpy(),
        "total_amount": pd.to_numeric(df_returns["total_amount"], errors="coerce").to_numpy(),
    })
    summary = (view.groupby("exclusion_reason", as_index=False)
               .agg(rows=("quantity", "size"), net_quantity=("quantity", "sum"), net_amount=("total_amount", "sum"))
               .sort_values("rows", ascending=False).reset_index(drop=True))
    summary.insert(2, "pct_of_excluded_rows", (100 * summary["rows"] / len(df_returns)).round(2))
    total = pd.DataFrame([{"exclusion_reason": "Total", "rows": int(summary["rows"].sum()),
                           "pct_of_excluded_rows": 100.0, "net_quantity": summary["net_quantity"].sum(),
                           "net_amount": round(float(summary["net_amount"].sum()), 2)}])
    return pd.concat([summary, total], ignore_index=True)


def validate_clean(df_clean, df_raw=None, cleaning_log=None, df_returns=None):
    """Table 5.5 - validation of df_clean against the schema contract (PASS / REVIEW / FAIL per check)."""
    checks = []

    def add(name, expected, observed, ok, warn_only=False):
        checks.append({"check": name, "expected": expected, "observed": observed,
                       "status": "PASS" if bool(ok) else ("REVIEW" if warn_only else "FAIL")})

    cols_ok = list(df_clean.columns) == CLEAN_COLS
    add("Column names and order", "8 columns exactly as in the contract",
        f"{len(df_clean.columns)} columns" if cols_ok else str(list(df_clean.columns)), cols_ok)
    for col in ID_COLS + TEXT_COLS:
        s = df_clean[col]
        n_null = int(s.isna().sum())
        n_bad = int(((s.astype(str) != s.astype(str).str.strip()) | (s.astype(str).str.strip() == "")).sum())
        is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
        add(f"{col}: text, no nulls / blanks / stray spaces", "string/object, 0 nulls, 0 blanks",
            f"{s.dtype}, {n_null} nulls, {n_bad} blank or padded", is_text and n_null == 0 and n_bad == 0)
    s = df_clean["transaction_date"]
    add("transaction_date: datetime, no nulls", "datetime64, 0 nulls", f"{s.dtype}, {int(s.isna().sum())} nulls",
        pd.api.types.is_datetime64_any_dtype(s) and not s.isna().any())
    for col in ("quantity", "unit_price", "total_amount"):
        s = df_clean[col]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        dtype_ok = numeric if col == "quantity" else pd.api.types.is_float_dtype(s)
        smin = float(s.min()) if len(s) else float("nan")
        add(f"{col}: numeric and strictly > 0", "int/float (float for prices), min > 0, 0 nulls",
            f"{s.dtype}, min = {smin:,.2f}, {int(s.isna().sum())} nulls",
            dtype_ok and not s.isna().any() and bool((s > 0).all()))
    calc = (df_clean["quantity"] * df_clean["unit_price"]).round(2)
    gap = float((df_clean["total_amount"] - calc).abs().max()) if len(df_clean) else 0.0
    add("total_amount = round(quantity * unit_price, 2)", "maximum absolute gap = 0",
        f"maximum absolute gap = {gap:.4f}", bool((df_clean["total_amount"] - calc).abs().max() < 0.005)
        if len(df_clean) else True)
    n_dup = int(df_clean.duplicated().sum())
    add("No exact duplicate rows left", "0 duplicate rows", f"{n_dup} duplicate rows", n_dup == 0, warn_only=True)
    add("Index and size", "RangeIndex 0..n-1 and at least one row",
        f"{type(df_clean.index).__name__}, {len(df_clean):,} rows",
        len(df_clean) > 0 and df_clean.index.equals(pd.RangeIndex(len(df_clean))))
    if df_raw is not None and cleaning_log is not None and len(cleaning_log) > 0:
        removed = int(cleaning_log["rows_removed"].sum())
        add("Row reconciliation", "len(df_raw) = len(df_clean) + rows removed in the log",
            f"{len(df_raw):,} = {len(df_clean):,} + {removed:,}", len(df_raw) == len(df_clean) + removed)
        chain_ok = (list(cleaning_log.columns) == LOG_COLS
                    and int(cleaning_log["rows_before"].iloc[0]) == len(df_raw)
                    and bool((cleaning_log["rows_before"].iloc[1:].to_numpy()
                              == cleaning_log["rows_after"].iloc[:-1].to_numpy()).all())
                    and int(cleaning_log["rows_after"].iloc[-1]) == len(df_clean)
                    and bool((cleaning_log["rows_removed"]
                              == cleaning_log["rows_before"] - cleaning_log["rows_after"]).all()))
        add("cleaning_log integrity", "8 columns, unbroken rows_before / rows_after chain",
            f"{cleaning_log.shape[0]} steps x {cleaning_log.shape[1]} columns", chain_ok)
        if df_returns is not None and (cleaning_log["step_no"] == 4).any():
            step4 = int(cleaning_log.loc[cleaning_log["step_no"] == 4, "rows_removed"].iloc[0])
            add("_df_returns bookkeeping", "rows in _df_returns = rows removed in Step 4",
                f"{len(df_returns):,} = {step4:,}", len(df_returns) == step4)
    return pd.DataFrame(checks)


# ===================================================================================================
# 9.  DATA LAYER  -  Section 4: data-quality audit of the raw table  (Member 3 - Sonam)
# ===================================================================================================
AUDIT_EXPECTED_KIND = {"transaction_id": "text", "customer_id": "text", "transaction_date": "datetime",
                       "product_name": "text", "category": "text", "quantity": "integer",
                       "unit_price": "float", "total_amount": "float"}
AUDIT_KIND_LABEL = {"text": "string (object / str)", "datetime": "datetime64 (ns / us)",
                    "integer": "integer (int64 / int32)", "float": "float (float64 / float32)"}
AUDIT_NULL_TOKENS = {"", "nan", "none", "null", "nat", "n/a"}
AUDIT_TOLERANCE = 0.01                                     # rounding tolerance for quantity x price = total


def _a_blank_mask(series):
    """True for non-null text cells that are empty, whitespace-only or a null-like token."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return pd.Series(False, index=series.index)
    text = series.astype(str).str.strip().str.lower()
    return series.notna() & text.isin(AUDIT_NULL_TOKENS)


def _a_missing_mask(series):
    return series.isna() | _a_blank_mask(series)


def _a_as_num(series):
    """Numeric view of a column (float64); non-numeric entries become NaN."""
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.Series(np.nan, index=series.index, dtype="float64")
        return pd.to_numeric(series, errors="coerce").astype("float64")
    except Exception:
        return pd.Series(np.nan, index=series.index, dtype="float64")


def _a_pct(part, whole):
    return (100.0 * part / whole) if whole else 0.0


def _a_cnt(count, noun):
    count = int(count)
    return f"{count:,} {noun}" + ("" if count == 1 else "s")


def _a_vb(count, singular, plural_form):
    return singular if int(count) == 1 else plural_form


def _a_kind_ok(series, kind):
    types = pd.api.types
    if kind == "text":
        return bool(types.is_object_dtype(series) or types.is_string_dtype(series)
                    or isinstance(series.dtype, pd.CategoricalDtype))
    if kind == "datetime":
        return bool(types.is_datetime64_any_dtype(series))
    if kind == "integer":
        return bool(types.is_integer_dtype(series))
    if kind == "float":
        return bool(types.is_float_dtype(series))
    return False


def _a_dtype_remark(series, kind, ok):
    if ok:
        return "As expected"
    non_null = int(series.notna().sum())
    try:
        if kind == "datetime":
            good = int(pd.to_datetime(series, errors="coerce").notna().sum())
            return f"Stored as {series.dtype}; {good:,} of {non_null:,} non-null values parse as dates"
        if kind in ("integer", "float"):
            good = int(_a_as_num(series).notna().sum())
            note = f"Stored as {series.dtype}; {good:,} of {non_null:,} non-null values are numeric"
            if kind == "integer" and pd.api.types.is_float_dtype(series):
                note += " (float usually results from NaN values)"
            return note
    except Exception:
        pass
    return f"Stored as {series.dtype}; convert to the expected type"


@st.cache_data(show_spinner=False)
def audit_raw(df_raw):
    """Section 4 - read-only audit of the raw table.  Returns every table, count and interpretation text."""
    audit = df_raw.copy()
    n_rows, n_cols = audit.shape
    out = {"n_rows": n_rows, "n_cols": n_cols}

    def get(name):
        if name in audit.columns:
            return audit[name]
        return pd.Series(np.nan, index=audit.index, dtype="float64")

    # ---- 4.1a dimensions ----
    mem_bytes = int(audit.memory_usage(deep=True).sum())
    mem_mb = mem_bytes / 1024 ** 2
    bytes_per_row = mem_bytes / max(n_rows, 1)
    absent = [c for c in CLEAN_COLS if c not in audit.columns]
    extra = [c for c in audit.columns if c not in CLEAN_COLS]
    try:
        dates_valid = pd.to_datetime(get("transaction_date"), errors="coerce")
        date_range = (f"{dates_valid.min():%Y-%m-%d} to {dates_valid.max():%Y-%m-%d}"
                      if dates_valid.notna().any() else "n/a (no valid dates)")
    except Exception:
        date_range = "n/a"
    out["dim_table"] = pd.DataFrame({
        "Metric": ["Rows (transactions)", "Columns", "Total cells", "Memory usage (deep)", "Memory per row",
                   "Expected columns present", "Missing expected columns", "Unexpected extra columns",
                   "Transaction date range (valid dates)"],
        "Value": [f"{n_rows:,}", f"{n_cols}", f"{n_rows * n_cols:,}", f"{mem_mb:.3f} MB",
                  f"{bytes_per_row:,.0f} bytes", f"{len(CLEAN_COLS) - len(absent)} of {len(CLEAN_COLS)}",
                  ", ".join(absent) if absent else "none", ", ".join(map(str, extra)) if extra else "none",
                  date_range]}).set_index("Metric")
    if not absent and not extra:
        schema_txt = (f"All {len(CLEAN_COLS)} expected columns are present and no unexpected column exists, "
                      f"so the delivered structure agrees with the agreed schema.")
    else:
        bits = []
        if absent:
            bits.append("missing expected column(s): " + ", ".join("`" + c + "`" for c in absent))
        if extra:
            bits.append("unexpected column(s): " + ", ".join("`" + str(c) + "`" for c in extra))
        schema_txt = "The delivered structure deviates from the agreed schema (" + "; ".join(bits) + ")."
    out["dim_text"] = [
        f"The raw extract holds {n_rows:,} rows and {n_cols} columns and occupies about {mem_mb:.2f} MB of memory "
        f"(roughly {bytes_per_row:,.0f} bytes per row).", schema_txt,
        "Profiling size and structure before any transformation follows the exploratory habit of looking at the "
        "data first (Tukey, 1977) and the data-analysis step that precedes cleaning (Rahm & Do, 2000)."]

    # ---- 4.1b expected versus actual dtypes ----
    recs = []
    for col in CLEAN_COLS:
        kind = AUDIT_EXPECTED_KIND[col]
        if col in audit.columns:
            s = audit[col]
            ok = _a_kind_ok(s, kind)
            recs.append({"Column": col, "Expected dtype": AUDIT_KIND_LABEL[kind], "Actual dtype": str(s.dtype),
                         "Status": "Match" if ok else "Mismatch", "Unique values": int(s.nunique(dropna=True)),
                         "Remark": _a_dtype_remark(s, kind, ok)})
        else:
            recs.append({"Column": col, "Expected dtype": AUDIT_KIND_LABEL[kind], "Actual dtype": "absent",
                         "Status": "Mismatch", "Unique values": 0, "Remark": "Column not found in df_raw"})
    dtype_table = pd.DataFrame(recs).set_index("Column")
    out["dtype_table"] = dtype_table
    n_ok = int((dtype_table["Status"] == "Match").sum())
    n_bad = len(dtype_table) - n_ok
    if n_bad == 0:
        t1 = f"All {len(dtype_table)} audited columns carry the expected data type, so no type conversion is required before analysis."
        t2 = ("In particular, the date column is a true datetime and the quantity and monetary fields are numeric, "
              "which permits time-based grouping and arithmetic without any parsing.")
    else:
        bad_names = [n for n in dtype_table.index if dtype_table.loc[n, "Status"] != "Match"]
        bad_text = "; ".join("`" + n + "` is " + str(dtype_table.loc[n, "Actual dtype"])
                             + " (expected: " + AUDIT_EXPECTED_KIND[n] + ")" for n in bad_names)
        t1 = f"{n_ok} of {len(dtype_table)} columns match the expected type, while {n_bad} deviate: {bad_text}."
        if any(dtype_table.loc[n, "Actual dtype"] == "absent" for n in bad_names):
            t2 = ("Absent columns cannot be repaired by cleaning and have to be re-supplied by the data source, "
                  "because every analysis that depends on them is blocked.")
        elif "transaction_date" in bad_names:
            t2 = ("Because the date column is not a true datetime, monthly, weekday and seasonal analysis is "
                  "impossible until it has been converted.")
        else:
            t2 = ("These columns have to be cast to their expected types during cleaning, because mis-typed "
                  "fields can silently break arithmetic and grouping.")
    out["dtype_text"] = [t1, t2, "Confirming types explicitly matters because missing integrity constraints at the "
                                 "source are a common cause of schema-level data-quality problems (Rahm & Do, 2000)."]

    # ---- 4.2 missing values ----
    recs, row_has_missing = [], pd.Series(False, index=audit.index)
    for col in audit.columns:
        s = audit[col]
        blank_flags = _a_blank_mask(s)
        n_null, n_blank = int(s.isna().sum()), int(blank_flags.sum())
        row_has_missing = row_has_missing | s.isna() | blank_flags
        recs.append({"Column": col, "Dtype": str(s.dtype), "Missing Count": n_null + n_blank,
                     "Missing %": round(_a_pct(n_null + n_blank, n_rows), 2),
                     "of which Null (NaN/None/NaT)": n_null, "of which Blank / placeholder text": n_blank})
    missing_table = pd.DataFrame(recs).set_index("Column")
    out["missing_table"] = missing_table
    total_missing = int(missing_table["Missing Count"].sum())
    cells_total = max(n_rows * n_cols, 1)
    cols_affected = [n for n in missing_table.index if missing_table.loc[n, "Missing Count"] > 0]
    rows_complete = int((~row_has_missing).sum())
    if total_missing == 0:
        m = [f"No missing value was detected: all {n_cols} columns are fully populated across {n_rows:,} rows.",
             "Completeness therefore poses no obstacle to analysis, and no imputation or row removal is needed on this ground.",
             "Even so, the check was made explicitly because missing data is a basic instance-level problem "
             "(Rahm & Do, 2000) that should always be ruled out before modelling (Tukey, 1977)."]
    else:
        worst_col = missing_table["Missing Count"].idxmax()
        worst_n = int(missing_table.loc[worst_col, "Missing Count"])
        m = [f"{len(cols_affected)} of {n_cols} columns {_a_vb(len(cols_affected), 'contains', 'contain')} missing "
             f"values, amounting to {total_missing:,} empty cells ({_a_pct(total_missing, cells_total):.2f}% of all cells).",
             f"The most affected column is `{worst_col}` with {_a_cnt(worst_n, 'row')} "
             f"({_a_pct(worst_n, n_rows):.2f}%), and {rows_complete:,} rows ({_a_pct(rows_complete, n_rows):.2f}%) "
             f"are complete in every column.",
             "Missing identifiers and dates are classic instance-level data-entry problems (Rahm & Do, 2000); they "
             "are only quantified here and left untouched, in keeping with the look-before-you-model principle of "
             "exploratory analysis (Tukey, 1977)."]
    out["missing_text"] = m

    # ---- 4.3a duplicates and transaction_id integrity ----
    mask_dup = audit.duplicated(keep="first")
    mask_dup_group = audit.duplicated(keep=False)
    n_dup_extra, n_dup_involved = int(mask_dup.sum()), int(mask_dup_group.sum())
    n_dup_records = n_dup_involved - n_dup_extra
    tid_raw = get("transaction_id")
    tid = tid_raw.where(~_a_missing_mask(tid_raw))
    n_tid_missing = int(tid.isna().sum())
    n_tid_unique = int(tid.nunique(dropna=True))
    tid_repeat_rows = tid.notna() & tid.duplicated(keep=False)
    n_tid_repeat_rows = int(tid_repeat_rows.sum())
    n_tid_repeat_ids = int(tid[tid_repeat_rows].nunique())
    tid_first = tid[~mask_dup]
    tid_conflict_rows = tid_first.notna() & tid_first.duplicated(keep=False)
    n_tid_conflict_ids = int(tid_first[tid_conflict_rows].nunique())
    cust_col = get("customer_id")
    owner = pd.DataFrame({"t": tid, "c": cust_col.where(~_a_missing_mask(cust_col)), "d": get("transaction_date")}).dropna(subset=["t"])
    owner_stats = owner.groupby("t").agg(nc=("c", "nunique"), nd=("d", "nunique"))
    n_tid_multi_owner = int(((owner_stats["nc"] > 1) | (owner_stats["nd"] > 1)).sum())
    dup_rows = [("Exact duplicate rows (extra copies)", n_dup_extra, True),
                ("Rows inside duplicate groups (incl. first copy)", n_dup_involved, True),
                ("Distinct records that are duplicated", n_dup_records, False),
                ("Rows with missing transaction_id", n_tid_missing, True),
                ("Distinct non-missing transaction_id values", n_tid_unique, False),
                ("transaction_id values occurring more than once", n_tid_repeat_ids, False),
                ("Rows sharing a repeated transaction_id", n_tid_repeat_rows, True),
                ("Repeated transaction_id with conflicting content", n_tid_conflict_ids, False),
                ("transaction_id shared by more than one customer or date", n_tid_multi_owner, False)]
    out["dup_table"] = pd.DataFrame({
        "Check": [r[0] for r in dup_rows], "Count": [f"{r[1]:,}" for r in dup_rows],
        "% of rows": [f"{_a_pct(r[1], n_rows):.2f}%" if r[2] else "-" for r in dup_rows]}).set_index("Check")
    if n_dup_extra == 0:
        d1 = "No fully identical rows were found, so the data contain no exact duplicate records."
    else:
        d1 = (f"Exact duplicates account for {_a_cnt(n_dup_extra, 'row')} ({_a_pct(n_dup_extra, n_rows):.2f}% of the "
              f"data) and stem from {_a_cnt(n_dup_records, 'distinct record')} that "
              f"{_a_vb(n_dup_records, 'appears', 'appear')} more than once.")
    if n_tid_missing == 0 and n_tid_repeat_ids == 0:
        d2 = "At the transaction_id level, every row carries an ID and every ID is unique, so the key identifies individual transactions reliably."
    else:
        d2 = (f"At the transaction_id level, {n_tid_missing:,} rows have no ID, {n_tid_repeat_ids:,} IDs occur on more than one row and "
              f"{n_tid_multi_owner:,} IDs are shared by more than one customer or date.")
        if n_tid_repeat_ids > 0 and n_tid_multi_owner == 0:
            d2 += (" Repeated IDs are expected here because one transaction_id is a basket with one row per product line; since no ID is "
                   "shared by two customers or dates, the key still identifies exactly one transaction.")
        elif n_tid_multi_owner > 0:
            d2 += " IDs shared by several customers or dates are a genuine integrity problem and have to be investigated."
    out["dup_text"] = [d1, d2, "A repeated key is a uniqueness violation at schema level, whereas duplicated records "
                                "are an instance-level defect (Rahm & Do, 2000)."]

    # ---- 4.3b negative and zero values ----
    qty_num, price_num, total_num = (_a_as_num(get("quantity")), _a_as_num(get("unit_price")),
                                     _a_as_num(get("total_amount")))
    recs = []
    for col, values in {"quantity": qty_num, "unit_price": price_num, "total_amount": total_num}.items():
        neg, zero = int((values < 0).sum()), int((values == 0).sum())
        has_values = bool(values.notna().any())
        recs.append({"Column": col, "Valid numeric values": int(values.notna().sum()), "Negative (< 0)": neg,
                     "Zero (= 0)": zero, "Non-positive (<= 0)": neg + zero,
                     "Non-positive % of rows": round(_a_pct(neg + zero, n_rows), 2),
                     "Minimum": round(float(values.min()), 2) if has_values else float("nan"),
                     "Maximum": round(float(values.max()), 2) if has_values else float("nan")})
    anomaly_table = pd.DataFrame(recs).set_index("Column")
    out["anomaly_table"] = anomaly_table
    mask_neg_return = (qty_num < 0) | (total_num < 0)
    mask_nonpos_other = (qty_num == 0) | (price_num <= 0) | (total_num == 0)
    n_neg_return, n_nonpos_other = int(mask_neg_return.sum()), int(mask_nonpos_other.sum())
    returns_value = float(total_num[total_num < 0].sum())
    cols_nonpos = [n for n in anomaly_table.index if anomaly_table.loc[n, "Non-positive (<= 0)"] > 0]
    if not cols_nonpos:
        a1 = ("No negative or zero value was found in `quantity`, `unit_price` or `total_amount`, so every audited "
              "numeric entry is strictly positive.")
        a2 = "The data therefore contain neither returns nor sign errors, and no row has to be excluded on range grounds."
    else:
        items = ["`" + n + "` (" + _a_cnt(anomaly_table.loc[n, "Non-positive (<= 0)"], "row") + ")"
                 for n in anomaly_table.index]
        a1 = (f"Non-positive entries occur in {len(cols_nonpos)} of the 3 numeric columns: "
              + ", ".join(items[:-1]) + " and " + items[-1] + ".")
        a2 = (f"Rows with a negative quantity or amount, which represent returns or entry errors, total "
              f"{n_neg_return:,} ({_a_pct(n_neg_return, n_rows):.2f}% of the data) and carry negative amounts worth "
              f"{CURRENCY_SYMBOL}{abs(returns_value):,.2f}, while rows with a zero quantity, a zero amount or a "
              f"non-positive price total {n_nonpos_other:,}.")
    out["anomaly_text"] = [a1, a2, "Range checks of this kind uncover illegal values (Rahm & Do, 2000), and screening "
                                   "a distribution for extreme or impossible values is a standard exploratory step "
                                   "(Tukey, 1977)."]

    # ---- 4.3c arithmetic reconciliation ----
    evaluable = qty_num.notna() & price_num.notna() & total_num.notna()
    abs_diff = (qty_num * price_num - total_num).abs()
    arith_ok = evaluable & (abs_diff <= AUDIT_TOLERANCE + 1e-9)
    mask_mismatch = evaluable & ~arith_ok
    n_eval, n_match, n_mis = int(evaluable.sum()), int(arith_ok.sum()), int(mask_mismatch.sum())
    n_not_eval = n_rows - n_eval
    max_diff = float(abs_diff[evaluable].max()) if n_eval else 0.0
    recon_rows = [("Rows evaluable (all three values present)", f"{n_eval:,}"),
                  ("Rows reconciling within tolerance", f"{n_match:,}"),
                  ("Rows where the equation fails", f"{n_mis:,}"),
                  ("Failure rate among evaluable rows", f"{_a_pct(n_mis, n_eval):.2f}%"),
                  ("Rows not evaluable (a value is missing)", f"{n_not_eval:,}"),
                  ("Largest absolute difference", f"{max_diff:,.2f}"),
                  ("Tolerance applied", f"{AUDIT_TOLERANCE:g}")]
    out["recon_table"] = pd.DataFrame({"Reconciliation check": [r[0] for r in recon_rows],
                                       "Result": [r[1] for r in recon_rows]}).set_index("Reconciliation check")
    out["recon_examples"] = None
    n_examples = 0
    if n_mis > 0:
        n_examples = min(5, n_mis)
        worst_pos = np.argsort(-abs_diff.where(mask_mismatch, -1.0).to_numpy(), kind="stable")[:n_examples]
        ex_cols = [c for c in ["transaction_id", "quantity", "unit_price", "total_amount"] if c in audit.columns]
        ex = audit.iloc[worst_pos][ex_cols].copy()
        ex["quantity * unit_price"] = (qty_num * price_num).iloc[worst_pos].round(2).to_numpy()
        ex["abs difference"] = abs_diff.iloc[worst_pos].round(2).to_numpy()
        out["recon_examples"] = ex
    if n_eval == 0:
        r1 = "No row has all three values present, so the equation `quantity * unit_price == total_amount` could not be tested."
        r2 = "The reconciliation has to be repeated once the missing values or columns have been resolved."
    elif n_mis == 0:
        r1 = (f"Of {_a_cnt(n_eval, 'evaluable row')}, all {n_match:,} satisfy `quantity * unit_price == total_amount` "
              f"within a tolerance of {AUDIT_TOLERANCE:g}.")
        r2 = ("The amount column is therefore arithmetically consistent with its components, including for return "
              "rows, so total_amount can be trusted as derived from the other two fields.")
    else:
        r1 = (f"Of {_a_cnt(n_eval, 'evaluable row')}, {n_match:,} ({_a_pct(n_match, n_eval):.2f}%) satisfy "
              f"`quantity * unit_price == total_amount` within a tolerance of {AUDIT_TOLERANCE:g}, and {n_mis:,} do not.")
        r2 = (f"The largest absolute deviation is {CURRENCY_SYMBOL}{max_diff:,.2f}, and the "
              f"{_a_cnt(n_examples, 'worst case')} {_a_vb(n_examples, 'is', 'are')} listed above for manual review.")
    out["recon_text"] = [r1, r2, "This test targets violated attribute dependencies (Rahm & Do, 2000) and applies the "
                                 "same observed-versus-expected residual logic that underlies exploratory analysis "
                                 "(Tukey, 1977)."]

    # ---- 4.4a headline issues ----
    mask_missing_id = _a_missing_mask(get("transaction_id")) | _a_missing_mask(get("customer_id"))
    mask_missing_date = _a_missing_mask(get("transaction_date"))
    labels = ["Missing IDs", "Missing Dates", "Exact Duplicates", "Negative Returns"]
    counts = [int(mask_missing_id.sum()), int(mask_missing_date.sum()), int(mask_dup.sum()), int(mask_neg_return.sum())]
    out["headline_labels"], out["headline_counts"] = labels, counts
    ranked = sorted(zip(labels, counts), key=lambda pair: pair[1], reverse=True)
    if sum(counts) == 0:
        v1 = "None of the four headline problems (missing IDs, missing dates, exact duplicates and negative returns) was detected in the raw data."
        v2 = "The chart is therefore flat, and no rows have to be removed on these grounds."
    else:
        ranking = "; ".join(f"{lab} ({cnt:,} rows, {_a_pct(cnt, n_rows):.2f}%)" for lab, cnt in ranked)
        v1 = f"Ranked from most to least frequent, the headline problems are: {ranking}."
        v2 = ("Because a single row can fall into several categories, the bars overlap and must not be added up; the "
              "combined number of affected rows is given in the audit report below.")
    out["headline_text"] = [v1, v2, "Presenting the counts graphically makes their relative scale visible at a glance "
                                    "(Tukey, 1977) and gives the cleaning stage a documented, evidence-based "
                                    "starting point (Rahm & Do, 2000)."]

    # ---- 4.4b consolidated report ----
    mask_other_missing = pd.Series(False, index=audit.index)
    for col in ["product_name", "category", "quantity", "unit_price", "total_amount"]:
        mask_other_missing = mask_other_missing | _a_missing_mask(get(col))
    items = [("Missing IDs (transaction_id or customer_id)", mask_missing_id,
              "Cannot be linked to a transaction or customer; candidate for removal."),
             ("Missing dates (transaction_date)", mask_missing_date, "Cannot enter time-based analysis; candidate for removal."),
             ("Exact duplicates (extra copies)", mask_dup, "Keep the first occurrence and drop the extra copies."),
             ("Negative returns (quantity or amount < 0)", mask_neg_return,
              "Exclude from spend analysis or handle separately; log the decision."),
             ("Zero quantity/amount or price <= 0", mask_nonpos_other, "Investigate; treat as invalid unless justified."),
             ("Amount mismatch (quantity x price != total)", mask_mismatch, "Review; recompute or exclude."),
             ("Other missing fields (product/category/numeric)", mask_other_missing,
              "Impute or drop, depending on analytical need.")]
    mask_any, recs = pd.Series(False, index=audit.index), []
    for lab, msk, advice in items:
        mask_any = mask_any | msk
        cnt = int(msk.sum())
        recs.append({"Audit check": lab, "Rows affected": f"{cnt:,}", "% of rows": f"{_a_pct(cnt, n_rows):.2f}%",
                     "Status": "Pass" if cnt == 0 else "Action needed", "Suggested handling in the cleaning stage": advice})
    n_any = int(mask_any.sum())
    n_pass = n_rows - n_any
    recs.append({"Audit check": "Rows with at least one issue (union)", "Rows affected": f"{n_any:,}",
                 "% of rows": f"{_a_pct(n_any, n_rows):.2f}%", "Status": "Pass" if n_any == 0 else "Action needed",
                 "Suggested handling in the cleaning stage": "Combined footprint of the checks above (overlaps counted once)."})
    recs.append({"Audit check": "Rows passing every check", "Rows affected": f"{n_pass:,}",
                 "% of rows": f"{_a_pct(n_pass, n_rows):.2f}%", "Status": "-",
                 "Suggested handling in the cleaning stage": "No intervention needed."})
    out["report_table"] = pd.DataFrame(recs).set_index("Audit check")
    out["n_any_issue"], out["n_pass"] = n_any, n_pass
    if n_any == 0:
        p1 = f"All {n_rows:,} rows pass every audit check, so the raw data show no flagged issue."
        p2 = ("No cleaning action is required on these grounds, although the unchanged raw data remain the "
              "reference point for the cleaning log.")
    else:
        p1 = (f"In total {n_any:,} of {n_rows:,} rows ({_a_pct(n_any, n_rows):.2f}%) fail at least one audit check, "
              f"while {n_pass:,} rows ({_a_pct(n_pass, n_rows):.2f}%) pass all of them.")
        p2 = ("The data are therefore not yet analysis-ready: the flagged rows are documented here but deliberately "
              "left in `df_raw`, so that the cleaning stage can remove or correct them and record every change in its log.")
    out["report_text"] = [p1, p2, "Systematic, evidence-based profiling of this kind is the first step of any "
                                  "data-cleaning process (Rahm & Do, 2000) and reflects the descriptive-first "
                                  "discipline of exploratory analysis (Tukey, 1977)."]
    return out


# ===================================================================================================
# 10.  CUSTOMER LAYER  -  Sections 6, 7, 10, 12.2, 12.5: customer table, labels, RFM, K-Means
# ===================================================================================================
def get_snapshot_date(df, date_col="transaction_date"):
    """snapshot_date = max(transaction_date) + 1 day  (single as-of date for every recency figure)."""
    return pd.to_datetime(df[date_col]).max().normalize() + pd.Timedelta(days=1)


def build_customer_table(tx, snapshot=None):
    """Section 6 - one row per customer with the 7 base columns of the schema contract."""
    snapshot = snapshot if snapshot is not None else get_snapshot_date(tx)
    grouped = (tx.groupby("customer_id", as_index=False)
               .agg(total_spend=("total_amount", "sum"), num_orders=("transaction_id", "nunique"),
                    first_purchase_date=("transaction_date", "min"),
                    last_purchase_date=("transaction_date", "max")))
    grouped["avg_order_value"] = grouped["total_spend"] / grouped["num_orders"]
    grouped["recency_days"] = (snapshot - grouped["last_purchase_date"]).dt.days.astype("int64")
    grouped["num_orders"] = grouped["num_orders"].astype("int64")
    grouped["total_spend"] = grouped["total_spend"].astype("float64")
    return grouped[CUSTOMER_COLS].sort_values("customer_id").reset_index(drop=True)


def spend_thresholds(total_spend):
    """Section 7 - tertile cut-offs T1 (33.3rd percentile) and T2 (66.7th percentile) of total_spend."""
    return float(total_spend.quantile(1 / 3)), float(total_spend.quantile(2 / 3))


def assign_spend_segment(cust, low_upper, med_upper):
    """Low if spend <= T1; Medium if T1 < spend <= T2; High if spend > T2."""
    return pd.Series(np.select([cust["total_spend"] <= low_upper, cust["total_spend"] <= med_upper],
                               ["Low", "Medium"], default="High"), index=cust.index).astype(str)


def assign_buyer_type(cust):
    """1-Time Buyers if num_orders == 1, Repeat Buyers if num_orders >= 2."""
    return pd.Series(np.where(cust["num_orders"] >= 2, RP_REP, RP_ONE), index=cust.index).astype(str)


def ensure_labels(cust):
    """Add spend_segment and buyer_type when they are missing or invalid.  Returns (customer table, thresholds)."""
    cust = cust.copy()
    low_upper, med_upper = spend_thresholds(cust["total_spend"])
    if "spend_segment" not in cust.columns or not set(cust["spend_segment"].astype(str)) <= set(SEGMENT_ORDER):
        cust["spend_segment"] = assign_spend_segment(cust, low_upper, med_upper)
    else:
        cust["spend_segment"] = cust["spend_segment"].astype(str)
    if "buyer_type" not in cust.columns or not set(cust["buyer_type"].astype(str)) <= set(RP_GROUPS):
        cust["buyer_type"] = assign_buyer_type(cust)
    else:
        cust["buyer_type"] = cust["buyer_type"].astype(str)
    return cust, {"low_upper": low_upper, "med_upper": med_upper}


def compute_rfm(cust):
    """Section 12.2 - R, F, M quintile scores and the six business segments (None if fewer than 5 customers)."""
    if len(cust) < 5:
        return None
    rfm = cust.copy()
    rfm["R_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["num_orders"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["total_spend"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_code"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    r, f, m = rfm["R_score"], rfm["F_score"], rfm["M_score"]
    rules = [("Champions", (r >= 4) & (f >= 4) & (m >= 4)),
             ("Loyal Customers", (r >= 3) & (f >= 4)),
             ("Potential Loyalists", (r >= 3) & (f == 3)),
             ("New / Promising", (r >= 3) & (f <= 2)),
             ("At Risk", (r <= 2) & (f >= 3)),
             ("Hibernating / Lost", (r <= 2) & (f <= 2))]
    rfm["rfm_segment"] = pd.Series(np.select([c for _, c in rules], [n for n, _ in rules], default="Unclassified"),
                                   index=rfm.index).astype(str)
    return rfm


# ---- K-Means (Section 12.5) ----------------------------------------------------------------------
CLUSTER_FEATURES = ["recency_days", "num_orders", "total_spend", "avg_order_value"]
K_MIN_ACTIONABLE, K_MAX_ACTIONABLE = 3, 6          # business-actionable window for the final choice
SIL_TOLERANCE = 0.90                               # elbow k accepted if its silhouette >= 90% of the window's best
SIL_SAMPLE = 10_000                                # silhouette on a random sample for very large customer bases
MIN_CUSTOMERS_KMEANS = 12                          # smallest customer base on which K-Means is run


def _elbow_k(ks, inertia):
    """k of the point farthest from the straight line joining the first and last (k, inertia) points."""
    ks = np.asarray(ks, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    if ks.max() == ks.min() or inertia.max() == inertia.min():
        return int(ks[0])
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min())
    return int(ks[np.argmax(np.abs(x + y - 1))])


@st.cache_data(show_spinner=False)
def kmeans_selection(feat):
    """12.5.1-12.5.2: log1p + StandardScaler, then k = 2..8 compared on inertia, silhouette and Davies-Bouldin."""
    feats = feat[CLUSTER_FEATURES].astype(float).copy()
    n = len(feats)
    n_missing = int(feats.isna().sum().sum())
    if n_missing:
        feats = feats.fillna(feats.median())
    feats = feats.clip(lower=0)                                   # log1p needs values > -1
    x_log = np.log1p(feats)
    x_scaled = StandardScaler().fit_transform(x_log)
    skew_df = pd.DataFrame({"skewness_raw": feats.skew(), "skewness_after_log1p": x_log.skew()}).round(2)
    skew_df.index.name = "feature"
    k_range = list(range(2, min(8, n - 1) + 1))
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, **KMEANS_PARAMS).fit(x_scaled)
        lab = km.labels_
        try:
            sil = silhouette_score(x_scaled, lab, random_state=RANDOM_STATE,
                                   sample_size=SIL_SAMPLE if n > SIL_SAMPLE else None)
            dbi = davies_bouldin_score(x_scaled, lab)
            chi = calinski_harabasz_score(x_scaled, lab)
        except ValueError:
            sil = dbi = chi = np.nan
        rows.append({"k": k, "inertia_wcss": km.inertia_, "silhouette": sil, "davies_bouldin": dbi,
                     "calinski_harabasz": chi, "smallest_cluster_pct": 100 * np.bincount(lab).min() / n})
    metrics = pd.DataFrame(rows)
    elbow = _elbow_k(metrics["k"], metrics["inertia_wcss"])
    elbow_y = float(metrics.loc[metrics["k"] == elbow, "inertia_wcss"].iloc[0])
    window = metrics[metrics["k"].between(K_MIN_ACTIONABLE, K_MAX_ACTIONABLE)].copy()
    if window.empty:
        window = metrics.copy()
    window["rank_silhouette"] = window["silhouette"].fillna(-1).rank(ascending=False, method="min")
    window["rank_davies_bouldin"] = window["davies_bouldin"].fillna(99).rank(ascending=True, method="min")
    window["avg_rank"] = (window["rank_silhouette"] + window["rank_davies_bouldin"]) / 2
    best = window.sort_values(["avg_rank", "silhouette"], ascending=[True, False]).iloc[0]
    elbow_row = window[window["k"] == elbow]
    elbow_ok = (not elbow_row.empty) and float(elbow_row["silhouette"].iloc[0]) >= SIL_TOLERANCE * float(window["silhouette"].max())
    optimal_k = int(elbow) if elbow_ok else int(best["k"])
    if elbow_ok:
        why = (f"elbow point k = {elbow} lies in the window and its silhouette is within "
               f"{round((1 - SIL_TOLERANCE) * 100)}% of the window's best")
    else:
        why = (f"elbow point k = {elbow} was not usable (outside the window or weaker silhouette), so the best "
               f"average rank of silhouette and Davies-Bouldin decides")
    return {"X": x_scaled, "feats": feats, "skew_df": skew_df, "metrics": metrics, "k_range": k_range,
            "elbow_k": elbow, "elbow_y": elbow_y, "optimal_k": optimal_k, "why": why, "n_missing": n_missing,
            "best_window_k": int(best["k"]), "n": n,
            "best_sil_k": int(metrics.loc[metrics["silhouette"].fillna(-9).idxmax(), "k"]),
            "best_dbi_k": int(metrics.loc[metrics["davies_bouldin"].fillna(99).idxmin(), "k"])}


def name_clusters(cz):
    """Business names from centroid z-scores (rows = clusters ordered best -> worst); names are always unique."""
    base_name = {}
    for cid in cz.index:
        r_, f_, m_, a_ = (float(cz.loc[cid, c]) for c in ["recency_days", "num_orders", "total_spend", "avg_order_value"])
        if cid == 0:
            base_name[int(cid)] = "Champions (VIP Loyalists)"
        elif r_ > 0.5:
            base_name[int(cid)] = "At-Risk Former Big Spenders" if (f_ + m_) / 2 > 0 else "Lost / Dormant Customers"
        elif f_ > 0.2:
            base_name[int(cid)] = "Loyal Regulars"
        elif a_ > 0.5:
            base_name[int(cid)] = "Big-Basket Occasional Buyers"
        elif r_ < -0.2:
            base_name[int(cid)] = "New & Recently Active Buyers"
        else:
            base_name[int(cid)] = "Casual Occasional Buyers"
    names = {}
    for cid, base in base_name.items():              # two clusters with the same base name -> add a value qualifier
        twins = [c for c, b in base_name.items() if b == base]
        if len(twins) == 1:
            names[cid] = base
        elif len(twins) == 2:
            names[cid] = f"{base} ({'Higher' if cid == twins[0] else 'Lower'} Value)"
        else:
            names[cid] = f"{base} (Tier {twins.index(cid) + 1})"
    return names


@st.cache_data(show_spinner=False)
def kmeans_final(feat, k):
    """12.5.3-12.5.5: final KMeans(n_clusters=k, n_init=10, random_state=42), value-ordered labels, PCA map."""
    sel = kmeans_selection(feat)
    x_scaled = sel["X"]
    km = KMeans(n_clusters=int(k), **KMEANS_PARAMS).fit(x_scaled)
    cent_raw = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATURES)
    value_score = (cent_raw["num_orders"] + cent_raw["total_spend"] - cent_raw["recency_days"]) / 3
    value_order = value_score.sort_values(ascending=False).index.to_list()
    relabel = {old: new for new, old in enumerate(value_order)}      # 0 = highest value
    centroids_z = cent_raw.loc[value_order].reset_index(drop=True)
    centroids_z.index.name = "cluster"
    labels = pd.Series(km.labels_).map(relabel).astype("int64").to_numpy()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(x_scaled)
    row = sel["metrics"].loc[sel["metrics"]["k"] == int(k)]
    quality = {c: (float(row[c].iloc[0]) if not row.empty else np.nan)
               for c in ["silhouette", "davies_bouldin", "calinski_harabasz", "inertia_wcss"]}
    return {"labels": labels, "centroids_z": centroids_z, "names": name_clusters(centroids_z), "pcs": pcs,
            "ev": pca.explained_variance_ratio_, "loadings": pd.DataFrame(pca.components_.T, index=CLUSTER_FEATURES,
                                                                          columns=["PC1", "PC2"]).round(3),
            "cent_pc": pca.transform(centroids_z.to_numpy()), "quality": quality, "k": int(k)}


def attach_advanced_labels(cust, k=None):
    """Add rfm_segment (Section 12.2) and cluster (Section 12.5) to a customer table.

    Returns (customer table, rfm table or None, K-Means selection dict or None, K-Means final dict or None).
    """
    cust = cust.copy()
    rfm = compute_rfm(cust)
    cust["rfm_segment"] = rfm["rfm_segment"].to_numpy() if rfm is not None else "Unscored"
    sel = fit = None
    if len(cust) >= MIN_CUSTOMERS_KMEANS:
        feat = cust[CLUSTER_FEATURES].reset_index(drop=True)
        sel = kmeans_selection(feat)
        use_k = int(k) if k else sel["optimal_k"]
        if use_k not in sel["k_range"]:
            use_k = sel["optimal_k"]
        fit = kmeans_final(feat, use_k)
        cust["cluster"] = fit["labels"]
    else:
        cust["cluster"] = -1
    return cust, rfm, sel, fit


# ===================================================================================================
# 11.  SCHEMA CONTRACT AND RECONCILIATION CHECKS  (Sections 2.11, 5.4, 6 and pre_merge_sanity_check.py)
# ===================================================================================================
def _kind_ok(series, kind):
    types = pd.api.types
    if kind in ("id", "text"):
        return bool(types.is_object_dtype(series) or types.is_string_dtype(series))
    if kind == "datetime":
        return bool(types.is_datetime64_any_dtype(series))
    if kind == "numeric":
        return bool(types.is_numeric_dtype(series) and not types.is_bool_dtype(series))
    return False


def schema_report(df, name):
    """PASS / FAIL table for one table against SCHEMA_CONTRACT (columns, dtype family, nulls, uniqueness)."""
    spec = SCHEMA_CONTRACT[name]
    rows = []
    for col, kind in spec["required"].items():
        present = col in df.columns
        type_ok = present and _kind_ok(df[col], kind)
        nulls = int(df[col].isna().sum()) if present else None
        null_ok = (nulls == 0) if (present and col in spec.get("no_null", [])) else True
        rows.append({"column": col, "expected type": kind, "present": "yes" if present else "NO",
                     "dtype found": str(df[col].dtype) if present else "-",
                     "nulls": "-" if nulls is None else nulls,
                     "status": "PASS" if (present and type_ok and null_ok) else "FAIL"})
    for col in spec.get("unique", []):
        dup = int(df[col].duplicated().sum()) if col in df.columns else -1
        rows.append({"column": f"{col} (unique key)", "expected type": "unique", "present": "yes" if col in df.columns else "NO",
                     "dtype found": "-", "nulls": "-", "status": "PASS" if dup == 0 else "FAIL"})
    return pd.DataFrame(rows)


def reconciliation_checks(tx, cust):
    """Cross-table identities between df_clean and customer_df (same customers, revenue, orders, dates)."""
    rows = []

    def add(name, expected, observed, ok):
        rows.append({"check": name, "expected": expected, "observed": observed, "status": "PASS" if ok else "FAIL"})

    tx_c, cu_c = set(tx["customer_id"]), set(cust["customer_id"])
    add("Same customers in both tables", f"{len(tx_c):,} customers in df_clean", f"{len(cu_c):,} in customer_df",
        tx_c == cu_c)
    rev_tx, rev_cu = round(float(tx["total_amount"].sum()), 2), round(float(cust["total_spend"].sum()), 2)
    add("Total revenue reconciles", f"{money(rev_tx, 2)}", f"{money(rev_cu, 2)}", abs(rev_tx - rev_cu) <= 0.01)
    ord_tx = int(tx.groupby("customer_id")["transaction_id"].nunique().sum())
    ord_cu = int(cust["num_orders"].sum())
    add("Total orders reconcile", f"{ord_tx:,} orders in df_clean", f"{ord_cu:,} in customer_df", ord_tx == ord_cu)
    per = tx.groupby("customer_id").agg(spend=("total_amount", "sum"), first=("transaction_date", "min"),
                                        last=("transaction_date", "max"))
    merged = cust.set_index("customer_id").join(per, how="inner")
    gap = float((merged["total_spend"] - merged["spend"]).abs().max()) if len(merged) else 0.0
    add("Customer-level spend equals the sum of its lines", "max |gap| <= 0.01", f"max |gap| = {gap:.4f}", gap <= 0.01)
    dates_ok = bool((merged["first_purchase_date"] == merged["first"]).all()
                    and (merged["last_purchase_date"] == merged["last"]).all()) if len(merged) else True
    add("First / last purchase dates match the transactions", "identical", "identical" if dates_ok else "DIFFERENT",
        dates_ok)
    snap = get_snapshot_date(tx)
    rec_gap = int((merged["recency_days"] - (snap - merged["last"]).dt.days).abs().max()) if len(merged) else 0
    add("recency_days = snapshot - last purchase", f"snapshot = {snap:%Y-%m-%d}", f"max |gap| = {rec_gap} days",
        rec_gap == 0)
    aov_gap = float((cust["avg_order_value"] - cust["total_spend"] / cust["num_orders"]).abs().max()) if len(cust) else 0.0
    add("avg_order_value = total_spend / num_orders", "max |gap| <= 0.01", f"max |gap| = {aov_gap:.4f}", aov_gap <= 0.01)
    return pd.DataFrame(rows)


# ===================================================================================================
# 12.  DATA LOADING  -  disk -> upload -> synthetic fallback
# ===================================================================================================
@st.cache_data(show_spinner=False)
def reference_pipeline(seed=RANDOM_STATE):
    """Generate the synthetic df_raw (Appendix A) and run the Section 5 cleaning pipeline on it."""
    df_raw = generate_retail_transactions(seed)
    df_clean, cleaning_log, df_returns = clean_raw_data(df_raw)
    return {"df_raw": df_raw, "df_clean": df_clean, "cleaning_log": cleaning_log, "df_returns": df_returns}


def disk_candidates(filename):
    """Folders searched for an exported table: the working folder first, then the folder of app.py."""
    folders = [Path.cwd() / "outputs" / "tables"]
    try:
        folders.append(Path(__file__).resolve().parent / "outputs" / "tables")
    except NameError:                                                   # pragma: no cover - __file__ undefined
        pass
    seen, out = set(), []
    for folder in folders:
        path = folder / filename
        if str(path) not in seen:
            seen.add(str(path))
            out.append(path)
    return out


def find_on_disk(filename):
    """Path of an exported CSV if it exists, else None."""
    for path in disk_candidates(filename):
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


@st.cache_data(show_spinner=False)
def read_csv_cached(path_str, mtime):
    """Read a CSV from disk (cached; the modification time invalidates the cache when the file changes)."""
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def read_csv_bytes(data):
    """Read an uploaded CSV (bytes) into a DataFrame."""
    return pd.read_csv(io.BytesIO(data))


def _norm_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def rename_to_contract(df):
    """Map alternative column spellings (order_id, date, revenue ...) onto the contract names."""
    df = df.copy()
    lookup = {_norm_name(c): c for c in df.columns}
    rename = {}
    for target, aliases in COLUMN_ALIASES.items():
        if target in df.columns:
            continue
        for candidate in [target] + aliases:
            key = _norm_name(candidate)
            if key in lookup and lookup[key] not in rename:
                rename[lookup[key]] = target
                break
    return df.rename(columns=rename)


def parse_dates_flex(series):
    """Parse dates robustly: ISO first, then day-first (27-02-2024), then per-element 'mixed' formats.

    The variant that parses the most values wins, so common non-ISO exports are not silently dropped.
    """
    n_in = int(series.notna().sum())
    best = pd.to_datetime(series, errors="coerce")
    if n_in and int(best.notna().sum()) < 0.98 * n_in:
        for kwargs in ({"dayfirst": True}, {"format": "mixed", "dayfirst": True}, {"format": "mixed"}):
            try:
                alt = pd.to_datetime(series, errors="coerce", **kwargs)
            except (TypeError, ValueError):
                continue
            if int(alt.notna().sum()) > int(best.notna().sum()):
                best = alt
    if isinstance(best.dtype, pd.DatetimeTZDtype):
        best = best.dt.tz_localize(None)
    return best.astype("datetime64[ns]")


def coerce_clean_table(df):
    """Make an uploaded / disk df_clean obey the schema contract.  Returns (table, list of notes)."""
    notes = []
    df = rename_to_contract(df)
    missing = [c for c in CLEAN_COLS if c not in df.columns and c != "total_amount"]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")
    df = df.copy()
    for col in ID_COLS + TEXT_COLS:
        df[col] = trim_text(df[col])
    df["transaction_date"] = parse_dates_flex(df["transaction_date"])
    for col in ("quantity", "unit_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "total_amount" not in df.columns:
        df["total_amount"] = (df["quantity"] * df["unit_price"]).round(2)
        notes.append("total_amount was missing and has been computed as round(quantity x unit_price, 2).")
    else:
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
        n_fill = int(df["total_amount"].isna().sum())
        if n_fill:
            df["total_amount"] = df["total_amount"].fillna((df["quantity"] * df["unit_price"]).round(2))
            notes.append(f"{n_fill:,} empty total_amount values were computed as quantity x unit_price.")
    bad = (is_blank(df["transaction_id"]) | is_blank(df["customer_id"]) | df["transaction_date"].isna()
           | ~(df["quantity"] > 0) | ~(df["unit_price"] > 0) | ~(df["total_amount"] > 0))
    if int(bad.sum()):
        notes.append(f"{int(bad.sum()):,} rows violate the df_clean contract (missing key / date, or quantity, price or "
                     f"amount <= 0) and were ignored.")
        df = df.loc[~bad].copy()
    for col in ID_COLS + TEXT_COLS:
        df[col] = df[col].astype(str)
    if (df["quantity"] % 1 == 0).all():
        df["quantity"] = df["quantity"].astype("int64")
    df["unit_price"] = df["unit_price"].astype("float64")
    df["total_amount"] = df["total_amount"].astype("float64")
    df = df[CLEAN_COLS].reset_index(drop=True)
    if df.empty:
        raise ValueError("no valid rows remain after applying the df_clean contract")
    return df, notes


def coerce_customer_table(df):
    """Make an uploaded / disk customer_df obey the schema contract (dtypes only; values are validated later)."""
    df = rename_to_contract(df.copy())
    missing = [c for c in CUSTOMER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")
    df["customer_id"] = trim_text(df["customer_id"]).astype(str)
    for col in ("first_purchase_date", "last_purchase_date"):
        df[col] = parse_dates_flex(df[col])
    for col in ("total_spend", "avg_order_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ("num_orders", "recency_days"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[CUSTOMER_COLS].isna().any().any() or df["customer_id"].duplicated().any():
        raise ValueError("customer_df contains missing values or duplicate customer_id values")
    df["num_orders"] = df["num_orders"].astype("int64")
    df["recency_days"] = df["recency_days"].astype("int64")
    return df.reset_index(drop=True)


def coerce_log_table(df):
    """cleaning_log must carry the 8 documented columns."""
    df = df.copy()
    df.columns = [_norm_name(c) for c in df.columns]
    missing = [c for c in LOG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")
    return df[LOG_COLS].reset_index(drop=True)


def load_bundle(upload_bytes):
    """Resolve the three project tables: upload > disk > synthetic.  Never raises; problems become notes.

    upload_bytes : dict {"df_clean": bytes | None, "customer_df": bytes | None, "cleaning_log": bytes | None}
    """
    notes, origins, frames = [], {}, {}
    for key, filename in DISK_FILES.items():
        raw_df, origin = None, None
        try:
            if upload_bytes.get(key):
                raw_df, origin = read_csv_bytes(upload_bytes[key]), "uploaded file"
            else:
                path = find_on_disk(filename)
                if path is not None:
                    raw_df = read_csv_cached(str(path), path.stat().st_mtime)
                    try:
                        origin = f"disk: {path.relative_to(Path.cwd())}"
                    except ValueError:
                        origin = f"disk: {path}"
        except Exception as exc:                                       # noqa: BLE001
            notes.append(f"{filename} could not be read ({type(exc).__name__}: {exc}); it was ignored.")
            raw_df = None
        frames[key], origins[key] = raw_df, origin

    ref = reference_pipeline(RANDOM_STATE)
    bundle = {"df_raw": ref["df_raw"], "df_returns": ref["df_returns"], "customer_provided": None,
              "customer_check": None, "customer_ok": False, "raw_is_reference": True}

    # ---- df_clean : provided or synthetic ----
    clean = None
    if frames["df_clean"] is not None:
        try:
            clean, clean_notes = coerce_clean_table(frames["df_clean"])
            notes += [f"cleaned_transactions.csv: {n}" for n in clean_notes]
        except Exception as exc:                                       # noqa: BLE001
            notes.append(f"cleaned_transactions.csv was rejected ({exc}); synthetic data are used instead.")
            origins["df_clean"] = None
    synthetic_mode = clean is None
    if synthetic_mode:
        clean = ref["df_clean"]
        origins["df_clean"] = "synthetic (RANDOM_STATE = 42)"
    bundle["df_clean"] = clean

    # ---- cleaning_log ----
    log = None
    if frames["cleaning_log"] is not None:
        try:
            log = coerce_log_table(frames["cleaning_log"])
        except Exception as exc:                                       # noqa: BLE001
            notes.append(f"data_cleaning_log.csv was rejected ({exc}); the log of the reference pipeline is used.")
            origins["cleaning_log"] = None
    bundle["log_is_reference"] = log is None
    if log is None:
        log = ref["cleaning_log"]
        origins["cleaning_log"] = "generated by the Section 5 pipeline" + ("" if synthetic_mode else " (reference raw data)")
    bundle["cleaning_log"] = log
    bundle["raw_is_reference"] = True

    # ---- customer_df : provided (validated against df_clean) or built from the transactions ----
    if frames["customer_df"] is not None:
        try:
            cust = coerce_customer_table(frames["customer_df"])
            check = reconciliation_checks(clean, cust)
            bundle["customer_check"] = check
            bundle["customer_ok"] = bool((check["status"] == "PASS").all())
            if bundle["customer_ok"]:
                bundle["customer_provided"] = cust
            else:
                notes.append("customer_analytics_master.csv does not reconcile with df_clean "
                             "(see the schema panel in Tab 1); customer_df is rebuilt from the transactions.")
        except Exception as exc:                                       # noqa: BLE001
            notes.append(f"customer_analytics_master.csv was rejected ({exc}); customer_df is rebuilt from df_clean.")
    if bundle["customer_provided"] is None:
        origins["customer_df"] = "built from df_clean (Section 6 logic)"
    bundle["mode"] = "synthetic" if synthetic_mode else "loaded"
    bundle["origins"] = origins
    bundle["notes"] = notes
    return bundle


# ===================================================================================================
# 13.  FILTER PIPELINE  -  turns the sidebar selections into the active transaction / customer tables
# ===================================================================================================
def build_active_dataset(bundle, flt):
    """Apply the sidebar filters.

    1. Date range and category filters select the transactions that are analysed.
    2. Inside that scope the customer table is (re)built and the Low/Medium/High and 1-Time/Repeat labels
       are assigned with the notebook logic (the exported customer file is used when nothing is narrowed).
    3. Spend-segment and buyer-type filters keep only the chosen labelled customers.
    """
    clean = bundle["df_clean"]
    start = pd.Timestamp(flt["start"])
    end_exclusive = pd.Timestamp(flt["end"]) + pd.Timedelta(days=1)
    in_scope = (clean["transaction_date"] >= start) & (clean["transaction_date"] < end_exclusive) & \
        clean["category"].isin(flt["categories"])
    base_tx = clean.loc[in_scope]
    full_range = (start <= clean["transaction_date"].min()) and (end_exclusive > clean["transaction_date"].max())
    narrowed = (not full_range) or (set(flt["categories"]) != set(clean["category"].unique()))
    empty = {"tx": base_tx.iloc[0:0], "cust": None, "thresholds": None, "snapshot": None, "narrowed": narrowed,
             "used_provided": False, "n_scope_customers": 0}
    if base_tx.empty:
        return empty
    use_provided = bundle["customer_provided"] is not None and not narrowed
    cust_base = bundle["customer_provided"].copy() if use_provided else build_customer_table(base_tx)
    cust_base, thresholds = ensure_labels(cust_base)
    keep = cust_base["spend_segment"].isin(flt["segments"]) & cust_base["buyer_type"].isin(flt["buyer_types"])
    cust = cust_base.loc[keep].reset_index(drop=True)
    tx = base_tx.loc[base_tx["customer_id"].isin(set(cust["customer_id"]))].reset_index(drop=True)
    return {"tx": tx, "cust": cust, "thresholds": thresholds, "snapshot": get_snapshot_date(base_tx),
            "narrowed": narrowed, "used_provided": use_provided, "n_scope_customers": len(cust_base)}


@st.cache_data(show_spinner=False)
def build_master_table(tx):
    """customer_analytics_master.csv: base columns + spend_segment + buyer_type + rfm_segment + cluster."""
    cust, _ = ensure_labels(build_customer_table(tx))
    cust, _, _, _ = attach_advanced_labels(cust)
    return cust


# ===================================================================================================
# 14.  TAB 1  -  Project overview, team allocation and data dictionary  (Sections 1-3, Appendix A)
#      Members 1 (Hasnain Raza Khan) and 2 (Aditi Sharma)
# ===================================================================================================
def kpi_summary(tx, cust):
    """Headline KPIs of the active selection."""
    revenue = float(tx["total_amount"].sum())
    orders = int(tx["transaction_id"].nunique())
    customers = int(cust["customer_id"].nunique())
    repeat_mask = cust["num_orders"] >= 2
    rpr = 100.0 * repeat_mask.mean() if customers else np.nan
    repeat_rev = float(cust.loc[repeat_mask, "total_spend"].sum())
    return {"revenue": revenue, "orders": orders, "customers": customers, "aov": safe_div(revenue, orders),
            "rpr": rpr, "repeat_customers": int(repeat_mask.sum()),
            "repeat_rev_share": 100.0 * safe_div(repeat_rev, revenue, 0.0),
            "lines": len(tx), "products": int(tx["product_name"].nunique()),
            "categories": int(tx["category"].nunique()),
            "first": tx["transaction_date"].min(), "last": tx["transaction_date"].max()}


def display_dates(df):
    """Copy of a DataFrame with datetime columns shown as YYYY-MM-DD text (tidier tables)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def generator_self_test(df_raw):
    """Section 3.6 - checks that the raw table has the documented structure and the planted impurities."""
    cols = list(CLEAN_COLS)
    text_cols = ["transaction_id", "customer_id", "product_name", "category"]
    valid = df_raw.drop_duplicates().dropna(subset=["customer_id", "transaction_date"])
    valid = valid[(valid["quantity"] > 0) & (valid["total_amount"] > 0)]
    per_txn = df_raw.groupby("transaction_id")[["customer_id", "transaction_date"]].nunique()
    basket = valid.groupby("transaction_id")["product_name"].nunique()
    dates = df_raw["transaction_date"].dropna()
    span = (dates.max() - dates.min()).days
    n_text = sum(pd.api.types.is_object_dtype(df_raw[c].dtype) or pd.api.types.is_string_dtype(df_raw[c].dtype)
                 for c in text_cols)
    n_ret = int(((df_raw["quantity"] < 0) & (df_raw["total_amount"] < 0)).sum())
    mixed = int(((df_raw["quantity"] < 0) != (df_raw["total_amount"] < 0)).sum())
    rows = []

    def add(check, expected, observed, passed):
        rows.append((check, expected, str(observed), "PASS" if passed else "FAIL"))

    add("8 lowercase columns in the agreed order", "exact match", "match" if list(df_raw.columns) == cols else "mismatch",
        list(df_raw.columns) == cols)
    add("Text columns hold text", "4 of 4", f"{n_text} of 4", n_text == 4)
    is_dt = pd.api.types.is_datetime64_any_dtype(df_raw["transaction_date"])
    add("transaction_date is a datetime column", "True", is_dt, is_dt)
    num_ok = (pd.api.types.is_integer_dtype(df_raw["quantity"]) and pd.api.types.is_numeric_dtype(df_raw["unit_price"])
              and pd.api.types.is_numeric_dtype(df_raw["total_amount"]))
    add("quantity is integer, prices and amounts are numeric", "True", num_ok, num_ok)
    add("Raw row count", "6,000 to 7,500", f"{len(df_raw):,}", 6000 <= len(df_raw) <= 7500)
    add("Distinct customers", "1,000", f"{df_raw['customer_id'].nunique():,}", df_raw["customer_id"].nunique() == 1000)
    add("Period covers about two years", "at least 700 days", f"{dates.min().date()} to {dates.max().date()} ({span} days apart)",
        span >= 700)
    n_multi = int((per_txn > 1).any(axis=1).sum())
    add("Transaction IDs with more than one customer or date", "0", n_multi, n_multi == 0)
    add("Different products per transaction", "1 to 4", f"{basket.min()} to {basket.max()}",
        basket.min() >= 1 and basket.max() <= 4)
    recon = bool(np.isclose(df_raw["total_amount"], df_raw["quantity"] * df_raw["unit_price"]).all())
    add("total_amount equals quantity x unit_price", "True", recon, recon)
    add("Exact duplicate rows", "120", int(df_raw.duplicated().sum()), int(df_raw.duplicated().sum()) == 120)
    add("Missing customer_id", "60", int(df_raw["customer_id"].isna().sum()), int(df_raw["customer_id"].isna().sum()) == 60)
    add("Missing transaction_date", "40", int(df_raw["transaction_date"].isna().sum()),
        int(df_raw["transaction_date"].isna().sum()) == 40)
    add("Returns (negative quantity and amount)", "90, no mixed signs", f"{n_ret}, mixed signs: {mixed}",
        n_ret == 90 and mixed == 0)
    add("Rows left after removing impurities", f"{len(df_raw) - 310:,}", f"{len(valid):,}", len(valid) == len(df_raw) - 310)
    add("Customers left after removing impurities", "1,000", f"{valid['customer_id'].nunique():,}",
        valid["customer_id"].nunique() == 1000)
    return pd.DataFrame(rows, columns=["Check", "Expected", "Observed", "Status"])


@chart
def chart_team_credit(members, shares, phases, credit):
    """Paper share per member (coloured by phase) against the equal 7.14% college credit."""
    phase_color = {"A": PALETTE["primary"], "B": PALETTE["secondary"], "C": PALETTE["accent"]}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    pos = np.arange(len(members))
    bars = ax.bar(pos, shares, color=[phase_color[p] for p in phases], width=0.68, zorder=3)
    ax.axhline(credit, color=PALETTE["negative"], ls="--", lw=1.5, zorder=4,
               label=f"Equal college credit per member = {credit:.2f}%")
    for rect, share in zip(bars, shares):
        ax.text(rect.get_x() + rect.get_width() / 2, share - 0.35, f"{share}%", ha="center", va="top", fontsize=9.5,
                fontweight="bold", color="white", zorder=5)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"M{i + 1}\n{short(n.split()[0], 9)}" for i, n in enumerate(members)], fontsize=8.5)
    ax.set_ylim(0, max(shares) * 1.3)
    ax.set_ylabel("Share of the final paper (%)")
    ax.set_title("Paper share per member (bar colour = phase) vs equal college credit")
    handles = [Patch(color=phase_color[p], label=f"Phase {p}: {PHASE_NAMES[p]}") for p in "ABC"]
    first = ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=1)
    ax.add_artist(first)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig_to_png(fig)


def render_tab1(ctx):
    """Tab 1 - KPI cards, Section 1 overview, team table, Section 3 data dictionary, Appendix A."""
    tx, cust, bundle = ctx["tx"], ctx["cust"], ctx["bundle"]
    full = bundle["df_clean"]
    kp = kpi_summary(tx, cust)
    st.markdown("## Project Overview, Team Allocation and Data Dictionary")
    st.caption("Sections 1-3 and Appendix A | Member 1 - Hasnain Raza Khan | Member 2 - Aditi Sharma")

    # ---------------- KPI cards ----------------
    cols = st.columns(5)
    kpi_card(cols[0], "Total Revenue", money(kp["revenue"]), "Sum of total_amount of the active selection (gross sales)")
    kpi_card(cols[1], "Total Orders", num(kp["orders"]), "Distinct transaction_id values")
    kpi_card(cols[2], "Total Customers", num(kp["customers"]), "Distinct customer_id values")
    kpi_card(cols[3], "Average Order Value (AOV)", money(kp["aov"]), "Total revenue / total orders")
    kpi_card(cols[4], "Repeat Purchase Rate (RPR)", pct(kp["rpr"]), "Customers with 2 or more orders / all customers x 100")
    full_rev = float(full["total_amount"].sum())
    share_txt = (f" This selection holds {pct(100 * safe_div(kp['revenue'], full_rev, 0), 1)} of the full dataset's revenue "
                 f"({money(full_rev)})." if ctx["narrowed"] or len(tx) != len(full) else "")
    interpret(f"The active selection contains {money(kp['revenue'])} of revenue from {num(kp['orders'])} orders placed by "
              f"{num(kp['customers'])} customers, which gives an average order value of {money(kp['aov'])}.",
              f"{pct(kp['rpr'])} of the customers ({num(kp['repeat_customers'])}) ordered at least twice, and these repeat "
              f"customers generate {pct(kp['repeat_rev_share'])} of the revenue.{share_txt}",
              f"The data cover {kp['first']:%d %b %Y} to {kp['last']:%d %b %Y} with {num(kp['lines'])} order lines, "
              f"{kp['products']} products and {kp['categories']} categories.")

    origins = bundle["origins"]
    with st.expander("Data source and loading status", expanded=bundle["mode"] == "loaded" or bool(bundle["notes"])):
        src = pd.DataFrame({"Table": ["df_clean", "customer_df", "cleaning_log"],
                            "Source": [origins["df_clean"], origins["customer_df"], origins["cleaning_log"]]})
        show_df(src, hide_index=True)
        for note in bundle["notes"]:
            st.warning(note)
        if bundle["mode"] == "synthetic":
            st.caption("No exported CSV files were found on disk or uploaded, so the project's data generator "
                       "(Appendix A, RANDOM_STATE = 42) created df_raw, which was cleaned and aggregated with the "
                       "Section 5 and Section 6 logic.")

    t1, t2, t3, t4 = st.tabs(["Section 1: Project overview", "Team allocation & credits",
                              "Section 3: Data dictionary & preview", "Appendix A: Generator & schema checks"])
    with t1:
        run_safely(render_sec1, ctx, kp)
    with t2:
        run_safely(render_team, ctx, kp)
    with t3:
        run_safely(render_sec3, ctx, kp)
    with t4:
        run_safely(render_appendix_a, ctx, kp)


def render_sec1(ctx, kp):
    """Section 1 (Member 1): background, business questions, objectives, workflow, techniques and formulas."""
    section_header("Section 1: Introduction, Objectives and Project Workflow", "Member 1 - Hasnain Raza Khan")
    st.markdown("**Business problem.** " + BUSINESS_PROBLEM)
    st.markdown("**Aim.** " + BUSINESS_AIM)
    caption_md("Table 1.1: Business questions and where they are answered")
    md_table(pd.DataFrame(BQ_TABLE, columns=["ID", "Business question", "Method", "Sections", "Members"]))
    cust, tx = ctx["cust"], ctx["tx"]
    n_seg = cust["spend_segment"].nunique()
    top_cat = tx.groupby("category")["total_amount"].sum().idxmax()
    interpret(f"With the current selection the ten questions are answered on {num(len(cust))} customers, "
              f"{num(tx['transaction_id'].nunique())} orders and {tx['category'].nunique()} categories; the largest category "
              f"by revenue is **{esc(top_cat)}**.",
              f"BQ3 compares {plural(n_seg, 'spend segment')} and BQ7-BQ9 build RFM scores, cohorts, baskets and clusters on the "
              f"same customer table, so every answer rests on one reconciled dataset.")

    with st.expander("Objectives by phase", expanded=False):
        for phase, items in OBJECTIVES.items():
            st.markdown(f"**{phase}**")
            for item in items:
                st.markdown("- " + item)
    caption_md("Table 1.2: Analytical techniques used in the study")
    md_table(pd.DataFrame(TECHNIQUE_TABLE, columns=["Technique", "Question answered", "Typical measure", "Section"]))
    interpret(f"Eleven techniques are chained in a fixed order: audit and cleaning feed the customer table "
              f"({num(len(cust))} rows here), which in turn feeds segmentation, RFM and K-Means.",
              f"The same order is visible in the data: {num(len(tx))} transaction lines collapse into {num(len(cust))} customer "
              f"rows, i.e. {safe_div(len(tx), len(cust), 0):.1f} lines per customer on average.")

    caption_md("Table 1.3: Twelve-step methodology")
    md_table(pd.DataFrame(WORKFLOW_STEPS, columns=["Step", "Activity", "Sections", "What is done", "Members"]))
    with st.expander("Project workflow diagram (shared objects: df_raw, df_clean, customer_df)"):
        st.code(WORKFLOW_DIAGRAM, language="text")
    caption_md("Key formulas used in the analysis")
    for title, latex in KEY_FORMULAS:
        st.markdown(f"*{title}*")
        st.latex(latex)
    interpret(f"The Pareto formula is evaluated on {num(len(cust))} customers, the RFM formula uses a snapshot date of "
              f"{ctx['snapshot']:%d %b %Y} for recency, and the cohort and basket formulas run on {num(tx['transaction_id'].nunique())} orders.",
              "All formulas are implemented exactly as written; the values they produce are shown in Tabs 3 to 6.")


def render_team(ctx, kp):
    """Credit assignment of the 14 members (7.14% college credit each)."""
    section_header("Team allocation and credit assignment", "Section 1.1 - 14 members, equal college credit")
    rows = []
    for no, name, part, sections, phase, share, tab in TEAM_ROWS:
        rows.append({"Member": f"Member {no}", "Name": f"**{name}**", "Part": f"Part {part}", "Notebook sections": sections,
                     "Phase": phase, "Paper share": f"{share}%", "College credit": f"{CREDIT_PER_MEMBER:.2f}%",
                     "Dashboard tab": tab})
    total_share = sum(r[5] for r in TEAM_ROWS)
    rows.append({"Member": "**Total**", "Name": "**14 members**", "Part": "", "Notebook sections": "Sections 1-17, Appendices A-B",
                 "Phase": "", "Paper share": f"**{total_share}%**", "College credit": f"**{CREDIT_PER_MEMBER * len(TEAM_ROWS):.2f}%**",
                 "Dashboard tab": "7 tabs"})
    caption_md("Table 1.4: Member credit assignment")
    md_table(pd.DataFrame(rows))
    phase_rows = []
    for ph in "ABC":
        mem = [r for r in TEAM_ROWS if r[4] == ph]
        phase_rows.append({"Phase": f"{ph}: {PHASE_NAMES[ph]}", "Members": len(mem),
                           "Member numbers": f"{mem[0][0]}-{mem[-1][0]}", "Sections": PHASE_SECTIONS[ph],
                           "Paper share": f"{sum(r[5] for r in mem)}%", "College credit": f"{CREDIT_PER_MEMBER * len(mem):.2f}%"})
    caption_md("Table 1.5: Credit by phase")
    show_df(pd.DataFrame(phase_rows), hide_index=True)
    png = chart_team_credit(tuple(r[1] for r in TEAM_ROWS), tuple(r[5] for r in TEAM_ROWS),
                            tuple(r[4] for r in TEAM_ROWS), CREDIT_PER_MEMBER)
    caption_md("Figure 1.1: Paper share per member against the equal college credit")
    show_png(png)
    cust, tx = ctx["cust"], ctx["tx"]
    interpret(f"All 14 members receive the same college credit of {CREDIT_PER_MEMBER:.2f}% (14 x {CREDIT_PER_MEMBER:.2f}% = 100%), "
              f"while the paper share varies slightly (7% for twelve members, 8% for Members 4 and 8).",
              f"Phase A (5 members) prepared the data that this dashboard currently shows for {num(len(cust))} customers and "
              f"{num(len(tx))} order lines, Phase B (5 members) explored spend, products and seasonality, and Phase C (4 members) "
              f"delivered the advanced analytics.")


def render_sec3(ctx, kp):
    """Section 3 (Member 2): dataset overview, data dictionary, built-in characteristics, raw vs clean preview."""
    bundle, tx = ctx["bundle"], ctx["tx"]
    raw, full = bundle["df_raw"], bundle["df_clean"]
    section_header("Section 3: Dataset Description and Data Dictionary", "Member 2 - Aditi Sharma")
    caption_md("Table 3.1: Dataset overview (documented properties and live values of the active selection)")
    live = {"Raw rows in df_raw": f"{len(raw):,} rows ({raw.shape[1]} columns)",
            "Customers": f"{kp['customers']:,} in the active selection ({full['customer_id'].nunique():,} in df_clean)",
            "Transaction IDs": f"{kp['orders']:,} in the active selection",
            "Products and categories": f"{kp['products']} products in {kp['categories']} categories (active selection)",
            "Period": f"{kp['first']:%Y-%m-%d} to {kp['last']:%Y-%m-%d} (active selection)"}
    overview = [(k, f"{v}  |  live: {live[k]}" if k in live else v) for k, v in DATASET_PROPERTIES]
    md_table(pd.DataFrame(overview, columns=["Property", "Description"]))
    removed = len(raw) - len(full) if bundle["mode"] == "synthetic" else None
    interpret(f"The raw extract has {len(raw):,} rows and {raw.shape[1]} columns; the cleaned table df_clean keeps "
              f"{len(full):,} valid sales lines" + (f" ({removed:,} rows or {100 * removed / len(raw):.2f}% were removed)." if removed is not None else "."),
              f"The active selection spans {(kp['last'] - kp['first']).days + 1} calendar days and {kp['lines']:,} lines, "
              f"i.e. {safe_div(kp['lines'], kp['orders'], 0):.2f} product lines per order.")

    caption_md("Table 3.2: Data dictionary of df_clean")
    md_table(pd.DataFrame(DATA_DICTIONARY, columns=["Column", "Expected dtype", "Description", "Example", "Business role"]))
    dt_found = pd.DataFrame({"Column": tx.columns, "dtype in the active df_clean": [str(t) for t in tx.dtypes],
                             "Distinct values": [int(tx[c].nunique()) for c in tx.columns]})
    show_df(dt_found, caption="Observed dtypes and distinct values in the active df_clean", hide_index=True)
    interpret(f"All {tx.shape[1]} documented columns are present in the active df_clean; {int(tx['customer_id'].nunique()):,} distinct "
              f"customer IDs, {int(tx['transaction_id'].nunique()):,} transaction IDs and {int(tx['product_name'].nunique())} product names "
              f"are stored as text, while quantity is {tx['quantity'].dtype} and the three monetary fields are numeric.",
              f"Because every quantity is positive (minimum {tx['quantity'].min()}) and total_amount equals "
              f"quantity x unit_price, revenue can be summed directly without further cleaning.")

    with st.expander("Table 3.3: Built-in data characteristics and intentional data-quality problems"):
        md_table(pd.DataFrame(BUSINESS_PATTERNS, columns=["Pattern", "How it is built into the data", "Analysed in"]))
        md_table(pd.DataFrame(DQ_PROBLEMS, columns=["Impurity", "Rows planted", "Signature", "Detected in", "Handled in"]))
        interpret(f"The raw table currently holds {int(raw.duplicated().sum()):,} exact duplicate rows, {int(raw['customer_id'].isna().sum()):,} rows without a "
                  f"customer_id, {int(raw['transaction_date'].isna().sum()):,} rows without a date and {int((pd.to_numeric(raw['quantity'], errors='coerce') < 0).sum()):,} "
                  f"returns (negative quantity).",
                  "These are the four planted impurity types of Table 3.3; Section 4 detects them and Section 5 removes or isolates them.")

    caption_md("Table 3.4: Raw data (df_raw) versus cleaned data (active df_clean) - first rows")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**df_raw** - {raw.shape[0]:,} rows x {raw.shape[1]} columns (untouched)")
        st.dataframe(display_dates(raw.head(10)), hide_index=True)
    with c2:
        st.markdown(f"**df_clean (active selection)** - {tx.shape[0]:,} rows x {tx.shape[1]} columns")
        st.dataframe(display_dates(tx.head(10)), hide_index=True)
    n_miss = int(raw.isna().any(axis=1).sum())
    n_neg = int((pd.to_numeric(raw["quantity"], errors="coerce") < 0).sum())
    interpret(f"df_raw holds {n_miss:,} rows with at least one missing value and {n_neg:,} rows with a negative quantity, "
              f"whereas the active df_clean contains none of them (every row has an ID, a date, a positive quantity and a price).",
              f"The cleaned table is {len(tx):,} rows long in this selection, so the preview shows the first 10 of "
              f"{len(tx):,} valid lines.")


def render_appendix_a(ctx, kp):
    """Appendix A (Member 2): generator source, parameters, self-test and schema-contract checks."""
    bundle, tx, cust = ctx["bundle"], ctx["tx"], ctx["cust"]
    section_header("Appendix A: Data generator and schema contract", "Member 2 - Aditi Sharma")
    st.markdown("**A.1 Generator source code** - the function below produced the synthetic `df_raw` used when no CSV "
                "files are available.")
    with st.expander("Show generate_retail_transactions() source code"):
        try:
            src = inspect.getsource(generate_retail_transactions)
        except (OSError, TypeError):
            src = "# The source code is not available in this runtime: see the generator function in app.py."
        st.code(src, language="python")
    caption_md("Table A.2: Generator parameters and reproducibility")
    md_table(pd.DataFrame(GENERATOR_PARAMS, columns=["Parameter", "Value"]))
    interpret(f"Calling the generator with seed {RANDOM_STATE} always returns {len(bundle['df_raw']):,} raw rows for "
              f"{bundle['df_raw']['customer_id'].nunique():,} customers, so any team member can reproduce this dashboard exactly.",
              "The 310 impurity rows are appended on purpose, which is why removing them in Section 5 leaves a fully "
              "reconciled table.")

    st.markdown("**Section 3.6 Generator self-test** (run on the reference raw data)")
    test = generator_self_test(bundle["df_raw"])
    show_df(test, hide_index=True)
    n_pass = int((test["Status"] == "PASS").sum())
    interpret(f"{n_pass} of {len(test)} self-test checks pass on the reference raw data "
              f"({'all structural and impurity counts agree with the documentation' if n_pass == len(test) else 'some documented counts differ'}).",
              f"For example, the raw table has {int(bundle['df_raw'].duplicated().sum())} exact duplicates, "
              f"{int(bundle['df_raw']['customer_id'].isna().sum())} missing customer IDs and "
              f"{int(bundle['df_raw']['transaction_date'].isna().sum())} missing dates.")

    st.markdown("**Schema contract and reconciliation of the active tables**")
    s1 = schema_report(tx, "df_clean")
    s2 = schema_report(cust[CUSTOMER_COLS], "customer_df")
    c1, c2 = st.columns(2)
    with c1:
        show_df(s1, caption="df_clean against the schema contract", hide_index=True)
    with c2:
        show_df(s2, caption="customer_df against the schema contract", hide_index=True)
    rec = reconciliation_checks(tx, cust)
    show_df(rec, caption="Reconciliation between df_clean and customer_df (active selection)", hide_index=True)
    n_ok = int((s1["status"] == "PASS").sum() + (s2["status"] == "PASS").sum() + (rec["status"] == "PASS").sum())
    n_all = len(s1) + len(s2) + len(rec)
    interpret(f"{n_ok} of {n_all} contract and reconciliation checks pass; revenue in df_clean ({money(tx['total_amount'].sum(), 2)}) "
              f"equals total_spend in customer_df ({money(cust['total_spend'].sum(), 2)}).",
              f"Both tables describe the same {num(len(cust))} customers and {num(int(cust['num_orders'].sum()))} orders, so Sections 6 to 16 "
              f"cannot disagree with each other.")
    if bundle["customer_check"] is not None:
        with st.expander("Check of the supplied customer_analytics_master.csv against df_clean"):
            show_df(bundle["customer_check"], hide_index=True)


# ===================================================================================================
# 15.  TAB 2  -  Data quality audit and cleaning pipeline  (Sections 4-5)   Members 3 and 4
# ===================================================================================================
@chart
def chart_audit_headline(labels, counts, n_rows):
    """Figure 4.4 - the four headline data-quality problems of the raw data."""
    colors = PALETTE_LIST[:4]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.2, width=0.62)
    top = max(max(counts), 1)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.015,
                f"{cnt:,}\n({_a_pct(cnt, n_rows):.2f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, top * 1.25)
    ax.set_ylabel("Number of rows affected")
    ax.set_title("Data Quality Issues Detected in the Raw Transaction Data", fontsize=14, fontweight="bold", pad=14)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, f"Base: {n_rows:,} raw rows. A row can fall into several categories, so the bars overlap "
                         f"and are not additive.", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig_to_png(fig)


@chart
def chart_cleaning_funnel(cleaning_log):
    """Figure 5.1 - two-panel cleaning funnel (rows retained vs removed per step)."""
    short_names = {1: "Exact\nduplicates", 2: "Missing\nID / date", 3: "Dates &\ntext trim",
                   4: "Returns &\ncancellations", 5: "Amount\ncheck"}
    flat = {k: v.replace("\n", " ") for k, v in short_names.items()}
    c_raw, c_step, c_final = PALETTE_LIST[:3]
    c_lost = "#C0392B"
    steps = cleaning_log["step_no"].astype(int).tolist()
    raw_rows = int(cleaning_log["rows_before"].iloc[0])
    kept = [raw_rows] + cleaning_log["rows_after"].astype(int).tolist()
    lost = [0] + cleaning_log["rows_removed"].astype(int).tolist()
    labels = ["Raw data\n(df_raw)"] + [f"Step {s}\n{short_names.get(s, '')}" for s in steps]
    bar_colors = [c_raw] + [c_step] * (len(steps) - 1) + [c_final]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.7, 1]})
    pos = np.arange(len(labels))
    ax1.bar(pos, kept, color=bar_colors, width=0.62, zorder=3)
    ax1.bar(pos, lost, bottom=kept, color=c_lost, width=0.62, zorder=3)
    ax1.set_ylim(0, max(k + l for k, l in zip(kept, lost)) * 1.28)
    for p, k, l in zip(pos, kept, lost):
        ax1.annotate(f"{k:,}\n({100 * k / max(raw_rows, 1):.1f}% of raw)", (p, k + l), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if l > 0:
            ax1.annotate(f"-{l:,}", (p, k + l), xytext=(0, 34), textcoords="offset points", ha="center", va="bottom",
                         fontsize=9, fontweight="bold", color=c_lost)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.yaxis.set_major_formatter(thousands_formatter)
    ax1.set_ylabel("Number of rows")
    ax1.set_title("Rows remaining after each step (red = removed at that step)", fontsize=11)
    removed_vals = cleaning_log["rows_removed"].astype(int).to_numpy()
    ypos = np.arange(len(steps))
    xmax = max(int(removed_vals.max()), 1)
    ax2.barh(ypos, removed_vals, color=c_lost, height=0.55, zorder=3)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f"Step {s}: {flat.get(s, '')}" for s in steps], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(0, xmax * 1.6)
    for p, v, pc in zip(ypos, removed_vals, cleaning_log["pct_removed"]):
        ax2.text(v + xmax * 0.02, p, f"{v:,} rows ({pc:.2f}%)", va="center", fontsize=9)
    ax2.xaxis.set_major_formatter(thousands_formatter)
    ax2.set_xlabel("Rows removed")
    ax2.set_title("Rows removed per step (% of rows entering the step)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax2.grid(axis="x", alpha=0.3, zorder=0)
    ax1.grid(axis="x", visible=False)
    ax2.grid(axis="y", visible=False)
    fig.suptitle("Figure 5.1 - Data cleaning funnel: from df_raw to df_clean", fontsize=14, fontweight="bold")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c_step), plt.Rectangle((0, 0), 1, 1, color=c_lost)],
               ["Rows retained after the step", "Rows removed at the step"], loc="lower center", ncol=2,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return fig_to_png(fig)


def filter_raw_pipeline(bundle, flt):
    """Re-run the raw-to-clean pipeline on the raw rows inside the sidebar Date range and Category selection.

    Spend-segment and buyer-type labels belong to customers of df_clean, so they cannot be applied to raw rows
    (guest-checkout rows have no customer).  Returns the bundle itself when the filters are at their defaults,
    a modified copy when they narrow the data, and None when no raw row is left.
    """
    raw, full = bundle["df_raw"], bundle["df_clean"]
    dmin, dmax = full["transaction_date"].min().date(), full["transaction_date"].max().date()
    date_full = flt["start"] <= dmin and flt["end"] >= dmax
    cat_full = set(flt["categories"]) == set(full["category"].unique())
    if date_full and cat_full:
        return bundle
    day = raw["transaction_date"]
    in_date = (day >= pd.Timestamp(flt["start"])) & (day < pd.Timestamp(flt["end"]) + pd.Timedelta(days=1))
    if date_full:
        in_date = in_date | day.isna()                       # undated rows can only be kept for the complete range
    in_cat = raw["category"].isin(flt["categories"]) | (cat_full & raw["category"].isna())
    raw_f = raw.loc[in_date & in_cat].reset_index(drop=True)
    if raw_f.empty:
        return None
    clean_f, log_f, rets_f = clean_raw_data(raw_f)
    return dict(bundle, df_raw=raw_f, df_clean=clean_f, cleaning_log=log_f, df_returns=rets_f)


def render_tab2(ctx):
    """Tab 2 - Section 4 audit (Member 3) and Section 5 cleaning (Member 4)."""
    bundle = ctx["bundle"]
    st.markdown("## Data Quality Audit and Cleaning Pipeline")
    st.caption("Sections 4-5 | Member 3 - Sonam (audit) and Member 4 - Samiya Khan (cleaning)")
    view = bundle
    if bundle["mode"] == "synthetic":
        apply_filters = st.checkbox("Apply the sidebar Date range and Category filters to Sections 4-5", value=True, key="tab2_apply_filters",
                                    help="On: the audit and the cleaning pipeline are re-run for the raw rows inside the selected dates and categories. "
                                         "Off: the complete raw dataset is shown. Spend-segment and Buyer-type filters cannot be applied to raw "
                                         "rows (guest rows have no customer). Raw rows without a date are kept only for the complete date range.")
        if apply_filters:
            view = filter_raw_pipeline(bundle, ctx["flt"])
            if view is None:
                st.info("No raw row falls inside the selected date range and categories; widen the sidebar filters.")
                return
        if view is bundle:
            st.info("Showing the **complete raw-to-clean pipeline** (filters at their defaults, or switched off above).")
        else:
            st.info(f"Showing the raw-to-clean pipeline for the **{len(view['df_raw']):,} raw rows** inside the selected date range and categories "
                    f"(of {len(bundle['df_raw']):,}).")
    else:
        st.warning("Only cleaned tables were supplied, so no raw file is available. Sections 4 and 5 are demonstrated on the "
                   "project's reference raw dataset (RANDOM_STATE = 42), which the sidebar filters do not apply to; your own df_clean is "
                   "validated in Table 5.5.")
    ctx2 = dict(ctx, bundle=view)
    s4, s5 = st.tabs(["Section 4: Data quality audit", "Section 5: Data cleaning & log"])
    with s4:
        run_safely(render_sec4, ctx2)
    with s5:
        run_safely(render_sec5, ctx2)


def render_sec4(ctx):
    """Section 4 (Member 3): audit of df_raw."""
    raw = ctx["bundle"]["df_raw"]
    a = audit_raw(raw)
    section_header("Section 4: Data Inspection and Quality Audit", "Member 3 - Sonam | df_raw is read-only here")

    caption_md("Figure 4.4: Data quality issues detected in the raw transaction data")
    show_png(chart_audit_headline(tuple(a["headline_labels"]), tuple(a["headline_counts"]), a["n_rows"]))
    interpret(*a["headline_text"])

    show_df(a["recon_table"], caption="Table 4.3c: Arithmetic reconciliation of total_amount (quantity x unit_price == total_amount, tolerance 0.01)")
    if a["recon_examples"] is not None:
        show_df(a["recon_examples"], caption="Table 4.3c (continued): rows with the largest reconciliation differences", hide_index=True)
    interpret(*a["recon_text"])

    show_df(a["report_table"], caption="Table 4.4: Consolidated data quality audit report")
    interpret(*a["report_text"])

    with st.expander("Table 4.1: Dimensions, memory and expected versus actual data types"):
        show_df(a["dim_table"], caption="Table 4.1a: Dataset dimensions and memory footprint")
        interpret(*a["dim_text"])
        show_df(a["dtype_table"], caption="Table 4.1b: Expected versus actual data types")
        interpret(*a["dtype_text"])
    with st.expander("Table 4.2: Missing values per column"):
        show_df(a["missing_table"])
        interpret(*a["missing_text"])
    with st.expander("Table 4.3a / 4.3b: Duplicates, transaction_id integrity, negative and zero values"):
        show_df(a["dup_table"], caption="Table 4.3a: Exact duplicates and transaction_id integrity")
        interpret(*a["dup_text"])
        show_df(a["anomaly_table"], caption="Table 4.3b: Negative and zero values in the numeric columns")
        interpret(*a["anomaly_text"])


def render_sec5(ctx):
    """Section 5 (Member 4): cleaning plan, issue profile, 5-step cleaning log, returns, funnel, validation."""
    bundle = ctx["bundle"]
    raw, log, rets, full = bundle["df_raw"], bundle["cleaning_log"], bundle["df_returns"], bundle["df_clean"]
    section_header("Section 5: Data Cleaning and Cleaning Log", "Member 4 - Samiya Khan")

    caption_md("Table 5.1: Planned cleaning pipeline")
    md_table(pd.DataFrame(PLANNED_PIPELINE, columns=["Step", "Cleaning stage", "Rule applied", "Reason"]))
    interpret(f"The five planned rules were applied in this order to {len(raw):,} raw rows and left {len(full):,} valid sales lines "
              f"({100 * len(full) / max(len(raw), 1):.2f}% retained); the order matters because duplicates are removed first, so that a repeated "
              f"row can never be counted twice in a later step.",
              "Every rule is logged with its row counts in Table 5.3, so the whole transformation can be audited step by step.")

    profile = raw_issue_profile(raw)
    show_df(profile, caption=f"Table 5.2: Data-quality profile of df_raw before cleaning ({len(raw):,} rows)", hide_index=True)
    top = profile.sort_values("rows_affected", ascending=False).iloc[0]
    n_issue_types = int((profile["rows_affected"] > 0).sum())
    interpret(f"Table 5.2 counts every problem independently, so one row can appear under several issues; "
              f"{n_issue_types} of the {len(profile)} issue types occur in the raw data and the largest is "
              f"**{esc(top['issue'])}** with {int(top['rows_affected']):,} rows ({top['pct_of_raw_rows']:.2f}% of {len(raw):,}).",
              "These counts are the baseline against which the cleaning log below is reconciled.")

    show_df(log, caption="Table 5.3: cleaning_log - rows before / after every cleaning step", hide_index=True)
    removed_total = int(log["rows_removed"].sum())
    big = log.sort_values("rows_removed", ascending=False).iloc[0]
    interpret(f"The five steps remove {removed_total:,} rows in total, taking the table from {int(log['rows_before'].iloc[0]):,} "
              f"to {int(log['rows_after'].iloc[-1]):,} rows ({100 * log['rows_after'].iloc[-1] / max(log['rows_before'].iloc[0], 1):.2f}% retained).",
              f"The largest single loss is Step {int(big['step_no'])} ({esc(big['cleaning_stage'])}) with {int(big['rows_removed']):,} rows "
              f"({big['pct_removed']:.2f}% of the rows entering that step); Step 5 only verifies amounts and normally removes nothing.")

    rs = returns_summary(rets)
    if len(rs):
        show_df(rs, caption="Table 5.4: Returns and cancellations moved out of df_clean (Step 4)", hide_index=True)
        tot = rs.iloc[-1]
        interpret(f"Step 4 set aside {int(tot['rows']):,} rows with a net quantity of {tot['net_quantity']:,.0f} units and a net "
                  f"amount of {money(tot['net_amount'], 0)}.",
                  f"These rows are kept in `_df_returns` instead of being deleted, so revenue in df_clean "
                  f"({money(full['total_amount'].sum(), 0)}) represents gross sales and the returns can be analysed separately.")
    else:
        st.info("Table 5.4 - no returns or cancellations were found, so _df_returns is empty.")

    caption_md("Figure 5.1: Cleaning funnel - rows remaining and removed after every step")
    show_png(chart_cleaning_funnel(log))
    kept = int(log["rows_after"].iloc[-1])
    interpret(f"The funnel starts with {int(log['rows_before'].iloc[0]):,} raw rows and ends with {kept:,} rows, equal to the "
              f"row count of df_clean ({len(full):,}).",
              f"The right-hand panel shows that Step {int(big['step_no'])} is the biggest source of data loss "
              f"({int(big['rows_removed']):,} rows), which confirms that cleaning was a controlled, step-wise reduction.")

    synthetic = bundle["mode"] == "synthetic"
    checks = validate_clean(full, raw if synthetic else None, log if synthetic else None, rets if synthetic else None)
    show_df(checks, caption="Table 5.5: Validation of df_clean against the schema contract", hide_index=True)
    n_pass, n_fail = int((checks["status"] == "PASS").sum()), int((checks["status"] == "FAIL").sum())
    interpret(f"{n_pass} of {len(checks)} validation checks pass and {n_fail} fail"
              f"{'' if synthetic else ' (row reconciliation with a raw file is skipped because only cleaned data were supplied)'}.",
              f"The validated table has {len(full):,} rows, {full['customer_id'].nunique():,} customers and strictly positive quantity, "
              f"price and amount, so it can be handed to Section 6 without further repair.")
    st.markdown("**Hand-off to the next parts**")
    st.markdown("- `df_clean` is the only transaction table; `df_raw` stays untouched for audit purposes and `_df_returns` holds "
                "the returns and cancellations.\n- Revenue and spend are gross sales (before returns).\n- Missing identifiers and "
                "dates were deleted rather than imputed, because an invented ID or date would create customers or periods that "
                "never existed.\n- No outlier filter was applied: large but positive purchases are valid sales.")


# ===================================================================================================
# 16.  TAB 3  -  Customer summary and spend segmentation  (Sections 6-8)
#      Members 5 (Aditya Singh), 6 (Saif Khan) and 7 (Divyanshi)
# ===================================================================================================
NUM_COLS = ["total_spend", "num_orders", "avg_order_value", "recency_days"]


def desc_stats(cust):
    """Table 6.3 - Mean, SD, Min, 25%, Median, 75%, Max and Skewness of the four customer metrics."""
    return pd.DataFrame({"Mean": cust[NUM_COLS].mean(), "SD": cust[NUM_COLS].std(),
                         "Min": cust[NUM_COLS].min(), "25%": cust[NUM_COLS].quantile(0.25),
                         "Median": cust[NUM_COLS].median(), "75%": cust[NUM_COLS].quantile(0.75),
                         "Max": cust[NUM_COLS].max(), "Skewness": cust[NUM_COLS].skew()})


def schema_rules_table(cust):
    """Table 6.2 - column-level rules of customer_df (dtype / nulls / ranges) with OK / FAIL status."""
    is_num = pd.api.types.is_numeric_dtype
    rules = {
        "customer_id": ("id: unique, non-null", lambda s: s.notna().all() and s.is_unique),
        "total_spend": ("numeric, non-null, >= 0", lambda s: is_num(s) and s.notna().all() and (s >= 0).all()),
        "num_orders": ("numeric, non-null, >= 1", lambda s: is_num(s) and s.notna().all() and (s >= 1).all()),
        "avg_order_value": ("numeric, non-null, = total_spend / num_orders",
                            lambda s: is_num(s) and s.notna().all() and np.allclose(s, cust["total_spend"] / cust["num_orders"])),
        "first_purchase_date": ("datetime64, non-null", lambda s: pd.api.types.is_datetime64_any_dtype(s) and s.notna().all()),
        "last_purchase_date": ("datetime64, non-null, >= first_purchase_date",
                               lambda s: pd.api.types.is_datetime64_any_dtype(s) and s.notna().all()
                               and (s >= cust["first_purchase_date"]).all()),
        "recency_days": ("numeric, non-null, >= 0", lambda s: is_num(s) and s.notna().all() and (s >= 0).all()),
    }
    rows = []
    for col, (rule, test) in rules.items():
        try:
            ok = bool(test(cust[col]))
        except Exception:
            ok = False
        rows.append({"column": col, "dtype": str(cust[col].dtype), "nulls": int(cust[col].isna().sum()),
                     "rule": rule, "status": "OK" if ok else "FAIL"})
    return pd.DataFrame(rows)


def hbox(ax, data, color):
    """Horizontal boxplot that works on old and new matplotlib versions."""
    kw = dict(patch_artist=True, widths=0.6, showmeans=True, boxprops=dict(facecolor=color, alpha=0.6),
              medianprops=dict(color="black", linewidth=1.5),
              meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4),
              flierprops=dict(marker="o", markersize=3, alpha=0.35))
    try:
        ax.boxplot(data, orientation="horizontal", **kw)               # matplotlib >= 3.10
    except TypeError:
        ax.boxplot(data, vert=False, **kw)                             # older matplotlib


@chart
def chart_customer_distributions(df3):
    """Figure 6.4 - histograms (top) with horizontal boxplots (bottom) for spend, orders and recency."""
    colors3 = PALETTE_LIST[:3]
    specs = [("total_spend", f"Total spend per customer ({CURRENCY_SYMBOL})", False),
             ("num_orders", "Orders per customer", True),
             ("recency_days", "Recency (days since last purchase)", False)]

    def legend_val(col, v):
        return f"{CURRENCY_SYMBOL}{v:,.0f}" if col == "total_spend" else f"{v:,.1f}"

    fig, axes = plt.subplots(2, 3, figsize=(16, 6.2), sharex="col", gridspec_kw={"height_ratios": [4, 1]})
    for j, (col, xlabel, discrete) in enumerate(specs):
        ax_hist, ax_box = axes[0, j], axes[1, j]
        vals = df3[col].astype(float)
        bins = np.arange(vals.min() - 0.5, vals.max() + 1.5, 1.0) if (discrete and vals.nunique() <= 30) else 30
        ax_hist.hist(vals, bins=bins, color=colors3[j], edgecolor="white", alpha=0.9)
        ax_hist.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.4, label=f"Mean {legend_val(col, vals.mean())}")
        ax_hist.axvline(vals.median(), color="#B22222", linestyle=":", linewidth=1.8,
                        label=f"Median {legend_val(col, vals.median())}")
        ax_hist.set_title(f"({'abc'[j]}) {xlabel}", loc="left", fontsize=11, fontweight="bold")
        ax_hist.legend(frameon=False, fontsize=9, loc="upper right")
        ax_hist.grid(axis="y", alpha=0.25)
        ax_hist.grid(axis="x", visible=False)
        if j == 0:
            ax_hist.set_ylabel("Number of customers")
        hbox(ax_box, vals, colors3[j])
        ax_box.set_yticks([])
        ax_box.set_xlabel(xlabel)
        ax_box.grid(axis="y", visible=False)
        if col == "total_spend":
            ax_box.xaxis.set_major_formatter(currency_formatter)
            ax_box.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        if discrete:
            ax_box.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.suptitle(f"Customer-level distributions (n = {len(df3):,} customers)", fontsize=14, fontweight="bold")
    fig.tight_layout(h_pad=0.4, rect=[0, 0, 1, 0.95])
    return fig_to_png(fig)


def render_tab3(ctx):
    """Tab 3 - Sections 6, 7 and 8."""
    st.markdown("## Customer Summary and Spend Segmentation")
    st.caption("Sections 6-8 | Member 5 - Aditya Singh | Member 6 - Saif Khan | Member 7 - Divyanshi")
    s6, s7, s8 = st.tabs(["Section 6: Customer-level summary", "Section 7: Spend segmentation",
                          "Section 8: Segment comparison"])
    with s6:
        run_safely(render_sec6, ctx)
    with s7:
        run_safely(render_sec7, ctx)
    with s8:
        run_safely(render_sec8, ctx)


# ---------------------------------------------------------------------------------------------------
# Section 6
# ---------------------------------------------------------------------------------------------------
def render_sec6(ctx):
    """Section 6 (Member 5): customer_df preview, schema checks, descriptive statistics and distributions."""
    cust, tx, snapshot = ctx["cust"], ctx["tx"], ctx["snapshot"]
    section_header("Section 6: Customer-Level Summary Table", "Member 5 - Aditya Singh")
    st.markdown("**Aggregation logic.** Each customer becomes one row: `total_spend` = sum of `total_amount`, `num_orders` = "
                "number of distinct `transaction_id`, `avg_order_value` = `total_spend / num_orders`, `first_purchase_date` / "
                "`last_purchase_date` = earliest / latest `transaction_date`, and `recency_days` = snapshot date minus "
                f"`last_purchase_date`, where snapshot date = latest transaction date + 1 day = **{snapshot:%d %b %Y}**.")

    base = cust[CUSTOMER_COLS]
    show_df(display_dates(base.head(10)), caption=f"Table 6.1: Preview of customer_df ({len(base):,} rows x {base.shape[1]} columns)",
            hide_index=True)
    interpret(f"The preview confirms the intended tidy structure: each row is one customer ({len(base):,} in the active selection) "
              f"and each column is one metric, aggregated from {len(tx):,} transaction lines.",
              f"For the first customer ({esc(base['customer_id'].iloc[0])}) the table shows {money(base['total_spend'].iloc[0])} spent over "
              f"{plural(base['num_orders'].iloc[0], 'order')} with {int(base['recency_days'].iloc[0])} days since the last purchase.")

    rules = schema_rules_table(base)
    show_df(rules, caption="Table 6.2: Schema contract of customer_df (column-level rules)", hide_index=True)
    rec = reconciliation_checks(tx, base)
    show_df(rec, caption="Cross-check reconciliation: df_clean versus customer_df", hide_index=True)
    n_ok = int((rules["status"] == "OK").sum()) + int((rec["status"] == "PASS").sum())
    interpret(f"{n_ok} of {len(rules) + len(rec)} checks pass: all seven base columns obey the contract and the two tables agree on "
              f"customers ({tx['customer_id'].nunique():,}), orders ({int(base['num_orders'].sum()):,}) and revenue "
              f"({money(base['total_spend'].sum(), 2)}).",
              "A failing line would stop the merge in the master notebook, which is why every later section can trust customer_df.")

    ds = desc_stats(base)
    shown = ds.rename(index={"total_spend": f"total_spend ({CURRENCY_SYMBOL})", "num_orders": "num_orders (count)",
                             "avg_order_value": f"avg_order_value ({CURRENCY_SYMBOL})", "recency_days": "recency_days (days)"}).round(2)
    show_df(shown, caption=f"Table 6.3: Descriptive statistics of customer_df (n = {len(base):,} customers)")
    st_ = lambda row, stat: float(ds.loc[row, stat])                       # noqa: E731 - tiny accessor
    sk = st_("total_spend", "Skewness")
    mean_sp, med_sp = st_("total_spend", "Mean"), st_("total_spend", "Median")
    rel = "above" if mean_sp > med_sp else ("below" if mean_sp < med_sp else "equal to")
    if pd.isna(sk):
        tail = "there is too little variation to judge the shape"
    elif sk >= 0.5:
        tail = ("a minority of high-spending customers pulls the mean above the typical customer, so the median "
                "describes the typical customer better than the mean")
    elif sk <= -0.5:
        tail = ("a minority of low-spending customers pulls the mean below the typical customer, so the median "
                "describes the typical customer better than the mean")
    else:
        tail = "mean and median are close, so the average is a fair description of a typical customer"
    interpret(f"The mean customer spend ({money(mean_sp)}) is {rel} the median ({money(med_sp)}) and a skewness of {num(sk, 2)} makes "
              f"`total_spend` {skew_label(sk)}: {tail}.",
              f"Customers place a median of {st_('num_orders', 'Median'):g} {'order' if st_('num_orders', 'Median') == 1 else 'orders'} (mean {num(st_('num_orders', 'Mean'), 2)}, "
              f"maximum {num(st_('num_orders', 'Max'))}) at an average order value of {money(st_('avg_order_value', 'Mean'))}; because "
              f"customer_df is tidy each summary is read directly from a single column (Wickham, 2014).",
              f"Recency spans {num(st_('recency_days', 'Min'))} to {num(st_('recency_days', 'Max'))} days before the snapshot date "
              f"(median {num(st_('recency_days', 'Median'))} days, {skew_label(st_('recency_days', 'Skewness'))}), which together with "
              f"frequency and spend gives the recency-frequency-monetary view of customer-base analysis (Fader et al., 2005).")

    caption_md("Figure 6.4: Distribution of spend, frequency and recency")
    show_png(chart_customer_distributions(base[["total_spend", "num_orders", "recency_days"]]))
    sp = base["total_spend"]
    q1, q3 = sp.quantile(0.25), sp.quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    n_out = int((sp > fence).sum())
    pct_once = 100 * float((base["num_orders"] == 1).mean())
    r1, r3 = base["recency_days"].quantile(0.25), base["recency_days"].quantile(0.75)
    interpret(f"In panel (a) `total_spend` is {skew_label(sp.skew())}: {n_out:,} customers ({100 * n_out / len(base):.1f}%) lie above the "
              f"upper whisker of the boxplot (Q3 + 1.5 x IQR = {money(fence)}), while the median customer spends {money(sp.median())}.",
              f"Panel (b) shows that {pct_once:.1f}% of customers placed exactly one order and the most active customer placed "
              f"{int(base['num_orders'].max()):,}, whereas panel (c) shows that the middle 50% of customers last purchased between "
              f"{r1:.0f} and {r3:.0f} days before the snapshot date ({snapshot:%d %b %Y}).",
              "Together the three panels give a frequency, monetary and recency view of every customer from one tidy table "
              "(Wickham, 2014) and motivate the segmentation steps in the following sections.")


# ---------------------------------------------------------------------------------------------------
# Section 7
# ---------------------------------------------------------------------------------------------------
def s7_skew_class(g):
    if g >= 1.0:
        return "strongly right-skewed"
    if g >= 0.5:
        return "moderately right-skewed"
    if g > -0.5:
        return "approximately symmetric"
    return "left-skewed"


def s7_diagnostics(spend):
    """Tables 7.1a / 7.1b and their interpretations (distribution diagnostics of total_spend)."""
    spend = spend.astype(float)
    positive = bool((spend > 0).all())
    n = int(spend.shape[0])
    mean, std, vmin, vmax = float(spend.mean()), float(spend.std()), float(spend.min()), float(spend.max())
    q1, p33, median, p67, q3, p90, p99 = (float(spend.quantile(p)) for p in (0.25, 1 / 3, 0.50, 2 / 3, 0.75, 0.90, 0.99))
    ratio = mean / median if median > 0 else float("nan")
    cv = std / mean if mean > 0 else float("nan")
    skew = float(spend.skew())
    kurt = float(spend.kurt())
    bowley = ((q3 + q1 - 2 * median) / (q3 - q1)) if q3 > q1 else float("nan")
    skew_log = float(np.log1p(spend).skew()) if positive else float("nan")
    m = lambda v: money(v, 2)                                              # noqa: E731
    stats_tbl = pd.DataFrame([("Customers (count)", f"{n:,}"), ("Mean", m(mean)), ("Standard deviation", m(std)),
                              ("Minimum", m(vmin)), ("25th percentile (Q1)", m(q1)),
                              ("33.33rd percentile  -> low_upper candidate", m(p33)), ("Median (50th percentile)", m(median)),
                              ("66.67th percentile  -> med_upper candidate", m(p67)), ("75th percentile (Q3)", m(q3)),
                              ("90th percentile", m(p90)), ("99th percentile", m(p99)), ("Maximum", m(vmax))],
                             columns=["Statistic", "total_spend"]).set_index("Statistic")
    has_tail = (skew >= 0.5) or (ratio >= 1.15)
    if has_tail:
        text_a = [f"The median customer spends {m(median)}, but the mean is {m(mean)} ({ratio:.2f}x the median) and the 99th percentile "
                  f"reaches {m(p99)} ({safe_div(p99, median, float('nan')):.1f}x the median).",
                  "A mean pulled well above the median is the signature of a long right tail, so resistant, rank-based summaries "
                  "(median, quartiles, percentiles) describe a typical customer better than the mean (Tukey, 1977).",
                  "A small group spending far more than everybody else is also the situation described by the 80/20 principle "
                  "(Koch, 1998), which is why the top spenders deserve a tier of their own."]
    else:
        text_a = [f"The median customer spends {m(median)} and the mean is {m(mean)} ({ratio:.2f}x the median), so no pronounced "
                  f"high-spending tail pulls the mean upward.",
                  "Rank-based summaries (median, quartiles, percentiles) are nevertheless the most resistant description of a typical "
                  "customer (Tukey, 1977).",
                  "Percentile cut-offs still give balanced, easily explained tiers, and any concentration of value in the top tier "
                  "(Koch, 1998) is quantified in the later sections."]
    ratio_read = "mean pulled up by high spenders" if ratio >= 1.15 else "mean and median are close"
    cv_read = "spread exceeds the average spend" if cv > 1 else "moderate relative spread"
    kurt_read = ("heavier tails than a normal curve" if kurt > 1 else "lighter tails than a normal curve" if kurt < -1
                 else "tails close to a normal curve")
    if not np.isfinite(bowley):
        bowley_read = "undefined (interquartile range is zero)"
    elif bowley > 0.1:
        bowley_read = "upper half of customers is more spread out (right-skewed centre)"
    elif bowley < -0.1:
        bowley_read = "lower half of customers is more spread out (left-skewed centre)"
    else:
        bowley_read = "roughly symmetric central half"
    if not np.isfinite(skew_log):
        log_read = "not computed (non-positive values)"
    elif abs(skew_log) < 0.5:
        log_read = "near-symmetric after log re-expression"
    else:
        log_read = "still skewed after log re-expression"
    shape_tbl = pd.DataFrame([("Mean / median ratio", f"{ratio:.2f}", ratio_read),
                              ("Coefficient of variation (SD / mean)", f"{cv:.2f}", cv_read),
                              ("Skewness (adjusted Fisher-Pearson, G1)", f"{skew:.2f}", s7_skew_class(skew)),
                              ("Excess kurtosis (G2)", f"{kurt:.2f}", kurt_read),
                              ("Bowley quartile skewness", f"{bowley:.2f}", bowley_read),
                              ("Skewness of log(1 + total_spend)", f"{skew_log:.2f}", log_read)],
                             columns=["Diagnostic", "Value", "Reading"]).set_index("Diagnostic")
    if np.isfinite(bowley):
        if abs(bowley) <= 0.05 and abs(skew) < 0.5:
            bw = "is also close to zero, confirming a symmetric central half"
        elif abs(bowley) <= 0.05:
            bw = "is close to zero, so the asymmetry sits in the extreme tail rather than in the central half"
        elif (bowley > 0) == (skew > 0):
            bw = "points the same way"
        else:
            bw = "points the opposite way, so the central half and the tail are skewed differently"
        bw_phrase = f", and Bowley's robust quartile skewness of {bowley:.2f} {bw}"
    else:
        bw_phrase = ""
    tb = f"The skewness of {skew:.2f} classifies `total_spend` as {s7_skew_class(skew)} (excess kurtosis {kurt:.1f}){bw_phrase}."
    if skew >= 0.5 and np.isfinite(skew_log) and abs(skew_log) < abs(skew):
        tb += (f" After a log re-expression the skewness falls to {skew_log:.2f}, i.e. differences between customers are "
               f"multiplicative rather than additive - the situation in which Tukey (1977) advocates re-expressing the data.")
    elif skew >= 0.5:
        tb += " A log re-expression does not remove the asymmetry here, so resistant, rank-based summaries are the more dependable route (Tukey, 1977)."
    else:
        tb += " The shape is not strongly right-skewed, so a re-expression is not essential here (Tukey, 1977)."
    tb += (" Because customer value is typically concentrated in a minority of buyers (Koch, 1998), tertile (33.33% / 66.67%) "
           "cut-offs stay balanced whatever the exact shape, which justifies the percentile rule.")
    return stats_tbl, text_a, shape_tbl, tb


@chart
def chart_spend_segmentation(df2, low_upper, med_upper, present):
    """Figure 7.4 - customers per segment (bars) and spend distribution per segment (box plots)."""
    segs = list(present)
    colors = [SEGMENT_COLORS[s] for s in segs]
    counts = df2["spend_segment"].value_counts().reindex(segs).fillna(0).astype(int)
    n_total = int(counts.sum())
    data = [df2.loc[df2["spend_segment"] == s, "total_spend"].to_numpy(dtype=float) for s in segs]
    lo, hi = min(float(d.min()) for d in data), max(float(d.max()) for d in data)
    use_log = (lo > 0) and (hi / lo >= 20)
    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    bars = ax_bar.bar(segs, counts.to_numpy(), color=colors, edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
    ax_bar.axhline(n_total / 3, color="#555555", linestyle="--", linewidth=1.1, zorder=2,
                   label=f"Equal split (N/3 = {n_total / 3:,.0f})")
    for bar, cnt in zip(bars, counts.to_numpy()):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, cnt + n_total * 0.012, f"{cnt:,}\n({cnt / n_total:.1%})",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222222")
    ax_bar.set_ylim(0, max(counts.max(), n_total / 3) * 1.35)
    ax_bar.set_title("Customers per Spend Segment", fontsize=13, fontweight="bold")
    ax_bar.set_xlabel("Spend segment")
    ax_bar.set_ylabel("Number of customers")
    ax_bar.legend(loc="upper center", fontsize=9, frameon=False)
    pos = list(range(1, len(segs) + 1))
    bp = ax_box.boxplot(data, positions=pos, widths=0.55, whis=1.5, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#222222", markersize=7),
                        medianprops=dict(color="#222222", linewidth=2.2), whiskerprops=dict(color="#444444", linewidth=1.3),
                        capprops=dict(color="#444444", linewidth=1.3),
                        flierprops=dict(marker="o", markersize=3.5, markeredgewidth=0, alpha=0.45))
    for i, col in enumerate(colors):
        bp["boxes"][i].set(facecolor=col, edgecolor="#333333", linewidth=1.3)
        bp["fliers"][i].set(markerfacecolor=col)
    ax_box.axhline(low_upper, color="#555555", linestyle="--", linewidth=1.1, zorder=1, label=f"low_upper = {money(low_upper)}")
    ax_box.axhline(med_upper, color="#555555", linestyle=":", linewidth=1.5, zorder=1, label=f"med_upper = {money(med_upper)}")
    ax_box.plot([], [], color="#222222", linewidth=2.2, label="Median")
    ax_box.plot([], [], linestyle="none", marker="D", markerfacecolor="white", markeredgecolor="#222222", markersize=7, label="Mean")
    for i, d in enumerate(data, start=1):
        mv = float(np.median(d))
        ax_box.text(i + 0.32, mv, f"Median\n{money(mv)}", va="center", ha="left", fontsize=9, color="#222222")
    ax_box.set_xlim(0.4, len(segs) + 0.95)
    ax_box.set_xticks(pos)
    ax_box.set_xticklabels(segs)
    if use_log:
        ax_box.set_yscale("log")
    ax_box.yaxis.set_major_formatter(currency_formatter)
    scale_txt = " (log scale)" if use_log else ""
    ax_box.set_title("Distribution of Total Spend by Segment" + scale_txt, fontsize=13, fontweight="bold")
    ax_box.set_xlabel("Spend segment")
    ax_box.set_ylabel("Total spend" + scale_txt)
    ax_box.legend(loc="upper left", fontsize=9, frameon=True)
    for ax in (ax_bar, ax_box):
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Customer Spend Segmentation: Low / Medium / High (tertiles of total_spend)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig_to_png(fig)


def render_sec7(ctx):
    """Section 7 (Member 6): distribution diagnostics, tertile thresholds, segment profile and figure."""
    cust = ctx["cust"]
    low_upper, med_upper = ctx["thresholds"]["low_upper"], ctx["thresholds"]["med_upper"]
    section_header("Section 7: Customer Spend Segmentation (Low / Medium / High)", "Member 6 - Saif Khan")
    if len(cust) < 3:
        st.info("At least three customers are needed for the spend diagnostics; widen the sidebar selection.")
        return
    stats_tbl, text_a, shape_tbl, text_b = s7_diagnostics(cust["total_spend"])
    show_df(stats_tbl, caption="Table 7.1a: Percentile ladder and summary statistics of total_spend")
    interpret(*text_a)
    show_df(shape_tbl, caption="Table 7.1b: Shape diagnostics of total_spend")
    interpret(text_b)

    present = [s for s in SEGMENT_ORDER if (cust["spend_segment"] == s).any()]
    st.markdown("**7.2 Thresholds and segment assignment**")
    st.markdown(f"Tertile cut-offs of `total_spend` (computed on all customers of the selected period and categories): "
                f"`low_upper` = **{money(low_upper, 2)}** (33.33rd percentile) and `med_upper` = **{money(med_upper, 2)}** "
                f"(66.67th percentile). Rule: **Low** if `total_spend <= low_upper`; **Medium** if `low_upper < total_spend <= "
                f"med_upper`; **High** if `total_spend > med_upper`.")
    checks = [("Cut-offs are strictly increasing (low_upper < med_upper)", "True", str(low_upper < med_upper),
               "PASS" if low_upper < med_upper else "FAIL"),
              ("spend_segment holds only the labels Low, Medium and High", "subset of {Low, Medium, High}",
               ", ".join(sorted(cust["spend_segment"].unique())), "PASS" if set(cust["spend_segment"]) <= set(SEGMENT_ORDER) else "FAIL"),
              ("spend_segment is plain string data", "str / object", str(cust["spend_segment"].dtype),
               "PASS" if (pd.api.types.is_object_dtype(cust["spend_segment"]) or pd.api.types.is_string_dtype(cust["spend_segment"])) else "FAIL")]
    if len(present) == 3:
        try:
            qc = pd.qcut(cust["total_spend"], q=3, labels=SEGMENT_ORDER).astype(str)
            same = bool((qc.to_numpy() == cust["spend_segment"].to_numpy()).all())
            checks.append(("Explicit percentile cut-offs reproduce pd.qcut(q=3)", "identical", "identical" if same else "differs (ties)",
                           "PASS" if same else "REVIEW"))
        except ValueError:
            checks.append(("Explicit percentile cut-offs reproduce pd.qcut(q=3)", "identical", "skipped (tied bin edges)", "REVIEW"))
    show_df(pd.DataFrame(checks, columns=["Validation check", "Expected", "Observed", "Status"]), hide_index=True)
    counts = cust["spend_segment"].value_counts().reindex(SEGMENT_ORDER).fillna(0).astype(int)
    interpret(f"About one third of customers spent at most {money(low_upper)} (Low: {counts['Low']:,}), the next third up to "
              f"{money(med_upper)} (Medium: {counts['Medium']:,}) and the top third more than {money(med_upper)} (High: {counts['High']:,}) "
              f"in the active selection.",
              "Right-closed intervals make the assignment identical to pd.qcut when all three segments are present; "
              "with a segment filter only the chosen tiers remain.")

    # ---- Table 7.3 profile ----
    grp = (cust.groupby("spend_segment")["total_spend"].agg(["count", "min", "max", "mean", "median"]).reindex(present))
    grp.columns = ["Customers", "Min Spend", "Max Spend", "Mean Spend", "Median Spend"]
    grp.insert(1, "Customer Share (%)", grp["Customers"] / grp["Customers"].sum() * 100)
    disp_rows = [[f"{int(r['Customers']):,}", f"{r['Customer Share (%)']:.2f}%", money(r["Min Spend"], 2), money(r["Max Spend"], 2),
                  money(r["Mean Spend"], 2), money(r["Median Spend"], 2)] for _, r in grp.iterrows()]
    sp = cust["total_spend"].astype(float)
    disp_rows.append([f"{len(sp):,}", "100.00%", money(sp.min(), 2), money(sp.max(), 2), money(sp.mean(), 2), money(sp.median(), 2)])
    disp = pd.DataFrame(disp_rows, index=pd.Index(list(grp.index) + ["All customers"], name="Spend Segment"), columns=list(grp.columns))
    show_df(disp, caption="Table 7.3: Profile of the Low / Medium / High spend segments")
    if len(present) == 3:
        n_, share_, med_, avg_ = grp["Customers"].astype(int), grp["Customer Share (%)"], grp["Median Spend"], grp["Mean Spend"]
        span = grp["Max Spend"] - grp["Min Spend"]
        gap = safe_div(med_["High"], med_["Low"])
        span_ratio = safe_div(span["High"], span["Low"])
        hi_gap = safe_div(avg_["High"], med_["High"]) - 1 if med_["High"] > 0 else float("nan")
        balanced = (n_.max() - n_.min()) <= max(1, 0.02 * n_.sum())
        bal_txt = "" if balanced else " (the small imbalance comes from ties at a cut-off)"
        if gap >= 2:
            koch = (f"a typical High customer spends {gap:.1f}x as much as a typical Low customer although every tier holds a similar "
                    f"number of customers - the unequal-value pattern described by the 80/20 principle (Koch, 1998)")
        else:
            koch = (f"a typical High customer spends only {gap:.1f}x as much as a typical Low customer - the tiers differ only "
                    f"moderately in typical spend, and Section 8 shows whether the 80/20 pattern (Koch, 1998) holds in revenue")
        tail = (f"in the High tier the mean is {hi_gap:.0%} above the median, so a few very large buyers still pull that tier upward"
                if hi_gap >= 0.05 else "in the High tier the mean and median are within 5% of each other, so spending there is fairly even")
        if span_ratio >= 2:
            tail += f", and its spend range is {span_ratio:.1f}x as wide as that of the Low tier"
        interpret(f"The tertile rule yields near-equal groups of {n_['Low']:,}, {n_['Medium']:,} and {n_['High']:,} customers "
                  f"({share_['Low']:.2f}%, {share_['Medium']:.2f}% and {share_['High']:.2f}%){bal_txt}, so every tier is large enough for stable "
                  f"means and medians.",
                  f"Median spend climbs from {money(med_['Low'])} (Low) to {money(med_['Medium'])} (Medium) and {money(med_['High'])} (High), "
                  f"so {koch}.",
                  f"Comparing mean and median inside a tier is a resistant check for remaining skew (Tukey, 1977): {tail}.")
    else:
        interpret(f"The active selection contains only the segment(s) {', '.join(present)}: {int(grp['Customers'].sum()):,} customers with a "
                  f"median spend of {money(sp.median())}.",
                  "Select all three spend segments in the sidebar to compare Low, Medium and High side by side.")

    caption_md("Figure 7.4: Segment sizes and spend distribution per segment")
    show_png(chart_spend_segmentation(cust[["total_spend", "spend_segment"]], low_upper, med_upper, tuple(present)))
    if len(present) == 3:
        data = [cust.loc[cust["spend_segment"] == s, "total_spend"].to_numpy(dtype=float) for s in SEGMENT_ORDER]
        meds, fence_hi, above, below = [], [], [], []
        for d in data:
            q1b, q3b = np.percentile(d, [25, 75])
            up, dn = q3b + 1.5 * (q3b - q1b), q1b - 1.5 * (q3b - q1b)
            meds.append(float(np.median(d)))
            fence_hi.append(float(up))
            above.append(int((d > up).sum()))
            below.append(int((d < dn).sum()))
        other_out = above[0] + above[1] + below[0] + below[1]
        out_txt = (f"{plural(above[2], 'High-tier customer')} {'lies' if above[2] == 1 else 'lie'} above the upper fence of "
                   f"{money(fence_hi[2])}" if above[2] > 0 else "no High-tier customer lies above the upper fence")
        out_txt += (", while no point is flagged in the Low and Medium tiers" if other_out == 0
                    else f", against {plural(other_out, 'flagged point')} in the Low and Medium tiers combined")
        gap2 = safe_div(meds[2], meds[0])
        interpret(f"In the left panel the bars are almost equally tall ({counts['Low']:,}, {counts['Medium']:,} and {counts['High']:,} "
                  f"customers), exactly as tertile cut-offs are designed to produce.",
                  f"In the right panel the median rises from {money(meds[0])} (Low) through {money(meds[1])} (Medium) to {money(meds[2])} "
                  f"(High) and the three boxes cannot overlap because the tiers are cut by rank; by the 1.5 x IQR rule {out_txt} (Tukey, 1977).",
                  f"A {gap2:.1f}x gap in median spend between High and Low with balanced tier sizes is the uneven value distribution the 80/20 "
                  f"principle describes (Koch, 1998); the revenue share of each tier is analysed in Section 8.")
    else:
        interpret(f"Only {', '.join(present)} appear in the figure because of the sidebar selection.",
                  "The cut-off lines still show the thresholds computed on the full selected period.")


# ---------------------------------------------------------------------------------------------------
# Section 8
# ---------------------------------------------------------------------------------------------------
def s8_summary(cust, present):
    """Table 8.1 - comprehensive segment comparison (shares always add up to 100% over the segments present)."""
    summary = (cust.groupby("spend_segment")
               .agg(customer_count=("customer_id", "nunique"), total_revenue=("total_spend", "sum"),
                    total_orders=("num_orders", "sum"), mean_spend=("total_spend", "mean"),
                    median_spend=("total_spend", "median"), avg_orders=("num_orders", "mean"),
                    mean_aov=("avg_order_value", "mean"), mean_recency_days=("recency_days", "mean"))
               .reindex(present))
    for number, share in (("customer_count", "customer_share_pct"), ("total_revenue", "revenue_share_pct"),
                          ("total_orders", "order_share_pct")):
        summary[share] = summary[number] / summary[number].sum() * 100
    cols = ["customer_count", "customer_share_pct", "total_revenue", "revenue_share_pct", "total_orders", "order_share_pct",
            "mean_spend", "median_spend", "avg_orders", "mean_aov", "mean_recency_days"]
    summary = summary[cols]
    summary.index.name = "spend_segment"
    return summary


def s8_format(summary, cust):
    """Display copy of Table 8.1 with an 'All customers' reference row."""
    allrow = pd.DataFrame([[cust["customer_id"].nunique(), 100.0, cust["total_spend"].sum(), 100.0, cust["num_orders"].sum(), 100.0,
                            cust["total_spend"].mean(), cust["total_spend"].median(), cust["num_orders"].mean(),
                            cust["avg_order_value"].mean(), cust["recency_days"].mean()]],
                          columns=summary.columns, index=["All customers"])
    table = pd.concat([summary, allrow])
    table.index.name = "spend_segment"
    fmt = {"customer_count": "{:,.0f}", "customer_share_pct": "{:.1f}%", "total_revenue": CURRENCY_SYMBOL + "{:,.0f}",
           "revenue_share_pct": "{:.1f}%", "total_orders": "{:,.0f}", "order_share_pct": "{:.1f}%",
           "mean_spend": CURRENCY_SYMBOL + "{:,.2f}", "median_spend": CURRENCY_SYMBOL + "{:,.2f}", "avg_orders": "{:.2f}",
           "mean_aov": CURRENCY_SYMBOL + "{:,.2f}", "mean_recency_days": "{:.0f}"}
    return pd.DataFrame({c: table[c].map(fmt[c].format) for c in table.columns})


@chart
def chart_segment_contribution(summary):
    """Figure 8.2 - grouped bars (customer share vs revenue share, concentration badges) + 100% stacked columns."""
    segs = list(summary.index)
    cp = summary["customer_share_pct"].to_numpy()
    op = summary["order_share_pct"].to_numpy()
    rp = summary["revenue_share_pct"].to_numpy()
    n = summary["customer_count"].to_numpy()
    idx = rp / cp
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.3), gridspec_kw={"width_ratios": [1.45, 1]})
    x, w = np.arange(len(segs)), 0.36
    top = float(max(cp.max(), rp.max()))
    for i, seg in enumerate(segs):
        col = SEGMENT_COLORS[seg]
        xc, xr = x[i] - w / 2 - 0.01, x[i] + w / 2 + 0.01
        ax1.bar(xc, cp[i], w, facecolor="white", edgecolor=col, hatch="///", linewidth=1.6)
        ax1.bar(xr, rp[i], w, color=col, edgecolor="white")
        for xx, vv in ((xc, cp[i]), (xr, rp[i])):
            ax1.text(xx, vv + top * 0.015, f"{vv:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        ax1.text(x[i], max(cp[i], rp[i]) + top * 0.12, f"{idx[i]:.2f}x", ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color=text_color_for(col), bbox={"boxstyle": "round,pad=0.3", "facecolor": col, "edgecolor": "none"})
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}\n(n = {int(c):,})" for s, c in zip(segs, n)], fontsize=11)
    ax1.set_ylim(0, top * 1.38)
    ax1.set_ylabel("Share of total (%)")
    ax1.set_title("Share of customers vs share of revenue", fontsize=13, fontweight="bold", loc="left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(pct_axis))
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555555", hatch="///", linewidth=1.4, label="% of customers"),
                        plt.Rectangle((0, 0), 1, 1, facecolor="#555555", label="% of revenue")],
               loc="upper left", frameon=False, fontsize=10.5)
    cols_b = ["Customers", "Orders", "Revenue"]
    base = np.zeros(3)
    for i, seg in enumerate(segs):
        vals = np.array([cp[i], op[i], rp[i]])
        ax2.bar(cols_b, vals, bottom=base, color=SEGMENT_COLORS[seg], edgecolor="white", width=0.62, label=seg)
        for j, vv in enumerate(vals):
            if vv >= 4:
                ax2.text(j, base[j] + vv / 2, f"{vv:.1f}%", ha="center", va="center", fontsize=10.5, fontweight="bold",
                         color=text_color_for(SEGMENT_COLORS[seg]))
        base = base + vals
    ax2.set_ylim(0, 122)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(pct_axis))
    ax2.set_title("How each metric splits across segments", fontsize=13, fontweight="bold", loc="left")
    ax2.legend(loc="upper center", ncol=3, frameon=False, fontsize=10.5)
    for ax in (ax1, ax2):
        ax.yaxis.grid(True, alpha=0.25)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
    fig.suptitle("Segment Revenue Contribution vs Customer Share", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.012, "Badge = concentration index (revenue share / customer share); 1.00x means a segment contributes "
                         "in proportion to its size.", ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig_to_png(fig)


def s8_category_tables(tx, cust, present, max_cats=8):
    """Segment x category value table and the within-segment mix (%) used by Figure 8.3."""
    seg_map = cust[["customer_id", "spend_segment"]].drop_duplicates(subset="customer_id")
    merged = tx.merge(seg_map, on="customer_id", how="inner", validate="many_to_one")
    merged = merged.assign(cat_raw=merged["category"].astype(str))
    cat_tot = merged.groupby("cat_raw")["total_amount"].sum().sort_values(ascending=False)
    keep = list(cat_tot.index[:max_cats]) if len(cat_tot) > max_cats + 1 else list(cat_tot.index)
    merged["cat_group"] = np.where(merged["cat_raw"].isin(keep), merged["cat_raw"], "Other")
    value = (merged.pivot_table(index="spend_segment", columns="cat_group", values="total_amount", aggfunc="sum", fill_value=0.0)
             .reindex(present).fillna(0.0))
    order = [c for c in value.sum().sort_values(ascending=False).index if c != "Other"]
    if "Other" in value.columns:
        order.append("Other")
    value = value[order]
    mix = value.div(value.sum(axis=1), axis=0) * 100
    overall = value.sum() / value.to_numpy().sum() * 100
    return value, mix, overall


@chart
def chart_category_mix(cat_value, cat_mix):
    """Figure 8.3 - absolute revenue (left) and 100% category mix (right) per spend segment."""
    segs = list(cat_value.index)
    order = list(cat_value.columns)
    if len([c for c in order if c != "Other"]) <= 9:
        palette = [plt.get_cmap("tab10")(i) for i in (0, 1, 2, 3, 4, 5, 6, 8, 9)]
    else:
        palette = [plt.get_cmap("tab20")(i) for i in range(20) if i not in (14, 15)]
    colors, k = {}, 0
    for c in order:
        if c == "Other":
            colors[c] = "#B5B5B5"
        else:
            colors[c] = palette[k % len(palette)]
            k += 1
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.6))
    base_abs, base_pct = np.zeros(len(segs)), np.zeros(len(segs))
    for c in order:
        av, pv = cat_value[c].to_numpy(), cat_mix[c].to_numpy()
        axL.bar(segs, av, bottom=base_abs, color=colors[c], edgecolor="white", width=0.62, label=c)
        axR.bar(segs, pv, bottom=base_pct, color=colors[c], edgecolor="white", width=0.62)
        for j, vv in enumerate(pv):
            if vv >= 5:
                axR.text(j, base_pct[j] + vv / 2, f"{vv:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold",
                         color=text_color_for(colors[c]))
        base_abs, base_pct = base_abs + av, base_pct + pv
    for j, t in enumerate(base_abs):
        axL.text(j, t * 1.012, money_compact(t), ha="center", va="bottom", fontsize=11, fontweight="bold")
    axL.set_ylim(0, base_abs.max() * 1.12)
    axL.yaxis.set_major_formatter(currency_formatter)
    axL.set_ylabel("Revenue")
    axL.set_title("Revenue by spend segment and product category", fontsize=13, fontweight="bold", loc="left")
    axR.set_ylim(0, 100)
    axR.yaxis.set_major_formatter(mticker.FuncFormatter(pct_axis))
    axR.set_ylabel("Share of segment total (%)")
    axR.set_title("Category mix inside each segment (100% stacked)", fontsize=13, fontweight="bold", loc="left")
    for ax in (axL, axR):
        ax.yaxis.grid(True, alpha=0.25)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xlabel("Spend segment")
    handles, labels = axL.get_legend_handles_labels()
    ncol = min(len(labels), 5)
    fig.legend(handles, labels, title="Product category", loc="lower center", ncol=ncol, frameon=False)
    fig.suptitle("Category Preference Across Spend Segments", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05 + 0.04 * (-(-len(labels) // ncol)), 1, 0.94))
    return fig_to_png(fig)


def render_sec8(ctx):
    """Section 8 (Member 7): segment comparison table, contribution charts and category mix."""
    cust, tx = ctx["cust"], ctx["tx"]
    section_header("Section 8: Segment Comparison and Revenue Contribution", "Member 7 - Divyanshi")
    present = [s for s in SEGMENT_ORDER if (cust["spend_segment"] == s).any()]
    if not present:
        st.info("No spend segment is available for the current selection.")
        return
    summary = s8_summary(cust, present)
    show_df(s8_format(summary, cust), caption="Table 8.1: Spend-segment comparison")
    full3 = len(present) == 3
    if full3:
        hi, md_, lo = summary.loc["High"], summary.loc["Medium"], summary.loc["Low"]
        conc = safe_div(hi["revenue_share_pct"], hi["customer_share_pct"])
        x_ord = safe_div(hi["avg_orders"], lo["avg_orders"])
        x_aov = safe_div(hi["mean_aov"], lo["mean_aov"])
        driver = "purchase frequency" if x_ord >= x_aov else "basket size (order value)"
        rec_txt = ("so they are also the most recently active customers" if hi["mean_recency_days"] < lo["mean_recency_days"]
                   else "so they are not the most recently active, which is a re-engagement risk")
        skew = safe_div(hi["mean_spend"], hi["median_spend"])
        if skew > 1.05:
            tail = (f"The High segment's mean spend ({money(hi['mean_spend'])}) is {(skew - 1) * 100:.0f}% above its median "
                    f"({money(hi['median_spend'])}), so a minority of very large buyers lifts the average and the median is the "
                    f"safer yardstick for a typical High customer (Tukey, 1977).")
        else:
            tail = (f"The High segment's mean ({money(hi['mean_spend'])}) and median ({money(hi['median_spend'])}) are close, which "
                    f"indicates a fairly symmetric spend distribution inside the segment (Tukey, 1977).")
        interpret(f"High-spend customers are {hi['customer_share_pct']:.1f}% of the base yet generate {hi['revenue_share_pct']:.1f}% of revenue "
                  f"({conc:.1f}x their headcount share), while Low-spend customers ({lo['customer_share_pct']:.1f}% of the base) add only "
                  f"{lo['revenue_share_pct']:.1f}% - the unequal pay-off described by the 80/20 principle (Koch, 1998).",
                  f"High spenders place {x_ord:.1f}x as many orders and spend {x_aov:.1f}x as much per order as Low spenders, so {driver} is the "
                  f"bigger lever behind the gap; their mean recency is {hi['mean_recency_days']:.0f} days versus {lo['mean_recency_days']:.0f} "
                  f"for Low spenders, {rec_txt}.", tail)
    else:
        st.info(f"The sidebar selection keeps only {', '.join(present)}; the comparison narrative needs all three spend segments, "
                f"so only the tables and charts are shown.")
        top = summary["revenue_share_pct"].idxmax()
        interpret(f"Within the selected segments, **{top}** contributes {summary.loc[top, 'revenue_share_pct']:.1f}% of revenue from "
                  f"{summary.loc[top, 'customer_share_pct']:.1f}% of the customers.")

    caption_md("Figure 8.2: Revenue contribution versus customer share")
    show_png(chart_segment_contribution(summary))
    if full3:
        cH, oH, rH = hi["customer_share_pct"], hi["order_share_pct"], hi["revenue_share_pct"]
        cL, rL = lo["customer_share_pct"], lo["revenue_share_pct"]
        if cH < oH < rH:
            chain = "High spenders earn their weight through both more frequent ordering and larger orders"
        elif oH <= cH:
            chain = "High spenders' weight comes mainly from larger orders rather than more frequent ordering"
        else:
            chain = "High spenders' weight comes mainly from more frequent ordering rather than larger orders"
        interpret(f"High-spend customers are {cH:.1f}% of the base but generate {rH:.1f}% of revenue (concentration index "
                  f"{safe_div(rH, cH):.2f}x), whereas Low-spend customers are {cL:.1f}% of the base and generate only {rL:.1f}% "
                  f"({safe_div(rL, cL):.2f}x) - the uneven pay-off that the 80/20 principle describes (Koch, 1998).",
                  f"Reading the stacked columns from customers to orders to revenue, the High slice moves from {cH:.1f}% to {oH:.1f}% to "
                  f"{rH:.1f}%: {chain} (Tukey, 1977).",
                  "Protecting High-spend customers is therefore the first revenue-protection priority, while the Low segment, which returns "
                  "less revenue than its headcount share, calls for low-cost activation rather than heavy marketing spend.")
    else:
        idx = (summary["revenue_share_pct"] / summary["customer_share_pct"]).round(2)
        interpret("Concentration index (revenue share / customer share) by segment: "
                  + ", ".join(f"{s} {idx[s]:.2f}x" for s in present) + ".")

    value, mix, overall = s8_category_tables(tx, cust, present)
    caption_md("Figure 8.3: Category preference across spend segments")
    show_png(chart_category_mix(value, mix))
    named = [c for c in value.columns if c != "Other"] or list(value.columns)
    lead = mix[named].idxmax(axis=1)
    lead_pct = mix[named].max(axis=1)
    dev = mix[named] - overall[named]
    over = dev.idxmax(axis=1)
    over_pp = dev.max(axis=1)
    if full3:
        s1 = (f"The category mix differs by spend segment: High spenders earn most of their revenue from **{esc(lead['High'])}** "
              f"({lead_pct['High']:.1f}%), Medium spenders from **{esc(lead['Medium'])}** ({lead_pct['Medium']:.1f}%) and Low spenders "
              f"from **{esc(lead['Low'])}** ({lead_pct['Low']:.1f}%).")
        if max(over_pp["High"], over_pp["Low"]) >= 1.0:
            s2 = (f"Against the all-customer mix, High spenders over-index most on **{esc(over['High'])}** ({over_pp['High']:+.1f} pp) and "
                  f"Low spenders on **{esc(over['Low'])}** ({over_pp['Low']:+.1f} pp), a group-wise contrast that is easiest to spot in a "
                  f"stacked-bar view before any modelling (Tukey, 1977).")
        else:
            s2 = ("Against the all-customer mix no category deviates by more than one percentage point, so product mix is a weak "
                  "differentiator of spend level (Tukey, 1977).")
        s3 = (f"Commercially, loyalty and cross-sell offers for the High segment should be anchored on **{esc(lead['High'])}**, where it earns "
              f"most (Koch, 1998), while Low spenders are best approached with entry-level bundles in **{esc(lead['Low'])}**.")
        interpret(s1, s2, s3)
    else:
        interpret("The leading category per segment is " + "; ".join(f"{s}: **{esc(lead[s])}** ({lead_pct[s]:.1f}%)" for s in present) + ".")

    st.markdown("**8.4 Key takeaways**")
    if full3:
        lever = "Purchase frequency" if x_ord >= x_aov else "Order value"
        st.markdown(
            f"1. **Value is concentrated.** High-spend customers are {hi['customer_share_pct']:.1f}% of the base but deliver "
            f"{hi['revenue_share_pct']:.1f}% of revenue; Low-spend customers are {lo['customer_share_pct']:.1f}% of the base and deliver "
            f"{lo['revenue_share_pct']:.1f}% (Koch, 1998).\n"
            f"2. **{lever} separates the segments most.** High spenders place {x_ord:.1f}x as many orders and spend {x_aov:.1f}x as much per "
            f"order as Low spenders.\n"
            f"3. **Engagement.** Mean recency (days since last purchase; lower = more recent) is {hi['mean_recency_days']:.0f} for High, "
            f"{md_['mean_recency_days']:.0f} for Medium and {lo['mean_recency_days']:.0f} for Low spenders.\n"
            f"4. **Category preference.** The leading category is **{esc(lead['High'])}** for High and **{esc(lead['Low'])}** for Low spenders; "
            f"High spenders over-index most on **{esc(over['High'])}** ({over_pp['High']:+.1f} pp versus the all-customer mix) (Tukey, 1977).\n"
            f"5. **Suggested focus.** Protect and grow the High segment (retention, loyalty); move Medium customers "
            f"({md_['customer_share_pct']:.1f}% of customers, {md_['revenue_share_pct']:.1f}% of revenue) towards High through cross-sell and "
            f"upsell; activate Low customers at low cost.")
    else:
        st.markdown("Select all three spend segments in the sidebar to see the five key takeaways of Section 8.")


# ===================================================================================================
# 17.  TAB 4  -  Product, repeat-purchase and seasonality EDA  (Sections 9-11)
#      Members 8 (Abhi Rastogi), 9 (Sanjay Rathour) and 10 (Dev Rastogi)
# ---------------------------------------------------------------------------------------------------
# 17.1  Section 9: Product, category and category-mix analysis                 (Member 8)
# ===================================================================================================
S9_TOP_N = 10


def mode_first(series):
    """Most frequent value (first one on ties) - the category of a product."""
    m = series.mode()
    return m.iat[0] if len(m) else "Unknown"


def n80(series, share=80.0):
    """How many of the largest items are needed to reach `share` % of the total."""
    s = series.sort_values(ascending=False)
    cum = s.cumsum() / s.sum() * 100
    return int((cum < share - 1e-9).sum()) + 1


def conc_phrase(n_top, n_all):
    """Noun phrase describing how concentrated a ranking is (n_top items reach 80% of the total)."""
    if n_all <= 2:
        return "not a meaningful concentration test with so few items"
    r = n_top / n_all
    if r <= 0.34:
        return "a clear 'vital few' pattern in the spirit of the 80/20 principle"
    if r <= 0.60:
        return "a moderate concentration, somewhat flatter than the classic 80/20 pattern"
    return "an even spread, much flatter than the classic 80/20 pattern"


def lift_phrase(share_pct, n_top, n_all):
    """Noun phrase comparing the share of the Top-N items with their share of all items."""
    if n_top >= n_all:
        return "not informative here because the table already covers the whole range"
    lift = share_pct / (n_top / n_all * 100)
    if lift >= 3:
        return "a strong concentration in the spirit of the 80/20 principle"
    if lift >= 1.8:
        return "a clear but moderate concentration, flatter than the classic 80/20 pattern"
    return "an even spread, much flatter than the classic 80/20 pattern"


def lead_cats(counts):
    """Tie-aware: (markdown text of the leading categories, their count, how many categories tie)."""
    mx = int(counts.max())
    names = [f"**{esc(k)}**" for k, v in counts.items() if int(v) == mx]
    if len(names) <= 2:
        txt = " and ".join(names)
    else:
        rest = len(names) - 2
        txt = f"{names[0]}, {names[1]} and {rest} other {'category' if rest == 1 else 'categories'}"
    return txt, mx, len(names)


def s9_display(df, money_cols=(), pct_cols=(), int_cols=(), index=None):
    """Display copy with formatted strings; the numeric source table stays untouched."""
    out = df.copy()
    for c in money_cols:
        out[c] = out[c].map(lambda v: "-" if pd.isna(v) else money(v, 2))
    for c in pct_cols:
        out[c] = out[c].map(lambda v: "-" if pd.isna(v) else f"{v:,.1f}%")
    for c in int_cols:
        out[c] = out[c].map(lambda v: "-" if pd.isna(v) else f"{int(round(v)):,}")
    return out.set_index(index) if index else out


def s9_tables(tx):
    """Category table, product table and the two Top-N tables of Section 9 (working copy; tx is untouched)."""
    work = tx.copy()
    for c in ("category", "product_name"):
        work[c] = work[c].astype(object).fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
    work["revenue"] = pd.to_numeric(work["total_amount"], errors="coerce").fillna(work["quantity"] * work["unit_price"])
    cat = (work.groupby("category", as_index=False)
           .agg(total_revenue=("revenue", "sum"), total_units=("quantity", "sum"), unique_orders=("transaction_id", "nunique"),
                unique_customers=("customer_id", "nunique")))
    cat["revenue_share_pct"] = cat["total_revenue"] / cat["total_revenue"].sum() * 100
    cat["order_share_pct"] = cat["unique_orders"] / cat["unique_orders"].sum() * 100
    cat["avg_unit_price"] = cat["total_revenue"] / cat["total_units"].where(cat["total_units"] > 0)
    cat = cat.sort_values(["total_revenue", "category"], ascending=[False, True]).reset_index(drop=True)
    cat = cat[["category", "total_revenue", "revenue_share_pct", "total_units", "unique_orders", "unique_customers",
               "avg_unit_price", "order_share_pct"]]
    prod = (work.groupby("product_name", as_index=False)
            .agg(category=("category", mode_first), total_revenue=("revenue", "sum"), total_units=("quantity", "sum"),
                 unique_orders=("transaction_id", "nunique"), unique_customers=("customer_id", "nunique")))
    prod["revenue_share_pct"] = prod["total_revenue"] / prod["total_revenue"].sum() * 100
    prod["units_share_pct"] = prod["total_units"] / prod["total_units"].sum() * 100
    prod["avg_unit_price"] = prod["total_revenue"] / prod["total_units"].where(prod["total_units"] > 0)
    top_n = min(S9_TOP_N, len(prod))
    top_rev = prod.sort_values(["total_revenue", "product_name"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    top_rev.insert(0, "rank", range(1, len(top_rev) + 1))
    top_rev["cum_revenue_share_pct"] = top_rev["revenue_share_pct"].cumsum()
    top_qty = (prod.sort_values(["total_units", "total_revenue", "product_name"], ascending=[False, False, True])
               .head(top_n).reset_index(drop=True))
    top_qty.insert(0, "rank", range(1, len(top_qty) + 1))
    top_qty["cum_units_share_pct"] = top_qty["units_share_pct"].cumsum()
    palette = (PALETTE_LIST + CLUSTER_PALETTE + [plt.get_cmap("tab10")(i) for i in range(10)])
    colors = {c: mcolors.to_hex(palette[i % len(palette)]) for i, c in enumerate(cat["category"])}
    return {"work": work, "cat": cat, "prod": prod, "top_rev": top_rev, "top_qty": top_qty, "top_n": top_n,
            "colors": colors, "total_rev": float(work["revenue"].sum()), "total_units": float(work["quantity"].sum())}


@chart
def chart_top_products(top_rev, top_qty, colors, cat_order):
    """Figure 9.2(c) - Top products by revenue (left) and by units (right), bars coloured by category."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))
    panels = [(axes[0], top_rev, "total_revenue", f"Top {len(top_rev)} Products by Revenue", f"Total revenue ({CURRENCY_SYMBOL})", True),
              (axes[1], top_qty, "total_units", f"Top {len(top_qty)} Products by Units Sold", "Units sold", False)]
    for ax, d, col, ttl, xl, is_money in panels:
        vals = d[col].astype(float).tolist()
        ypos = list(range(len(vals)))
        ax.barh(ypos, vals, height=0.72, edgecolor="white", color=[colors.get(c, "#888888") for c in d["category"]])
        ax.set_yticks(ypos)
        ax.set_yticklabels([short(p, 30) for p in d["product_name"]], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, (max(vals) * 1.25) if vals else 1)
        for y, v in zip(ypos, vals):
            lab = money_compact(v) if is_money else f"{v:,.0f}"
            ax.text(v, y, "  " + mpl_text(lab), va="center", ha="left", fontsize=9, color="#333333")
        ax.set_title(ttl, fontsize=13, fontweight="bold", loc="left")
        ax.set_xlabel(mpl_text(xl))
        ax.xaxis.set_major_formatter(currency_formatter if is_money else thousands_formatter)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
    present = set(top_rev["category"]) | set(top_qty["category"])
    handles = [Patch(facecolor=colors[c], label=c) for c in cat_order if c in present]
    leg_rows = -(-len(handles) // 6)
    fig.legend(handles=handles, title="Category", loc="lower center", ncol=min(len(handles), 6), frameon=False)
    fig.suptitle("Top-Selling Products: Revenue Leaders vs. Volume Leaders", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04 + 0.04 * leg_rows, 1, 0.95])
    return fig_to_png(fig)


@chart
def chart_category_pareto(cat, par, n80_prod, k_show):
    """Figure 9.3 - category mix (revenue share vs order share) and the product Pareto chart."""
    n_prod = len(par)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(18, 7.6), gridspec_kw={"width_ratios": [1, 1.15]})
    cats = cat["category"].tolist()
    ypos = list(range(len(cats)))
    c_rev, c_ord = PALETTE_LIST[0], PALETTE_LIST[2]
    axA.barh([y - 0.2 for y in ypos], cat["revenue_share_pct"], height=0.38, color=c_rev, edgecolor="white", label="Share of revenue")
    axA.barh([y + 0.2 for y in ypos], cat["order_share_pct"], height=0.38, color=c_ord, edgecolor="white", label="Share of orders")
    for y, v in zip(ypos, cat["revenue_share_pct"]):
        axA.text(v, y - 0.2, f"  {v:.1f}%", va="center", ha="left", fontsize=9, color="#333333")
    for y, v in zip(ypos, cat["order_share_pct"]):
        axA.text(v, y + 0.2, f"  {v:.1f}%", va="center", ha="left", fontsize=9, color="#333333")
    axA.set_yticks(ypos)
    axA.set_yticklabels([short(c, 26) for c in cats], fontsize=10)
    axA.invert_yaxis()
    axA.set_xlim(0, max(cat["revenue_share_pct"].max(), cat["order_share_pct"].max()) * 1.22)
    axA.xaxis.set_major_formatter(mticker.FuncFormatter(pct_axis))
    axA.set_xlabel("Share of total (%)")
    axA.set_title("Category Mix: Share of Revenue vs. Share of Orders\n(categories sorted by revenue)", fontsize=13,
                  fontweight="bold", loc="left")
    axA.legend(loc="lower right", frameon=False)
    axA.grid(axis="x", alpha=0.3)
    axA.grid(axis="y", visible=False)
    axA.set_axisbelow(True)
    bar_cols = [c_rev if k < n80_prod else "#C3C9D3" for k in range(k_show)]
    axB.bar(range(k_show), par["total_revenue"].iloc[:k_show], color=bar_cols, edgecolor="white")
    axB.set_xticks(range(k_show))
    axB.set_xticklabels([short(p, 22) for p in par["product_name"].iloc[:k_show]], rotation=50, ha="right",
                        fontsize=9 if k_show <= 18 else 8)
    axB.set_xlim(-0.6, k_show - 0.4)
    axB.yaxis.set_major_formatter(currency_formatter)
    axB.set_ylabel(mpl_text(f"Revenue per product ({CURRENCY_SYMBOL})"))
    axB.grid(axis="y", alpha=0.3)
    axB.grid(axis="x", visible=False)
    axB.set_axisbelow(True)
    axB2 = axB.twinx()
    axB2.plot(range(k_show), par["cum_share"].iloc[:k_show], color="#2E3440", marker="o", linewidth=2)
    axB2.axhline(80, linestyle="--", color="#555555", linewidth=1)
    axB2.set_ylim(0, 105)
    axB2.yaxis.set_major_formatter(mticker.FuncFormatter(pct_axis))
    axB2.set_ylabel("Cumulative share of revenue (line)")
    axB2.grid(False)
    axB.set_title(f"Product Pareto: Top {k_show} of {n_prod} Products by Revenue\n"
                  f"{n80_prod} products ({n80_prod / n_prod * 100:.0f}%) generate 80% of revenue (dark bars)",
                  fontsize=13, fontweight="bold", loc="left")
    for a, hide in ((axA, ("top", "right")), (axB, ("top", "right")), (axB2, ("top",))):
        for s in hide:
            a.spines[s].set_visible(False)
    fig.suptitle("Category Mix & Product Concentration", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_png(fig)


def render_sec9(ctx):
    """Section 9 (Member 8): category performance, top products by revenue / volume, category mix, scorecard."""
    tx = ctx["tx"]
    section_header("Section 9: Product, Category and Category-Mix Analysis", "Member 8 - Abhi Rastogi")
    t = s9_tables(tx)
    cat, prod, top_rev, top_qty, top_n = t["cat"], t["prod"], t["top_rev"], t["top_qty"], t["top_n"]
    n_cat, n_prod = len(cat), len(prod)
    ref_rev = float(pd.to_numeric(tx["total_amount"], errors="coerce").sum())
    recon_ok = abs(float(cat["total_revenue"].sum()) - ref_rev) <= max(0.01, 1e-9 * abs(ref_rev))

    # ---- 9.1 category table ----
    all_row = pd.DataFrame([{"category": "ALL CATEGORIES", "total_revenue": t["total_rev"], "revenue_share_pct": 100.0,
                             "total_units": t["total_units"], "unique_orders": tx["transaction_id"].nunique(),
                             "unique_customers": tx["customer_id"].nunique(),
                             "avg_unit_price": safe_div(t["total_rev"], t["total_units"]), "order_share_pct": 100.0}])
    shown = s9_display(pd.concat([cat, all_row], ignore_index=True), money_cols=("total_revenue", "avg_unit_price"),
                       pct_cols=("revenue_share_pct", "order_share_pct"),
                       int_cols=("total_units", "unique_orders", "unique_customers"), index="category")
    show_df(shown, caption="Table 9.1: Category-level performance, ranked by total revenue")
    n80_cat = n80(cat["total_revenue"])
    top80 = cat["revenue_share_pct"].iloc[:n80_cat].sum()
    lead = cat.iloc[0]
    priciest = cat.loc[cat["avg_unit_price"].idxmax()]
    biggest_vol = cat.loc[cat["total_units"].idxmax()]
    if priciest["category"] == biggest_vol["category"]:
        pv = (f"**{esc(priciest['category'])}** has both the highest average unit price ({money(priciest['avg_unit_price'], 2)}) and the "
              f"highest unit volume ({biggest_vol['total_units']:,.0f}), so its position rests on price and volume together.")
    else:
        pv = (f"**{esc(priciest['category'])}** has the highest average unit price ({money(priciest['avg_unit_price'], 2)}) while "
              f"**{esc(biggest_vol['category'])}** sells the most units ({biggest_vol['total_units']:,.0f}), so category revenue is driven "
              f"by price in some categories and by volume in others.")
    interpret(f"**{esc(lead['category'])}** is the largest category with {money(lead['total_revenue'], 2)} in revenue "
              f"({lead['revenue_share_pct']:.1f}% of the total), and the top {n80_cat} of {n_cat} categories together generate "
              f"{top80:.1f}% of revenue, which is {conc_phrase(n80_cat, n_cat)} (Koch, 1998).",
              pv, "Reading a ranked summary like this before any modelling follows the exploratory-data-analysis habit of letting "
                  "simple tables reveal structure first (Tukey, 1977).")

    # ---- 9.2 (a) top products by revenue ----
    sh = s9_display(top_rev[["rank", "product_name", "category", "total_revenue", "revenue_share_pct", "cum_revenue_share_pct",
                             "total_units", "unique_orders", "unique_customers", "avg_unit_price"]],
                    money_cols=("total_revenue", "avg_unit_price"), pct_cols=("revenue_share_pct", "cum_revenue_share_pct"),
                    int_cols=("total_units", "unique_orders", "unique_customers"), index="rank")
    show_df(sh, caption=f"Table 9.2(a): Top {top_n} products by total revenue")
    p1 = top_rev.iloc[0]
    top_rev_share = top_rev["revenue_share_pct"].sum()
    dom_txt, dom_n, dom_ties = lead_cats(top_rev["category"].value_counts())
    if dom_n >= 2 and dom_ties == 1:
        cat_sentence = (f"{dom_txt} supplies {dom_n} of these {top_n} products, so it is the category where stock availability, "
                        f"pricing and promotions matter most.")
    elif dom_n >= 2:
        cat_sentence = (f"{dom_txt} supply {dom_n} of these {top_n} products each, so these categories deserve the most attention "
                        f"in stock availability, pricing and promotions.")
    else:
        cat_sentence = (f"These {top_n} products come from {plural(top_rev['category'].nunique(), 'category', 'categories')}, so stock "
                        f"and promotion decisions are needed across several categories at once.")
    interpret(f"**{esc(p1['product_name'])}** is the best-selling product by revenue ({money(p1['total_revenue'], 2)}, "
              f"{p1['revenue_share_pct']:.1f}% of total revenue), and the Top {top_n} products together bring in {top_rev_share:.1f}% of "
              f"revenue while making up {top_n / n_prod * 100:.1f}% of the {n_prod} products on sale, which is "
              f"{lift_phrase(top_rev_share, top_n, n_prod)} (Koch, 1998).",
              cat_sentence, "A simple ranked table like this is the first look at the data that exploratory analysis recommends "
                            "before any deeper modelling (Tukey, 1977).")

    # ---- 9.2 (b) top products by volume ----
    sh = s9_display(top_qty[["rank", "product_name", "category", "total_units", "units_share_pct", "cum_units_share_pct",
                             "total_revenue", "unique_orders", "unique_customers", "avg_unit_price"]],
                    money_cols=("total_revenue", "avg_unit_price"), pct_cols=("units_share_pct", "cum_units_share_pct"),
                    int_cols=("total_units", "unique_orders", "unique_customers"), index="rank")
    show_df(sh, caption=f"Table 9.2(b): Top {top_n} products by quantity (units) sold")
    q1 = top_qty.iloc[0]
    top_units_share = top_qty["units_share_pct"].sum()
    rev_set = set(top_rev["product_name"])
    both_names = [p for p in top_qty["product_name"] if p in rev_set]
    if both_names:
        both_txt = ", ".join(esc(p) for p in both_names[:3]) + (" and others" if len(both_names) > 3 else "")
        overlap = (f"{plural(len(both_names), 'product')} {'appears' if len(both_names) == 1 else 'appear'} in both the revenue and the "
                   f"volume Top {top_n} ({both_txt}); these are 'hero' items that are popular and valuable at the same time, whereas "
                   f"products found only in the volume list are lower-priced traffic drivers.")
    else:
        overlap = (f"No product appears in both Top-{top_n} lists, so the volume leaders are low-ticket items while the revenue leaders "
                   f"are high-ticket items and need different pricing and stocking plans.")
    interpret(f"**{esc(q1['product_name'])}** leads on volume with {q1['total_units']:,.0f} units, and the Top {top_n} products account "
              f"for {top_units_share:.1f}% of all units sold while making up {top_n / n_prod * 100:.1f}% of the {n_prod} products, which is "
              f"{lift_phrase(top_units_share, top_n, n_prod)} (Koch, 1998).",
              overlap, "Comparing two rankings of the same data is a standard exploratory device for spotting such differences (Tukey, 1977).")

    # ---- 9.2 (c) two-panel figure ----
    caption_md("Figure 9.2(c): Top products by revenue and by volume")
    show_png(chart_top_products(top_rev[["product_name", "category", "total_revenue", "total_units"]],
                                top_qty[["product_name", "category", "total_revenue", "total_units"]],
                                t["colors"], tuple(cat["category"])))
    rev_txt, rev_n, rev_ties = lead_cats(top_rev["category"].value_counts())
    qty_txt, qty_n, qty_ties = lead_cats(top_qty["category"].value_counts())
    qty_set = set(top_qty["product_name"])
    both2 = [p for p in top_rev["product_name"] if p in qty_set]
    if both2:
        both_txt = ", ".join(esc(p) for p in both2[:3]) + (" and others" if len(both2) > 3 else "")
        fig_s2 = (f"{plural(len(both2), 'product')} {'appears' if len(both2) == 1 else 'appear'} in both panels ({both_txt}), so "
                  f"{'it combines' if len(both2) == 1 else 'they combine'} high volume with high value and deserve"
                  f"{'s' if len(both2) == 1 else ''} priority in stocking and promotion.")
    else:
        fig_s2 = ("No product appears in both panels, so the volume leaders and the revenue leaders are different products and need "
                  "separate pricing and stocking strategies.")
    drop = safe_div(top_rev["total_revenue"].iloc[0], top_rev["total_revenue"].iloc[-1])
    if drop >= 3:
        drop_txt = (f"The best revenue product earns {drop:.1f} times as much as the number {len(top_rev)} product, a steep drop that "
                    f"is the visual signature of a 'vital few' (Koch, 1998)")
    elif drop >= 1.5:
        drop_txt = (f"The best revenue product earns {drop:.1f} times as much as the number {len(top_rev)} product, a gradual decline "
                    f"that points to a moderately concentrated range (Koch, 1998)")
    else:
        drop_txt = (f"The revenue bars are of similar length ({drop:.1f} times between first and last), so no single product dominates, "
                    f"unlike the classic 80/20 pattern (Koch, 1998)")
    interpret(f"Bars are coloured by category: {rev_txt} {'supplies' if rev_ties == 1 else 'each supply'} {rev_n} of the {len(top_rev)} "
              f"revenue leaders, while {qty_txt} {'supplies' if qty_ties == 1 else 'each supply'} {qty_n} of the {len(top_qty)} volume leaders.",
              fig_s2, f"{drop_txt}, and showing both rankings side by side lets the graphics reveal patterns that a table can hide (Tukey, 1977).")

    # ---- 9.3 category mix + product Pareto ----
    par = prod.sort_values(["total_revenue", "product_name"], ascending=[False, True]).reset_index(drop=True)
    par["cum_share"] = par["total_revenue"].cumsum() / par["total_revenue"].sum() * 100
    n80_prod = n80(par["total_revenue"])
    k_show = min(n_prod, 25, max(15, n80_prod + 2))
    caption_md("Figure 9.3: Category mix and product concentration")
    show_png(chart_category_pareto(cat, par[["product_name", "total_revenue", "cum_share"]], n80_prod, k_show))
    gap = cat["revenue_share_pct"] - cat["order_share_pct"]
    hi, lo = cat.loc[gap.idxmax()], cat.loc[gap.idxmin()]
    if len(cat) > 1 and hi["category"] != lo["category"] and gap.abs().max() >= 1.0:
        mix_txt = (f"**{esc(hi['category'])}** earns {hi['revenue_share_pct']:.1f}% of revenue from {hi['order_share_pct']:.1f}% of orders "
                   f"(higher value per order), whereas **{esc(lo['category'])}** brings {lo['order_share_pct']:.1f}% of orders but only "
                   f"{lo['revenue_share_pct']:.1f}% of revenue (frequency-driven, lower value per order).")
    else:
        mix_txt = ("Revenue share and order share differ by less than one percentage point in every category, so no category stands "
                   "out by value per order.")
    interpret(mix_txt,
              f"In the Pareto panel, {n80_prod} of {n_prod} products ({n80_prod / n_prod * 100:.1f}%) are enough to reach 80% of revenue, "
              f"which is {conc_phrase(n80_prod, n_prod)} (Koch, 1998).",
              "Reading the two panels together shows where revenue comes from and how concentrated it is, exactly the kind of pattern "
              "that exploratory graphics are meant to reveal before modelling (Tukey, 1977).")

    # ---- 9.4 scorecard + implications ----
    lead_cat, lead_cat_share = cat.iloc[0]["category"], cat.iloc[0]["revenue_share_pct"]
    price_name = cat.loc[cat["avg_unit_price"].idxmax(), "category"]
    both_n = len(set(top_rev["product_name"]) & set(top_qty["product_name"]))
    score = pd.DataFrame([("Total revenue analysed", money(t["total_rev"], 2)),
                          ("Categories / products analysed", f"{n_cat} / {n_prod}"),
                          ("Leading category (revenue share)", f"{lead_cat} ({lead_cat_share:.1f}%)"),
                          ("Categories needed for 80% of revenue", f"{n80_cat} of {n_cat}"),
                          ("Leading product by revenue (share)", f"{top_rev.iloc[0]['product_name']} ({top_rev.iloc[0]['revenue_share_pct']:.1f}%)"),
                          ("Leading product by units sold", f"{top_qty.iloc[0]['product_name']} ({top_qty.iloc[0]['total_units']:,.0f} units)"),
                          ("Products needed for 80% of revenue", f"{n80_prod} of {n_prod} ({n80_prod / n_prod * 100:.1f}%)"),
                          (f"Products in both Top-{top_n} lists", f"{both_n}"),
                          ("Category revenue reconciles with df_clean", "PASSED" if recon_ok else "CHECK")],
                         columns=["Metric", "Value"]).set_index("Metric")
    show_df(score, caption="Table 9.4: Product and category scorecard")
    if price_name == lead_cat:
        price_txt = f" (**{esc(price_name)}**, the leading category, also has the highest average unit price)"
    else:
        price_txt = f" (for premium cross-selling, look first at **{esc(price_name)}**, the category with the highest average unit price)"
    if both_n > 0:
        anchor = (f"Use the {plural(both_n, 'product')} that rank high on both revenue and volume as anchor items for bundles and "
                  f"promotions, and let lower-priced volume leaders attract traffic towards higher-priced items{price_txt}.")
    else:
        anchor = (f"Because no product leads on both revenue and volume, build separate playbooks for premium items and for "
                  f"traffic-driving items{price_txt}.")
    interpret(f"Protect availability and marketing support for **{esc(lead_cat)}** and for the {plural(n80_prod, 'product')} that deliver 80% "
              f"of revenue, because lost sales in these top-revenue products reduce total revenue the most (Koch, 1998).",
              anchor, "The long tail of low-revenue products should be reviewed for pruning, bundling or repricing, and the tables and "
                      "charts above are the exploratory starting point for that review (Tukey, 1977).", title="Business implications")


# ---------------------------------------------------------------------------------------------------
# 17.2  Section 10: Repeat purchase analysis                                  (Member 9 - Sanjay Rathour)
# ---------------------------------------------------------------------------------------------------
RP_TOP_N = 8                                            # categories shown in Table 10.3
RP_RC = {"figure.dpi": 110, "font.size": 10, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelsize": 10.5,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
         "grid.alpha": 0.25, "grid.linestyle": "--", "axes.axisbelow": True, "legend.frameon": False}


def rp_working_table(tx):
    """df_clean copy with the short column names used by the Section 10 logic."""
    return pd.DataFrame({"cust": tx["customer_id"].astype(str), "order": tx["transaction_id"].astype(str),
                         "rev": pd.to_numeric(tx["total_amount"], errors="coerce"),
                         "units": pd.to_numeric(tx["quantity"], errors="coerce"), "cat": tx["category"].astype(str).str.strip()})


def rp_rollup(work):
    """item-lines -> orders -> customers, with the 1-Time / Repeat label."""
    orders = (work.groupby(["cust", "order"], sort=False)
              .agg(order_value=("rev", "sum"), units=("units", "sum"), lines=("rev", "size")).reset_index())
    cb = (orders.groupby("cust", sort=False)
          .agg(n_orders=("order", "size"), total_spend=("order_value", "sum"), total_units=("units", "sum")).reset_index())
    cb["buyer_type"] = np.where(cb["n_orders"] >= 2, RP_REP, RP_ONE)
    return orders, cb


def rp_build_summary(cb):
    """Table 10.1 - 1-Time Buyers vs Repeat Buyers: core comparison."""
    g = (cb.groupby("buyer_type").agg(c=("cust", "size"), o=("n_orders", "sum"), r=("total_spend", "sum"))
         .reindex(RP_GROUPS).fillna(0.0))
    g.loc["All Customers"] = g.sum()
    tot_c, tot_r = g.loc["All Customers", "c"], g.loc["All Customers", "r"]
    t = pd.DataFrame(index=g.index)
    t["Customer Count"] = g["c"]
    t["Customer Share (%)"] = g["c"] / tot_c * 100
    t["Total Orders"] = g["o"]
    t["Orders per Customer"] = g["o"] / g["c"].replace(0, np.nan)
    t["Total Revenue"] = g["r"]
    t["Revenue Share (%)"] = g["r"] / tot_r * 100
    t["AOV"] = g["r"] / g["o"].replace(0, np.nan)
    t["Spend per Customer"] = g["r"] / g["c"].replace(0, np.nan)
    t.index.name = "Buyer Type"
    return t


def rp_format_summary(S):
    fmt = {"Customer Count": "{:,.0f}", "Customer Share (%)": "{:.2f}", "Total Orders": "{:,.0f}", "Orders per Customer": "{:.2f}",
           "Total Revenue": CURRENCY_SYMBOL + "{:,.2f}", "Revenue Share (%)": "{:.2f}", "AOV": CURRENCY_SYMBOL + "{:,.2f}",
           "Spend per Customer": CURRENCY_SYMBOL + "{:,.2f}"}
    return pd.DataFrame({c: S[c].map(lambda v, f=fmt[c]: "-" if pd.isna(v) else f.format(v)) for c in S.columns})


def rp_freq_table(cb):
    """Table 10.2 - repeat buyers by number of orders (2, 3, 4+)."""
    rep = cb[cb["buyer_type"] == RP_REP].copy()
    if rep.empty:
        return None
    labels = ["2 orders", "3 orders", "4+ orders"]
    rep["bucket"] = np.select([rep["n_orders"] == 2, rep["n_orders"] == 3], labels[:2], default=labels[2])
    f = (rep.groupby("bucket").agg(customers=("cust", "size"), revenue=("total_spend", "sum"), orders=("n_orders", "sum"))
         .reindex(labels).fillna(0.0))
    out = pd.DataFrame(index=labels)
    out["Customers"] = f["customers"]
    out["% of Repeat Buyers"] = f["customers"] / f["customers"].sum() * 100
    out["Revenue"] = f["revenue"]
    out["Revenue Share within Repeat (%)"] = f["revenue"] / f["revenue"].sum() * 100
    out["Share of Total Revenue (%)"] = f["revenue"] / cb["total_spend"].sum() * 100
    out["Spend per Customer"] = f["revenue"] / f["customers"].replace(0, np.nan)
    out["Avg Orders per Customer"] = f["orders"] / f["customers"].replace(0, np.nan)
    out.index.name = "Order Frequency"
    return out


def rp_category_table(work, top_n=RP_TOP_N):
    """Table 10.3 - category preference of 1-Time vs Repeat buyers with the Preference Index."""
    g = work.groupby(["buyer_type", "cat"]).agg(rev=("rev", "sum"), units=("units", "sum")).reset_index()
    tot = g.groupby("buyer_type")[["rev", "units"]].sum()
    g["rev_p"] = g["rev"] / g["buyer_type"].map(tot["rev"]) * 100
    g["unit_p"] = g["units"] / g["buyer_type"].map(tot["units"]) * 100
    rev_p = g.pivot(index="cat", columns="buyer_type", values="rev_p").reindex(columns=RP_GROUPS).fillna(0.0)
    unit_p = g.pivot(index="cat", columns="buyer_type", values="unit_p").reindex(columns=RP_GROUPS).fillna(0.0)
    order = work.groupby("cat")["rev"].sum().sort_values(ascending=False).index
    tbl = pd.DataFrame({"1-Time Units %": unit_p[RP_ONE], "1-Time Revenue %": rev_p[RP_ONE],
                        "Repeat Units %": unit_p[RP_REP], "Repeat Revenue %": rev_p[RP_REP]}).reindex(order)
    if len(tbl) > top_n:
        rest = tbl.iloc[top_n:].sum().to_frame("All other categories").T
        tbl = pd.concat([tbl.iloc[:top_n], rest])
    tbl["Preference Index"] = tbl["Repeat Revenue %"] / tbl["1-Time Revenue %"].replace(0, np.nan)
    lean = np.select([tbl["Preference Index"] >= 1.10, tbl["Preference Index"] <= 0.90], [RP_REP, RP_ONE], default="Balanced")
    tbl["Leans Toward"] = np.where(tbl["Preference Index"].isna(), "n/a", lean)
    tbl.index.name = "Category"
    return tbl


def rp_basket_table(orders):
    """Table 10.4 - basket size (items per order) of 1-Time vs Repeat buyers."""
    g = orders.groupby("buyer_type")
    n = g.size()
    out = pd.DataFrame({"Avg items (units) per order": g["units"].sum() / n, "Median items per order": g["units"].median(),
                        "Avg item-lines per order": g["lines"].sum() / n,
                        "Avg value per item": g["order_value"].sum() / g["units"].sum().replace(0, np.nan)}).reindex(RP_GROUPS)
    cmp_ = out.T.copy()
    cmp_["Difference (Repeat - 1-Time)"] = cmp_[RP_REP] - cmp_[RP_ONE]
    cmp_["Difference (%)"] = cmp_["Difference (Repeat - 1-Time)"] / cmp_[RP_ONE].replace(0, np.nan) * 100
    cmp_.index.name = "Metric"
    cmp_.columns.name = None
    return cmp_


@chart
def chart_rp_share(S):
    """Figure 10.1 - customer share vs revenue share: dual bars with pp-gap badges plus two donuts."""
    cs = S.loc[RP_GROUPS, "Customer Share (%)"].to_numpy(dtype=float)
    rs = S.loc[RP_GROUPS, "Revenue Share (%)"].to_numpy(dtype=float)
    counts = S.loc[RP_GROUPS, "Customer Count"].to_numpy(dtype=float)
    revenue = S.loc[RP_GROUPS, "Total Revenue"].to_numpy(dtype=float)
    with plt.rc_context(RP_RC):
        fig = plt.figure(figsize=(14, 6.8), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
        ax = fig.add_subplot(gs[:, 0])
        x, w = np.arange(len(RP_GROUPS)), 0.36
        for i, grp in enumerate(RP_GROUPS):
            ax.bar(x[i] - w / 2, cs[i], w, color=RP_COLORS[grp], alpha=0.45, edgecolor="white")
            ax.bar(x[i] + w / 2, rs[i], w, color=RP_COLORS[grp], edgecolor="white")
            for xx, v in ((x[i] - w / 2, cs[i]), (x[i] + w / 2, rs[i])):
                ax.text(xx, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
            gap = rs[i] - cs[i]
            ax.text(x[i], max(cs[i], rs[i]) + 11, f"Gap: {gap:+.1f} pp", ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                    color="#2E7D32" if gap >= 0 else "#C62828", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC"))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n({int(c):,} customers)" for g, c in zip(RP_GROUPS, counts)])
        ax.set_ylim(0, max(cs.max(), rs.max()) + 34)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "%d%%" % v))
        ax.set_ylabel("Share of total (%)")
        ax.set_title("(a) Customer share vs revenue share")
        ax.legend(handles=[Patch(facecolor="#555555", alpha=0.45, label="Customer share (%)"),
                           Patch(facecolor="#555555", label="Revenue share (%)")], loc="upper center", ncol=2)
        for row, (vals, name, tag) in enumerate([(counts, "Customer share", "b"), (revenue, "Revenue share", "c")]):
            axd = fig.add_subplot(gs[row, 1])
            axd.grid(False)
            total = vals.sum()
            wedges, _ = axd.pie(vals, colors=[RP_COLORS[g] for g in RP_GROUPS], startangle=90, counterclock=False,
                                wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2))
            for wd, v in zip(wedges, vals):
                if total > 0 and v / total >= 0.05:
                    ang = np.deg2rad((wd.theta1 + wd.theta2) / 2.0)
                    axd.text(0.75 * np.cos(ang), 0.75 * np.sin(ang), f"{v / total * 100:.1f}%", ha="center", va="center",
                             color="white", fontsize=10.5, fontweight="bold")
            axd.text(0, 0, name.replace(" ", "\n"), ha="center", va="center", fontsize=10.5, fontweight="bold")
            axd.set_title(f"({tag}) {name}", fontsize=11)
            axd.set_aspect("equal")
            if row == 1:
                axd.legend(handles=[Patch(facecolor=RP_COLORS[g], label=g) for g in RP_GROUPS], loc="upper center",
                           bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9.5)
        fig.suptitle("Figure 10.1 - Customer Share vs Revenue Share: 1-Time vs Repeat Buyers", fontsize=14, fontweight="bold")
        return fig_to_png(fig)


@chart
def chart_rp_spend(cb, orders, S):
    """Figure 10.2 - spend distribution (log-scale boxplots) and average bars for spend per customer and AOV."""
    present = [g for g in RP_GROUPS if (cb["buyer_type"] == g).any()]
    cust_spend = {g: cb.loc[cb["buyer_type"] == g, "total_spend"].to_numpy(dtype=float) for g in present}
    order_val = {g: orders.loc[orders["buyer_type"] == g, "order_value"].to_numpy(dtype=float) for g in present}

    def money_axis(ax):
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{CURRENCY_SYMBOL}{v:,.0f}" if (abs(v) >= 1 or v == 0) else f"{CURRENCY_SYMBOL}{v:,.2f}"))

    def box(ax, data, title, ylabel):
        arrays = [d[np.isfinite(d) & (d > 0)] for d in (data[g] for g in present)]
        pos = np.arange(1, len(present) + 1)
        bp = ax.boxplot(arrays, positions=pos, widths=0.55, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.30, markerfacecolor="gray", markeredgecolor="none"))
        for patch, g in zip(bp["boxes"], present):
            patch.set_facecolor(RP_COLORS[g])
            patch.set_alpha(0.85)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        money_axis(ax)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{g}\nmedian {money(np.median(a), 0)} | mean {money(np.mean(a), 0)}" for g, a in zip(present, arrays)],
                           fontsize=9.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    def bar(ax, col, title, ylabel):
        vals = S.loc[RP_GROUPS, col].to_numpy(dtype=float)
        top = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
        xs = np.arange(len(RP_GROUPS))
        bars = ax.bar(xs, np.nan_to_num(vals), color=[RP_COLORS[g] for g in RP_GROUPS], width=0.55, edgecolor="white")
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, money(v, 2), ha="center", va="bottom", fontsize=11,
                        fontweight="bold")
        if np.isfinite(vals).all() and vals[0] > 0:
            ax.text(0.5, 0.97, f"Repeat = {vals[1] / vals[0]:.2f}x of 1-Time", transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color="#444444", bbox=dict(boxstyle="round,pad=0.3", fc="#F3F6FA", ec="#D0D7DE"))
        ax.set_xticks(xs)
        ax.set_xticklabels(RP_GROUPS)
        ax.set_ylim(0, top * 1.28)
        money_axis(ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    with plt.rc_context(RP_RC):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.6), constrained_layout=True)
        box(axes[0, 0], cust_spend, "(a) Total spend per customer - distribution", "Total spend per customer (log scale)")
        box(axes[0, 1], order_val, "(b) Order value (AOV) - distribution", "Value per order (log scale)")
        bar(axes[1, 0], "Spend per Customer", "(c) Average total spend per customer", "Average spend per customer")
        bar(axes[1, 1], "AOV", "(d) Average Order Value (AOV)", "Average order value")
        fig.suptitle("Figure 10.2 - Spend Distribution and AOV: 1-Time vs Repeat Buyers", fontsize=14, fontweight="bold")
        return fig_to_png(fig)


@chart
def chart_rp_freq(F):
    """Figure 10.3 - order-frequency breakdown of repeat buyers (headcount vs revenue share)."""
    labels = list(F.index)
    shades = ["#9DB9D6", "#4E79A7", "#2B4C75"]
    with plt.rc_context(RP_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})
        top = max(F["Customers"].max(), 1)
        bars = a1.bar(labels, F["Customers"], color=shades, width=0.6, edgecolor="white")
        for b, n_, p in zip(bars, F["Customers"], F["% of Repeat Buyers"]):
            a1.text(b.get_x() + b.get_width() / 2, n_ + top * 0.02, f"{int(n_):,}\n({p:.1f}%)", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold")
        a1.set_ylim(0, top * 1.25)
        a1.set_ylabel("Number of repeat buyers")
        a1.set_title("(a) Repeat buyers by number of orders")
        a1.yaxis.set_major_formatter(thousands_formatter)
        x, w = np.arange(len(labels)), 0.38
        c_share, r_share = F["% of Repeat Buyers"].to_numpy(float), F["Revenue Share within Repeat (%)"].to_numpy(float)
        a2.bar(x - w / 2, c_share, w, color=RP_COLORS[RP_REP], alpha=0.45, edgecolor="white", label="% of repeat buyers")
        a2.bar(x + w / 2, r_share, w, color=RP_COLORS[RP_REP], edgecolor="white", label="% of repeat-buyer revenue")
        for xx, v in list(zip(x - w / 2, c_share)) + list(zip(x + w / 2, r_share)):
            a2.text(xx, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        a2.set_xticks(x)
        a2.set_xticklabels(labels)
        a2.set_ylim(0, max(c_share.max(), r_share.max()) * 1.32 + 1)
        a2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "%d%%" % v))
        a2.set_ylabel("Share within repeat buyers (%)")
        a2.set_title("(b) Headcount share vs revenue share")
        a2.legend(loc="upper right")
        fig.suptitle("Figure 10.3 - Order-Frequency Breakdown of Repeat Buyers", fontsize=14, fontweight="bold")
        return fig_to_png(fig)


@chart
def chart_rp_category(tbl):
    """Figure 10.4 - category mix of 1-Time vs Repeat buyers (share of each group's own revenue)."""
    core = tbl.drop(index="All other categories", errors="ignore")
    cats = list(core.index)[::-1]
    one = core["1-Time Revenue %"].to_numpy(dtype=float)[::-1]
    rep = core["Repeat Revenue %"].to_numpy(dtype=float)[::-1]
    y, h = np.arange(len(cats)), 0.38
    with plt.rc_context(RP_RC):
        fig, ax = plt.subplots(figsize=(11.5, 0.8 * len(cats) + 3.4), constrained_layout=True)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.barh(y + h / 2, one, h, color=RP_COLORS[RP_ONE], label=RP_ONE, edgecolor="white")
        ax.barh(y - h / 2, rep, h, color=RP_COLORS[RP_REP], label=RP_REP, edgecolor="white")
        top = max(one.max(), rep.max(), 1.0)
        for yy, v in list(zip(y + h / 2, one)) + list(zip(y - h / 2, rep)):
            ax.text(v + top * 0.01, yy, f"{v:.1f}%", va="center", fontsize=9.5)
        ax.set_yticks(y)
        ax.set_yticklabels(cats)
        ax.set_xlim(0, top * 1.15)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "%d%%" % v))
        ax.set_xlabel("Share of the group's own revenue (%)")
        ax.set_title("Top categories by revenue share within each buyer group")
        ax.legend(loc="lower right")
        fig.suptitle("Figure 10.4 - Category Mix: 1-Time vs Repeat Buyers", fontsize=14, fontweight="bold")
        return fig_to_png(fig)


def rp_numbers(S, cb, freq, cat_tbl, basket):
    """Every number used by the key findings, collected from Tables 10.1 - 10.4."""
    n = {}
    one, rep, tot = S.loc[RP_ONE], S.loc[RP_REP], S.loc["All Customers"]
    n["N_CUST"], n["N_ORD"], n["REV_TOT"] = tot["Customer Count"], tot["Total Orders"], tot["Total Revenue"]
    n["N_ONE"], n["N_REP"] = one["Customer Count"], rep["Customer Count"]
    n["P_ONE"], n["P_REP"] = one["Customer Share (%)"], rep["Customer Share (%)"]
    n["R_ONE"], n["R_REP"] = one["Revenue Share (%)"], rep["Revenue Share (%)"]
    n["IDX_REV"] = n["R_REP"] / n["P_REP"] if n["P_REP"] > 0 else np.nan
    n["AOV_ONE"], n["AOV_REP"] = one["AOV"], rep["AOV"]
    n["AOV_RATIO"] = n["AOV_REP"] / n["AOV_ONE"] if n["AOV_ONE"] > 0 else np.nan
    n["SPC_ONE"], n["SPC_REP"] = one["Spend per Customer"], rep["Spend per Customer"]
    n["SPC_RATIO"] = n["SPC_REP"] / n["SPC_ONE"] if n["SPC_ONE"] > 0 else np.nan
    n["OPC_REP"] = rep["Orders per Customer"]
    med = cb.groupby("buyer_type")["total_spend"].median()
    n["MED_ONE"], n["MED_REP"] = med.get(RP_ONE, np.nan), med.get(RP_REP, np.nan)
    if freq is not None:
        n["F2"], n["F3"], n["F4"] = (freq.loc["2 orders", "% of Repeat Buyers"], freq.loc["3 orders", "% of Repeat Buyers"],
                                     freq.loc["4+ orders", "% of Repeat Buyers"])
        n["F4_REV"] = freq.loc["4+ orders", "Revenue Share within Repeat (%)"]
        n["F4_TOT"] = freq.loc["4+ orders", "Share of Total Revenue (%)"]
        n["F_TOP"] = freq["Customers"].idxmax()
    else:
        for k in ("F2", "F3", "F4", "F4_REV", "F4_TOT"):
            n[k] = np.nan
        n["F_TOP"] = "n/a"
    if cat_tbl is not None:
        core = cat_tbl.drop(index="All other categories", errors="ignore")
        n["TOP_CAT_ONE"], n["TOP_CAT_ONE_P"] = core["1-Time Revenue %"].idxmax(), core["1-Time Revenue %"].max()
        n["TOP_CAT_REP"], n["TOP_CAT_REP_P"] = core["Repeat Revenue %"].idxmax(), core["Repeat Revenue %"].max()
        pref = core["Preference Index"].dropna().sort_values(ascending=False)
        n["OVER_CATS"] = ", ".join(f"{c} ({v:.2f}x)" for c, v in pref.head(2).items() if v >= 1.10) or "none"
        n["UNDER_CATS"] = ", ".join(f"{c} ({v:.2f}x)" for c, v in pref.sort_values().head(2).items() if v <= 0.90) or "none"
    b = basket.loc["Avg items (units) per order"]
    n["BASKET_ONE"], n["BASKET_REP"] = b[RP_ONE], b[RP_REP]
    n["BASKET_DIFF"], n["BASKET_DIFF_PCT"] = b["Difference (Repeat - 1-Time)"], b["Difference (%)"]
    v = basket.loc["Avg value per item"]
    n["ITEMVAL_ONE"], n["ITEMVAL_REP"] = v[RP_ONE], v[RP_REP]
    return n


def rp_findings(n):
    """The six key findings as a list of Markdown sentences (wording adapts to the direction of the numbers)."""
    m2 = lambda v: money(v, 2)                                              # noqa: E731

    def f1():
        concl = ("**Most customers buy only once**, so acquisition is not yet turning into loyalty." if n["P_ONE"] > 50
                 else "**Most customers already come back**, so retention is a relative strength.")
        return (f"**Finding 1 - Retention baseline.** {pct(n['P_REP'])} of customers ({num(n['N_REP'])} of {num(n['N_CUST'])}) are repeat "
                f"buyers - this is the Repeat Purchase Rate - while {num(n['N_ONE'])} customers ({pct(n['P_ONE'])}) bought only once. {concl}")

    def f2():
        concl = ("they earn more than their headcount suggests, so protecting and growing this group protects most of the revenue"
                 if n["R_REP"] > n["P_REP"] else "they do not (yet) earn more than their headcount suggests, so repeat behaviour is not "
                 "translating into extra revenue")
        return (f"**Finding 2 - Revenue concentration.** Repeat buyers are {pct(n['P_REP'])} of customers but generate {pct(n['R_REP'])} of "
                f"revenue - a revenue-to-customer-share index of **{n['IDX_REV']:.2f}x**; {concl}.")

    def f3():
        freq_x, aov_x = n["OPC_REP"], n["AOV_RATIO"]
        if pd.isna(freq_x) or pd.isna(aov_x) or freq_x <= 0 or aov_x <= 0:
            return "**Finding 3 - Spend per customer.** Could not be computed (check Table 10.1)."
        driver = ("**purchase frequency** (more orders per customer)" if abs(np.log(freq_x)) >= abs(np.log(aov_x))
                  else "**larger orders** (higher AOV)")
        aov_txt = f"{(aov_x - 1) * 100:.1f}% higher" if aov_x >= 1 else f"{(1 - aov_x) * 100:.1f}% lower"
        return (f"**Finding 3 - Spend per customer.** A repeat buyer spends {m2(n['SPC_REP'])} on average versus {m2(n['SPC_ONE'])} for a "
                f"1-time buyer (**{n['SPC_RATIO']:.2f}x**; median {m2(n['MED_REP'])} vs {m2(n['MED_ONE'])}). Repeat buyers place {freq_x:.2f} "
                f"orders each (1-time buyers: exactly 1) and their AOV is {aov_txt} ({m2(n['AOV_REP'])} vs {m2(n['AOV_ONE'])}) - so the gap "
                f"is driven mainly by {driver}.")

    def f4():
        if pd.isna(n["F2"]):
            return "**Finding 4 - Order-frequency depth.** Not available (no repeat buyers)."
        link = "yet brings" if n["F4_REV"] > n["F4"] else "and brings"
        extra = (" The largest repeat group stops after the 2nd order, so **moving customers from 2 to 3 orders is a key lever**."
                 if n["F_TOP"] == "2 orders" else "")
        return (f"**Finding 4 - Order-frequency depth.** Among repeat buyers, {pct(n['F2'])} placed exactly 2 orders, {pct(n['F3'])} placed 3 "
                f"and {pct(n['F4'])} placed 4 or more. The 4+ group is {pct(n['F4'])} of repeat buyers {link} {pct(n['F4_REV'])} of "
                f"repeat-buyer revenue ({pct(n['F4_TOT'])} of total revenue).{extra}")

    def f5():
        if "TOP_CAT_ONE" not in n:
            return "**Finding 5 - Category preference.** Not available."
        if n["TOP_CAT_ONE"] == n["TOP_CAT_REP"]:
            lead = (f"Both groups spend most on **{esc(n['TOP_CAT_ONE'])}** ({pct(n['TOP_CAT_ONE_P'])} of 1-time and "
                    f"{pct(n['TOP_CAT_REP_P'])} of repeat revenue)")
        else:
            lead = (f"1-time buyers spend most on **{esc(n['TOP_CAT_ONE'])}** ({pct(n['TOP_CAT_ONE_P'])} of their revenue) while repeat "
                    f"buyers spend most on **{esc(n['TOP_CAT_REP'])}** ({pct(n['TOP_CAT_REP_P'])} of theirs)")
        return (f"**Finding 5 - Category preference.** {lead}. Categories over-represented among repeat buyers: {esc(n['OVER_CATS'])}; "
                f"over-represented among 1-time buyers: {esc(n['UNDER_CATS'])}. Repeat-leaning categories are candidates for cross-selling "
                f"on the first order, while 1-time-leaning categories look like one-off entry purchases.")

    def f6():
        b1, b2, i1, i2 = n["BASKET_ONE"], n["BASKET_REP"], n["ITEMVAL_ONE"], n["ITEMVAL_REP"]
        if any(pd.isna(x) or x <= 0 for x in (b1, b2, i1, i2)):
            return "**Finding 6 - Basket size.** Could not be computed (check Table 10.4)."
        driver = ("the number of items per order (basket size)" if abs(np.log(b2 / b1)) >= abs(np.log(i2 / i1))
                  else "the value of each item")
        return (f"**Finding 6 - Basket size.** A repeat buyer's order holds {b2:.2f} items versus {b1:.2f} for a 1-time buyer "
                f"({n['BASKET_DIFF']:+.2f} items, {n['BASKET_DIFF_PCT']:+.1f}%), with an average value per item of {m2(i2)} vs {m2(i1)}. "
                f"AOV bridge: AOV ratio **{n['AOV_RATIO']:.2f}x** = basket-size ratio **{b2 / b1:.2f}x** x value-per-item ratio "
                f"**{i2 / i1:.2f}x**, so the AOV difference is explained mainly by **{driver}**.")

    out = []
    for title, fn in (("Finding 1", f1), ("Finding 2", f2), ("Finding 3", f3), ("Finding 4", f4), ("Finding 5", f5), ("Finding 6", f6)):
        try:
            out.append(fn())
        except Exception as err:                                           # noqa: BLE001
            out.append(f"**{title}:** could not be generated automatically ({err}).")
    return out


def render_sec10(ctx):
    """Section 10 (Member 9): 1-Time vs Repeat buyers - tables 10.1-10.4, figures 10.1-10.4, findings, actions."""
    tx, cust = ctx["tx"], ctx["cust"]
    section_header("Section 10: Repeat Purchase Analysis (1-Time vs Repeat Buyers)", "Member 9 - Sanjay Rathour")
    work = rp_working_table(tx)
    orders, cb = rp_rollup(work)
    type_of = cb.set_index("cust")["buyer_type"]
    work["buyer_type"] = work["cust"].map(type_of)
    orders["buyer_type"] = orders["cust"].map(type_of)
    S = rp_build_summary(cb)

    # ---- analysis base and reconciliation ----
    cu_match = cust.set_index("customer_id")[["num_orders", "total_spend"]].join(cb.set_index("cust")[["n_orders", "total_spend"]],
                                                                                  rsuffix="_rp", how="outer")
    ord_ok = bool((cu_match["num_orders"] == cu_match["n_orders"]).all())
    spend_ok = bool(np.allclose(cu_match["total_spend"].to_numpy(float), cu_match["total_spend_rp"].to_numpy(float), rtol=1e-4, atol=0.01))
    st.markdown(f"**Analysis base:** {len(cb):,} customers | {len(orders):,} orders | {len(work):,} item-lines | total revenue "
                f"{money(orders['order_value'].sum(), 2)}. **Definition:** one order = one unique `transaction_id` of one customer; "
                f"1-Time Buyers have exactly 1 order, Repeat Buyers have 2 or more. **Reconciliation with customer_df:** order counts "
                f"{'match' if ord_ok else 'differ'}, total spend {'matches' if spend_ok else 'differs'}.")
    preview = (cb.rename(columns={"cust": "Customer", "n_orders": "Orders", "total_spend": "Total Spend", "buyer_type": "Buyer Type"})
               [["Customer", "Orders", "Total Spend", "Buyer Type"]])
    with st.expander("Classification preview (4 customers of each type)"):
        show_df(preview.groupby("Buyer Type").head(4).sort_values(["Buyer Type", "Orders"]).reset_index(drop=True), hide_index=True)

    show_df(rp_format_summary(S), caption="Table 10.1: 1-Time Buyers vs Repeat Buyers - core comparison")
    both = bool((cb["buyer_type"] == RP_ONE).any() and (cb["buyer_type"] == RP_REP).any())
    if not both:
        only = RP_ONE if (cb["buyer_type"] == RP_ONE).any() else RP_REP
        st.info(f"The current selection contains only **{only}**, so the comparison tables, charts and findings of Section 10 need "
                f"both buyer types. Select both in the sidebar (Buyer type) to see the full analysis.")
        return
    one, rep = S.loc[RP_ONE], S.loc[RP_REP]
    interpret(f"{pct(rep['Customer Share (%)'])} of customers are repeat buyers (the Repeat Purchase Rate) while {pct(one['Customer Share (%)'])} "
              f"bought only once; repeat buyers are {pct(rep['Customer Share (%)'])} of customers but bring {pct(rep['Revenue Share (%)'])} of revenue.",
              f"A repeat buyer is worth {money(rep['Spend per Customer'])} against {money(one['Spend per Customer'])} for a 1-time buyer "
              f"({safe_div(rep['Spend per Customer'], one['Spend per Customer']):.2f}x), and their AOV is {money(rep['AOV'])} versus {money(one['AOV'])}.",
              "AOV = total revenue / total orders and spend per customer = total revenue / customers, so the table separates "
              "'how often' from 'how much' customers buy.")

    caption_md("Figure 10.1: Customer share vs revenue share")
    show_png(chart_rp_share(S))
    gap = rep["Revenue Share (%)"] - rep["Customer Share (%)"]
    interpret(f"Repeat buyers are {pct(rep['Customer Share (%)'])} of customers but bring {pct(rep['Revenue Share (%)'])} of revenue, a gap of "
              f"{gap:+.1f} percentage points (revenue-to-customer-share index {safe_div(rep['Revenue Share (%)'], rep['Customer Share (%)']):.2f}x).",
              f"The donut charts show the same split as shares of the whole: {int(rep['Customer Count']):,} repeat and {int(one['Customer Count']):,} "
              f"one-time customers account for {money(rep['Total Revenue'])} and {money(one['Total Revenue'])} respectively.")

    caption_md("Figure 10.2: Spend distribution and AOV")
    show_png(chart_rp_spend(cb[["buyer_type", "total_spend"]], orders[["buyer_type", "order_value"]], S))
    med = cb.groupby("buyer_type")["total_spend"].agg(["median", "mean"]).reindex(RP_GROUPS)
    interpret(f"Median spend per customer is {money(med.loc[RP_ONE, 'median'])} for 1-time buyers and {money(med.loc[RP_REP, 'median'])} for repeat "
              f"buyers (means {money(med.loc[RP_ONE, 'mean'])} and {money(med.loc[RP_REP, 'mean'])}); the log-scale boxplots show that both "
              f"distributions are right-skewed.",
              f"AOV is {money(S.loc[RP_ONE, 'AOV'])} for 1-time and {money(S.loc[RP_REP, 'AOV'])} for repeat buyers, so the bars in panels (c) and (d) "
              f"show how much of the spend gap comes from order size rather than order count.")

    freq = rp_freq_table(cb)
    if freq is not None:
        shown = freq.copy()
        fmt = {"Customers": "{:,.0f}", "% of Repeat Buyers": "{:.2f}", "Revenue": CURRENCY_SYMBOL + "{:,.2f}",
               "Revenue Share within Repeat (%)": "{:.2f}", "Share of Total Revenue (%)": "{:.2f}",
               "Spend per Customer": CURRENCY_SYMBOL + "{:,.2f}", "Avg Orders per Customer": "{:.2f}"}
        show_df(pd.DataFrame({c: shown[c].map(lambda v, f=fmt[c]: "-" if pd.isna(v) else f.format(v)) for c in shown.columns}),
                caption="Table 10.2: Repeat buyers by order frequency")
        caption_md("Figure 10.3: Order-frequency breakdown of repeat buyers")
        show_png(chart_rp_freq(freq))
        interpret(f"The largest repeat group is '{freq['Customers'].idxmax()}' ({pct(freq['% of Repeat Buyers'].max())} of repeat buyers); "
                  f"customers with 4+ orders are {pct(freq.loc['4+ orders', '% of Repeat Buyers'])} of repeat buyers and bring "
                  f"{pct(freq.loc['4+ orders', 'Revenue Share within Repeat (%)'])} of repeat-buyer revenue "
                  f"({pct(freq.loc['4+ orders', 'Share of Total Revenue (%)'])} of total revenue).",
                  "A revenue share above the headcount share for the 4+ group shows that depth of loyalty, not only the second order, drives revenue.")

    cat_tbl = rp_category_table(work)
    fmt = {"1-Time Units %": "{:.2f}", "1-Time Revenue %": "{:.2f}", "Repeat Units %": "{:.2f}", "Repeat Revenue %": "{:.2f}",
           "Preference Index": "{:.2f}"}
    disp = pd.DataFrame({c: (cat_tbl[c].map(lambda v, f=fmt[c]: "-" if pd.isna(v) else f.format(v)) if c in fmt else cat_tbl[c])
                         for c in cat_tbl.columns})
    show_df(disp, caption="Table 10.3: Category preference of 1-Time vs Repeat buyers (shares are % of each group's own total; "
                          "Preference Index = Repeat revenue % / 1-Time revenue %)")
    caption_md("Figure 10.4: Category mix of 1-Time vs Repeat buyers")
    show_png(chart_rp_category(cat_tbl))
    core = cat_tbl.drop(index="All other categories", errors="ignore").dropna(subset=["Preference Index"])
    over = core[core["Leans Toward"] == RP_REP].sort_values("Preference Index", ascending=False)
    under = core[core["Leans Toward"] == RP_ONE].sort_values("Preference Index")
    interpret("Over-represented among repeat buyers: " + (", ".join(f"{esc(c)} ({v:.2f}x)" for c, v in over["Preference Index"].head(3).items()) or "none")
              + "; over-represented among 1-time buyers: "
              + (", ".join(f"{esc(c)} ({v:.2f}x)" for c, v in under["Preference Index"].head(3).items()) or "none") + ".",
              "A Preference Index above 1.10 marks a category that repeat buyers favour, below 0.90 a category that one-time buyers favour; "
              "values in between are balanced.")

    basket = rp_basket_table(orders)
    bfmt = basket.copy()
    for c in basket.columns:
        bfmt[c] = [(f"{money(v, 2)}" if (name == "Avg value per item" and c in (RP_ONE, RP_REP)) else
                    (f"{v:+,.2f}" if c == "Difference (Repeat - 1-Time)" else (f"{v:+.1f}%" if c == "Difference (%)" else f"{v:,.2f}")))
                   if pd.notna(v) else "-" for name, v in zip(basket.index, basket[c])]
    show_df(bfmt, caption="Table 10.4: Basket-size comparison (per order)")
    b, v_, aov = basket.loc["Avg items (units) per order"], basket.loc["Avg value per item"], S["AOV"]
    if all(pd.notna(x) and x > 0 for x in [b[RP_ONE], b[RP_REP], v_[RP_ONE], v_[RP_REP], aov[RP_ONE], aov[RP_REP]]):
        bridge = pd.DataFrame([("AOV of 1-Time Buyers", money(aov[RP_ONE], 2)), ("AOV of Repeat Buyers", money(aov[RP_REP], 2)),
                               ("AOV ratio (Repeat / 1-Time)", f"{aov[RP_REP] / aov[RP_ONE]:.3f}x"),
                               ("Basket-size ratio (items per order)", f"{b[RP_REP] / b[RP_ONE]:.3f}x"),
                               ("Value-per-item ratio", f"{v_[RP_REP] / v_[RP_ONE]:.3f}x"),
                               ("Check: basket-size ratio x value-per-item ratio", f"{(b[RP_REP] / b[RP_ONE]) * (v_[RP_REP] / v_[RP_ONE]):.3f}x")],
                              columns=["AOV bridge metric", "Value"])
        show_df(bridge, caption="AOV bridge: AOV = items per order x value per item (the last row must equal the AOV ratio)", hide_index=True)
        interpret(f"A repeat buyer's order holds {b[RP_REP]:.2f} items versus {b[RP_ONE]:.2f} for a 1-time buyer "
                  f"({b[RP_REP] - b[RP_ONE]:+.2f} items, {b['Difference (%)']:+.1f}%).",
                  f"AOV bridge: AOV ratio {aov[RP_REP] / aov[RP_ONE]:.2f}x = basket-size ratio {b[RP_REP] / b[RP_ONE]:.2f}x x value-per-item "
                  f"ratio {v_[RP_REP] / v_[RP_ONE]:.2f}x, which tells whether larger baskets or pricier items explain the AOV gap.")

    st.markdown("#### Key findings")
    nums = rp_numbers(S, cb, freq, cat_tbl, basket)
    for line in rp_findings(nums):
        st.markdown("- " + line.replace("$", r"\$"))
    st.markdown("#### Recommendations: how to turn 1-time buyers into repeat buyers")
    md_table(pd.DataFrame(RP_RECOMMENDATIONS, columns=["#", "Strategy", "What to do", "Based on", "KPI to track"]))
    interpret(f"With {nums['N_ONE']:,.0f} one-time buyers ({pct(nums['P_ONE'])} of customers), converting even 10% of them would add about "
              f"{0.10 * nums['N_ONE']:,.0f} repeat buyers, each worth on average {money(nums['SPC_REP'])} instead of {money(nums['SPC_ONE'])}.",
              "The seven strategies target the three leaks visible above: no second order, a shallow order history and small first baskets.")
    with st.expander("Assumptions and limitations of Section 10"):
        for item in RP_LIMITATIONS:
            st.markdown("- " + item)


# ---------------------------------------------------------------------------------------------------
# 17.3  Section 11: Monthly, weekday and seasonal sales trends                    (Member 10 - Dev Rastogi)
# ---------------------------------------------------------------------------------------------------
S11_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
S11_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
S11_QUARTER_OF_MONTH = {m: "Q" + str((m - 1) // 3 + 1) for m in range(1, 13)}
S11_SEASONS = ["Winter", "Spring/Summer", "Monsoon", "Festive/Autumn"]
S11_SEASON_MONTHS = {"Winter": "Dec-Feb", "Spring/Summer": "Mar-Jun", "Monsoon": "Jul-Sep", "Festive/Autumn": "Oct-Nov"}
S11_SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring/Summer", 4: "Spring/Summer", 5: "Spring/Summer",
                       6: "Spring/Summer", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Festive/Autumn", 11: "Festive/Autumn"}
S11_TOP_N_CATEGORIES = 8


def s11_strength(gap):
    return "pronounced" if gap >= 25 else ("moderate" if gap >= 10 else "mild")


def s11_best_worst(tbl):
    idx = tbl["seasonal_index"].dropna()
    return idx.idxmax(), idx.idxmin(), float(idx.max()), float(idx.min())


def s11_period_table(work, group_col, order, month_to_label, monthly, avg_month):
    """Quarter / season table with the seasonal index (average monthly revenue of the period / overall average month x 100)."""
    tbl = (work.groupby(group_col).agg(revenue=("revenue_amt", "sum"), orders=("transaction_id", "nunique"))
           .reindex(order).fillna(0))
    tbl["avg_order_value"] = tbl["revenue"] / tbl["orders"].replace(0, np.nan)
    tbl["revenue_share_pct"] = tbl["revenue"] / tbl["revenue"].sum() * 100
    tbl["months_in_data"] = (pd.Series(monthly.index.month).map(month_to_label).value_counts().reindex(order).fillna(0).to_numpy())
    tbl["avg_monthly_revenue"] = tbl["revenue"] / tbl["months_in_data"].replace(0, np.nan)
    tbl["seasonal_index"] = tbl["avg_monthly_revenue"] / avg_month * 100
    return tbl


def s11_period_view(tbl, label, month_span=None):
    out = pd.DataFrame({label: list(tbl.index)})
    if month_span:
        out["Calendar Months"] = [month_span[k] for k in tbl.index]
    out["Months in Data"] = tbl["months_in_data"].astype(int).to_numpy()
    out["Revenue"] = tbl["revenue"].map(lambda v: money(v, 0)).to_numpy()
    out["Revenue Share (%)"] = tbl["revenue_share_pct"].map(lambda v: pct(v, 1)).to_numpy()
    out["Orders"] = tbl["orders"].map(num).to_numpy()
    out["Avg Order Value"] = tbl["avg_order_value"].map(lambda v: money(v, 2)).to_numpy()
    out["Avg Monthly Revenue"] = tbl["avg_monthly_revenue"].map(lambda v: money(v, 0)).to_numpy()
    out["Seasonal Index"] = tbl["seasonal_index"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}").to_numpy()
    return out


def s11_period_insight(tbl, kind, month_span=None):
    """Three-sentence insight on the strongest / weakest period of a quarter or season table."""
    if tbl["seasonal_index"].notna().sum() < 2:
        return [f"Only one {kind} is covered by the data, so a seasonal comparison is not yet possible.",
                "A longer date range is needed to study seasonality (Tukey, 1977)."]
    best, worst, best_i, worst_i = s11_best_worst(tbl)
    gap = best_i - worst_i
    best_txt = f"**{best}** ({month_span[best]})" if month_span else f"**{best}**"
    out = [f"{best_txt} is the strongest {kind} (seasonal index {best_i:.0f}, {tbl.loc[best, 'revenue_share_pct']:.1f}% of revenue) and "
           f"**{worst}** the weakest (index {worst_i:.0f})."]
    if gap >= 25:
        out.append(f"A spread of {gap:.0f} index points signals {s11_strength(gap)} seasonality, so stock, staffing and campaign budgets "
                   f"should be built up ahead of {best} and clearance or loyalty offers planned for {worst}.")
    elif gap >= 10:
        out.append(f"A spread of {gap:.0f} index points signals {s11_strength(gap)} seasonality, so targets and stock should tilt towards "
                   f"{best} without over-committing.")
    else:
        out.append(f"A spread of only {gap:.0f} index points signals {s11_strength(gap)} seasonality, so demand is steady across {kind}s "
                   f"and growth has to come from customer acquisition and basket size rather than timing.")
    out.append(f"Reading the {kind}s side by side is a simple exploratory check for repeating patterns (Tukey, 1977).")
    return out


def s11_tables(tx):
    """All Section 11 tables built from the active transactions (working copy only)."""
    work = tx[["transaction_id", "customer_id", "transaction_date", "category", "total_amount"]].copy()
    work["revenue_amt"] = pd.to_numeric(work["total_amount"], errors="coerce")
    work = work.dropna(subset=["transaction_date", "revenue_amt"]).copy()
    work["category"] = work["category"].astype(object).fillna("Unknown").astype(str)
    work["year_month"] = work["transaction_date"].dt.to_period("M")
    work["month_num"] = work["transaction_date"].dt.month
    work["weekday"] = work["transaction_date"].dt.day_name()
    work["quarter"] = "Q" + work["transaction_date"].dt.quarter.astype(str)
    work["season"] = work["month_num"].map(S11_SEASON_OF_MONTH)
    monthly = (work.groupby("year_month").agg(revenue=("revenue_amt", "sum"), orders=("transaction_id", "nunique"),
                                              active_customers=("customer_id", "nunique")))
    monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"), fill_value=0)
    monthly.index.name = "year_month"
    monthly["avg_order_value"] = monthly["revenue"] / monthly["orders"].replace(0, np.nan)
    for metric, col in (("revenue", "mom_revenue_pct"), ("orders", "mom_orders_pct")):
        prev = monthly[metric].shift(1)
        monthly[col] = ((monthly[metric] - prev) / prev * 100).replace([np.inf, -np.inf], np.nan)
    weekday = (work.groupby("weekday").agg(revenue=("revenue_amt", "sum"), orders=("transaction_id", "nunique"))
               .reindex(S11_WEEKDAYS).fillna(0))
    weekday.index.name = "weekday"
    weekday["avg_order_value"] = weekday["revenue"] / weekday["orders"].replace(0, np.nan)
    weekday["revenue_share_pct"] = weekday["revenue"] / weekday["revenue"].sum() * 100
    span = pd.date_range(work["transaction_date"].min().normalize(), work["transaction_date"].max().normalize(), freq="D")
    weekday["calendar_days"] = pd.Series(span.day_name()).value_counts().reindex(S11_WEEKDAYS).fillna(0).to_numpy()
    weekday["avg_daily_revenue"] = weekday["revenue"] / weekday["calendar_days"].replace(0, np.nan)
    total_months = len(monthly)
    avg_month = monthly["revenue"].sum() / total_months
    quarter = s11_period_table(work, "quarter", S11_QUARTERS, S11_QUARTER_OF_MONTH, monthly, avg_month)
    season = s11_period_table(work, "season", S11_SEASONS, S11_SEASON_OF_MONTH, monthly, avg_month)
    cat_rev = (work.pivot_table(index="category", columns="season", values="revenue_amt", aggfunc="sum", fill_value=0)
               .reindex(columns=S11_SEASONS, fill_value=0))
    cat_rev["Total"] = cat_rev.sum(axis=1)
    cat_rev = cat_rev.sort_values("Total", ascending=False)
    top_cats = list(cat_rev.index[:min(S11_TOP_N_CATEGORIES, len(cat_rev))])
    cat_idx = (cat_rev[S11_SEASONS].div(season["months_in_data"].replace(0, np.nan), axis=1)
               .div(cat_rev["Total"] / total_months, axis=0) * 100)
    return {"work": work, "monthly": monthly, "weekday": weekday, "quarter": quarter, "season": season, "cat_rev": cat_rev,
            "cat_idx": cat_idx, "top_cats": top_cats, "avg_month": avg_month, "total_months": total_months,
            "rev_all": float(work["revenue_amt"].sum())}


def s11_weekend_stats(weekday):
    """(weekend revenue share %, weekend share of calendar days %, weekend revenue-per-day lift vs weekdays %)."""
    we, wk = S11_WEEKDAYS[5:], S11_WEEKDAYS[:5]
    rev_we, rev_wk = weekday.loc[we, "revenue"].sum(), weekday.loc[wk, "revenue"].sum()
    day_we, day_wk = weekday.loc[we, "calendar_days"].sum(), weekday.loc[wk, "calendar_days"].sum()
    share = rev_we / weekday["revenue"].sum() * 100
    day_share = day_we / weekday["calendar_days"].sum() * 100
    lift = ((rev_we / day_we) / (rev_wk / day_wk) - 1) * 100 if (day_we > 0 and day_wk > 0 and rev_wk > 0) else np.nan
    return share, day_share, lift


@chart
def chart_monthly_trend(monthly):
    """Figure 11.1 - monthly revenue (left axis) and orders (right axis) on one time axis."""
    x = monthly.index.to_timestamp()
    pk = monthly["revenue"].idxmax()
    pk_rev = monthly.loc[pk, "revenue"]
    c_rev, c_ord = PALETTE_LIST[0], PALETTE_LIST[1]
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(x, monthly["revenue"], color=c_rev, marker="o", linewidth=2.2, label="Revenue (left axis)")
    ax2.plot(x, monthly["orders"], color=c_ord, marker="s", linewidth=2.0, linestyle="--", label="Orders (right axis)")
    ax1.set_ylabel(mpl_text(f"Monthly revenue ({CURRENCY_SYMBOL})"))
    ax2.set_ylabel("Orders (count)")
    ax1.yaxis.set_major_formatter(currency_formatter)
    ax2.yaxis.set_major_formatter(thousands_formatter)
    ax1.set_ylim(0, max(monthly["revenue"].max() * 1.18, 1))
    ax2.set_ylim(0, max(monthly["orders"].max() * 1.18, 1))
    step = max(1, int(np.ceil(len(monthly) / 12)))                       # about 12 labelled months, anchored on real data months
    ax1.set_xticks(list(x[::step]))
    ax1.set_xticklabels([t.strftime("%b %Y") for t in x[::step]], rotation=45, ha="right")
    ax1.annotate(mpl_text(f"Peak: {money_compact(pk_rev)} ({pk.strftime('%b %Y')})"), xy=(pk.to_timestamp(), pk_rev), xytext=(0, 14),
                 textcoords="offset points", ha="center", fontsize=9, arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8))
    ax1.set_title("Monthly Revenue and Order Volume", loc="left", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.grid(axis="x", visible=False)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig_to_png(fig)


@chart
def chart_weekday_pattern(weekday):
    """Figure 11.2 - weekday revenue, orders and average order value (darker bar = highest day)."""
    short_days = [d[:3] for d in S11_WEEKDAYS]
    panels = (("revenue", "Total revenue", True), ("orders", "Number of orders", False), ("avg_order_value", "Average order value", True))
    base = mcolors.to_rgba(PALETTE_LIST[0])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (col, title, is_money) in zip(axes, panels):
        vals = weekday[col].to_numpy(dtype=float)
        top = int(np.nanargmax(vals))
        cols = [base if i == top else (base[0], base[1], base[2], 0.5) for i in range(len(vals))]
        bars = ax.bar(short_days, np.nan_to_num(vals), color=cols, edgecolor="white")
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                lab = money_compact(v) if is_money else f"{v:,.0f}"
                ax.text(bar.get_x() + bar.get_width() / 2, v, mpl_text(lab), ha="center", va="bottom", fontsize=8.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, max(np.nanmax(vals) * 1.15, 1))
        ax.yaxis.set_major_formatter(currency_formatter if is_money else thousands_formatter)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Weekday shopping pattern (darker bar = highest day in each panel)", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig_to_png(fig)


@chart
def chart_quarter_season(quarter, season, avg_month):
    """Figure 11.3 - average monthly revenue by calendar quarter and by retail season."""
    base = mcolors.to_rgba(PALETTE_LIST[0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    views = ((axes[0], quarter, S11_QUARTERS, "Calendar quarters"), (axes[1], season, S11_SEASONS, "Retail seasons"))
    for ax, tbl, labels, title in views:
        vals = tbl["avg_monthly_revenue"].to_numpy(dtype=float)
        top = int(np.nanargmax(vals))
        cols = [base if i == top else (base[0], base[1], base[2], 0.5) for i in range(len(vals))]
        ticks = [f"{l}\n({S11_SEASON_MONTHS[l]})" if l in S11_SEASON_MONTHS else l for l in labels]
        bars = ax.bar(ticks, np.nan_to_num(vals), color=cols, edgecolor="white")
        for bar, v, ix in zip(bars, vals, tbl["seasonal_index"]):
            if np.isfinite(v):
                ax.annotate(mpl_text(f"{money_compact(v)}\nindex {ix:.0f}"), xy=(bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        ax.axhline(avg_month, color="0.35", linestyle="--", linewidth=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", visible=False)
    ymax = max(np.nanmax(quarter["avg_monthly_revenue"]), np.nanmax(season["avg_monthly_revenue"]), avg_month)
    axes[0].set_ylim(0, ymax * 1.28)
    axes[0].set_ylabel(mpl_text(f"Average monthly revenue ({CURRENCY_SYMBOL})"))
    axes[0].yaxis.set_major_formatter(currency_formatter)
    axes[0].text(0.01, 0.98, "dashed line = overall average month (index 100)", transform=axes[0].transAxes, fontsize=8.5,
                 color="0.35", va="top")
    fig.suptitle("Sales by quarter and season: average monthly revenue (darker bar = strongest period)", fontsize=13,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig_to_png(fig)


@chart
def chart_season_heatmap(hm, hm_rev):
    """Figure 11.4 - category x season heatmap (colour = seasonal index, text = revenue)."""
    cats = list(hm.index)
    vals = hm.to_numpy(dtype=float)
    norm = mcolors.TwoSlopeNorm(vmin=min(float(np.nanmin(vals)), 99.0), vcenter=100.0, vmax=max(float(np.nanmax(vals)), 101.0))
    cmap = plt.get_cmap("BrBG")
    fig, ax = plt.subplots(figsize=(11, 0.7 * len(cats) + 3.0))
    img = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(cats)):
        for j in range(len(S11_SEASONS)):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", color="0.4")
                continue
            r, g, b, _ = cmap(norm(v))
            txt_col = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"
            ax.text(j, i, mpl_text(f"{money_compact(hm_rev.iloc[i, j])}\nindex {v:.0f}"), ha="center", va="center", fontsize=9, color=txt_col)
    ax.set_xticks(range(len(S11_SEASONS)))
    ax.set_xticklabels([f"{s}\n({S11_SEASON_MONTHS[s]})" for s in S11_SEASONS])
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xticks(np.arange(-0.5, len(S11_SEASONS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.grid(False, which="major")
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Category x Season heatmap: seasonal index (colour) and revenue (text)", loc="left", fontweight="bold")
    cbar = fig.colorbar(img, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Seasonal index (100 = category's average month)")
    fig.tight_layout()
    return fig_to_png(fig)


def render_sec11(ctx):
    """Section 11 (Member 10): monthly, weekday, quarter / season and category x season analysis."""
    tx = ctx["tx"]
    section_header("Section 11: Monthly, Weekday and Seasonal Sales Trends", "Member 10 - Dev Rastogi")
    t = s11_tables(tx)
    monthly, weekday, quarter, season = t["monthly"], t["weekday"], t["quarter"], t["season"]
    rev_all, avg_month = t["rev_all"], t["avg_month"]

    # ---- 11.1 monthly ----
    m = monthly.reset_index()
    view = pd.DataFrame({"Month": m["year_month"].dt.strftime("%b %Y"), "Revenue": m["revenue"].map(lambda v: money(v, 0)),
                         "Orders": m["orders"].map(num), "Active Customers": m["active_customers"].map(num),
                         "Avg Order Value": m["avg_order_value"].map(lambda v: money(v, 2)),
                         "MoM Revenue (%)": m["mom_revenue_pct"].map(lambda v: signed_pct(v, 1)),
                         "MoM Orders (%)": m["mom_orders_pct"].map(lambda v: signed_pct(v, 1))})
    orders_all = tx["transaction_id"].nunique()
    view.loc[len(view)] = ["Total / Overall", money(monthly["revenue"].sum(), 0), num(orders_all), num(tx["customer_id"].nunique()),
                           money(safe_div(monthly["revenue"].sum(), orders_all), 2), "-", "-"]
    show_df(view, caption="Table 11.1: Monthly revenue, orders, active customers and month-over-month (MoM) growth", hide_index=True)
    peak, low = monthly["revenue"].idxmax(), monthly["revenue"].idxmin()
    peak_rev, low_rev = monthly.loc[peak, "revenue"], monthly.loc[low, "revenue"]
    mom = monthly["mom_revenue_pct"].dropna()
    s1 = (f"Revenue peaked in **{peak.strftime('%b %Y')}** ({money(peak_rev)}, {peak_rev / monthly['revenue'].sum() * 100:.1f}% of all sales) and "
          f"was lowest in **{low.strftime('%b %Y')}** ({money(low_rev)})"
          + (f", a {peak_rev / low_rev:.1f}x gap between the strongest and the weakest month." if low_rev > 0 else "."))
    sents = [s1]
    if len(mom) >= 1:
        avg_mom = mom.mean()
        trend = "upward" if avg_mom > 2 else ("downward" if avg_mom < -2 else "broadly flat")
        shape = "volatile" if mom.std() > 20 else "fairly stable"
        sents.append(f"Revenue rose month-over-month in {int((mom > 0).sum())} of {len(mom)} comparisons (average change {avg_mom:+.1f}%), "
                     f"which describes {'an' if trend == 'upward' else 'a'} {trend}, {shape} trajectory.")
    sents.append("These peaks and dips should be treated as hypotheses and checked against promotions, festivals and stock availability "
                 "before they are used for forecasting (Tukey, 1977).")
    interpret(*sents)

    caption_md("Figure 11.1: Monthly revenue and order volume")
    show_png(chart_monthly_trend(monthly[["revenue", "orders"]]))
    pk_orders = monthly["orders"].idxmax()
    r = monthly["revenue"].corr(monthly["orders"]) if len(monthly) >= 3 else np.nan
    sents = []
    if pd.notna(r):
        link = "almost in lockstep" if r >= 0.8 else ("broadly together" if r >= 0.5 else "only loosely together")
        driver = ("mainly by how many orders are placed" if r >= 0.8 else
                  ("by both order volume and basket size" if r >= 0.5 else "more by basket size than by order volume"))
        sents.append(f"Revenue and orders move {link} (correlation r = {r:.2f}), so monthly sales swings are driven {driver}.")
    if peak == pk_orders:
        sents.append(f"Both measures peak in **{peak.strftime('%b %Y')}**, which makes it the month to stock, staff and promote for first.")
    else:
        sents.append(f"Orders peak in **{pk_orders.strftime('%b %Y')}** while revenue peaks in **{peak.strftime('%b %Y')}**, which suggests the "
                     f"revenue peak is lifted by larger baskets rather than more orders; traffic campaigns fit the first month and bundle or "
                     f"upsell offers the second.")
    sents.append("Plotting both series on one time axis shows the trend, the turning points and the co-movement at a glance, in the spirit "
                 "of exploratory graphics (Tukey, 1977).")
    interpret(*sents)

    # ---- 11.2 weekday ----
    wd = weekday.reset_index()
    wview = pd.DataFrame({"Weekday": wd["weekday"], "Revenue": wd["revenue"].map(lambda v: money(v, 0)),
                          "Revenue Share (%)": wd["revenue_share_pct"].map(lambda v: pct(v, 1)), "Orders": wd["orders"].map(num),
                          "Avg Order Value": wd["avg_order_value"].map(lambda v: money(v, 2)),
                          "Calendar Days": wd["calendar_days"].map(num), "Avg Daily Revenue": wd["avg_daily_revenue"].map(lambda v: money(v, 0))})
    for label, days in (("Weekdays (Mon-Fri)", S11_WEEKDAYS[:5]), ("Weekend (Sat-Sun)", S11_WEEKDAYS[5:]), ("All days", S11_WEEKDAYS)):
        part = weekday.loc[days]
        rev, orders, cal = part["revenue"].sum(), part["orders"].sum(), part["calendar_days"].sum()
        wview.loc[len(wview)] = [label, money(rev, 0), pct(rev / weekday["revenue"].sum() * 100, 1), num(orders),
                                 money(safe_div(rev, orders), 2), num(cal), money(safe_div(rev, cal), 0)]
    show_df(wview, caption="Table 11.2: Revenue, orders and average order value by day of week", hide_index=True)
    best, worst = weekday["revenue"].idxmax(), weekday["revenue"].idxmin()
    we_share, we_day_share, lift = s11_weekend_stats(weekday)
    sents = [f"**{best}** is the strongest trading day ({weekday.loc[best, 'revenue_share_pct']:.1f}% of revenue, "
             f"{plural(weekday.loc[best, 'orders'], 'order')}) and **{worst}** the weakest ({weekday.loc[worst, 'revenue_share_pct']:.1f}%)."]
    if pd.notna(lift):
        if lift >= 10:
            action = "so campaigns, staffing and stock should lean towards Saturday and Sunday"
        elif lift <= -10:
            action = "so weekday traffic is the main revenue engine and weekend-specific promotions could close the gap"
        else:
            action = "so demand is spread fairly evenly across the week"
        sents.append(f"Weekends bring {we_share:.1f}% of revenue from {we_day_share:.1f}% of the calendar days, i.e. an average weekend day earns "
                     f"{abs(lift):.0f}% {'more' if lift >= 0 else 'less'} than an average weekday, {action}.")
    sents.append("Comparing all seven days side by side is a simple exploratory check (Tukey, 1977); any day-specific promotion should be "
                 "tested on a small scale before it is rolled out.")
    interpret(*sents)

    caption_md("Figure 11.2: Weekday revenue, orders and average order value")
    show_png(chart_weekday_pattern(weekday[["revenue", "orders", "avg_order_value"]]))
    orders_top = weekday["orders"].idxmax()
    aov_top, aov_low = weekday["avg_order_value"].idxmax(), weekday["avg_order_value"].idxmin()
    if orders_top == aov_top:
        sents = [f"**{orders_top}** leads on both order count and average order value ({money(weekday.loc[aov_top, 'avg_order_value'], 2)}), "
                 f"which makes it the single most valuable trading day.",
                 f"Stock, staffing and marketing spend should therefore be protected for {orders_top}."]
    else:
        sents = [f"Order volume peaks on **{orders_top}**, whereas the largest baskets appear on **{aov_top}** (average order value "
                 f"{money(weekday.loc[aov_top, 'avg_order_value'], 2)} versus {money(weekday.loc[aov_low, 'avg_order_value'], 2)} on {aov_low}).",
                 f"Volume and traffic campaigns therefore fit {orders_top}, while bundles, upselling and premium offers fit {aov_top}."]
    sents.append("Showing the three measures in matching panels separates busy days from valuable days, which is exactly the kind of pattern "
                 "exploratory graphics are meant to reveal (Tukey, 1977).")
    interpret(*sents)

    # ---- 11.3 quarters and seasons ----
    show_df(s11_period_view(quarter, "Quarter"), caption="Table 11.3a: Sales by calendar quarter (Q1 = Jan-Mar)", hide_index=True)
    interpret(*s11_period_insight(quarter, "calendar quarter"))
    show_df(s11_period_view(season, "Season", S11_SEASON_MONTHS), caption="Table 11.3b: Sales by Indian retail season", hide_index=True)
    interpret(*s11_period_insight(season, "retail season", S11_SEASON_MONTHS))

    cat_rev, cat_idx, top_cats = t["cat_rev"], t["cat_idx"], t["top_cats"]
    cview = pd.DataFrame({"Category": top_cats})
    for sea in S11_SEASONS:
        cview[sea] = [money(cat_rev.loc[c, sea], 0) for c in top_cats]
    cview["Total"] = [money(cat_rev.loc[c, "Total"], 0) for c in top_cats]
    cview["Share (%)"] = [pct(cat_rev.loc[c, "Total"] / cat_rev["Total"].sum() * 100, 1) for c in top_cats]
    cview["Peak Season (index)"] = [f"{cat_idx.loc[c].idxmax()} ({cat_idx.loc[c].max():.0f})" if cat_idx.loc[c].notna().any() else "-"
                                    for c in top_cats]
    cview.loc[len(cview)] = (["All categories"] + [money(cat_rev[sea].sum(), 0) for sea in S11_SEASONS]
                             + [money(cat_rev["Total"].sum(), 0), "100.0%", "-"])
    show_df(cview, caption=f"Table 11.3c: Revenue by category and season (top {len(top_cats)} categories; peak season by seasonal index)",
            hide_index=True)
    active = [sea for sea in S11_SEASONS if season.loc[sea, "months_in_data"] > 0]
    leaders = cat_rev[active].idxmax(axis=0)
    if leaders.nunique() == 1:
        first = (f"**{esc(leaders.iloc[0])}** is the largest category in every season ({cat_rev['Total'].iloc[0] / cat_rev['Total'].sum() * 100:.1f}% "
                 f"of total revenue), so its availability matters all year round.")
    else:
        first = ("The leading category changes with the season (" + "; ".join(f"{sea}: {esc(cat)}" for sea, cat in leaders.items())
                 + "), so the product mix shifts through the year.")
    if len(cat_rev) > 3:
        top3 = cat_rev["Total"].head(3).sum() / cat_rev["Total"].sum() * 100
        second = (f"The top three categories generate {top3:.0f}% of revenue, so stock-outs there cost the most, while the remaining "
                  f"categories are natural candidates for cross-sell bundles.")
    else:
        second = (f"Revenue comes from only {len(cat_rev)} categories, so the range is narrow and every category needs a clear seasonal stocking plan.")
    interpret(first, second, "Cross-tabulating categories against seasons is a classic exploratory step (Tukey, 1977) that shows where the "
                             "product mix, and not just the volume, changes.")

    caption_md("Figure 11.3: Sales by quarter and season (average month)")
    show_png(chart_quarter_season(quarter[["avg_monthly_revenue", "seasonal_index"]], season[["avg_monthly_revenue", "seasonal_index"]], avg_month))
    n_q_above, n_q_valid = int((quarter["seasonal_index"] > 100).sum()), int(quarter["seasonal_index"].notna().sum())
    n_s_above, n_s_valid = int((season["seasonal_index"] > 100).sum()), int(season["seasonal_index"].notna().sum())
    if n_s_valid < 2:
        interpret("Only one retail season is covered by the data, so seasonal targets cannot be derived yet.",
                  "A longer date range is needed to study the shape of the year (Tukey, 1977).")
    else:
        sb, sw, sb_i, sw_i = s11_best_worst(season)
        interpret(f"{n_s_above} of {n_s_valid} seasons and {n_q_above} of {n_q_valid} quarters run above the overall average month (index 100), "
                  f"led by **{sb}** at an index of {sb_i:.0f}.",
                  f"Seasonal targets can therefore be set from the index instead of a flat annual average, e.g. roughly {sb_i / 100:.2f}x the "
                  f"average monthly revenue ({money(avg_month)}) in {sb} and {sw_i / 100:.2f}x in {sw}.",
                  "Standardising every period to an average month keeps the comparison fair and shows the shape of the year at a glance (Tukey, 1977).")

    hm = cat_idx.loc[top_cats, S11_SEASONS].astype(float)
    hm_rev = cat_rev.loc[top_cats, S11_SEASONS]
    if hm.notna().to_numpy().any():
        caption_md("Figure 11.4: Category x Season heatmap")
        show_png(chart_season_heatmap(hm, hm_rev))
        spread = hm.max(axis=1) - hm.min(axis=1)
        lead_cat, swing_cat, steady_cat = top_cats[0], spread.idxmax(), spread.idxmin()
        lead_peak = hm.loc[lead_cat].idxmax()
        swing_lo, swing_hi = hm.loc[swing_cat].idxmin(), hm.loc[swing_cat].idxmax()
        if swing_cat == lead_cat:
            first = (f"The largest category, **{esc(lead_cat)}**, peaks in **{lead_peak}** (index {hm.loc[lead_cat].max():.0f}) and is also the most "
                     f"season-sensitive: its index swings from {hm.loc[swing_cat].min():.0f} in {swing_lo} to {hm.loc[swing_cat].max():.0f} in {swing_hi}.")
        else:
            first = (f"The largest category, **{esc(lead_cat)}**, peaks in **{lead_peak}** (index {hm.loc[lead_cat].max():.0f}), whereas "
                     f"**{esc(swing_cat)}** is the most season-sensitive: its index swings from {hm.loc[swing_cat].min():.0f} in {swing_lo} to "
                     f"{hm.loc[swing_cat].max():.0f} in {swing_hi}.")
        if len(top_cats) >= 2 and steady_cat != swing_cat:
            second = (f"Categories with a wide swing need seasonal stock build-ups and timed campaigns, while steadier rows such as "
                      f"**{esc(steady_cat)}** (range {spread[steady_cat]:.0f} index points) can run on regular replenishment.")
        else:
            second = "A category with a wide swing needs seasonal stock build-ups and timed campaigns rather than flat replenishment."
        interpret(first, second, "Reading the heatmap row by row separates seasonal from year-round categories at a glance (Tukey, 1977).")

    # ---- 11.4 scorecard ----
    pk_m, lw_m = monthly["revenue"].idxmax(), monthly["revenue"].idxmin()
    mom_all = monthly["mom_revenue_pct"].dropna()
    best_d, worst_d = weekday["revenue"].idxmax(), weekday["revenue"].idxmin()
    we_share, we_day_share, lift = s11_weekend_stats(weekday)
    rows = [("Strongest month", f"{pk_m.strftime('%b %Y')} - {money(monthly.loc[pk_m, 'revenue'])} ({monthly.loc[pk_m, 'revenue'] / rev_all * 100:.1f}% of revenue)"),
            ("Weakest month", f"{lw_m.strftime('%b %Y')} - {money(monthly.loc[lw_m, 'revenue'])} ({monthly.loc[lw_m, 'revenue'] / rev_all * 100:.1f}% of revenue)"),
            ("Average MoM revenue change", signed_pct(mom_all.mean(), 1) if len(mom_all) else "-"),
            ("Strongest / weakest weekday", f"{best_d} / {worst_d}"),
            ("Weekend vs weekday revenue per day", signed_pct(lift, 1))]
    if quarter["seasonal_index"].notna().sum() >= 1 and season["seasonal_index"].notna().sum() >= 1:
        qb, qw, qb_i, qw_i = s11_best_worst(quarter)
        sb, sw, sb_i, sw_i = s11_best_worst(season)
        rows += [("Strongest / weakest quarter", f"{qb} (index {qb_i:.0f}) / {qw} (index {qw_i:.0f})"),
                 ("Strongest / weakest season", f"{sb} (index {sb_i:.0f}) / {sw} (index {sw_i:.0f})")]
    rows.append(("Largest category and its peak season",
                 f"{top_cats[0]} - peaks in {cat_idx.loc[top_cats[0]].idxmax()} (index {cat_idx.loc[top_cats[0]].max():.0f})"
                 if cat_idx.loc[top_cats[0]].notna().any() else f"{top_cats[0]}"))
    show_df(pd.DataFrame(rows, columns=["Theme", "Finding"]), caption="Table 11.4: Scorecard of sales trends and seasonality", hide_index=True)
    checks = [("Monthly table adds up to total revenue", np.isclose(monthly["revenue"].sum(), rev_all)),
              ("Weekday table adds up to total revenue", np.isclose(weekday["revenue"].sum(), rev_all)),
              ("Quarter table adds up to total revenue", np.isclose(quarter["revenue"].sum(), rev_all)),
              ("Season table adds up to total revenue", np.isclose(season["revenue"].sum(), rev_all)),
              ("Category x Season table adds up to total revenue", np.isclose(cat_rev["Total"].sum(), rev_all))]
    n_pass = sum(bool(ok) for _, ok in checks)
    sb_txt = f"the **{sb}** season" if season["seasonal_index"].notna().sum() >= 1 else "the strongest season"
    interpret(f"Sales concentrate around **{pk_m.strftime('%b %Y')}**, **{best_d}** and {sb_txt}, while {lw_m.strftime('%b %Y')} and {worst_d} "
              f"are the soft spots; {n_pass} of {len(checks)} reconciliation checks confirm that every table adds up to {money(rev_all, 0)}.",
              "Stock, staffing and promotions should be sequenced around the peaks, and targeted offers used to lift the soft periods.",
              "All patterns reported here are descriptive; they point to hypotheses worth testing rather than proven causes (Tukey, 1977).")


# ---------------------------------------------------------------------------------------------------
# Tab 4 container
# ---------------------------------------------------------------------------------------------------
def render_tab4(ctx):
    """Tab 4 - Sections 9, 10 and 11."""
    st.markdown("## Product, Repeat Purchase and Seasonality EDA")
    st.caption("Sections 9-11 | Member 8 - Abhi Rastogi | Member 9 - Sanjay Rathour | Member 10 - Dev Rastogi")
    s9, s10, s11 = st.tabs(["Section 9: Products & categories", "Section 10: Repeat purchase", "Section 11: Trends & seasonality"])
    with s9:
        run_safely(render_sec9, ctx)
    with s10:
        run_safely(render_sec10, ctx)
    with s11:
        run_safely(render_sec11, ctx)


# ===================================================================================================
# 18.  TAB 5  -  Advanced analytics: Pareto, RFM, cohort retention and market basket  (Sections 12.1-12.4)
#      Members 11 (Vidhan Agarwal) and 12 (Anshu)
# ===================================================================================================
# ---------------------------------------------------------------------------------------------------
# 18.1  Section 12.1: Pareto (80/20) analysis                                    (Member 11)
# ---------------------------------------------------------------------------------------------------
def pareto_tables(cust):
    """Cumulative table, exact 80/20 numbers, Gini coefficient and decile table (cust is not modified)."""
    pdf = (cust[["customer_id", "total_spend"]].copy().sort_values("total_spend", ascending=False, kind="mergesort")
           .reset_index(drop=True))
    n = len(pdf)
    cum_rev = pdf["total_spend"].cumsum()
    total = float(cum_rev.iloc[-1])
    if n < 5 or total <= 0:
        return None
    pdf["customer_rank"] = np.arange(1, n + 1)
    pdf["cum_customer_pct"] = pdf["customer_rank"] / n * 100                              # k / N
    pdf["cum_revenue_pct"] = cum_rev / total * 100                                        # C_k in percent
    k20 = int(np.ceil(n * 20 / 100))
    top20 = float(pdf.loc[k20 - 1, "cum_revenue_pct"])
    k80 = int(np.argmax(pdf["cum_revenue_pct"].to_numpy() >= 80 - 1e-9)) + 1              # 1e-9 guards float round-off
    k50 = int(np.ceil(n * 50 / 100))
    bottom50 = 100 - float(pdf.loc[k50 - 1, "cum_revenue_pct"])
    lorenz = np.r_[0.0, np.sort(pdf["total_spend"].to_numpy()).cumsum()]
    lorenz = lorenz / lorenz[-1]
    gini = 1 - (lorenz[:-1] + lorenz[1:]).sum() / n                                       # trapezoid rule
    pdf["decile"] = (pdf["customer_rank"] - 1) * 10 // n + 1
    dec = (pdf.groupby("decile").agg(customers=("customer_id", "size"), avg_spend=("total_spend", "mean"),
                                     revenue=("total_spend", "sum")).reset_index())
    dec["customer_band"] = dec["decile"].map(lambda d: "Top 10%" if d == 1 else f"{(d - 1) * 10}-{d * 10}%")
    dec["revenue_share_pct"] = dec["revenue"] / total * 100
    dec["cum_revenue_pct"] = dec["revenue_share_pct"].cumsum()
    return {"pareto": pdf, "n": n, "total": total, "k20": k20, "top20": top20, "k80": k80, "pct80": k80 / n * 100,
            "bottom50": bottom50, "gini": float(gini), "deciles": dec}


@chart
def chart_pareto(curve, top20, pct80):
    """Figure 12.1 - Pareto (Lorenz-style) curve with the 20% / 80% reference lines."""
    x = np.r_[0.0, curve["cum_customer_pct"].to_numpy()]
    y = np.r_[0.0, curve["cum_revenue_pct"].to_numpy()]
    c = PALETTE_LIST
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.fill_between(x, y, x, color=c[0], alpha=0.10)
    ax.plot(x, y, color=c[0], lw=2.6, label="Cumulative revenue (customers ranked by spend)")
    ax.plot([0, 100], [0, 100], color="grey", ls=":", lw=1.6, label="Line of equality (every customer spends the same)")
    ax.axvline(20, color="#555555", ls="--", lw=1.1, label="20% of customers")
    ax.axhline(80, color="#555555", ls="-.", lw=1.1, label="80% of revenue")
    ax.scatter([20], [top20], s=95, color=c[2], edgecolor="black", zorder=5, label=f"Top 20% of customers = {top20:.1f}% of revenue")
    ax.scatter([pct80], [80], s=95, marker="D", color=c[3], edgecolor="black", zorder=5, label=f"80% of revenue = top {pct80:.1f}% of customers")
    ticks = list(range(0, 101, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}%" for t in ticks])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Cumulative share of customers (highest spenders first)")
    ax.set_ylabel("Cumulative share of revenue")
    ax.set_title("Pareto Curve: Customer Revenue Concentration", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    return fig_to_png(fig)


def render_sec12_1(ctx):
    """Section 12.1 (Member 11): Pareto 80/20 analysis."""
    cust = ctx["cust"]
    section_header("Section 12.1: Pareto Analysis (80/20 Rule)", "Member 11 - Vidhan Agarwal")
    st.markdown("Customers are ranked by `total_spend` from highest to lowest. With $r_{(j)}$ the spend of the customer ranked $j$ and "
                "$N$ customers, the cumulative revenue share after the top $k$ customers is:")
    st.latex(r"C_k = \frac{\sum_{j=1}^{k} r_{(j)}}{\sum_{j=1}^{N} r_{(j)}}, \qquad r_{(1)} \ge r_{(2)} \ge \dots \ge r_{(N)}")
    P = pareto_tables(cust)
    if P is None:
        st.info("At least five customers with positive spend are needed for the Pareto analysis; widen the sidebar selection.")
        return
    dec = P["deciles"]
    headline = pd.DataFrame([("Customers analysed", f"{P['n']:,}"), ("Total revenue", money(P["total"])),
                             ("Top 20% of customers generate", f"{P['top20']:.1f}% of revenue  ({P['k20']:,} customers)"),
                             ("80% of revenue comes from the top", f"{P['pct80']:.1f}% of customers  ({P['k80']:,} customers)"),
                             ("Bottom 50% of customers generate", f"{P['bottom50']:.1f}% of revenue"),
                             ("Gini coefficient of customer spend", f"{P['gini']:.3f}")], columns=["Pareto headline result", "Value"])
    show_df(headline, caption="Table 12.1a: Pareto headline results", hide_index=True)
    verdict = ("at least as concentrated as" if P["top20"] >= 80 else "less concentrated than")
    interpret(f"The top 20% of customers ({P['k20']:,} of {P['n']:,}) generate {P['top20']:.1f}% of the {money(P['total'])} revenue, so the base is "
              f"{verdict} the classic 80/20 benchmark (Koch, 1998); conversely, 80% of revenue is reached with only the top {P['pct80']:.1f}% of customers.",
              f"The bottom half of the customers contributes just {P['bottom50']:.1f}% of revenue and the Gini coefficient of {P['gini']:.2f} "
              f"(0 = equal spend, 1 = one customer pays everything) confirms {'a very high' if P['gini'] >= 0.6 else ('a moderate' if P['gini'] >= 0.4 else 'a low')} "
              f"concentration.")
    show_df(pd.DataFrame({"Decile": dec["decile"], "Customer band": dec["customer_band"], "Customers": dec["customers"].map("{:,}".format),
                          "Avg spend per customer": dec["avg_spend"].map(lambda v: money(v)), "Revenue": dec["revenue"].map(lambda v: money(v)),
                          "Revenue share": dec["revenue_share_pct"].map("{:.1f}%".format),
                          "Cumulative revenue share": dec["cum_revenue_pct"].map("{:.1f}%".format)}).set_index("Decile"),
            caption="Table 12.1b: Decile table - customers ranked by total_spend, highest spenders first")
    d1 = dec.iloc[0]
    d_last = dec.iloc[-1]
    interpret(f"The top decile ({int(d1['customers']):,} customers, average spend {money(d1['avg_spend'])}) alone contributes {d1['revenue_share_pct']:.1f}% of "
              f"revenue, whereas the lowest decile ({int(d_last['customers']):,} customers, average spend {money(d_last['avg_spend'])}) contributes "
              f"{d_last['revenue_share_pct']:.2f}%.",
              f"The first three deciles together already cover {dec['cum_revenue_pct'].iloc[min(2, len(dec) - 1)]:.1f}% of revenue, so retention and VIP service "
              f"effort should be concentrated on the top bands.")
    caption_md("Figure 12.1: Pareto curve of customer revenue concentration")
    show_png(chart_pareto(P["pareto"][["cum_customer_pct", "cum_revenue_pct"]], P["top20"], P["pct80"]))
    interpret(f"The curve bows far above the diagonal: the shaded gap between the curve and the line of equality is the visual measure of "
              f"concentration, and the markers show that the data cross the 20% line at {P['top20']:.1f}% of revenue and the 80% line at "
              f"{P['pct80']:.1f}% of customers.",
              "A strongly concentrated base means that losing a few top customers is costly, so retention, VIP service and personalised offers "
              "for the top bands deserve priority; a flatter curve would instead call for broad-based activation (Koch, 1998).")


# ---------------------------------------------------------------------------------------------------
# 18.2  Section 12.2: RFM segmentation                                            (Member 11)
# ---------------------------------------------------------------------------------------------------
def rfm_score_ranges(rfm):
    """Raw value range behind each score (5 = best)."""
    out = pd.DataFrame({
        "R: recency_days": rfm.groupby("R_score")["recency_days"].agg(lambda s: f"{s.min():,.0f} to {s.max():,.0f}"),
        "F: num_orders": rfm.groupby("F_score")["num_orders"].agg(lambda s: f"{s.min():,.0f} to {s.max():,.0f}"),
        "M: total_spend": rfm.groupby("M_score")["total_spend"].agg(lambda s: f"{CURRENCY_SYMBOL}{s.min():,.0f} to {CURRENCY_SYMBOL}{s.max():,.0f}"),
    }).sort_index(ascending=False)
    out.index.name = "Score"
    return out


def rfm_profile(rfm):
    """Table 12.2c - segment profile (fixed business order; empty segments kept)."""
    n, rev = len(rfm), float(rfm["total_spend"].sum())
    summ = (rfm.groupby("rfm_segment").agg(customers=("customer_id", "size"), revenue=("total_spend", "sum"),
                                           avg_recency_days=("recency_days", "mean"), avg_frequency=("num_orders", "mean"),
                                           avg_monetary=("total_spend", "mean"), avg_R_score=("R_score", "mean"),
                                           avg_F_score=("F_score", "mean"), avg_M_score=("M_score", "mean")).reindex(RFM_ORDER))
    summ[["customers", "revenue"]] = summ[["customers", "revenue"]].fillna(0)
    summ["customer_share_pct"] = summ["customers"] / n * 100
    summ["revenue_share_pct"] = summ["revenue"] / rev * 100 if rev else 0.0
    return summ[["customers", "customer_share_pct", "revenue", "revenue_share_pct", "avg_recency_days", "avg_frequency", "avg_monetary",
                 "avg_R_score", "avg_F_score", "avg_M_score"]]


@chart
def chart_rfm_segments(summ):
    """Figure 12.2 - customer share vs revenue share (left) and average RFM scores (right) per segment."""
    c = PALETTE_LIST
    ypos = np.arange(len(RFM_ORDER))
    bar_h = 0.38
    ylabels = [f"{seg}\n(n = {int(n):,})" for seg, n in zip(RFM_ORDER, summ["customers"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.45, 1]})
    bars_c = ax1.barh(ypos - bar_h / 2, summ["customer_share_pct"], bar_h, color=c[1], label="Share of customers")
    bars_r = ax1.barh(ypos + bar_h / 2, summ["revenue_share_pct"], bar_h, color=c[0], label="Share of revenue")
    ax1.bar_label(bars_c, fmt="%.1f%%", padding=3, fontsize=8.5)
    ax1.bar_label(bars_r, fmt="%.1f%%", padding=3, fontsize=8.5)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(ylabels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(summ["customer_share_pct"].max(), summ["revenue_share_pct"].max()) * 1.18)
    ax1.set_xlabel("Share of total (%)")
    ax1.set_title("Share of customers vs share of revenue by RFM segment", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    ax1.legend(loc="lower right")
    mat = summ[["avg_R_score", "avg_F_score", "avg_M_score"]].to_numpy(dtype=float)
    im = ax2.imshow(np.ma.masked_invalid(mat), cmap="YlGnBu", vmin=1, vmax=5, aspect="auto")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["R\n(recency)", "F\n(frequency)", "M\n(monetary)"])
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(RFM_ORDER)
    for r in range(mat.shape[0]):
        for k in range(3):
            v = mat[r, k]
            ax2.text(k, r, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=10, fontweight="bold",
                     color="white" if (not np.isnan(v) and v >= 3.6) else "black")
    ax2.set_title("Average RFM score by segment (5 = best)", fontweight="bold")
    ax2.tick_params(length=0)
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Average score")
    fig.tight_layout()
    return fig_to_png(fig)


def render_sec12_2(ctx):
    """Section 12.2 (Member 11): RFM scoring and named customer groups."""
    rfm = ctx["rfm"]
    section_header("Section 12.2: RFM Segmentation", "Member 11 - Vidhan Agarwal")
    st.markdown("**Recency** = `recency_days` (fewer days = better), **Frequency** = `num_orders`, **Monetary** = `total_spend`. Each is ranked and "
                "cut into quintiles (score 5 = best); the three scores are combined into six business segments with rules that are "
                "evaluated top to bottom (Hughes, 1994; Fader et al., 2005).")
    if rfm is None:
        st.info("At least five customers are needed for quintile scoring; widen the sidebar selection.")
        return
    caption_md("Table 12.2a: Segment rules (the first matching rule wins)")
    md_table(pd.DataFrame(RFM_RULES_TABLE, columns=["Segment", "Rule on R, F, M scores", "Meaning"]))
    show_df(rfm_score_ranges(rfm), caption="Table 12.2b: Raw value range behind each score (5 = best)")
    r5 = rfm.loc[rfm["R_score"] == 5, "recency_days"]
    f5 = rfm.loc[rfm["F_score"] == 5, "num_orders"]
    m5 = rfm.loc[rfm["M_score"] == 5, "total_spend"]
    interpret(f"A top recency score (R = 5) means the last purchase was at most {r5.max():,.0f} days before the snapshot date; a top frequency score "
              f"(F = 5) means {f5.min():,.0f} or more orders; a top monetary score (M = 5) means total spend of at least {money(m5.min())}.",
              "Because scores are rank-based quintiles, each score holds one fifth of the customers, so the segments below are comparable "
              "whatever the shape of the underlying distributions.")
    sample = rfm[["customer_id", "recency_days", "num_orders", "total_spend", "R_score", "F_score", "M_score", "rfm_code", "rfm_segment"]].head(8)
    show_df(sample.assign(total_spend=sample["total_spend"].map(lambda v: money(v, 2))), caption="Sample of scored customers", hide_index=True)

    summ = rfm_profile(rfm)
    n_rfm, rev_rfm = len(rfm), float(rfm["total_spend"].sum())
    allrow = pd.DataFrame({"customers": [n_rfm], "customer_share_pct": [100.0], "revenue": [rev_rfm], "revenue_share_pct": [100.0],
                           "avg_recency_days": [rfm["recency_days"].mean()], "avg_frequency": [rfm["num_orders"].mean()],
                           "avg_monetary": [rfm["total_spend"].mean()], "avg_R_score": [rfm["R_score"].mean()],
                           "avg_F_score": [rfm["F_score"].mean()], "avg_M_score": [rfm["M_score"].mean()]}, index=["All customers"])
    tab = pd.concat([summ, allrow])
    shown = pd.DataFrame({"Customers": tab["customers"].map("{:,.0f}".format), "Customer share": tab["customer_share_pct"].map("{:.1f}%".format),
                          "Revenue": tab["revenue"].map(lambda v: money(v)), "Revenue share": tab["revenue_share_pct"].map("{:.1f}%".format),
                          "Avg R (days since last order)": tab["avg_recency_days"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}"),
                          "Avg F (orders)": tab["avg_frequency"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}"),
                          "Avg M (total spend)": tab["avg_monetary"].map(lambda v: "-" if pd.isna(v) else money(v)),
                          "Avg R score": tab["avg_R_score"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}"),
                          "Avg F score": tab["avg_F_score"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}"),
                          "Avg M score": tab["avg_M_score"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")})
    shown.index.name = "RFM segment"
    show_df(shown, caption="Table 12.2c: RFM segment profile")
    champ, loyal, risk, lost = summ.loc["Champions"], summ.loc["Loyal Customers"], summ.loc["At Risk"], summ.loc["Hibernating / Lost"]
    top_seg = summ["revenue_share_pct"].idxmax()
    interpret(f"**Champions** are {champ['customer_share_pct']:.1f}% of customers but generate {champ['revenue_share_pct']:.1f}% of revenue; together with "
              f"Loyal Customers they are {champ['customer_share_pct'] + loyal['customer_share_pct']:.1f}% of customers and "
              f"{champ['revenue_share_pct'] + loyal['revenue_share_pct']:.1f}% of revenue.",
              f"**At Risk** customers ({int(risk['customers']):,}) still hold {risk['revenue_share_pct']:.1f}% of revenue and are the priority group for win-back "
              f"campaigns, while **Hibernating / Lost** customers are {lost['customer_share_pct']:.1f}% of the base but only {lost['revenue_share_pct']:.1f}% of revenue.",
              f"The largest revenue share belongs to **{top_seg}** ({summ.loc[top_seg, 'revenue_share_pct']:.1f}%).")
    caption_md("Figure 12.2: RFM segments - customer share vs revenue share and average scores")
    show_png(chart_rfm_segments(summ))
    interpret("The left panel shows how far revenue share departs from customer share in each segment; the right panel confirms that the segments are "
              "ordered as intended (Champions have the highest average R, F and M scores, Hibernating / Lost the lowest).",
              f"{int(summ['customers'].sum()):,} customers are classified into {int((summ['customers'] > 0).sum())} non-empty segments with no customer left unclassified.")
    caption_md("Table 12.2d: Suggested action per segment")
    md_table(pd.DataFrame(RFM_ACTIONS, columns=["Segment", "Profile", "Suggested action"]))
    lead_plan = summ.loc[["Champions", "At Risk", "Hibernating / Lost"]]
    interpret(f"Effort should follow the revenue at stake: Champions ({int(lead_plan.loc['Champions', 'customers']):,} customers, {lead_plan.loc['Champions', 'revenue_share_pct']:.1f}% of revenue) "
              f"get reward and retention, At Risk ({int(lead_plan.loc['At Risk', 'customers']):,} customers, {lead_plan.loc['At Risk', 'revenue_share_pct']:.1f}%) get win-back campaigns, "
              f"and Hibernating / Lost ({int(lead_plan.loc['Hibernating / Lost', 'customers']):,} customers, {lead_plan.loc['Hibernating / Lost', 'revenue_share_pct']:.1f}%) only low-cost reactivation.",
              "Re-running the scoring after every data refresh shows which customers moved between segments, which is the signal to adjust the campaign.")


# ---------------------------------------------------------------------------------------------------
# 18.3  Section 12.3: Cohort retention                                            (Member 12)
# ---------------------------------------------------------------------------------------------------
def cohort_tables(tx):
    """Cohort count matrix and retention-rate matrix (cohort = month of the customer's first purchase)."""
    tmp = tx[["customer_id", "transaction_date"]].dropna().copy()
    tmp["customer_id"] = tmp["customer_id"].astype(str)
    first = tmp.groupby("customer_id")["transaction_date"].transform("min")
    tmp["order_month"] = tmp["transaction_date"].dt.to_period("M")
    tmp["cohort_month"] = first.dt.to_period("M")
    tmp["cohort_index"] = ((tmp["order_month"].dt.year - tmp["cohort_month"].dt.year) * 12
                           + (tmp["order_month"].dt.month - tmp["cohort_month"].dt.month))
    n_periods = int(tmp["cohort_index"].max()) + 1
    counts = (tmp.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique().unstack("cohort_index")
              .reindex(columns=range(n_periods)))
    last_month = tmp["order_month"].max()
    months_left = np.array([(last_month.year - p.year) * 12 + (last_month.month - p.month) for p in counts.index])
    observed = np.arange(n_periods)[None, :] <= months_left[:, None]
    counts = counts.fillna(0).where(observed)                              # periods in the future stay blank, not 0%
    counts.index = counts.index.astype(str)
    counts.index.name, counts.columns.name = "cohort_month", "cohort_index"
    size = counts[0].astype(int)
    retention = counts.div(size, axis=0) * 100
    pooled = pd.Series({k: counts[k].sum() / size[counts[k].notna()].sum() * 100 for k in counts.columns})
    return {"counts": counts, "retention": retention, "size": size, "pooled": pooled, "n_customers": int(tmp["customer_id"].nunique()),
            "n_periods": n_periods}


@chart
def chart_cohort_heatmap(retention, size):
    """Figure 12.3 - cohort retention heatmap (colour scale capped at the Month-1+ maximum)."""
    cmap = sns.light_palette(PALETTE_LIST[0], as_cmap=True)
    n_rows, n_cols = retention.shape
    wide = n_cols > 12
    later = retention.drop(columns=0)
    vmax = 100.0 if later.empty or later.isna().all().all() else float(min(100, max(10, np.ceil(later.max().max() / 10) * 10)))
    fig, ax = plt.subplots(figsize=(min(18, max(9, 0.62 * n_cols + 3)), max(4.5, 0.42 * n_rows + 2.2)))
    sns.heatmap(retention, mask=retention.isna(), cmap=cmap, vmin=0, vmax=vmax, annot=True, fmt=".0f" if wide else ".1f",
                annot_kws={"size": 7.5 if wide else 9}, linewidths=0.6, linecolor="white", cbar_kws={"label": "Retention rate (%)"},
                xticklabels=list(retention.columns), yticklabels=[f"{c}  (n={n:,})" for c, n in zip(retention.index, size)], ax=ax)
    ax.set_xlabel("Months since first purchase (cohort_index)")
    ax.set_ylabel("Cohort (first-purchase month)")
    note = f"\n(Month 0 = 100% by definition; colour scale capped at {vmax:.0f}% so that Month 1+ stays readable)" if vmax < 100 else ""
    ax.set_title("Cohort Retention Rate (%): share of each cohort still buying in month k" + note, fontsize=12, pad=12)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig_to_png(fig)


def render_sec12_3(ctx):
    """Section 12.3 (Member 12): monthly cohort retention."""
    tx = ctx["tx"]
    section_header("Section 12.3: Cohort Retention Analysis", "Member 12 - Anshu")
    st.markdown("Each customer belongs to the **cohort** of the month of their first purchase; `cohort_index` counts the months since that "
                "purchase. Retention(c, k) is the share of cohort c that buys again in month k (Fader & Hardie, 2007). Periods that have not "
                "happened yet are left blank, not shown as 0%.")
    st.latex(r"\mathrm{Retention}_{c,k} = \frac{\text{customers of cohort } c \text{ active in month } k}{n_c} \times 100")
    C = cohort_tables(tx)
    counts, retention, size, pooled = C["counts"], C["retention"], C["size"], C["pooled"]
    st.markdown(f"**{C['n_customers']:,} customers in {len(size)} monthly cohorts** ({counts.index[0]} to {counts.index[-1]}); cohort_index runs from 0 to "
                f"{C['n_periods'] - 1}. Cohort sizes add up to {int(size.sum()):,} customers "
                f"({'matches' if int(size.sum()) == ctx['cust']['customer_id'].nunique() else 'differs from'} the {ctx['cust']['customer_id'].nunique():,} customers of the active customer table).")
    cnt_show = counts.copy().astype("Float64").map(lambda v: "-" if pd.isna(v) else f"{int(v):,}")
    show_df(cnt_show, caption="Table 12.3a: Cohort count matrix - distinct active customers per cohort and month (- = period not yet observed)")
    ret_show = retention.map(lambda v: "-" if pd.isna(v) else f"{v:.1f}")
    show_df(ret_show, caption="Table 12.3b: Cohort retention rate (%) matrix")
    show_k = [k for k in (1, 2, 3, 6, 9, 12) if k in pooled.index]
    m1_txt = ""
    if 1 in retention.columns and retention[1].notna().any():
        m1 = retention[1].dropna()
        m1_txt = f" Month-1 retention is best for the {m1.idxmax()} cohort ({m1.max():.1f}%) and weakest for the {m1.idxmin()} cohort ({m1.min():.1f}%)."
    interpret("Pooled retention (all cohorts that have reached the month): " + ", ".join(f"Month {k}: {pooled[k]:.1f}%" for k in show_k) + "." + m1_txt,
              f"The largest cohort is {size.idxmax()} (n = {size.max():,}) and the smallest {size.idxmin()} (n = {size.min():,}); small cohorts produce noisier "
              f"percentages, so single cells should be read with care.")
    caption_md("Figure 12.3: Cohort retention heatmap")
    show_png(chart_cohort_heatmap(retention, size))
    first_drop = (100 - pooled[1]) if 1 in pooled.index else np.nan
    interpret("Cohort retention curves typically fall fastest in the first months and then flatten: "
              + (f"here {first_drop:.1f}% of an average cohort is not active again in month 1" if pd.notna(first_drop) else "month 1 is not yet observed")
              + (f", and {pooled[3]:.1f}% are still buying in month 3." if 3 in pooled.index else "."),
              "A retention matrix is more informative than a single repeat rate because it separates how fast customers drop away from how many "
              "stay for good (Fader & Hardie, 2007); the weakest early months are the natural target for onboarding and second-order campaigns.")


# ---------------------------------------------------------------------------------------------------
# 18.4  Section 12.4: Market basket analysis                                      (Member 12)
# ---------------------------------------------------------------------------------------------------
MBA_MIN_SUPPORT, MBA_MIN_CONFIDENCE, MBA_MIN_COUNT, MBA_TOP_N = 0.005, 0.05, 3, 15
MBA_RULE_COLS = ["antecedent", "consequent", "pair_count", "support", "confidence", "lift"]


@st.cache_data(show_spinner=False)
def basket_counts(pairs):
    """One basket = the SET of distinct products of one transaction_id.  Returns the counts used by all rules."""
    uniq = pairs.dropna().astype(str).drop_duplicates()
    sets = {}
    for t, p in zip(uniq["transaction_id"], uniq["product_name"]):
        sets.setdefault(t, set()).add(p)
    baskets = [tuple(sorted(s)) for s in sets.values()]
    item_counts = Counter(item for b in baskets for item in b)
    pair_counts = Counter(pair for b in baskets for pair in combinations(b, 2))
    sizes = [len(b) for b in baskets]
    return {"n_baskets": len(baskets), "sizes": sizes, "item_counts": dict(item_counts), "pair_counts": dict(pair_counts),
            "n_products": int(uniq["product_name"].nunique())}


def mba_rules_exact(bc, min_support):
    """Rules x => y for BOTH directions of every pair whose support is at least `min_support` (exact counting)."""
    n_b, items, out = bc["n_baskets"], bc["item_counts"], []
    for (a, b), n_ab in bc["pair_counts"].items():
        if n_ab / n_b < min_support:
            continue
        for x, y in ((a, b), (b, a)):
            conf = n_ab / items[x]                                   # confidence(x => y) = support(x, y) / support(x)
            lift = conf / (items[y] / n_b)                           # lift(x => y)       = confidence(x => y) / support(y)
            out.append((x, y, n_ab, n_ab / n_b, conf, lift))
    return pd.DataFrame(out, columns=MBA_RULE_COLS)


def mba_find_rules(bc, min_support, min_conf, min_count):
    """Keep rules with lift > 1; if fewer than 10 survive, halve the minimum support (up to 4 times)."""
    used, rules = min_support, pd.DataFrame(columns=MBA_RULE_COLS)
    for attempt in range(5):
        used = min_support / 2 ** attempt
        cand = mba_rules_exact(bc, used)
        rules = cand[(cand["pair_count"] >= min_count) & (cand["confidence"] >= min_conf) & (cand["lift"] > 1)]
        if len(rules) >= 10:
            break
    return rules, used


def mba_rank(rules, top_n):
    """Rank by lift, then confidence (rounding stops float noise from reordering the two directions of a pair)."""
    top = (rules.assign(_l=rules["lift"].round(9), _c=rules["confidence"].round(9))
           .sort_values(["_l", "_c", "antecedent", "consequent"], ascending=[False, False, True, True])
           .drop(columns=["_l", "_c"]).head(top_n).reset_index(drop=True))
    top.index = top.index + 1
    top.index.name = "rank"
    return top


@chart
def chart_mba_rules(top, others):
    """Figure 12.4 - top rules by lift (a) and support-vs-confidence scatter (b)."""
    accent = PALETTE_LIST[0]
    sh = lambda s: s if len(s) <= 22 else s[:21] + "\u2026"                              # noqa: E731
    labels = [f"#{r.Index}  {sh(r.antecedent)} \u21d2 {sh(r.consequent)}" for r in top.itertuples()]
    ypos = np.arange(len(top))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 0.42 * len(top) + 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax1.barh(ypos, top["lift"], color=accent, edgecolor="white")
    ax1.axvline(1, color="grey", linestyle="--", linewidth=1)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    for y, r in zip(ypos, top.itertuples()):
        ax1.text(r.lift, y, f"  conf {r.confidence:.0%} | sup {r.support:.1%}", va="center", fontsize=8)
    ax1.set_xlim(0, top["lift"].max() * 1.4)
    ax1.set_xlabel("Lift (x the co-purchase frequency expected by chance)")
    ax1.set_title("(a) Top rules ranked by lift", fontsize=11)
    ax2.scatter(others["support"] * 100, others["confidence"] * 100, s=18, color="lightgrey", label="other rules")
    sc = ax2.scatter(top["support"] * 100, top["confidence"] * 100, c=top["lift"], cmap="viridis", s=90, edgecolor="black",
                     linewidth=0.5, label="top rules")
    for r in top.itertuples():
        ax2.annotate(str(r.Index), (r.support * 100, r.confidence * 100), textcoords="offset points", xytext=(5, 4), fontsize=8)
    fig.colorbar(sc, ax=ax2, label="Lift")
    ax2.set_xlabel("Support (% of baskets containing both products)")
    ax2.set_ylabel("Confidence (%) of A \u21d2 B")
    ax2.set_title("(b) Support vs confidence (numbers = rank in the table)", fontsize=11)
    ax2.legend(loc="best", fontsize=8)
    fig.suptitle("Market Basket Analysis: top association rules", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig_to_png(fig)


def render_sec12_4(ctx):
    """Section 12.4 (Member 12): frequent product pairs with support, confidence and lift."""
    tx = ctx["tx"]
    section_header("Section 12.4: Market Basket Analysis", "Member 12 - Anshu")
    st.markdown("Each `transaction_id` is one **basket**: the set of distinct products bought together. For every product pair the app counts the "
                "baskets that contain both products and derives support, confidence and lift (Agrawal et al., 1993).")
    st.latex(r"\mathrm{Support}(A \cup B)=\frac{\text{baskets with both }A\text{ and }B}{N},\quad "
             r"\mathrm{Conf}(A \Rightarrow B)=\frac{\mathrm{Support}(A \cup B)}{\mathrm{Support}(A)},\quad "
             r"\mathrm{Lift}(A \Rightarrow B)=\frac{\mathrm{Conf}(A \Rightarrow B)}{\mathrm{Support}(B)}")
    bc = basket_counts(tx[["transaction_id", "product_name"]])
    n_b = bc["n_baskets"]
    n_multi = sum(s >= 2 for s in bc["sizes"])
    stats = pd.DataFrame([("Baskets (transactions)", f"{n_b:,}"), ("Distinct products", f"{bc['n_products']:,}"),
                          ("Average basket size", f"{np.mean(bc['sizes']):.2f} products"),
                          ("Baskets with 2+ different products", f"{n_multi:,} ({n_multi / n_b:.1%})"),
                          ("Distinct product pairs seen together", f"{len(bc['pair_counts']):,}")], columns=["Basket statistic", "Value"])
    show_df(stats, caption="Table 12.4a: Basket construction", hide_index=True)
    interpret(f"The {n_b:,} transactions hold {bc['n_products']} different products with an average of {np.mean(bc['sizes']):.2f} products per basket, and "
              f"{n_multi / n_b:.1%} of the baskets contain at least two different products.",
              f"Only these multi-product baskets can create product pairs; {len(bc['pair_counts']):,} distinct pairs were observed.")
    if n_multi == 0:
        st.info("No basket contains two different products, so no product pairs can be analysed for the current selection.")
        return
    key = f"{len(tx)}_{n_b}"
    with st.expander("Rule filters (defaults = values of the Section 12.4 notebook)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        min_sup = c1.slider("Minimum support (%)", 0.1, 5.0, MBA_MIN_SUPPORT * 100, 0.1, key=f"mba_sup_{key}") / 100
        min_conf = c2.slider("Minimum confidence (%)", 0, 60, int(MBA_MIN_CONFIDENCE * 100), 1, key=f"mba_conf_{key}") / 100
        min_cnt = c3.slider("Pair seen in at least N baskets", 1, 20, MBA_MIN_COUNT, 1, key=f"mba_cnt_{key}")
        top_n = c4.slider("Rules shown", 5, 25, MBA_TOP_N, 1, key=f"mba_top_{key}")
    rules_all, used = mba_find_rules(bc, min_sup, min_conf, min_cnt)
    top = mba_rank(rules_all, top_n)
    st.markdown(f"**Engine:** exact counting (itertools.combinations + Counter), identical to mlxtend's apriori for pairs. **Filters:** support >= "
                f"{used:.2%}, confidence >= {min_conf:.0%}, lift > 1, pair seen in >= {min_cnt} baskets. **Directed rules found:** {len(rules_all):,}; "
                f"the table shows the top {len(top)} (highest lift first, ties broken by confidence).")
    if top.empty:
        st.info("No rule passed the filters; lower the minimum support or confidence in the filter panel above.")
        return
    shown = pd.DataFrame({"antecedent": top["antecedent"], "consequent": top["consequent"], "pair_count": top["pair_count"].map("{:,}".format),
                          "support": top["support"].map("{:.2%}".format), "confidence": top["confidence"].map("{:.1%}".format),
                          "lift": top["lift"].map("{:.2f}".format)})
    show_df(shown, caption="Table 12.4b: Top association rules (support, confidence, lift)")
    best = top.iloc[0]
    reliable = top.sort_values("confidence", ascending=False).iloc[0]
    interpret(f"The strongest rule is **{esc(best['antecedent'])} \u21d2 {esc(best['consequent'])}**: lift {best['lift']:.2f}, confidence {best['confidence']:.1%} "
              f"and support {best['support']:.2%} ({int(best['pair_count']):,} baskets), i.e. the consequent is bought {best['lift']:.1f} times more often "
              f"than chance with the antecedent in the basket (Agrawal et al., 1993).",
              f"The most reliable single rule is **{esc(reliable['antecedent'])} \u21d2 {esc(reliable['consequent'])}** with a confidence of {reliable['confidence']:.1%}; "
              f"rules with a high lift but a small pair_count rest on few baskets and are better treated as hypotheses for a pilot promotion.")
    caption_md("Figure 12.4: Top association rules")
    show_png(chart_mba_rules(top, rules_all[["support", "confidence"]]))
    interpret(f"Rules in the upper right of panel (b) with a bright colour are both frequent and reliable, so they are the safest candidates for "
              f"store-wide cross-selling such as bundles, combo pricing or checkout recommendations; the best rule reaches {best['confidence']:.0%} confidence.",
              "Because a rule is directional, the direction with the higher confidence shows which product should trigger the recommendation of the other.")


def render_tab5(ctx):
    """Tab 5 - Sections 12.1 to 12.4."""
    st.markdown("## Advanced Analytics: Pareto, RFM, Cohort Retention and Market Basket")
    st.caption("Sections 12.1-12.4 | Member 11 - Vidhan Agarwal | Member 12 - Anshu")
    t1, t2, t3, t4 = st.tabs(["12.1 Pareto (80/20)", "12.2 RFM segmentation", "12.3 Cohort retention", "12.4 Market basket"])
    with t1:
        run_safely(render_sec12_1, ctx)
    with t2:
        run_safely(render_sec12_2, ctx)
    with t3:
        run_safely(render_sec12_3, ctx)
    with t4:
        run_safely(render_sec12_4, ctx)


# ===================================================================================================
# 19.  TAB 6  -  Machine learning: K-Means customer clustering  (Section 12.5)   Member 13 - Rudra Pratap Singh
# ===================================================================================================
CLUSTER_EXTRA_COLORS = ["#8E6BBF", "#E9C46A", "#264653", "#B5838D"]


def cluster_colors(n):
    """First n colours of the project PALETTE; extra distinct colours are appended when k > 6."""
    base = PALETTE_LIST + CLUSTER_EXTRA_COLORS
    return [base[i % len(base)] for i in range(n)]


def cluster_profile_table(cl, names):
    """Table 12.5d - one row per cluster with size, revenue share and average behaviour."""
    total_rev = float(cl["total_spend"].sum())
    prof = (cl.groupby("cluster").agg(customers=("customer_id", "count"), revenue=("total_spend", "sum"),
                                      avg_recency_days=("recency_days", "mean"), avg_frequency=("num_orders", "mean"),
                                      avg_spend=("total_spend", "mean"), avg_aov=("avg_order_value", "mean")).reset_index())
    prof.insert(1, "cluster_name", prof["cluster"].map(names))
    prof["customer_share_pct"] = 100 * prof["customers"] / prof["customers"].sum()
    prof["revenue_share_pct"] = 100 * prof["revenue"] / total_rev if total_rev else 0.0
    return prof[["cluster", "cluster_name", "customers", "customer_share_pct", "revenue", "revenue_share_pct",
                 "avg_recency_days", "avg_frequency", "avg_spend", "avg_aov"]]


@chart
def chart_kmeans_selection(metrics, optimal_k, elbow_k, elbow_y):
    """Figure 12.5a - elbow curve (inertia) and silhouette score for every k."""
    pal = cluster_colors(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.plot(metrics["k"], metrics["inertia_wcss"], marker="o", lw=2, color=pal[0])
    ax1.axvline(optimal_k, ls="--", lw=1.3, color="grey", label=f"Selected k = {optimal_k}")
    ax1.scatter([elbow_k], [elbow_y], s=170, facecolors="none", edgecolors="crimson", linewidths=2, zorder=3,
                label=f"Elbow point k = {elbow_k}")
    ax1.set(title="Elbow Curve: Inertia (WCSS)", xlabel="Number of clusters (k)", ylabel="Inertia (WCSS)")
    ax1.set_xticks(metrics["k"])
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.plot(metrics["k"], metrics["silhouette"], marker="s", lw=2, color=pal[1])
    for kx, sy in zip(metrics["k"], metrics["silhouette"]):
        if pd.notna(sy):
            ax2.annotate(f"{sy:.3f}", (kx, sy), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax2.axvline(optimal_k, ls="--", lw=1.3, color="grey", label=f"Selected k = {optimal_k}")
    ax2.set(title="Silhouette Score by k", xlabel="Number of clusters (k)", ylabel="Mean silhouette score")
    ax2.set_xticks(metrics["k"])
    ax2.margins(y=0.15)
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.suptitle(f"K-Means Model Selection: Elbow and Silhouette (k = {int(metrics['k'].min())} to {int(metrics['k'].max())})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_png(fig)


@chart
def chart_kmeans_pca(pcs, labels, cent_pc, ev, name_list, k):
    """Figure 12.5b - customers and centroids of the final model in PCA space."""
    cols = cluster_colors(k)
    idx = np.arange(len(pcs))
    if len(idx) > 12_000:                                       # reproducible sample for drawing only
        idx = np.random.default_rng(RANDOM_STATE).choice(idx, 12_000, replace=False)
    fig, ax = plt.subplots(figsize=(10, 6.6))
    for cid in range(k):
        sel = idx[labels[idx] == cid]
        ax.scatter(pcs[sel, 0], pcs[sel, 1], s=16, alpha=0.55, color=cols[cid],
                   label=f"C{cid}: {name_list[cid]} (n = {int((labels == cid).sum()):,})")
    ax.scatter(cent_pc[:, 0], cent_pc[:, 1], marker="X", s=260, c=cols, edgecolors="black", linewidths=1.5, zorder=5)
    for cid in range(k):
        ax.annotate(f"C{cid}", (cent_pc[cid, 0], cent_pc[cid, 1]), xytext=(9, 9), textcoords="offset points", fontsize=9,
                    fontweight="bold", zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.9))
    ax.set(xlabel=f"PC1 ({ev[0] * 100:.1f}% of variance)", ylabel=f"PC2 ({ev[1] * 100:.1f}% of variance)",
           title=f"K-Means Customer Clusters (k = {k}) in PCA Space")
    ax.grid(alpha=0.25)
    ax.legend(title="Cluster", fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    return fig_to_png(fig)


@chart
def chart_cluster_value(prof):
    """Figure 12.5c - customer share versus revenue share of every cluster."""
    cols = cluster_colors(len(prof))
    x = np.arange(len(prof))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    b1 = ax.bar(x - w / 2, prof["customer_share_pct"], w, color="#B8C4D1", edgecolor="white", label="Share of customers")
    b2 = ax.bar(x + w / 2, prof["revenue_share_pct"], w, color=cols, edgecolor="white", label="Share of revenue")
    ax.bar_label(b1, fmt="%.1f%%", padding=2, fontsize=8.5)
    ax.bar_label(b2, fmt="%.1f%%", padding=2, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{int(c)}\n{short(n, 22)}" for c, n in zip(prof["cluster"], prof["cluster_name"])], fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_axis))
    ax.set_ylim(0, max(prof["customer_share_pct"].max(), prof["revenue_share_pct"].max()) * 1.18)
    ax.set_ylabel("Share of total (%)")
    ax.set_title("Share of customers vs share of revenue by K-Means cluster", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig_to_png(fig)


def render_tab6(ctx):
    """Tab 6 - Section 12.5 K-Means clustering with an interactive choice of k."""
    cust, sel = ctx["cust"], ctx["km_sel"]
    st.markdown("## Machine Learning: K-Means Customer Clustering")
    st.caption("Section 12.5 | Member 13 - Rudra Pratap Singh")
    if sel is None:
        st.info(f"K-Means needs at least {MIN_CUSTOMERS_KMEANS} customers; the current selection has {len(cust):,}. Widen the sidebar filters.")
        return
    run_safely(render_sec12_5, ctx, cust, sel)


def render_sec12_5(ctx, cust, sel):
    """Section 12.5: feature preparation, choice of k, final model, profiles and PCA map."""
    section_header("Section 12.5: K-Means Customer Clustering", "Member 13 - Rudra Pratap Singh", level=3)
    st.markdown("K-Means groups customers with similar behaviour into `k` clusters without using any pre-defined label (MacQueen, 1967). The four "
                "behaviour features are log-transformed on a copy, standardised with `StandardScaler`, and clustered with "
                "`KMeans(n_clusters=k, n_init=10, random_state=42)`.")
    caption_md("Table 12.5a: Method and feature preparation")
    md_table(pd.DataFrame(KMEANS_METHOD, columns=["Step", "Choice", "Why"]))

    skew = sel["skew_df"].copy()
    show_df(skew, caption="Table 12.5b: Skewness of the features before and after log1p (|skew| near 0 = symmetric)")
    worst = skew["skewness_raw"].abs().idxmax()
    interpret(f"The most skewed feature is `{worst}` (skewness {skew.loc[worst, 'skewness_raw']:.2f} before and {skew.loc[worst, 'skewness_after_log1p']:.2f} "
              f"after the log transform), so a few very large customers would dominate the distance calculations without it.",
              f"After log1p and StandardScaler every feature has mean 0 and standard deviation 1 across the {sel['n']:,} customers "
              f"({'no missing value had to be filled' if sel['n_missing'] == 0 else str(sel['n_missing']) + ' missing values were filled with the median'}), "
              f"so days, orders and rupees contribute equally to the clustering.")

    # ---- choice of k ----
    metrics = sel["metrics"]
    mshow = metrics.set_index("k").round({"inertia_wcss": 1, "silhouette": 4, "davies_bouldin": 4, "calinski_harabasz": 1,
                                          "smallest_cluster_pct": 1})
    show_df(mshow, caption=f"Table 12.5c: Cluster quality for k = {sel['k_range'][0]} to {sel['k_range'][-1]}")
    st.markdown(f"**Data-driven choice:** best silhouette (all k) is **k = {sel['best_sil_k']}**, best Davies-Bouldin is **k = {sel['best_dbi_k']}**, the elbow "
                f"point is **k = {sel['elbow_k']}** and the best average rank inside the actionable window k = {K_MIN_ACTIONABLE}-{K_MAX_ACTIONABLE} is "
                f"**k = {sel['best_window_k']}**. The rule therefore selects **k = {sel['optimal_k']}** ({sel['why']}).")
    sig = f"{len(cust)}_{int(round(float(cust['total_spend'].sum())))}_{sel['optimal_k']}"
    k = st.slider("Number of clusters k (default = data-driven choice)", min_value=int(sel["k_range"][0]), max_value=int(sel["k_range"][-1]),
                  value=int(sel["optimal_k"]), step=1, key=f"km_k_{sig}",
                  help="Move the slider to re-run K-Means with another k; every table and chart below updates.")
    ctx["k_final"] = int(k)
    caption_md("Figure 12.5a: Elbow curve and silhouette score")
    show_png(chart_kmeans_selection(metrics, int(k), int(sel["elbow_k"]), float(sel["elbow_y"])))
    row_k = metrics.loc[metrics["k"] == k].iloc[0]
    interpret(f"Inertia falls from {metrics['inertia_wcss'].iloc[0]:,.0f} at k = {int(metrics['k'].iloc[0])} to {metrics['inertia_wcss'].iloc[-1]:,.0f} at "
              f"k = {int(metrics['k'].iloc[-1])}, and the bend (elbow) is at k = {sel['elbow_k']}; the selected k = {k} has a silhouette of {row_k['silhouette']:.3f} "
              f"and a Davies-Bouldin index of {row_k['davies_bouldin']:.3f}.",
              f"The smallest cluster at k = {k} holds {row_k['smallest_cluster_pct']:.1f}% of the customers, so "
              f"{'no cluster is tiny' if row_k['smallest_cluster_pct'] >= 3 else 'one cluster is small and may need a closer look'}; a higher silhouette and a lower "
              f"Davies-Bouldin value mean better separated clusters (Rousseeuw, 1987).")

    # ---- final model ----
    feat = cust[CLUSTER_FEATURES].reset_index(drop=True)
    fit = kmeans_final(feat, int(k))
    names, q = fit["names"], fit["quality"]
    cl = cust.copy()
    cl["cluster"] = fit["labels"]
    ctx["cust_k"] = cl
    ctx["km_fit_k"] = fit
    c = st.columns(5)
    kpi_card(c[0], "Clusters (k)", f"{int(k)}")
    kpi_card(c[1], "Silhouette", f"{q['silhouette']:.3f}", "Higher is better (range -1 to +1)")
    kpi_card(c[2], "Davies-Bouldin", f"{q['davies_bouldin']:.3f}", "Lower is better")
    kpi_card(c[3], "Calinski-Harabasz", f"{q['calinski_harabasz']:,.0f}", "Higher is better")
    kpi_card(c[4], "Inertia (WCSS)", f"{q['inertia_wcss']:,.0f}", "Within-cluster sum of squares")
    prof = cluster_profile_table(cl, names)
    allrow = pd.DataFrame([{"cluster": "All", "cluster_name": "Whole customer base", "customers": len(cl), "customer_share_pct": 100.0,
                            "revenue": float(cl["total_spend"].sum()), "revenue_share_pct": 100.0, "avg_recency_days": cl["recency_days"].mean(),
                            "avg_frequency": cl["num_orders"].mean(), "avg_spend": cl["total_spend"].mean(), "avg_aov": cl["avg_order_value"].mean()}])
    view = pd.concat([prof.assign(cluster=prof["cluster"].astype(str)), allrow], ignore_index=True)
    shown = pd.DataFrame({"Cluster": view["cluster"], "Business name": view["cluster_name"], "Customers": view["customers"].map("{:,.0f}".format),
                          "Customers %": view["customer_share_pct"].map("{:.1f}".format), "Revenue": view["revenue"].map(lambda v: money(v)),
                          "Revenue share %": view["revenue_share_pct"].map("{:.1f}".format),
                          "Avg recency (days)": view["avg_recency_days"].map("{:.1f}".format),
                          "Avg frequency (orders)": view["avg_frequency"].map("{:.2f}".format),
                          "Avg spend": view["avg_spend"].map(lambda v: money(v)), "Avg AOV": view["avg_aov"].map(lambda v: money(v))})
    show_df(shown, caption=f"Table 12.5d: Cluster profiles at k = {int(k)} (cluster 0 = highest value)", hide_index=True)
    top_c = prof.sort_values("revenue_share_pct", ascending=False).iloc[0]
    stale = prof.sort_values("avg_recency_days", ascending=False).iloc[0]
    interpret(f"Value is concentrated: cluster C{int(top_c['cluster'])} (**{esc(top_c['cluster_name'])}**) is {top_c['customer_share_pct']:.1f}% of customers but "
              f"{top_c['revenue_share_pct']:.1f}% of revenue ({safe_div(top_c['revenue_share_pct'], top_c['customer_share_pct']):.1f}x its headcount share).",
              f"The largest inactivity pool is cluster C{int(stale['cluster'])} (**{esc(stale['cluster_name'])}**): {int(stale['customers']):,} customers "
              f"({stale['customer_share_pct']:.1f}%) whose last purchase was on average {stale['avg_recency_days']:,.0f} days ago.")
    cz = fit["centroids_z"].round(2).assign(business_name=pd.Series(names))
    show_df(cz, caption="Table 12.5e: Centroids in standardised units (log1p + z-score): how far each cluster sits above (+) or below (-) the average customer")
    c0 = fit["centroids_z"].iloc[0]
    far = fit["centroids_z"]["recency_days"].idxmax()
    interpret(f"The top-value cluster C0 sits {c0['total_spend']:+.2f} standard deviations from the average customer on spend and {c0['num_orders']:+.2f} on order count, "
              f"with a recency of {c0['recency_days']:+.2f} (negative = more recent than average).",
              f"The least recent cluster is C{int(far)} at {fit['centroids_z'].loc[far, 'recency_days']:+.2f} on recency; values near 0 mean a cluster behaves like the average customer on that feature.")

    caption_md("Figure 12.5b: Clusters in two PCA dimensions")
    show_png(chart_kmeans_pca(fit["pcs"], fit["labels"], fit["cent_pc"], fit["ev"], tuple(names[i] for i in range(int(k))), int(k)))
    load = fit["loadings"]
    pc1_f, pc2_f = load["PC1"].abs().idxmax(), load["PC2"].abs().idxmax()
    interpret(f"The first two principal components capture {100 * fit['ev'].sum():.1f}% of the total variance in the four scaled features "
              f"(PC1 {fit['ev'][0] * 100:.1f}%, PC2 {fit['ev'][1] * 100:.1f}%), so the map is a faithful but simplified picture of the clusters.",
              f"PC1 is driven mostly by `{pc1_f}` (loading {load.loc[pc1_f, 'PC1']:+.2f}) and PC2 by `{pc2_f}` (loading {load.loc[pc2_f, 'PC2']:+.2f}); clusters that "
              f"sit far apart on the map differ strongly on these features, while overlapping clusters are separated mainly by the other dimensions.")
    show_df(load, caption="Table 12.5f: PCA feature loadings (which features drive each axis)")

    caption_md("Figure 12.5c: Customer share versus revenue share by cluster")
    show_png(chart_cluster_value(prof))
    n_above = int((prof['revenue_share_pct'] > prof['customer_share_pct']).sum())
    interpret(f"The bars compare each cluster's size with its revenue: {n_above} of {len(prof)} clusters "
              f"{'earns' if n_above == 1 else 'earn'} more revenue than their headcount suggests.",
              "Clusters with a revenue share far above their customer share are the groups to protect; clusters with the highest recency are the groups to win back.")
    st.markdown("**Plain-language reading of the cluster profile**")
    for r in prof.itertuples(index=False):
        st.markdown(f"- **C{r.cluster} - {esc(r.cluster_name)}:** {r.customers:,} customers ({r.customer_share_pct:.1f}%) generate {r.revenue_share_pct:.1f}% of revenue; "
                    f"last purchase {r.avg_recency_days:,.0f} days ago, {r.avg_frequency:.1f} orders, spend {money(r.avg_spend)}, AOV {money(r.avg_aov)}.")
    with st.expander("Assumptions and limitations of K-Means"):
        for item in KMEANS_ASSUMPTIONS:
            st.markdown("- " + item)


# ===================================================================================================
# 20.  TAB 7 ENGINES  -  Section 13 key findings and Section 14 recommendations   (Member 13)
#      Every number is computed live from the active transactions (tx) and customers (cust).
# ===================================================================================================
def month_name(m):
    return pd.Timestamp(2000, int(m), 1).strftime("%B")


def q(name):
    """Quoted, Markdown-safe name for use inside sentences."""
    return f"'{esc(name)}'"


def compute_findings(tx, cust, names, bundle):
    """Return (KEY_METRICS dict, list of finding rows).  A finding that cannot be computed is flagged, never fatal."""
    d = tx.copy()
    d["dt"] = pd.to_datetime(d["transaction_date"], errors="coerce")
    d["rev"] = pd.to_numeric(d["total_amount"], errors="coerce")
    ref_rev = float(cust["total_spend"].sum())
    rev_gap = (float(d["rev"].sum()) - ref_rev) / ref_rev if ref_rev else np.nan
    full_clean, raw = bundle["df_clean"], bundle["df_raw"]
    use_raw = bundle["mode"] == "synthetic"

    def f01():                                                  # BQ1 - data quality (Sections 4-5)
        m = {"rows_clean": int(len(full_clean)), "customers": int(cust["customer_id"].nunique()), "orders": int(cust["num_orders"].sum()),
             "recon_gap": float(rev_gap) if is_num(rev_gap) else None}
        m["null_cells"] = int(d[["customer_id", "transaction_date", "transaction_id"]].isna().sum().sum())
        m["dup_rows"] = int(d[CLEAN_COLS].duplicated().sum())
        m["nonpos_lines"] = int((d["rev"] <= 0).sum())
        m["date_min"], m["date_max"] = d["dt"].min(), d["dt"].max()
        if use_raw:
            m["rows_raw"] = int(len(raw))
            m["removed"] = m["rows_raw"] - m["rows_clean"]
            m["removed_pct"] = m["removed"] / m["rows_raw"] if m["rows_raw"] else None
            m["raw_dups"] = int(raw.duplicated().sum())
            m["raw_no_id"] = int(raw["customer_id"].isna().sum())
            m["raw_no_date"] = int(raw["transaction_date"].isna().sum())
            m["raw_returns"] = int((pd.to_numeric(raw["quantity"], errors="coerce") < 0).sum())
            finding = (f"The raw data needed cleaning: {frac(m['removed_pct'])} of the raw rows ({m['removed']:,} of {m['rows_raw']:,}) were invalid "
                       f"and removed, leaving a reconciled analytical base.") if m["removed"] > 0 else \
                "No raw rows had to be removed; the analytical base is complete and reconciled."
            raw_part = (f"Raw rows {m['rows_raw']:,} -> clean rows {m['rows_clean']:,} ({m['removed']:,} removed, {frac(m['removed_pct'])}); flagged in the raw "
                        f"data: duplicate rows {m['raw_dups']:,}, missing customer IDs {num(m['raw_no_id'])}, missing dates {num(m['raw_no_date'])}, "
                        f"returns / negative quantities {num(m['raw_returns'])} (categories can overlap). ")
        else:
            finding = "The analytical base is clean and internally consistent (the raw-data comparison is not available for supplied cleaned tables)."
            raw_part = ""
        if m["recon_gap"] is not None and abs(m["recon_gap"]) >= 0.01:
            finding += f" Note: transaction revenue differs from customer_df total_spend by {m['recon_gap']:+.2%} and should be reviewed."
        span = f"{m['date_min']:%d %b %Y} to {m['date_max']:%d %b %Y}"
        evidence = (raw_part + f"Active table: {m['dup_rows']:,} duplicate rows, {m['null_cells']:,} null key cells, {num(m['nonpos_lines'])} non-positive "
                    f"revenue lines; {m['customers']:,} customers, {m['orders']:,} orders, {span}; transaction revenue vs customer_df total_spend gap = "
                    f"{(m['recon_gap'] or 0):+.3%}.")
        impl = "All later shares, ratios and clusters rest on a reconciled base; keep the cleaning rules as an automatic gate for future data loads."
        return m, finding, evidence, impl

    def f02():                                                  # BQ2 - customer profile (Section 6)
        s, o, a, r = (cust[c] for c in ["total_spend", "num_orders", "avg_order_value", "recency_days"])
        m = dict(customers=int(len(cust)), revenue=float(s.sum()), mean_spend=float(s.mean()), median_spend=float(s.median()),
                 mean_orders=float(o.mean()), median_orders=float(o.median()), mean_aov=float(a.mean()), median_aov=float(a.median()),
                 mean_recency=float(r.mean()), median_recency=float(r.median()), skew=float(s.skew()))
        m["ratio"] = m["mean_spend"] / m["median_spend"] if m["median_spend"] else float("nan")
        if m["ratio"] >= 1.25:
            shape = f"highly right-skewed: the average customer spends {m['ratio']:.2f}x the median customer"
        elif m["ratio"] >= 1.10:
            shape = f"moderately right-skewed: the average customer spends {m['ratio']:.2f}x the median customer"
        else:
            shape = f"fairly evenly spread: mean spend is only {m['ratio']:.2f}x the median"
        finding = f"Customer value is {shape}."
        evidence = (f"{m['customers']:,} customers; total revenue {money(m['revenue'])}; spend mean {money(m['mean_spend'])} vs median {money(m['median_spend'])} "
                    f"(skewness {m['skew']:.2f}); orders mean {m['mean_orders']:.2f} (median {m['median_orders']:.0f}); AOV mean {money(m['mean_aov'])} "
                    f"(median {money(m['median_aov'])}); recency mean {m['mean_recency']:.0f} days (median {m['median_recency']:.0f}).")
        impl = ("Averages overstate what a typical customer does, so targets and budgets should be set by value tier (medians / percentiles), not on the overall mean."
                if m["ratio"] >= 1.10 else "Because spend is fairly even, broad programmes can work, but tier-based targeting still avoids waste.")
        return m, finding, evidence, impl

    def f03():                                                  # BQ3 - spend segments (Sections 7-8)
        g = (cust.groupby("spend_segment").agg(customers=("customer_id", "count"), revenue=("total_spend", "sum"),
                                               avg_spend=("total_spend", "mean"), avg_orders=("num_orders", "mean"))
             .sort_values("avg_spend", ascending=False))
        g["cust_share"] = g["customers"] / g["customers"].sum()
        g["rev_share"] = g["revenue"] / g["revenue"].sum()
        segs = [dict(segment=str(i), **{k: float(v) for k, v in row.items()}) for i, row in g.iterrows()]
        top, bottom = segs[0], segs[-1]
        m = dict(segments=segs, top=top, bottom=bottom, lift=top["rev_share"] / top["cust_share"])
        finding = (f"{q(top['segment'])} spenders are {frac(top['cust_share'])} of customers but deliver {frac(top['rev_share'])} of revenue "
                   f"({m['lift']:.1f}x their headcount share), while {q(bottom['segment'])} spenders are {frac(bottom['cust_share'])} of customers "
                   f"and {frac(bottom['rev_share'])} of revenue.")
        evidence = "; ".join(f"{s_['segment']}: {frac(s_['cust_share'])} of customers -> {frac(s_['rev_share'])} of revenue "
                             f"(avg spend {money(s_['avg_spend'])}, avg orders {s_['avg_orders']:.1f})" for s_ in segs) + "."
        impl = "Value sits in the top spend tier: protect it, and build a ladder that moves Medium customers upward."
        return m, finding, evidence, impl

    def f04():                                                  # BQ4 - products and categories (Section 9)
        pr = d.groupby("product_name")["rev"].sum().sort_values(ascending=False)
        tot = float(pr.sum())
        m = dict(n_products=int(len(pr)), top_product=str(pr.index[0]), top_product_rev=float(pr.iloc[0]), top_product_share=float(pr.iloc[0] / tot),
                 top5_share=float(pr.iloc[:5].sum() / tot), total=tot)
        ca = d.groupby("category")["rev"].sum().sort_values(ascending=False)
        m.update(n_categories=int(len(ca)), top_cat=str(ca.index[0]), top_cat_rev=float(ca.iloc[0]), top_cat_share=float(ca.iloc[0] / tot),
                 top3_cat_share=float(ca.iloc[:3].sum() / tot), low_cat=str(ca.index[-1]), low_cat_share=float(ca.iloc[-1] / tot))
        finding = (f"{q(m['top_cat'])} is the leading category with {frac(m['top_cat_share'])} of revenue and the top 3 categories deliver "
                   f"{frac(m['top3_cat_share'])}; the best-selling product, {q(m['top_product'])}, alone brings {frac(m['top_product_share'])}.")
        evidence = (f"{m['n_products']} products in {m['n_categories']} categories; top category {q(m['top_cat'])} = {money(m['top_cat_rev'])} "
                    f"({frac(m['top_cat_share'])}); top-3 categories {frac(m['top3_cat_share'])}; weakest category {q(m['low_cat'])} = {frac(m['low_cat_share'])}; "
                    f"top product {q(m['top_product'])} = {money(m['top_product_rev'])} ({frac(m['top_product_share'])}); top-5 products = {frac(m['top5_share'])}.")
        impl = ("A few categories and products carry the sales: keep them in stock and visible, and use them to pull weaker lines through bundles."
                if m["top3_cat_share"] >= 0.50 else "Sales are spread across many lines: manage the range broadly and focus effort on lifting the weakest categories.")
        return m, finding, evidence, impl

    def f05():                                                  # BQ5 - repeat vs one-time buyers (Section 10)
        rep = cust["num_orders"] >= 2
        if rep.sum() == 0 or (~rep).sum() == 0:
            raise ValueError("only one buyer type present")
        tot = float(cust["total_spend"].sum())
        many, one = cust[rep], cust[~rep]
        m = dict(repeat_n=int(rep.sum()), one_n=int((~rep).sum()), repeat_rate=float(rep.mean()), one_rate=float((~rep).mean()),
                 repeat_rev_share=float(many["total_spend"].sum() / tot), avg_spend_repeat=float(many["total_spend"].mean()),
                 avg_spend_one=float(one["total_spend"].mean()), aov_repeat=float(many["avg_order_value"].mean()),
                 aov_one=float(one["avg_order_value"].mean()), avg_orders_repeat=float(many["num_orders"].mean()),
                 recency_repeat=float(many["recency_days"].mean()), recency_one=float(one["recency_days"].mean()))
        m["spend_ratio"] = m["avg_spend_repeat"] / m["avg_spend_one"] if m["avg_spend_one"] else float("nan")
        finding = (f"{frac(m['repeat_rate'])} of customers are repeat buyers and they generate {frac(m['repeat_rev_share'])} of revenue; "
                   f"a repeat buyer is worth {m['spend_ratio']:.1f}x a one-time buyer.")
        evidence = (f"Repeat buyers {m['repeat_n']:,} ({frac(m['repeat_rate'])}) vs one-time buyers {m['one_n']:,} ({frac(m['one_rate'])}); average spend "
                    f"{money(m['avg_spend_repeat'])} vs {money(m['avg_spend_one'])}; average AOV {money(m['aov_repeat'])} vs {money(m['aov_one'])}; repeat "
                    f"buyers place {m['avg_orders_repeat']:.1f} orders on average; mean recency {m['recency_repeat']:.0f} vs {m['recency_one']:.0f} days.")
        impl = ("The second order is the key turning point: converting one-time buyers into repeat buyers is the most direct way to raise customer value."
                if m["spend_ratio"] >= 1.5 else "Repeat and one-time buyers spend similar amounts, so growth should come from frequency and basket size rather than from the second order alone.")
        return m, finding, evidence, impl

    def f06():                                                  # BQ6 - monthly seasonality (Section 11)
        dd = d.dropna(subset=["dt", "rev"])
        mon = dd.groupby(dd["dt"].dt.to_period("M"))["rev"].sum()
        dmin, dmax = dd["dt"].min(), dd["dt"].max()
        start = 0 if dmin.day <= 3 else 1                                    # drop partial first / last month
        end = len(mon) if dmax.day >= dmax.days_in_month - 3 else len(mon) - 1
        full = mon.iloc[start:end]
        if len(full) < 3:
            full = mon
        if len(full) < 2:
            raise ValueError("data span too short for a monthly pattern")
        moy = full.groupby(full.index.month).mean()
        mean_m = float(full.mean())
        m = dict(months_used=int(len(full)), mean_month=mean_m, peak_month=str(full.idxmax()), peak_rev=float(full.max()), low_month=str(full.idxmin()),
                 low_rev=float(full.min()), peak_moy=int(moy.idxmax()), peak_idx=float(moy.max() / moy.mean()), low_moy=int(moy.idxmin()),
                 low_idx=float(moy.min() / moy.mean()), moy_cover=int(len(moy)),
                 slope_pct=float(np.polyfit(np.arange(len(full)), full.to_numpy(dtype=float), 1)[0] / mean_m), total=float(dd["rev"].sum()))
        note = "" if m["moy_cover"] == 12 else f" (only {m['moy_cover']} calendar months are covered by the data)"
        lead = "Sales are seasonal" if m["peak_idx"] >= 1.15 else "Seasonality is mild"
        finding = (f"{lead}: {month_name(m['peak_moy'])} is the strongest calendar month ({m['peak_idx']:.2f}x the average month) and "
                   f"{month_name(m['low_moy'])} the weakest ({m['low_idx']:.2f}x){note}.")
        evidence = (f"{m['months_used']} complete months analysed; average month {money(mean_m)}; best month {m['peak_month']} = {money(m['peak_rev'])} "
                    f"({m['peak_rev'] / mean_m:.2f}x average); weakest month {m['low_month']} = {money(m['low_rev'])} ({m['low_rev'] / mean_m:.2f}x); "
                    f"calendar-month index: peak {month_name(m['peak_moy'])} {m['peak_idx']:.2f}, trough {month_name(m['low_moy'])} {m['low_idx']:.2f}; "
                    f"linear trend {m['slope_pct']:+.2%} of the average monthly revenue per month.")
        impl = ("Inventory, staffing and campaign calendars should follow the seasonal curve instead of a flat monthly plan."
                if m["peak_idx"] >= 1.15 else "A flat monthly plan is acceptable, but the strongest and weakest months still deserve targeted promotions.")
        return m, finding, evidence, impl

    def f07():                                                  # BQ6 - weekday pattern (Section 11)
        dd = d.dropna(subset=["dt", "rev"])
        wd_tot = dd.groupby(dd["dt"].dt.dayofweek)["rev"].sum().reindex(range(7), fill_value=0.0)
        n_days = (pd.Series(pd.date_range(dd["dt"].min().normalize(), dd["dt"].max().normalize()).dayofweek)
                  .value_counts().reindex(range(7), fill_value=0))
        per_day = (wd_tot / n_days.replace(0, np.nan)).fillna(0.0)          # average revenue per occurrence of each weekday
        share = per_day / per_day.sum()
        best, worst = int(share.idxmax()), int(share.idxmin())
        m = dict(best=S11_WEEKDAYS[best], worst=S11_WEEKDAYS[worst], best_share=float(share.loc[best]), worst_share=float(share.loc[worst]),
                 best_idx=float(share.loc[best] * 7), weekend_share=float(share.loc[5] + share.loc[6]),
                 shares={S11_WEEKDAYS[i]: float(share.loc[i]) for i in range(7)}, avg_day_rev=float(per_day.mean()),
                 worst_day_rev=float(per_day.loc[worst]), total=float(dd["rev"].sum()))
        finding = (f"{m['best']} is the strongest sales day ({frac(m['best_share'])} of weekly revenue, {m['best_idx']:.2f}x an average day) and "
                   f"{m['worst']} the weakest ({frac(m['worst_share'])}); weekends carry {frac(m['weekend_share'])} of revenue versus 28.6% if sales were even.")
        evidence = "Share of weekly revenue (per day-of-week occurrence): " + ", ".join(f"{k[:3]} {frac(v)}" for k, v in m["shares"].items()) + "."
        impl = "Campaign timing and support staffing can be aligned to the weekday rhythm; the weakest day is the natural place to test offers."
        return m, finding, evidence, impl

    def f08():                                                  # BQ7 - Pareto / revenue concentration (Section 12.1)
        s = np.sort(cust["total_spend"].to_numpy(dtype=float))[::-1]
        cum = np.cumsum(s) / s.sum()
        n = len(s)
        n80_ = int(np.searchsorted(cum, 0.80) + 1)
        top = lambda p: float(s[: max(1, int(np.ceil(p * n)))].sum() / s.sum())                  # noqa: E731
        m = dict(n=n, n80=n80_, pct80=n80_ / n, top1=top(0.01), top10=top(0.10), top20=top(0.20), n_top20=int(np.ceil(0.20 * n)),
                 top20_rev=float(s[: int(np.ceil(0.20 * n))].sum()), total=float(s.sum()))
        p = np.sort(d.groupby("product_name")["rev"].sum().to_numpy(dtype=float))[::-1]
        m["pct_products_80"] = float((np.searchsorted(np.cumsum(p) / p.sum(), 0.80) + 1) / len(p))
        verdict = ("meets or exceeds the classic 80/20 rule" if m["top20"] >= 0.80 else
                   "is close to the 80/20 rule" if m["top20"] >= 0.60 else "is less concentrated than the 80/20 rule")
        finding = f"Revenue is concentrated: the top 20% of customers generate {frac(m['top20'])} of revenue, which {verdict}."
        evidence = (f"Top 1% of customers = {frac(m['top1'])}, top 10% = {frac(m['top10'])}, top 20% = {frac(m['top20'])} of revenue; 80% of revenue comes "
                    f"from {m['n80']:,} of {m['n']:,} customers ({frac(m['pct80'])}); on the product side {frac(m['pct_products_80'])} of products make 80% of revenue.")
        impl = ("A small VIP base carries the business, so keeping those customers active matters more than acquiring many small ones."
                if m["top20"] >= 0.60 else "Revenue is spread across the base, so broad-based programmes are worthwhile alongside VIP care.")
        return m, finding, evidence, impl

    def f09():                                                  # BQ7 - RFM segments (Section 12.2)
        g = (cust.groupby("rfm_segment").agg(customers=("customer_id", "count"), revenue=("total_spend", "sum"), orders=("num_orders", "sum"),
                                             avg_recency=("recency_days", "mean"), avg_spend=("total_spend", "mean")))
        g["cust_share"] = g["customers"] / g["customers"].sum()
        g["rev_share"] = g["revenue"] / g["revenue"].sum()
        g["aov"] = g["revenue"] / g["orders"]
        top, stale = g.sort_values("revenue", ascending=False).iloc[0], g.sort_values("avg_recency", ascending=False).iloc[0]
        top_name, stale_name = g["revenue"].idxmax(), g["avg_recency"].idxmax()
        pack = lambda name, r: dict(name=str(name), **{k: float(v) for k, v in r.items()})        # noqa: E731
        m = dict(n_segments=int(len(g)), top=pack(top_name, top), stale=pack(stale_name, stale))
        finding = (f"RFM segment {q(m['top']['name'])} is the biggest revenue pool ({frac(m['top']['rev_share'])} of revenue from {frac(m['top']['cust_share'])} "
                   f"of customers), while {q(m['stale']['name'])} is the least recently active (mean recency {m['stale']['avg_recency']:.0f} days; "
                   f"{frac(m['stale']['cust_share'])} of customers, {frac(m['stale']['rev_share'])} of revenue).")
        evidence = "; ".join(f"{esc(i)}: {frac(r.cust_share)} of customers -> {frac(r.rev_share)} of revenue (recency {r.avg_recency:.0f} d, avg spend {money(r.avg_spend)})"
                             for i, r in g.sort_values("revenue", ascending=False).iterrows()) + "."
        impl = "Treat RFM segments differently: reward and protect the value segments, and trigger win-back flows for the segments that have gone quiet."
        return m, finding, evidence, impl

    def f10():                                                  # BQ8 - cohort retention (Section 12.3)
        a = d.dropna(subset=["customer_id", "dt"])[["customer_id", "dt"]].copy()
        a["mi"] = a["dt"].dt.year * 12 + a["dt"].dt.month
        a = a.drop_duplicates(["customer_id", "mi"])
        a["c0"] = a.groupby("customer_id")["mi"].transform("min")
        a["age"] = a["mi"] - a["c0"]
        last_mi = int(a["mi"].max())
        size = a[a["age"] == 0].groupby("c0")["customer_id"].nunique()

        def ret(h):
            elig = size.index[size.index <= last_mi - h]
            if len(elig) == 0:
                return None, 0
            kept = a[(a["age"] == h) & a["c0"].isin(elig)]["customer_id"].nunique()
            return float(kept / size.loc[elig].sum()), int(size.loc[elig].sum())

        r1, n1 = ret(1)
        r3, n3 = ret(3)
        if r1 is None:
            raise ValueError("data span too short for cohort retention")
        m = dict(ret_m1=r1, ret_m3=r3, n_m1=n1, n_m3=n3, cohorts=int(len(size)), new_per_month=float(size.mean()))
        finding = (f"{frac(r1)} of new customers buy again in the next calendar month" + (f" and {frac(r3)} in the third month" if r3 is not None else "")
                   + f", so about {frac(1 - r1, 0)} do not return in the month after their first purchase.")
        evidence = (f"{m['cohorts']} monthly cohorts, on average {m['new_per_month']:,.0f} new customers per month; month-1 retention {frac(r1)} "
                    f"(base {n1:,} customers); month-3 retention {frac(r3) if r3 is not None else 'n/a'} (base {n3:,} customers).")
        impl = ("The first 90 days decide whether a new customer becomes a repeat buyer, so onboarding and second-purchase nudges deserve dedicated effort."
                if r1 < 0.50 else "Early retention is healthy; keep the onboarding journey and focus on lifting frequency among customers who return.")
        return m, finding, evidence, impl

    def f11():                                                  # BQ8 - market basket (Section 12.4)
        b = d[["transaction_id", "product_name"]].dropna().drop_duplicates()
        b["product_name"] = b["product_name"].astype(str)
        n_orders = int(b["transaction_id"].nunique())
        size = b.groupby("transaction_id")["product_name"].transform("size")
        multi_share = float((b.groupby("transaction_id").size() >= 2).mean())
        multi = b[(size >= 2) & (size <= 15)]
        if multi.empty:
            raise ValueError("no multi-item orders in the active data")
        pairs = multi.merge(multi, on="transaction_id", suffixes=("_a", "_b"))
        pairs = pairs[pairs["product_name_a"] < pairs["product_name_b"]]
        pc = pairs.groupby(["product_name_a", "product_name_b"]).size().rename("n").reset_index()
        item = b.groupby("product_name")["transaction_id"].nunique()
        pc["na"], pc["nb"] = pc["product_name_a"].map(item), pc["product_name_b"].map(item)
        pc["lift"] = (pc["n"] / n_orders) / ((pc["na"] / n_orders) * (pc["nb"] / n_orders))
        pc["conf_ab"], pc["conf_ba"] = pc["n"] / pc["na"], pc["n"] / pc["nb"]
        min_n = max(5, int(np.ceil(0.002 * n_orders)))
        ok = pc[pc["n"] >= min_n]
        if ok.empty:
            raise ValueError("no product pair is frequent enough")
        t = ok.sort_values(["lift", "n"], ascending=False).iloc[0]
        f = pc.sort_values("n", ascending=False).iloc[0]
        a_first = t["conf_ab"] >= t["conf_ba"]
        ante, cons = (t["product_name_a"], t["product_name_b"]) if a_first else (t["product_name_b"], t["product_name_a"])
        m = dict(n_orders=n_orders, multi_share=multi_share, lift=float(t["lift"]), pair_n=int(t["n"]), pair_support=float(t["n"] / n_orders),
                 ante=str(ante), cons=str(cons), conf=float(max(t["conf_ab"], t["conf_ba"])), n_ante=int(t["na"] if a_first else t["nb"]),
                 freq_pair=f"{f['product_name_a']} + {f['product_name_b']}", freq_n=int(f["n"]), min_n=min_n)
        finding = (f"The strongest product association is {q(m['ante'])} -> {q(m['cons'])} (lift {m['lift']:.2f}, confidence {frac(m['conf'])}); "
                   f"{frac(multi_share)} of orders contain two or more products.")
        evidence = (f"Strongest association (at least {min_n} joint orders): {q(m['ante'])} -> {q(m['cons'])} lift {m['lift']:.2f}, confidence {frac(m['conf'])}, "
                    f"{m['pair_n']:,} joint orders ({frac(m['pair_support'])} of {n_orders:,} orders); most frequent pair: {esc(m['freq_pair'])} in "
                    f"{m['freq_n']:,} orders; multi-product orders = {frac(multi_share)}.")
        impl = ("Bundles and 'frequently bought together' prompts for the strongest pairs raise basket size at low cost."
                if m["lift"] >= 1.5 else "Associations are weak (lift below 1.5): prefer broad recommendations over fixed bundles.")
        return m, finding, evidence, impl

    def cluster_table():                                        # helper for F12 / F13
        if "cluster" not in cust.columns or (cust["cluster"] < 0).all():
            raise KeyError("K-Means was not run (too few customers in the current selection)")
        g = (cust.groupby("cluster").agg(customers=("customer_id", "count"), revenue=("total_spend", "sum"), orders=("num_orders", "sum"),
                                         avg_recency=("recency_days", "mean"), avg_orders=("num_orders", "mean"), avg_spend=("total_spend", "mean"),
                                         avg_aov=("avg_order_value", "mean")))
        g["cust_share"] = g["customers"] / g["customers"].sum()
        g["rev_share"] = g["revenue"] / g["revenue"].sum()
        return [dict(cluster=int(i), name=names.get(int(i), f"Cluster {int(i)}"), **{k: float(v) for k, v in row.items()}) for i, row in g.iterrows()]

    def f12():                                                  # BQ9 - where the value is (Section 12.5)
        cl = cluster_table()
        top = max(cl, key=lambda c: c["rev_share"])
        m = dict(k=len(cl), top=top, clusters=cl, lift=top["rev_share"] / top["cust_share"])
        finding = (f"K-Means separates customers into {m['k']} behavioural clusters; {q(top['name'])} is {frac(top['cust_share'])} of customers "
                   f"but {frac(top['rev_share'])} of revenue.")
        evidence = "; ".join(f"C{c['cluster']} {esc(c['name'])}: {c['customers']:,.0f} customers ({frac(c['cust_share'])}) -> {frac(c['rev_share'])} of revenue "
                             f"(recency {c['avg_recency']:.0f} d, {c['avg_orders']:.1f} orders, spend {money(c['avg_spend'])}, AOV {money(c['avg_aov'])})"
                             for c in cl) + "."
        impl = "Clusters give an operational customer map: one playbook per cluster is more effective than one campaign for everyone."
        return m, finding, evidence, impl

    def f13():                                                  # BQ9 - inactivity pool (Section 12.5)
        cl = cluster_table()
        stale = max(cl, key=lambda c: c["avg_recency"])
        m = dict(stale=stale)
        finding = (f"The least active cluster, {q(stale['name'])}, holds {frac(stale['cust_share'])} of customers ({stale['customers']:,.0f}) whose last purchase "
                   f"was {stale['avg_recency']:.0f} days ago on average; it accounts for {frac(stale['rev_share'])} of revenue.")
        evidence = (f"C{stale['cluster']} {esc(stale['name'])}: {stale['customers']:,.0f} customers, mean recency {stale['avg_recency']:.0f} days, "
                    f"{stale['avg_orders']:.1f} orders, average spend {money(stale['avg_spend'])}, AOV {money(stale['avg_aov'])}, revenue "
                    f"{money(stale['revenue'])} ({frac(stale['rev_share'])} of total).")
        impl = ("This is the main win-back pool; a staged reactivation effort is usually cheaper than acquiring the same number of new customers."
                if stale["cust_share"] >= 0.10 else "The inactive group is small; a light-touch reactivation campaign is enough.")
        return m, finding, evidence, impl

    specs = [("F01", "BQ1", "Sections 4-5", f01), ("F02", "BQ2", "Section 6", f02), ("F03", "BQ3", "Sections 7-8", f03),
             ("F04", "BQ4", "Section 9", f04), ("F05", "BQ5", "Section 10", f05), ("F06", "BQ6", "Section 11", f06),
             ("F07", "BQ6", "Section 11", f07), ("F08", "BQ7", "Section 12.1", f08), ("F09", "BQ7", "Section 12.2", f09),
             ("F10", "BQ8", "Section 12.3", f10), ("F11", "BQ8", "Section 12.4", f11), ("F12", "BQ9", "Section 12.5", f12),
             ("F13", "BQ9", "Section 12.5", f13)]
    metrics, rows = {}, []
    for fid, bq, sec, fn in specs:
        try:
            met, finding, evidence, impl = fn()
            metrics[fid] = met
        except Exception as err:                                            # noqa: BLE001 - flag, never break
            metrics[fid] = None
            finding, evidence, impl = ("Evidence unavailable for this finding.", f"{type(err).__name__}: {err}",
                                       "Widen the sidebar selection or fix the missing input.")
        rows.append({"id": fid, "bq": bq, "section": sec, "finding": finding, "evidence": evidence, "implication": impl})
    return metrics, rows


# ---------------------------------------------------------------------------------------------------
# Section 14: one recommendation per finding (scenario assumptions are editable in the app)
# ---------------------------------------------------------------------------------------------------
def build_recommendations(K, SA, total_rev, mean_aov):
    """R01-R13 from the key metrics K and the scenario assumptions SA.  Revenue effects are scenarios, not forecasts."""
    def r01(m):
        rp, gap = m.get("removed_pct"), m.get("recon_gap")
        gap_txt = f"{gap:+.3%}" if gap is not None else "n/a"
        impact = (f"Keeps invalid rows ({frac(rp)} of the raw load) out of every KPI and holds the revenue reconciliation gap at {gap_txt}; no direct revenue "
                  f"uplift, but every other recommendation depends on it." if rp is not None else
                  f"Holds the revenue reconciliation gap at {gap_txt}; no direct revenue uplift, but every other recommendation depends on it.")
        return dict(target="Data pipeline: all customers and transactions",
                    action="Turn the Section 5 cleaning rules into a permanent load gate: de-duplicate on the transaction key, reject rows with a missing customer ID or date, "
                           "route returns to a separate table and report net revenue; publish a weekly data-quality scorecard (duplicate %, null %, return %, reconciliation gap).",
                    impact=impact, priority="Medium" if (rp or 0) >= 0.05 else "Low", phase=1, owner="Data / BI team",
                    kpi=(f"Invalid-row rate {frac(rp)} -> below 1%" if rp is not None else "Reconciliation gap kept within +/-1%"))

    def r02(m):
        return dict(target="Management reporting: all customers",
                    action="Report the median and percentiles (P25 / P75 / P90) of spend, orders and AOV beside every average, publish a value-tier dashboard "
                           "(share of customers vs share of revenue) and set KPI targets by tier instead of on the overall mean.",
                    impact=f"Removes the {m['ratio']:.2f}x mean-versus-median distortion (mean {money(m['mean_spend'])} vs median {money(m['median_spend'])}) from "
                           f"targets and budgets, so money is not spent on an 'average' customer who rarely exists.",
                    priority="Low", phase=1, owner="BI / Finance", kpi=f"Mean-to-median spend ratio ({m['ratio']:.2f}x) reported monthly by value tier")

    def r03(m):
        top, segs = m["top"], m["segments"]
        nxt = segs[1] if len(segs) > 1 else None
        if nxt:
            moved = nxt["customers"] * SA["vip_upgrade_rate"]
            gap = max(top["avg_spend"] - nxt["avg_spend"], 0)
            inc = moved * gap
            tgt = f"{q(top['segment'])} spend segment ({top['customers']:,.0f} customers) and {q(nxt['segment'])} segment ({nxt['customers']:,.0f}) as the upgrade pool"
            step = f" show 'spend {money(gap)} more to unlock' progress (the average gap between the two tiers) to move {q(nxt['segment'])} customers up."
            scen = (f" Scenario: lifting {SA['vip_upgrade_rate']:.0%} of {q(nxt['segment'])} customers ({moved:,.0f}) to the {q(top['segment'])} average spend "
                    f"adds about {money(inc)} (+{inc / total_rev:.1%} of revenue).")
            kpi = f"Customers in the {q(top['segment'])} tier {top['customers']:,.0f} -> {top['customers'] + moved:,.0f}"
        else:
            tgt, step, scen = f"{q(top['segment'])} spend segment ({top['customers']:,.0f} customers)", " keep the tier criteria visible to customers.", ""
            kpi = f"Customers in the {q(top['segment'])} tier: {top['customers']:,.0f}"
        return dict(target=tgt, action="Launch a tiered loyalty programme anchored on the top spend tier (early access, free premium delivery, dedicated support and "
                                       "personalised bundles) and" + step,
                    impact=f"Protects the {frac(top['rev_share'])} of revenue that sits in the top tier ({m['lift']:.1f}x its headcount share)." + scen,
                    priority="High", phase=2, owner="CRM / Loyalty", kpi=kpi)

    def r04(m):
        inc = m["top_cat_rev"] * SA["category_lift"]
        conc = m["top3_cat_share"] >= 0.50
        return dict(target=f"Category {q(m['top_cat'])} (top-3 categories = {frac(m['top3_cat_share'])} of revenue), weakest category {q(m['low_cat'])} and hero product {q(m['top_product'])}",
                    action="Ring-fence availability, pricing and visibility of the top categories and hero products (safety stock, no stock-outs, prime placement), build "
                           "cross-category bundles that pair top sellers with the weakest lines, and review the long tail of low-revenue products for range rationalisation.",
                    impact=f"Secures the {frac(m['top3_cat_share'])} of revenue held by the top-3 categories. Scenario: a {SA['category_lift']:.0%} lift in {q(m['top_cat'])} adds "
                           f"about {money(inc)} (+{inc / total_rev:.1%} of revenue).",
                    priority="Medium" if conc else "Low", phase=3, owner="Merchandising / Category management",
                    kpi=f"Top-3 category share {frac(m['top3_cat_share'])} protected; stock-out rate on hero products tracked (new KPI)")

    def r05(m):
        gain = max(m["avg_spend_repeat"] - m["avg_spend_one"], 0)
        inc = m["one_n"] * SA["second_order_rate"] * gain
        new_rate = m["repeat_rate"] + m["one_rate"] * SA["second_order_rate"]
        return dict(target=f"One-time buyers ({m['one_n']:,} customers = {frac(m['one_rate'])} of the base)",
                    action="Build an automated second-order journey: thank-you and product-care message at day 3, complementary-product or replenishment offer at day 14-21, "
                           "and a time-boxed second-order incentive (for example free delivery or a small credit) at day 30-45; run it against a holdout group to prove the lift.",
                    impact=f"Repeat buyers spend {m['spend_ratio']:.1f}x one-time buyers and generate {frac(m['repeat_rev_share'])} of revenue. Scenario: converting "
                           f"{SA['second_order_rate']:.0%} of one-time buyers adds about {money(inc)} (+{inc / total_rev:.1%} of revenue) at the average spend gap of {money(gain)}.",
                    priority="High" if m["spend_ratio"] >= 1.5 else "Medium", phase=2, owner="CRM / Lifecycle marketing",
                    kpi=f"Repeat-buyer rate {frac(m['repeat_rate'])} -> {frac(new_rate)}")

    def r06(m):
        inc = m["peak_rev"] * SA["peak_month_lift"]
        return dict(target=f"Peak season ({month_name(m['peak_moy'])}) and weakest season ({month_name(m['low_moy'])}); all categories",
                    action="Plan inventory, staffing and marketing calendars around the seasonal curve: build stock and launch campaigns 4-6 weeks before the peak month, "
                           "and use bundles or loyalty bonuses in the weakest month to smooth demand.",
                    impact=f"The strongest month runs at {m['peak_idx']:.2f}x and the weakest at {m['low_idx']:.2f}x an average month. Scenario: capturing an extra "
                           f"{SA['peak_month_lift']:.0%} in the best month ({m['peak_month']}: {money(m['peak_rev'])}) adds about {money(inc)} (+{inc / m['total']:.1%} of revenue).",
                    priority="Medium" if m["peak_idx"] >= 1.15 else "Low", phase=3, owner="Marketing / Supply chain",
                    kpi=f"Peak-month revenue {money(m['peak_rev'])} -> {money(m['peak_rev'] + inc)}; stock-out days in the peak month tracked (new KPI)")

    def r07(m):
        gap = max(1 / 7 - m["worst_share"], 0)
        return dict(target=f"Weakest sales day ({m['worst']}) and strongest sales day ({m['best']})",
                    action="Schedule e-mail / app campaigns and ad budgets to land 12-24 hours before the strongest day, test day-specific offers or free delivery on the weakest day, "
                           "and align support and fulfilment staffing to the weekday profile.",
                    impact=f"{m['best']} carries {frac(m['best_share'])} of weekly revenue. If {m['worst']} ({frac(m['worst_share'])}) matched an average day (14.3%), "
                           f"revenue would rise by up to {gap:.1%} (an upper bound, not an expectation).",
                    priority="Low", phase=1, owner="Marketing / Operations", kpi=f"{m['worst']} revenue share {frac(m['worst_share'])} -> 14.3% (average day)")

    def r08(m):
        prot = m["top20_rev"] * SA["top_customer_attrition"]
        return dict(target=f"Top 20% of customers ({m['n_top20']:,} customers; {frac(m['top20'])} of revenue)",
                    action="Create a key-account / VIP desk for the top 20%: a named relationship owner for the top 1-5%, quarterly personalised offers, and an early-warning churn alert "
                           "when a customer's days since the last order exceed their usual purchase gap.",
                    impact=f"This group generates {money(m['top20_rev'])} ({frac(m['top20'])} of revenue). Scenario: preventing {SA['top_customer_attrition']:.0%} attrition "
                           f"protects about {money(prot)} ({prot / m['total']:.1%} of revenue).",
                    priority="High" if m["top20"] >= 0.60 else "Medium", phase=2, owner="CRM / Key-account management",
                    kpi=f"Top-20% customer attrition at or below {SA['top_customer_attrition']:.0%}; their revenue share held at {frac(m['top20'])}")

    def r09(m):
        st_, tp = m["stale"], m["top"]
        inc = st_["customers"] * SA["reactivation_rate"] * st_["aov"]
        return dict(target=f"RFM segment {q(st_['name'])} ({frac(st_['cust_share'])} of customers, mean recency {st_['avg_recency']:.0f} days) and the value segment {q(tp['name'])}",
                    action=f"Automate recency-triggered lifecycle flows in the RFM engine: win-back sequences for {q(st_['name'])} when recency passes the segment median, reward and "
                           f"early-access offers for {q(tp['name'])}, monthly refresh of RFM scores and tracking of customers moving between segments.",
                    impact=f"{q(st_['name'])} holds {frac(st_['rev_share'])} of revenue from {frac(st_['cust_share'])} of customers. Scenario: re-activating "
                           f"{SA['reactivation_rate']:.0%} of them with one order at the segment AOV ({money(st_['aov'])}) adds about {money(inc)} (+{inc / total_rev:.1%} of revenue).",
                    priority="Medium", phase=2, owner="CRM / Marketing automation",
                    kpi=f"Re-activation rate of {q(st_['name'])} 0% -> {SA['reactivation_rate']:.0%}; monthly segment-migration report")

    def r10(m):
        extra = m["new_per_month"] * SA["retention_lift_pp"]
        r3 = f" (month-3: {frac(m['ret_m3'])})" if m.get("ret_m3") is not None else ""
        return dict(target=f"New customers in their first 90 days (about {m['new_per_month']:,.0f} per month)",
                    action="Design a first-90-day onboarding programme: welcome series, personalised category recommendations, a second-purchase nudge before day 30 and check-ins at "
                           "day 60 and 90; track month-1 / month-3 retention weekly by acquisition channel.",
                    impact=f"Month-1 retention is {frac(m['ret_m1'])}{r3}. Scenario: +{SA['retention_lift_pp'] * 100:.0f} percentage points in month-1 retention brings back "
                           f"about {extra:,.0f} more customers every month, worth about {money(extra * mean_aov)} per month at the average AOV ({money(mean_aov)}).",
                    priority="High" if m["ret_m1"] < 0.50 else "Medium", phase=2, owner="CRM / Growth",
                    kpi=f"Month-1 retention {frac(m['ret_m1'])} -> {frac(m['ret_m1'] + SA['retention_lift_pp'])}")

    def r11(m):
        extra = m["pair_n"] * SA["attach_rate_lift"]
        return dict(target=f"Product pair {q(m['ante'])} -> {q(m['cons'])} (lift {m['lift']:.2f}) and multi-product baskets ({frac(m['multi_share'])} of orders)",
                    action="Deploy 'frequently bought together' prompts and bundle offers for the strongest pairs on product and cart pages, add a one-click add-on at checkout "
                           "and review bundle margins monthly.",
                    impact=f"{m['pair_n']:,} orders ({frac(m['pair_support'])}) already contain this pair, out of {m['n_ante']:,} orders with {q(m['ante'])}. Scenario: a "
                           f"{SA['attach_rate_lift']:.0%} relative lift in the attach rate adds about {extra:,.0f} bundle orders over the data period.",
                    priority="Medium" if m["lift"] >= 1.5 else "Low", phase=1, owner="Merchandising / Product",
                    kpi=f"Attach rate of {q(m['cons'])} on {q(m['ante'])} orders {frac(m['conf'])} -> {frac(m['conf'] * (1 + SA['attach_rate_lift']))}")

    def r12(m):
        top = m["top"]
        inc = top["revenue"] * SA["champion_frequency_lift"]
        return dict(target=f"Cluster {q(top['name'])} ({top['customers']:,.0f} customers = {frac(top['cust_share'])}; {frac(top['rev_share'])} of revenue)",
                    action=f"Treat this cluster as the advocacy engine: referral rewards, early access and limited editions, community or beta-tester invitations, and premium bundles "
                           f"designed around its basket profile (AOV {money(top['avg_aov'])}).",
                    impact=f"Generates {frac(top['rev_share'])} of revenue from {frac(top['cust_share'])} of customers. Scenario: +{SA['champion_frequency_lift']:.0%} purchase "
                           f"frequency adds about {money(inc)} (+{inc / total_rev:.1%} of revenue).",
                    priority="Medium", phase=3, owner="Loyalty / Brand",
                    kpi=f"Average orders per customer in the cluster {top['avg_orders']:.1f} -> {top['avg_orders'] * (1 + SA['champion_frequency_lift']):.1f}")

    def r13(m):
        st_ = m["stale"]
        inc = st_["customers"] * SA["reactivation_rate"] * st_["avg_aov"]
        return dict(target=f"Cluster {q(st_['name'])} ({st_['customers']:,.0f} customers = {frac(st_['cust_share'])}; mean recency {st_['avg_recency']:.0f} days)",
                    action=f"Run a staged win-back ladder: (1) a personalised 'we miss you' message with best-sellers, (2) a time-limited incentive sized to the cluster's AOV "
                           f"({money(st_['avg_aov'])}), (3) a final offer; stop contacting non-responders after three attempts to avoid wasted spend and collect exit feedback.",
                    impact=f"The cluster holds {frac(st_['cust_share'])} of customers and {frac(st_['rev_share'])} of revenue. Scenario: re-activating {SA['reactivation_rate']:.0%} "
                           f"with one order at the cluster AOV adds about {money(inc)} (+{inc / total_rev:.1%} of revenue); suppressing non-responders also cuts campaign cost.",
                    priority=("High" if st_["rev_share"] >= 0.10 else "Medium" if (st_["cust_share"] >= 0.10 or st_["rev_share"] >= 0.05) else "Low"),
                    phase=2, owner="CRM", kpi=f"Re-activated customers 0 -> {st_['customers'] * SA['reactivation_rate']:,.0f} ({SA['reactivation_rate']:.0%} of the cluster)")

    specs = [("R01", "F01", r01), ("R02", "F02", r02), ("R03", "F03", r03), ("R04", "F04", r04), ("R05", "F05", r05), ("R06", "F06", r06),
             ("R07", "F07", r07), ("R08", "F08", r08), ("R09", "F09", r09), ("R10", "F10", r10), ("R11", "F11", r11), ("R12", "F12", r12),
             ("R13", "F13", r13)]
    out = []
    for rid, fid, fn in specs:
        fm = K.get(fid)
        try:
            if fm is None:
                raise ValueError(f"finding {fid} has no evidence")
            rec = fn(fm)
        except Exception:                                                    # noqa: BLE001 - flag, never break
            rec = dict(target="n/a", action="Widen the sidebar selection so that the linked finding can be computed, then this recommendation is generated automatically.",
                       impact="n/a", priority="Low", phase=3, owner="Data / BI team", kpi="n/a")
        rec.update(id=rid, finding=fid)
        out.append(rec)
    return out


# ===================================================================================================
# 21.  TAB 7  -  Findings, recommendations, exports and glossary  (Sections 13-17, Appendix B)
#      Members 13 (Rudra Pratap Singh), 14 (Ayan Khan) and 1 (Hasnain Raza Khan)
# ===================================================================================================
PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}
ASSUMPTION_LIMITS = {"vip_upgrade_rate": (0, 50), "category_lift": (0, 30), "second_order_rate": (0, 30), "peak_month_lift": (0, 30),
                     "top_customer_attrition": (0, 30), "reactivation_rate": (0, 50), "retention_lift_pp": (0, 30),
                     "attach_rate_lift": (0, 50), "champion_frequency_lift": (0, 30)}


def reset_scenario():
    """Callback of the 'reset assumptions' button: drop the slider states so that the defaults return."""
    for key in [k for k in st.session_state.keys() if str(k).startswith("sa_")]:
        del st.session_state[key]


def scenario_assumptions_editor():
    """Sliders for the nine what-if assumptions of Section 14 (defaults = the notebook values)."""
    SA = {}
    with st.expander("Edit the scenario assumptions (what-if) - every revenue effect below is illustrative, not a forecast"):
        cols = st.columns(3)
        for i, (key, default) in enumerate(DEFAULT_SCENARIO.items()):
            lo, hi = ASSUMPTION_LIMITS[key]
            unit = "percentage points" if key.endswith("_pp") else "%"
            used_in, meaning = ASSUMPTION_NOTES[key]
            with cols[i % 3]:
                val = st.slider(f"{key} ({used_in})", lo, hi, int(round(default * 100)), 1, key=f"sa_{key}",
                                help=f"{meaning} - in {unit}")
            SA[key] = val / 100.0
        st.button("Reset assumptions to the notebook defaults", on_click=reset_scenario)
    return SA


def render_tab7(ctx):
    """Tab 7 container."""
    st.markdown("## Findings, Recommendations, Exports and Glossary")
    st.caption("Sections 13-17 and Appendix B | Member 13 - Rudra Pratap Singh | Member 14 - Ayan Khan | Member 1 - Hasnain Raza Khan")
    cust7 = ctx.get("cust_k") if ctx.get("cust_k") is not None else ctx["cust"]
    fit7 = ctx.get("km_fit_k") or ctx.get("km_fit")
    names7 = fit7["names"] if fit7 else {}
    K, rows = compute_findings(ctx["tx"], cust7, names7, ctx["bundle"])
    ctx["K"], ctx["finding_rows"], ctx["cust7"] = K, rows, cust7
    t13, t14, t15, t16, t17 = st.tabs(["13. Key findings", "14. Recommendations", "15-16.1 Conclusion & future scope",
                                       "16.2 Export", "App. B Glossary & 17 References"])
    with t13:
        run_safely(render_sec13, ctx)
    with t14:
        run_safely(render_sec14, ctx)
    with t15:
        run_safely(render_sec15_16, ctx)
    with t16:
        run_safely(render_sec16_2, ctx)
    with t17:
        run_safely(render_appendix_b, ctx)


def render_sec13(ctx):
    """Section 13: business-question map, 13 evidence-based findings, KPI snapshot, executive summary, coverage check."""
    K, rows, tx, cust = ctx["K"], ctx["finding_rows"], ctx["tx"], ctx["cust7"]
    section_header("Section 13: Consolidated Key Findings", "Member 13 - Rudra Pratap Singh | all numbers are computed live from the active selection")
    caption_md("Table 13.1: Business-question map")
    md_table(pd.DataFrame([(k, qn, s) for k, (qn, s) in BQ_CONFIG.items()], columns=["Business question", "What we want to know", "Evidence produced in"]))
    rev_gap = (float(tx["total_amount"].sum()) - float(cust["total_spend"].sum())) / max(float(cust["total_spend"].sum()), 1e-9)
    interpret(f"The nine business questions are answered from {len(tx):,} transaction lines and {len(cust):,} customers; transaction revenue and customer "
              f"total_spend differ by {rev_gap:+.4%}, so every share and ratio below rests on one reconciled base.",
              "Each finding carries one business question, the section that produced the evidence and the exact figures behind it.")
    df = pd.DataFrame([{"ID": r["id"], "BQ": r["bq"], "Section": r["section"], "Key finding": r["finding"], "Numerical evidence": r["evidence"],
                        "Business implication": r["implication"]} for r in rows])
    caption_md("Table 13.2: Thirteen evidence-based key findings (F01-F13)")
    md_table(df)
    ok = [r for r in rows if not r["finding"].startswith("Evidence unavailable")]
    interpret(f"{len(ok)} of {len(rows)} findings could be computed for the current selection"
              f"{'' if len(ok) == len(rows) else ' (the others need a wider selection, see the evidence column)'}.",
              (f"The strongest concentration signal is F08: {K['F08']['top20']:.1%} of revenue from the top 20% of customers, and the strongest behaviour "
               f"signal is F05: repeat buyers are {K['F05']['spend_ratio']:.1f}x as valuable as one-time buyers." if K.get("F08") and K.get("F05")
               else "Some of the headline findings are unavailable for the current selection."))

    # ---- 13.3 KPI snapshot, executive summary, coverage ----
    kpi = []
    if K.get("F02"):
        kpi += [("Customers", f"{K['F02']['customers']:,}", "Section 6"), ("Total revenue", money(K["F02"]["revenue"]), "Section 6"),
                ("Mean vs median spend per customer", f"{money(K['F02']['mean_spend'])} vs {money(K['F02']['median_spend'])}", "Section 6")]
    if K.get("F01"):
        kpi.append(("Orders", f"{K['F01']['orders']:,}", "Section 6"))
    if K.get("F03"):
        kpi.append((f"Revenue share of '{K['F03']['top']['segment']}' spend segment", frac(K["F03"]["top"]["rev_share"]), "Sections 7-8"))
    if K.get("F05"):
        kpi += [("Repeat-buyer rate", frac(K["F05"]["repeat_rate"]), "Section 10"), ("Revenue from repeat buyers", frac(K["F05"]["repeat_rev_share"]), "Section 10")]
    if K.get("F06"):
        kpi.append(("Strongest calendar month", f"{month_name(K['F06']['peak_moy'])} ({K['F06']['peak_idx']:.2f}x average)", "Section 11"))
    if K.get("F08"):
        kpi.append(("Revenue from top 20% of customers", frac(K["F08"]["top20"]), "Section 12.1"))
    if K.get("F10"):
        kpi.append(("Month-1 retention of new customers", frac(K["F10"]["ret_m1"]), "Section 12.3"))
    if K.get("F12"):
        kpi.append(("K-Means clusters (value cluster revenue share)", f"{K['F12']['k']} ({frac(K['F12']['top']['rev_share'])})", "Section 12.5"))
    show_df(pd.DataFrame(kpi, columns=["KPI", "Value", "Source"]), caption="Table 13.3: Headline KPI snapshot", hide_index=True)
    interpret(f"The snapshot condenses {len(kpi)} headline KPIs from Sections 6 to 12.5 into one table"
              + (f"; for example {money(K['F02']['revenue'])} of revenue from {K['F02']['customers']:,} customers" if K.get("F02") else "")
              + (f", with {frac(K['F08']['top20'])} of it coming from the top 20% of customers." if K.get("F08") else "."),
              "Because every value is computed from the active selection, the snapshot is the quickest way to see how a filter change moves the business picture.")
    st.markdown("**Executive summary**")
    bullets = []
    if K.get("F08") and K.get("F03"):
        lead = "Revenue is concentrated" if K["F08"]["top20"] >= 0.60 else "Revenue is spread more evenly than the 80/20 rule"
        bullets.append(f"{lead}: the top 20% of customers generate {frac(K['F08']['top20'])} of revenue and the {q(K['F03']['top']['segment'])} spend segment "
                       f"delivers {frac(K['F03']['top']['rev_share'])} from {frac(K['F03']['top']['cust_share'])} of customers (F03, F08).")
    if K.get("F05"):
        bullets.append(f"Repeat behaviour drives value: {frac(K['F05']['repeat_rate'])} of customers buy more than once and provide "
                       f"{frac(K['F05']['repeat_rev_share'])} of revenue; a repeat buyer is worth {K['F05']['spend_ratio']:.1f}x a one-time buyer (F05)"
                       + (f", yet only {frac(K['F10']['ret_m1'])} of new customers return the next month (F10)." if K.get("F10") else "."))
    if K.get("F04") and K.get("F06"):
        bullets.append(f"Product and seasonal mix: {q(K['F04']['top_cat'])} leads the product mix ({frac(K['F04']['top_cat_share'])} of revenue) and "
                       f"{month_name(K['F06']['peak_moy'])} peaks at {K['F06']['peak_idx']:.2f}x the average month (F04, F06).")
    if K.get("F09") and K.get("F12"):
        bullets.append(f"Customer groups differ sharply: K-Means cluster {q(K['F12']['top']['name'])} holds {frac(K['F12']['top']['rev_share'])} of revenue from "
                       f"{frac(K['F12']['top']['cust_share'])} of customers; the top RFM segment {q(K['F09']['top']['name'])} holds {frac(K['F09']['top']['rev_share'])} "
                       f"from {frac(K['F09']['top']['cust_share'])} (F09, F12).")
    if K.get("F13"):
        bullets.append(f"Inactivity pool: {q(K['F13']['stale']['name'])} contains {frac(K['F13']['stale']['cust_share'])} of customers with "
                       f"{K['F13']['stale']['avg_recency']:.0f} days since the last purchase (F13).")
    if K.get("F11"):
        bullets.append(f"Cross-sell signal: {q(K['F11']['ante'])} -> {q(K['F11']['cons'])} has lift {K['F11']['lift']:.2f}, while {frac(K['F11']['multi_share'])} "
                       f"of orders hold 2+ products (F11).")
    if K.get("F01"):
        gap = abs(K["F01"]["recon_gap"] or 0)
        bullets.append(f"Data foundation: {K['F01']['customers']:,} customers and {K['F01']['orders']:,} orders; transaction revenue vs customer_df differs by "
                       f"{gap:.3%}" + (" - reconciled (F01)." if gap < 0.01 else " - needs review (F01)."))
    for b in bullets:
        st.markdown("- " + b.replace("$", r"\$"))
    if not bullets:
        st.info("The executive summary needs at least some computable findings; widen the sidebar selection.")
    cov = pd.DataFrame(rows).assign(ok=lambda x: ~x["finding"].str.startswith("Evidence unavailable"))
    cov = (cov.groupby("bq").agg(findings=("id", "count"), with_evidence=("ok", "sum"), ids=("id", lambda x: ", ".join(x))).reset_index()
           .rename(columns={"bq": "Business question"}))
    cov["Section(s)"] = cov["Business question"].map(lambda b: BQ_CONFIG.get(b, ("", ""))[1])
    show_df(cov, caption="Coverage check: every business question needs at least one evidence-backed finding", hide_index=True)
    all_ok = bool((cov["with_evidence"] >= 1).all()) and set(BQ_CONFIG) <= set(cov["Business question"])
    interpret(f"{int(cov['with_evidence'].sum())} evidence-backed findings cover {int((cov['with_evidence'] >= 1).sum())} of {len(BQ_CONFIG)} business questions"
              f"{' - every question is answered.' if all_ok else ' - at least one question has no evidence for this selection.'}",
              "The executive summary lines are generated from the same live metrics, so they change whenever a sidebar filter changes.")


def render_sec14(ctx):
    """Section 14: prioritised recommendation matrix, roadmap, scenario assumptions and risks."""
    K, cust = ctx["K"], ctx["cust7"]
    section_header("Section 14: Business Recommendations", "Member 13 - Rudra Pratap Singh | one linked recommendation per finding")
    SA = scenario_assumptions_editor()
    total_rev, mean_aov = float(cust["total_spend"].sum()), float(cust["avg_order_value"].mean())
    recs = build_recommendations(K, SA, total_rev, mean_aov)
    ctx["recs"] = recs
    caption_md("How priority is set")
    md_table(pd.DataFrame(PRIORITY_MEANING, columns=["Priority", "Meaning"]))
    ordered = sorted(recs, key=lambda r: (PRIORITY_RANK[r["priority"]], r["id"]))
    matrix = pd.DataFrame([{"ID": r["id"], "Linked finding": f"{r['finding']}: {FINDING_TOPIC.get(r['finding'], '')}",
                            "Target customer segment / category": r["target"], "Concrete action plan": r["action"],
                            "Expected business impact": r["impact"], "Priority": f"**{r['priority']}**"} for r in ordered])
    caption_md("Table 14.1: Prioritised recommendation matrix (R01-R13)")
    md_table(matrix)
    by_p = {p: [r["id"] for r in recs if r["priority"] == p] for p in ["High", "Medium", "Low"]}
    interpret("Priority summary: " + " | ".join(f"{p}: {', '.join(v) or '-'}" for p, v in by_p.items()) + ".",
              f"The active base is {len(cust):,} customers with {money(total_rev)} of revenue and an average AOV of {money(mean_aov)}; scenario effects multiply an assumed "
              f"rate by these measured bases, so they scale with the sidebar selection and with the sliders above.",
              "Scenario effects overlap (R03, R08 and R12 partly address the same customers; R09 and R13 partly the same inactive customers) and must not be added up.")

    road = pd.DataFrame([{"Phase": PHASES[r["phase"]][0], "Timeline": PHASES[r["phase"]][1], "Rec. ID": r["id"], "Priority": r["priority"],
                          "Owner": r["owner"], "KPI: baseline -> scenario target": r["kpi"]}
                         for r in sorted(recs, key=lambda r: (r["phase"], PRIORITY_RANK[r["priority"]], r["id"]))])
    caption_md("Table 14.2: Roadmap - phase, owner and KPI (baseline -> scenario target)")
    md_table(road)
    md_table(pd.DataFrame(PHASE_LOGIC, columns=["Phase", "Timeline", "Logic"]))
    interpret(" ".join(f"{PHASES[ph][0].split(' - ')[0]} ({PHASES[ph][1]}): " + (", ".join(f"{r['id']} ({r['priority']})" for r in sorted(recs, key=lambda r: (PRIORITY_RANK[r['priority']], r['id'])) if r['phase'] == ph) or "-") + "."
                       for ph in (1, 2, 3)),
              "Cheap, fast actions deliver evidence first, and the larger programmes of Phases 2 and 3 build on that evidence and on the cleaner data and customer map.")

    assump = pd.DataFrame([(k, (f"{v * 100:.0f} pp" if k.endswith("_pp") else f"{v:.0%}"), *ASSUMPTION_NOTES.get(k, ("", ""))) for k, v in SA.items()],
                          columns=["Assumption", "Value", "Used in", "Meaning"])
    caption_md("Table 14.3: Scenario assumptions used in the recommendations (editable above)")
    md_table(assump)
    interpret(f"Base figures used: total revenue {money(total_rev)}, mean AOV {money(mean_aov)} and {len(cust):,} customers.",
              "The risks below explain why the scenario numbers are planning aids, not predictions.")
    st.markdown("**Risks and limitations**")
    for r in RISKS:
        st.markdown("- " + r)
    st.markdown("**Measuring success.** Review the KPIs of Table 14.2 weekly in Phase 1 and fortnightly afterwards, compare every programme with its holdout "
                "group, and refresh this dashboard after each data load so findings, priorities and baselines stay current.")


def render_sec15_16(ctx):
    """Sections 15 and 16.1: conclusion, limitations (with the synthetic-data disclaimer) and future scope."""
    K, tx, cust, bundle = ctx["K"], ctx["tx"], ctx["cust7"], ctx["bundle"]
    section_header("Section 15: Conclusion and Limitations", "Member 14 - Ayan Khan")
    st.markdown("#### 15.1 Overall study conclusion")
    st.markdown(CONCLUSION_TEXT)
    md_table(pd.DataFrame(CONCLUSION_PHASES, columns=["Phase", "Sections", "What it contributed"]))
    st.markdown(CONCLUSION_SYNTHESIS)
    pieces = []
    if K.get("F08"):
        pieces.append(f"the top 20% of customers generate {frac(K['F08']['top20'])} of revenue")
    if K.get("F05"):
        pieces.append(f"{frac(K['F05']['repeat_rate'])} of customers are repeat buyers worth {K['F05']['spend_ratio']:.1f}x a one-time buyer")
    if K.get("F10"):
        pieces.append(f"only {frac(K['F10']['ret_m1'])} of new customers return in month 1")
    if K.get("F12"):
        pieces.append(f"K-Means finds {K['F12']['k']} behavioural clusters")
    interpret(f"The conclusion rests on {len(tx):,} order lines and {len(cust):,} customers: " + ("; ".join(pieces) if pieces else "the findings need a wider selection") + ".",
              f"Spend tiers, RFM and K-Means act as cross-checks: the RFM top segment and the K-Means value cluster hold "
              f"{frac(K['F09']['top']['rev_share']) if K.get('F09') else 'n/a'} and {frac(K['F12']['top']['rev_share']) if K.get('F12') else 'n/a'} of revenue respectively.")

    st.markdown("#### 15.2 Methodological and data limitations")
    st.warning(SYNTHETIC_DISCLAIMER + ("" if bundle["mode"] == "synthetic" else
                                       " (In this session cleaned tables were loaded from files, so the disclaimer applies to the project's reference dataset; "
                                       "validate the findings separately on your own data.)"))
    for i, item in enumerate(LIMITATIONS, start=1):
        st.markdown(f"{i}. {item}")
    last_win = tx["transaction_date"].max() - pd.Timedelta(days=90)
    n_recent = int((cust["first_purchase_date"] > last_win).sum())
    rets = bundle["df_returns"]
    interpret(f"{n_recent:,} of the {len(cust):,} customers ({100 * n_recent / max(len(cust), 1):.1f}%) made their first purchase within the last 90 days of the window, "
              f"so their repeat behaviour is still censored (limitation 2).",
              f"The cleaning step set aside {len(rets):,} return / cancellation rows worth {money(abs(pd.to_numeric(rets['total_amount'], errors='coerce').sum()), 0)} "
              f"that are therefore not analysed (limitation 3); every customer is assumed to be one person (limitation 7).")

    section_header("Section 16.1: Future Scope and Analytical Extensions", "Member 14 - Ayan Khan")
    for item in FUTURE_SCOPE:
        with st.expander(f"{item['title']}  |  horizon: {item['horizon']}"):
            st.markdown(f"**Objective.** {item['objective']}")
            st.markdown(f"**Method.** {item['method']}")
            st.markdown(f"**Reuses.** {item['reuses']}")
            st.markdown(f"**Validation.** {item['validation']}")
            st.markdown(f"**Business use.** {item['use']}")
    st.markdown("**Enablers.** " + FUTURE_ENABLERS)
    n_hist = int((cust["num_orders"] >= 3).sum())
    stale = K["F13"]["stale"] if K.get("F13") else None
    interpret(f"For predictive CLV, {n_hist:,} customers ({100 * n_hist / max(len(cust), 1):.1f}%) already have three or more orders, which is the kind of purchase history "
              f"that BG/NBD and Gamma-Gamma models need to estimate purchase and value rates.",
              (f"For churn prediction, the inactive cluster {q(stale['name'])} ({stale['customers']:,.0f} customers, {stale['avg_recency']:.0f} days since the last purchase) "
               f"is the natural source of churned examples; A/B tests should then confirm which win-back offer works." if stale else
               "Churn prediction needs a defined inactivity window; A/B tests should then confirm which win-back offer works."))


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def render_sec16_2(ctx):
    """Section 16.2: reproducible export of the final analytical tables with an export manifest."""
    bundle, tx = ctx["bundle"], ctx["tx"]
    section_header("Section 16.2: Reproducible Export of the Final Analytical Tables", "Member 14 - Ayan Khan")
    st.markdown("The three tables below are the project's deliverables. They are exported with `index=False` (no key column is lost) and verified by "
                "reading them back, exactly as in the notebook.")
    scope = st.radio("Export scope", ["Full dataset (project deliverable)", "Current filtered view"], horizontal=True, key="export_scope")
    if scope.startswith("Full"):
        clean_df, log_df = bundle["df_clean"], bundle["cleaning_log"]
        master = build_master_table(bundle["df_clean"])
    else:
        clean_df, log_df = tx, bundle["cleaning_log"]
        master = ctx["cust7"]
    files = [("cleaned_transactions.csv", clean_df), ("customer_analytics_master.csv", master), ("data_cleaning_log.csv", log_df)]
    payloads = {name: csv_bytes(frame) for name, frame in files}
    records = []
    for name, frame in files:
        data = payloads[name]
        rows_back = len(pd.read_csv(io.BytesIO(data), usecols=[0], skip_blank_lines=False))
        on_disk = find_on_disk(name)
        records.append({"file_name": name, "path": f"outputs/tables/{name}", "rows": rows_back, "columns": frame.shape[1],
                        "size_kb": round(len(data) / 1024, 2), "status": "OK" if rows_back == len(frame) else "ROW MISMATCH",
                        "already on disk": "yes" if on_disk is not None else "no"})
    manifest = pd.DataFrame(records)
    cols = st.columns(3)
    for col, (name, _) in zip(cols, files):
        with col:
            st.download_button(f"Download {name}", data=payloads[name], file_name=name, mime="text/csv", key=f"dl_{name}")
    show_df(manifest, caption="Export manifest (rows verified by reading each file back)", hide_index=True)
    n_ok = int((manifest["status"] == "OK").sum())
    interpret(f"{n_ok} of {len(manifest)} files pass the read-back check; the export holds {len(clean_df):,} transaction rows, {len(master):,} customer rows "
              f"({master.shape[1]} columns including spend_segment, buyer_type, rfm_segment and cluster) and {len(log_df)} cleaning-log steps.",
              f"Total revenue in the exported transactions is {money(clean_df['total_amount'].sum(), 2)} and total_spend in the customer master is "
              f"{money(master['total_spend'].sum(), 2)}, so the two files reconcile.")
    if st.button("Save the three files to outputs/tables/ on the machine that runs the app"):
        try:
            target = Path.cwd() / "outputs" / "tables"
            target.mkdir(parents=True, exist_ok=True)
            for name, _ in files:
                (target / name).write_bytes(payloads[name])
            st.success(f"Saved {len(files)} files to {target}. The next time the app starts it loads them automatically.")
        except Exception as exc:                                           # noqa: BLE001
            st.error(f"The files could not be written ({type(exc).__name__}: {exc}); use the download buttons instead.")
    with st.expander("Preview of customer_analytics_master.csv"):
        show_df(display_dates(master.head(15)), hide_index=True)


def render_appendix_b(ctx):
    """Appendix B glossary (26 terms) and Section 17 references (APA 7th edition)."""
    K, cust = ctx["K"], ctx["cust7"]
    section_header("Appendix B: Glossary of Technical and Business Terms", "Member 14 - Ayan Khan")
    query = st.text_input("Search the glossary (term or definition)", key="glossary_query")
    gdf = pd.DataFrame(GLOSSARY, columns=["Term", "Definition", "Used in"])
    if query.strip():
        mask = gdf["Term"].str.contains(query, case=False, regex=False) | gdf["Definition"].str.contains(query, case=False, regex=False)
        gdf = gdf[mask]
    md_table(gdf.assign(Term="**" + gdf["Term"] + "**"))
    if gdf.empty:
        st.info("No glossary entry matches the search text.")
    live = []
    if K.get("F11"):
        live.append(f"the strongest market-basket rule has lift {K['F11']['lift']:.2f} and confidence {frac(K['F11']['conf'])}")
    if K.get("F08"):
        live.append(f"the Pareto top-20% share is {frac(K['F08']['top20'])}")
    interpret(f"The glossary defines {len(GLOSSARY)} terms ({len(gdf)} shown for the current search); the formulas in the definitions are the ones used by "
              f"this dashboard.",
              ("In the active data " + " and ".join(live) + ", which are the live values of the terms Lift, Confidence and Pareto 80/20 Rule." if live
               else "Widen the sidebar selection to see live values for the Lift, Confidence and Pareto terms."))

    section_header("Section 17: References (APA 7th edition)", "Member 1 - Hasnain Raza Khan")
    for ref in sorted(REFERENCES, key=lambda s: s.lower()):
        st.markdown(f"- {ref}")
    caption_md("Where each source is used")
    md_table(pd.DataFrame(REFERENCE_USE, columns=["Topic", "Source(s)", "What it supports", "Sections"]))
    st.markdown("**How to cite**")
    for item in CITATION_GUIDE:
        st.markdown("- " + item)
    interpret(f"{len(REFERENCES)} sources support the methods: for example Koch (1998) for the 80/20 result of {frac(K['F08']['top20']) if K.get('F08') else 'n/a'} revenue from "
              f"the top 20% of {len(cust):,} customers, and Agrawal et al. (1993) for the market-basket measures.",
              "All references are listed alphabetically in APA 7th edition style; software documentation uses 'n.d.' where no date is given.")


# ===================================================================================================
# 22.  MAIN  -  page set-up, sidebar (uploads + global filters), tabs
# ===================================================================================================
APP_CSS = """
<style>
div[data-testid="stMetric"] {background: rgba(31,78,121,0.07); border: 1px solid rgba(128,128,128,0.25); border-radius: 10px; padding: 10px 14px;}
div[data-testid="stMetricLabel"] p {font-weight: 600; white-space: normal;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
div[data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip;}
.block-container {padding-top: 2.2rem;}
</style>
"""


def reset_filters():
    """Callback of the 'Reset filters' button: removing the widget states restores every default."""
    for key in [k for k in st.session_state.keys() if str(k).startswith("flt_")]:
        del st.session_state[key]


def get_bundle(uploads):
    """Load the data once per session (upload > disk > synthetic); reload only when the uploads change."""
    payload = {key: (f.getvalue() if f is not None else None) for key, f in uploads.items()}
    sig = tuple((k, hashlib.md5(v).hexdigest() if v else None) for k, v in payload.items())
    if st.session_state.get("_bundle_sig") != sig or "_bundle" not in st.session_state:
        st.session_state["_bundle"] = load_bundle(payload)
        st.session_state["_bundle_sig"] = sig
    return st.session_state["_bundle"], sig


def sidebar_filters(bundle, sig):
    """Global filters: date range, category, spend segment and buyer type."""
    full = bundle["df_clean"]
    dmin, dmax = full["transaction_date"].min().date(), full["transaction_date"].max().date()
    cats = sorted(full["category"].unique())
    token = hashlib.md5(repr((sig, len(full), str(dmin), str(dmax), len(cats))).encode()).hexdigest()[:8]   # new data -> fresh widgets
    st.sidebar.markdown("### Global filters")
    st.sidebar.caption("Every chart and table in all seven tabs follows these selections (Sections 4-5 always show the full raw-to-clean pipeline).")
    picked = st.sidebar.date_input("Date range (transaction_date)", value=(dmin, dmax), min_value=dmin, max_value=dmax, key=f"flt_date_{token}")
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = picked
    elif isinstance(picked, (tuple, list)) and len(picked) == 1:
        start, end = picked[0], dmax
    else:
        start, end = picked, dmax
    categories = st.sidebar.multiselect("Category", cats, default=cats, key=f"flt_cat_{token}")
    segments = st.sidebar.multiselect("Spend segment", SEGMENT_ORDER, default=SEGMENT_ORDER, key=f"flt_seg_{token}")
    buyer_types = st.sidebar.multiselect("Buyer type", RP_GROUPS, default=RP_GROUPS, key=f"flt_buy_{token}")
    st.sidebar.button("Reset filters", on_click=reset_filters)
    with st.sidebar.expander("How the filters work"):
        st.markdown("- **Date range** and **Category** define the transactions that are analysed.\n"
                    "- Inside that scope the customer table and the **Low / Medium / High** (tertiles of total spend) and **1-Time / Repeat** "
                    "labels are computed with the notebook logic; recency uses the last date in the scope + 1 day.\n"
                    "- **Spend segment** and **Buyer type** keep only the chosen labelled customers.\n"
                    "- With the default selection every number equals the notebooks' output.")
    return {"start": start, "end": end, "categories": categories, "segments": segments, "buyer_types": buyer_types}


def main():
    """Entry point: build the page, load the data, apply the global filters and render the seven tabs."""
    st.set_page_config(page_title="Customer Spend, Product & Sales Analytics", page_icon="\U0001F4CA", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.sidebar.title("Customer Analytics")
    st.sidebar.caption("14-member college project | RANDOM_STATE = 42 | " + CURRENCY_SYMBOL + " INR")
    with st.sidebar.expander("Data source: upload the exported CSV files (optional)", expanded=False):
        st.caption("Without uploads the app reads outputs/tables/*.csv from disk and falls back to synthetic data.")
        uploads = {"df_clean": st.file_uploader("cleaned_transactions.csv (df_clean)", type=["csv"], key="up_clean"),
                   "customer_df": st.file_uploader("customer_analytics_master.csv (customer_df)", type=["csv"], key="up_cust"),
                   "cleaning_log": st.file_uploader("data_cleaning_log.csv (cleaning_log)", type=["csv"], key="up_log")}
        if st.button("Reload data files"):
            st.session_state.pop("_bundle", None)
    bundle, sig = get_bundle(uploads)
    flt = sidebar_filters(bundle, sig)
    st.sidebar.markdown("---")
    st.sidebar.caption("Data: " + bundle["origins"]["df_clean"])

    st.title(PROJECT_TITLE)
    st.caption(PROJECT_SUBTITLE)
    if not (flt["categories"] and flt["segments"] and flt["buyer_types"]):
        st.warning("Select at least one value in every filter (Category, Spend segment and Buyer type) to see the analysis.")
        st.stop()
    act = build_active_dataset(bundle, flt)
    if act["cust"] is None or act["tx"].empty or len(act["cust"]) == 0:
        st.warning("No transactions match the current filters. Widen the date range or the selections in the sidebar.")
        st.stop()
    cust, rfm, km_sel, km_fit = attach_advanced_labels(act["cust"])
    ctx = {"bundle": bundle, "flt": flt, "tx": act["tx"], "cust": cust, "rfm": rfm, "km_sel": km_sel, "km_fit": km_fit,
           "thresholds": act["thresholds"], "snapshot": act["snapshot"], "narrowed": act["narrowed"],
           "used_provided": act["used_provided"], "k_final": None}
    st.caption(f"Active selection: {len(act['tx']):,} order lines | {len(cust):,} customers | {flt['start']:%d %b %Y} to {flt['end']:%d %b %Y} | "
               f"{len(flt['categories'])} categories | recency snapshot {act['snapshot']:%d %b %Y}")
    if len(cust) < 20:
        st.info("The selection is small (fewer than 20 customers); some analyses show warnings or are skipped.")

    tabs = st.tabs(["1. Overview & Data", "2. Quality & Cleaning", "3. Customers & Segments", "4. Products, Repeat & Seasonality",
                    "5. Pareto, RFM, Cohort & Basket", "6. K-Means Clustering", "7. Findings & Export"])
    renderers = [render_tab1, render_tab2, render_tab3, render_tab4, render_tab5, render_tab6, render_tab7]
    for tab, fn in zip(tabs, renderers):
        with tab:
            run_safely(fn, ctx)
    st.markdown("---")
    st.caption("Customer Spend, Product & Sales Analytics | Sections 1-17 and Appendices A-B | all 14 members | built with Streamlit, pandas, "
               "NumPy, Matplotlib, Seaborn and scikit-learn")


if __name__ == "__main__":
    main()
