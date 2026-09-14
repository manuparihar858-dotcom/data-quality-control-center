import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Data Quality Control Center",
    page_icon="🧹",
    layout="wide"
)

@st.cache_data
def create_sample_data():
    rng = np.random.default_rng(42)

    n = 1000
    customers = pd.DataFrame({
        "Customer ID": [f"C{i+1:04d}" for i in range(n)],
        "Customer Name": [f"Customer {i+1}" for i in range(n)],
        "Email": [f"customer{i+1}@example.com" for i in range(n)],
        "Age": rng.integers(18, 71, n),
        "Region": rng.choice(["North", "South", "East", "West"], n),
        "Signup Date": pd.date_range("2025-01-01", periods=n, freq="D")
    })
    missing_idx = rng.choice(n, 28, replace=False)
    customers.loc[missing_idx[:10], "Email"] = np.nan
    customers.loc[missing_idx[10:18], "Region"] = np.nan
    customers.loc[missing_idx[18:], "Age"] = np.nan
    bad_email_idx = rng.choice(n, 18, replace=False)
    customers.loc[bad_email_idx, "Email"] = "invalid-email"
    customers = pd.concat(
        [customers, customers.iloc[rng.choice(n, 22, replace=False)]],
        ignore_index=True
    )

    n2 = 1200
    sales = pd.DataFrame({
        "Order ID": [f"O{i+1:05d}" for i in range(n2)],
        "Customer ID": [f"C{rng.integers(1, n+1):04d}" for _ in range(n2)],
        "Product": rng.choice(["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"], n2),
        "Quantity": rng.integers(1, 8, n2),
        "Unit Price": rng.integers(500, 95000, n2),
        "Order Date": pd.date_range("2025-01-01", periods=n2, freq="12h")
    })
    sales["Sales Amount"] = sales["Quantity"] * sales["Unit Price"]
    sales.loc[rng.choice(n2, 25, replace=False), "Customer ID"] = np.nan
    sales.loc[rng.choice(n2, 15, replace=False), "Quantity"] = 0
    sales.loc[rng.choice(n2, 14, replace=False), "Unit Price"] = -500
    sales = pd.concat(
        [sales, sales.iloc[rng.choice(n2, 18, replace=False)]],
        ignore_index=True
    )

    n3 = 900
    orders = pd.DataFrame({
        "Order ID": [f"ORD{i+1:05d}" for i in range(n3)],
        "Customer ID": [f"C{rng.integers(1, n+1):04d}" for _ in range(n3)],
        "Status": rng.choice(["Completed", "Pending", "Cancelled", "Returned"], n3),
        "Payment Method": rng.choice(["UPI", "Card", "Cash", "Net Banking"], n3),
        "Amount": rng.integers(500, 75000, n3).astype(float)
    })
    orders.loc[rng.choice(n3, 20, replace=False), "Status"] = np.nan
    orders.loc[rng.choice(n3, 12, replace=False), "Payment Method"] = "unknown"
    orders.loc[rng.choice(n3, 10, replace=False), "Amount"] = -100
    orders = pd.concat(
        [orders, orders.iloc[rng.choice(n3, 16, replace=False)]],
        ignore_index=True
    )

    return {
        "Customer Records": customers,
        "Sales Records": sales,
        "Orders Records": orders
    }


def read_uploaded_file(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def normalize_columns(df):
    result = df.copy()
    result.columns = [
        str(c).strip().replace("_", " ").replace("-", " ")
        for c in result.columns
    ]
    return result


def quality_report(df):
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    formatting = 0
    invalid = 0

    email_cols = [c for c in df.columns if str(c).lower() in {"email", "email address"}]
    age_cols = [c for c in df.columns if str(c).lower() in {"age"}]
    quantity_cols = [c for c in df.columns if str(c).lower() in {"quantity", "qty"}]
    amount_cols = [c for c in df.columns if str(c).lower() in {"amount", "sales amount", "unit price"}]

    if email_cols:
        col = email_cols[0]
        formatting += int((
            df[col].notna() &
            ~df[col].astype(str).str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)
        ).sum())

    if age_cols:
        invalid += int((
            pd.to_numeric(df[age_cols[0]], errors="coerce").notna() &
            ~pd.to_numeric(df[age_cols[0]], errors="coerce").between(0, 120)
        ).sum())

    for col in quantity_cols:
        invalid += int((pd.to_numeric(df[col], errors="coerce").fillna(1) <= 0).sum())

    for col in amount_cols:
        invalid += int((pd.to_numeric(df[col], errors="coerce").fillna(1) < 0).sum())

    issues = missing + duplicates + formatting + invalid
    quality_score = round((1 - min(issues / total_cells, 0.40)) * 100, 1)
    completeness = round((1 - missing / total_cells) * 100, 1)

    cleaned = df.copy()

    for col in cleaned.columns:
        if cleaned[col].dtype == "object" or pd.api.types.is_string_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].astype("string").str.strip()

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            if cleaned[col].isna().any():
                median = cleaned[col].median()
                if pd.notna(median):
                    cleaned[col] = cleaned[col].fillna(median)

    cleaned_missing = int(cleaned.isna().sum().sum())
    cleaned_duplicates = int(cleaned.duplicated().sum())
    cleaned_cells = max(cleaned.shape[0] * cleaned.shape[1], 1)
    after_score = round(
        (1 - min((cleaned_missing + cleaned_duplicates) / cleaned_cells, 0.40)) * 100,
        1
    )

    return {
        "missing": missing,
        "duplicates": duplicates,
        "formatting": formatting,
        "invalid": invalid,
        "issues": issues,
        "quality_score": quality_score,
        "completeness": completeness,
        "cleaned": cleaned,
        "after_score": after_score,
        "issues_resolved": max(issues - cleaned_missing - cleaned_duplicates, 0)
    }


st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.dashboard-title {font-size: 2rem; font-weight: 800; color: #f8fafc;}
.dashboard-subtitle {color: #94a3b8; margin-bottom: 1rem;}
[data-testid="stAppViewContainer"] {background: #0b1120;}
[data-testid="stHeader"] {background: rgba(11,17,32,.95);}
[data-testid="stSidebar"] {background: #111827;}
div[data-testid="stMetric"] {
    background: #111827; border: 1px solid #243044;
    border-radius: 12px; padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="dashboard-title">DATA QUALITY CONTROL CENTER</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="dashboard-subtitle">Data validation, issue tracking and before-and-after quality analysis</div>',
    unsafe_allow_html=True
)

st.sidebar.header("Data Source")
source = st.sidebar.radio(
    "Choose data",
    ["Use sample data", "Upload CSV / Excel"]
)

if source == "Upload CSV / Excel":
    uploaded = st.sidebar.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a tabular dataset for automated quality checks."
    )

    if uploaded is None:
        st.info("Upload a CSV or Excel file from the sidebar to run a quality check.")
        st.caption("You can also choose 'Use sample data' to preview the dashboard.")
        st.stop()

    try:
        df = normalize_columns(read_uploaded_file(uploaded))
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

    st.success(f"Loaded {len(df):,} rows from {uploaded.name}")
else:
    sample_sets = create_sample_data()
    dataset_name = st.sidebar.selectbox("Sample dataset", list(sample_sets.keys()))
    df = sample_sets[dataset_name]
    st.sidebar.caption("Using simulated portfolio data.")

report = quality_report(df)

st.sidebar.header("Quality Controls")
quality_area = st.sidebar.selectbox(
    "Quality Area",
    ["All Issues", "Completeness", "Consistency", "Duplicates"]
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall Quality", f"{report['quality_score']:.1f}%")
k2.metric("Raw Records", f"{len(df):,}")
k3.metric("Clean Records", f"{len(report['cleaned']):,}")
k4.metric("Issues Resolved", f"{report['issues_resolved']:,}")
k5.metric("Completeness", f"{report['completeness']:.1f}%")

st.divider()
st.subheader("Quality Issue Breakdown")

issue_df = pd.DataFrame({
    "Issue Type": ["Missing Values", "Duplicates", "Formatting", "Invalid Values"],
    "Count": [
        report["missing"], report["duplicates"],
        report["formatting"], report["invalid"]
    ]
})

if quality_area == "Completeness":
    issue_df = issue_df[issue_df["Issue Type"] == "Missing Values"]
elif quality_area == "Consistency":
    issue_df = issue_df[issue_df["Issue Type"].isin(["Formatting", "Invalid Values"])]
elif quality_area == "Duplicates":
    issue_df = issue_df[issue_df["Issue Type"] == "Duplicates"]

left, right = st.columns([1.2, 1])

with left:
    fig = px.bar(
        issue_df, x="Issue Type", y="Count", text="Count",
        template="plotly_dark",
        labels={"Issue Type": "", "Count": "Issues"}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#111827", plot_bgcolor="#111827"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.pie(
        issue_df, names="Issue Type", values="Count",
        hole=.58, template="plotly_dark"
    )
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#111827"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Before vs After Data Quality")

comparison = pd.DataFrame({
    "Stage": ["Before Cleaning", "After Cleaning"],
    "Quality Score": [report["quality_score"], report["after_score"]]
})

fig = px.bar(
    comparison, x="Stage", y="Quality Score", text="Quality Score",
    range_y=[0, 100], template="plotly_dark",
    labels={"Stage": "", "Quality Score": "Quality Score (%)"}
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(
    height=330, margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="#111827", plot_bgcolor="#111827"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Validation Checks")

checks = pd.DataFrame({
    "Validation Check": [
        "Missing value check",
        "Duplicate record check",
        "Formatting consistency",
        "Numeric range validation"
    ],
    "Issues Found": [
        report["missing"], report["duplicates"],
        report["formatting"], report["invalid"]
    ]
})
checks["Status"] = np.where(checks["Issues Found"] == 0, "PASS", "REVIEW")

st.dataframe(checks, use_container_width=True, hide_index=True, height=190)

st.subheader("Cleaned Data Preview")
st.dataframe(
    report["cleaned"].head(100),
    use_container_width=True,
    hide_index=True,
    height=320
)

csv_bytes = report["cleaned"].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Cleaned CSV",
    data=csv_bytes,
    file_name="cleaned_dataset.csv",
    mime="text/csv"
)

st.caption(
    "Portfolio project • Supports CSV and Excel uploads. Sample data is simulated "
    "for demonstration. Uploaded files are processed by the running application."
)
