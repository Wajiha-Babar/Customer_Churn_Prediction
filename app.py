import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent

TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"
METRICS_PATH = ROOT_DIR / "outputs" / "metrics.json"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "outputs" / "feature_importance.csv"
BATCH_OUTPUT_PATH = ROOT_DIR / "outputs" / "dashboard_batch_predictions.csv"


# =========================================================
# PREMIUM CLEAN CSS
# =========================================================

st.markdown(
    """
    <style>
    /* ===============================
       GLOBAL THEME
    =============================== */

    .stApp {
        background: #f6efe7 !important;
        color: #25181c !important;
        font-family: "Segoe UI", sans-serif !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-left: 2.7rem !important;
        padding-right: 2.7rem !important;
        max-width: 1550px !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #25181c;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* ===============================
       SIDEBAR
    =============================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4d0717 0%, #6d1028 50%, #3b0611 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.18);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] label {
        font-weight: 800 !important;
        font-size: 14px !important;
    }

    .sidebar-title {
        color: #ffffff !important;
        font-size: 25px;
        font-weight: 950;
        margin-bottom: 8px;
    }

    .sidebar-subtitle {
        color: #ffe8d5 !important;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 24px;
    }

    .sidebar-section {
        color: #ffffff !important;
        font-size: 19px;
        font-weight: 900;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    /* Sidebar select boxes */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        border-radius: 14px !important;
        min-height: 44px !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #25181c !important;
        font-weight: 650 !important;
    }

    section[data-testid="stSidebar"] input {
        color: #25181c !important;
        background: #ffffff !important;
    }

    /* Dropdown menu readability */
    div[role="listbox"] {
        background: #ffffff !important;
        border-radius: 14px !important;
        border: 1px solid rgba(91,9,28,0.15) !important;
    }

    div[role="option"] {
        background: #ffffff !important;
        color: #25181c !important;
    }

    div[role="option"] span {
        color: #25181c !important;
    }

    div[role="option"]:hover {
        background: #f4e5dc !important;
    }

    /* Sliders */
    section[data-testid="stSidebar"] .stSlider {
        padding-top: 4px;
        padding-bottom: 10px;
    }

    /* ===============================
       HERO
    =============================== */

    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #fffaf5 50%, #f3e2d4 100%);
        border: 1px solid rgba(91,9,28,0.12);
        border-radius: 30px;
        padding: 34px 40px;
        box-shadow: 0 20px 55px rgba(91,9,28,0.12);
        margin-bottom: 28px;
    }

    .main-title {
        color: #5b091c !important;
        font-size: 46px;
        font-weight: 950;
        line-height: 1.12;
        margin-bottom: 12px;
        letter-spacing: -0.9px;
    }

    .sub-title {
        color: #66565b !important;
        font-size: 18px;
        line-height: 1.75;
        max-width: 1050px;
    }

    .model-pill {
        display: inline-block;
        background: #5b091c;
        color: #ffffff !important;
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 850;
        font-size: 14px;
        margin-top: 18px;
        margin-right: 8px;
        border: 1px solid #d8ae57;
    }

    /* ===============================
       TABS
    =============================== */

    button[data-baseweb="tab"] {
        background: #ffffff !important;
        border: 1px solid rgba(91,9,28,0.12) !important;
        border-radius: 14px 14px 0px 0px !important;
        padding: 12px 18px !important;
        margin-right: 5px !important;
    }

    button[data-baseweb="tab"] p {
        color: #5b091c !important;
        font-weight: 850 !important;
        font-size: 15px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #5b091c !important;
        border-bottom: 4px solid #d8ae57 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ffffff !important;
    }

    /* ===============================
       CARDS
    =============================== */

    .section-heading {
        color: #5b091c !important;
        font-size: 28px;
        font-weight: 950;
        margin-top: 20px;
        margin-bottom: 16px;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 22px;
        border: 1px solid rgba(91,9,28,0.12);
        box-shadow: 0 16px 38px rgba(91,9,28,0.08);
        padding: 22px;
        min-height: 142px;
    }

    .metric-label {
        color: #74666b !important;
        font-size: 13px;
        font-weight: 900;
        letter-spacing: 0.7px;
        text-transform: uppercase;
    }

    .metric-value {
        color: #5b091c !important;
        font-size: 31px;
        font-weight: 950;
        line-height: 1.25;
        margin-top: 12px;
    }

    .metric-note {
        color: #74666b !important;
        font-size: 13px;
        margin-top: 8px;
        line-height: 1.45;
    }

    .insight-box {
        background: #ffffff;
        border-radius: 20px;
        border-left: 7px solid #5b091c;
        border-top: 1px solid rgba(91,9,28,0.10);
        border-right: 1px solid rgba(91,9,28,0.10);
        border-bottom: 1px solid rgba(91,9,28,0.10);
        box-shadow: 0 14px 35px rgba(91,9,28,0.08);
        padding: 22px 24px;
        margin-bottom: 14px;
        color: #25181c !important;
        line-height: 1.75;
        font-size: 16px;
    }

    .insight-box b {
        color: #5b091c !important;
    }

    /* ===============================
       FORM / INPUTS
    =============================== */

    input, textarea {
        background: #ffffff !important;
        color: #25181c !important;
    }

    div[data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: #25181c !important;
    }

    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid rgba(91,9,28,0.12) !important;
        border-radius: 14px !important;
        min-height: 45px !important;
    }

    div[data-baseweb="select"] span {
        color: #25181c !important;
        font-weight: 600 !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #5b091c 0%, #7a1730 100%) !important;
        color: #ffffff !important;
        border-radius: 15px !important;
        border: 1px solid #d8ae57 !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 900 !important;
        font-size: 15px !important;
        box-shadow: 0 12px 28px rgba(91,9,28,0.20) !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #7a1730 0%, #5b091c 100%) !important;
        border: 1px solid #f0c96c !important;
        color: #ffffff !important;
    }

    /* Radio buttons */
    div[role="radiogroup"] label {
        background: #ffffff !important;
        border: 1px solid rgba(91,9,28,0.14) !important;
        border-radius: 14px !important;
        padding: 8px 14px !important;
        margin-right: 8px !important;
    }

    div[role="radiogroup"] label p {
        color: #25181c !important;
        font-weight: 700 !important;
    }

    /* ===============================
       PREDICTION RESULT
    =============================== */

    .risk-high {
        background: linear-gradient(135deg, #fff1f1 0%, #ffdcdc 100%);
        border: 1px solid #b71c1c;
        border-radius: 24px;
        padding: 25px;
        color: #7f1010 !important;
        font-weight: 950;
        font-size: 22px;
        box-shadow: 0 16px 42px rgba(183,28,28,0.12);
    }

    .risk-low {
        background: linear-gradient(135deg, #effaf1 0%, #dff2e3 100%);
        border: 1px solid #2e7d32;
        border-radius: 24px;
        padding: 25px;
        color: #1b5e20 !important;
        font-weight: 950;
        font-size: 22px;
        box-shadow: 0 16px 42px rgba(46,125,50,0.12);
    }

    .recommend-card {
        background: #ffffff;
        border-radius: 20px;
        border: 1px solid rgba(91,9,28,0.12);
        box-shadow: 0 14px 35px rgba(91,9,28,0.08);
        padding: 20px;
        margin-top: 14px;
        color: #25181c !important;
        font-size: 16px;
        line-height: 1.7;
    }

    .recommend-card b {
        color: #5b091c !important;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        background: #ffffff !important;
        border-radius: 18px !important;
        border: 1px solid rgba(91,9,28,0.12) !important;
        padding: 8px !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 16px !important;
        color: #25181c !important;
    }

    .footer-text {
        text-align: center;
        color: #74666b !important;
        font-size: 14px;
        margin-top: 36px;
        padding-bottom: 28px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA LOADING FUNCTIONS
# =========================================================

@st.cache_data
def load_default_data():
    return pd.read_csv(TRAIN_PATH)


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    return None


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def prepare_dataset(dataframe):
    df = dataframe.copy()

    if "Exited" in df.columns:
        df["Status"] = df["Exited"].map(
            {
                0: "Not Churned",
                1: "Churned",
            }
        )

    return df


def get_best_model_info(metrics_data):
    if metrics_data is None:
        return "Unknown", 0.50, {}

    best_model_name = metrics_data.get("best_model", "Unknown")
    model_metrics = metrics_data.get("all_model_metrics", {}).get(best_model_name, {})
    optimal_threshold = model_metrics.get("optimal_threshold", 0.50)

    return best_model_name, optimal_threshold, model_metrics


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_plot_layout(fig, height=430):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#25181c", size=13),
        title=dict(
            font=dict(color="#5b091c", size=20),
            x=0.02,
        ),
        margin=dict(l=30, r=30, t=70, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(91,9,28,0.12)",
            borderwidth=1,
            font=dict(color="#25181c"),
        ),
    )

    fig.update_xaxes(
        color="#25181c",
        gridcolor="rgba(91,9,28,0.08)",
        linecolor="rgba(91,9,28,0.18)",
    )

    fig.update_yaxes(
        color="#25181c",
        gridcolor="rgba(91,9,28,0.08)",
        linecolor="rgba(91,9,28,0.18)",
    )

    return fig


def create_gauge_chart(value, title):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": "%",
                "font": {"size": 42, "color": "#5b091c"},
            },
            title={
                "text": title,
                "font": {"size": 22, "color": "#5b091c"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#25181c",
                    "tickfont": {"color": "#25181c"},
                },
                "bar": {"color": "#5b091c"},
                "bgcolor": "#ffffff",
                "borderwidth": 2,
                "bordercolor": "#5b091c",
                "steps": [
                    {"range": [0, 30], "color": "#dff3e5"},
                    {"range": [30, 60], "color": "#fff1cc"},
                    {"range": [60, 100], "color": "#f8caca"},
                ],
                "threshold": {
                    "line": {"color": "#b71c1c", "width": 5},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )

    fig.update_layout(
        height=370,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#25181c"),
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return fig


def create_prediction_input_dataframe(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_card,
    active_member,
    salary,
):
    return pd.DataFrame(
        {
            "id": [0],
            "CustomerId": [0],
            "Surname": ["NewCustomer"],
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [float(age)],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_card],
            "IsActiveMember": [active_member],
            "EstimatedSalary": [salary],
        }
    )


# =========================================================
# LOAD PROJECT DATA
# =========================================================

default_df = load_default_data()
default_df = prepare_dataset(default_df)

model = load_model()
metrics = load_metrics()
importance_df = load_feature_importance()

best_model_name, optimal_threshold, best_model_metrics = get_best_model_info(metrics)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    f"""
    <div class="hero-card">
        <div class="main-title">🏦 Customer Churn Prediction Dashboard</div>
        <div class="sub-title">
            A clean and professional machine learning dashboard for bank customer churn analysis,
            customer risk scoring, model performance monitoring, feature importance, and batch predictions.
        </div>
        <span class="model-pill">Best Model: {best_model_name}</span>
        <span class="model-pill">Optimal Threshold: {optimal_threshold}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR FILTERS ONLY
# =========================================================

st.sidebar.markdown('<div class="sidebar-title">🏦 Churn Control Panel</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-subtitle">Apply clean filters to update the dashboard instantly.</div>',
    unsafe_allow_html=True,
)

df = default_df.copy()

required_columns = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

missing_required = [col for col in required_columns if col not in df.columns]

if missing_required:
    st.error(f"Dataset is missing required columns: {missing_required}")
    st.stop()

st.sidebar.markdown('<div class="sidebar-section">🔎 Filters</div>', unsafe_allow_html=True)

geography_options = ["All"] + sorted(df["Geography"].dropna().unique().tolist())
gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())

selected_geography = st.sidebar.selectbox(
    "Geography",
    options=geography_options,
)

selected_gender = st.sidebar.selectbox(
    "Gender",
    options=gender_options,
)

if "Status" in df.columns:
    status_options = ["All"] + sorted(df["Status"].dropna().unique().tolist())
    selected_status = st.sidebar.selectbox(
        "Customer Status",
        options=status_options,
    )
else:
    selected_status = "All"

age_min = int(df["Age"].min())
age_max = int(df["Age"].max())

selected_age = st.sidebar.slider(
    "Age Range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max),
)

credit_min = int(df["CreditScore"].min())
credit_max = int(df["CreditScore"].max())

selected_credit = st.sidebar.slider(
    "Credit Score Range",
    min_value=credit_min,
    max_value=credit_max,
    value=(credit_min, credit_max),
)

balance_min = float(df["Balance"].min())
balance_max = float(df["Balance"].max())

selected_balance = st.sidebar.slider(
    "Balance Range",
    min_value=balance_min,
    max_value=balance_max,
    value=(balance_min, balance_max),
)


# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df.copy()

if selected_geography != "All":
    filtered_df = filtered_df[filtered_df["Geography"] == selected_geography]

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]

if "Status" in filtered_df.columns and selected_status != "All":
    filtered_df = filtered_df[filtered_df["Status"] == selected_status]

filtered_df = filtered_df[
    (filtered_df["Age"].between(selected_age[0], selected_age[1]))
    & (filtered_df["CreditScore"].between(selected_credit[0], selected_credit[1]))
    & (filtered_df["Balance"].between(selected_balance[0], selected_balance[1]))
].copy()


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📌 Executive Overview",
        "📊 Customer Insights",
        "🤖 Predict Customer",
        "📈 Model Performance",
        "📁 Batch Prediction",
    ]
)


# =========================================================
# TAB 1: EXECUTIVE OVERVIEW
# =========================================================

with tab1:
    st.markdown('<div class="section-heading">Executive Summary</div>', unsafe_allow_html=True)

    total_customers = len(filtered_df)

    if "Exited" in filtered_df.columns and total_customers > 0:
        churned_customers = int(filtered_df["Exited"].sum())
        churn_rate = churned_customers / total_customers * 100
    else:
        churned_customers = "N/A"
        churn_rate = 0

    avg_age = filtered_df["Age"].mean() if total_customers > 0 else 0
    avg_credit = filtered_df["CreditScore"].mean() if total_customers > 0 else 0
    avg_balance = filtered_df["Balance"].mean() if total_customers > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Total Customers", f"{total_customers:,}", "Filtered customer records")

    with c2:
        metric_card("Churned Customers", f"{churned_customers}", "Customers who exited")

    with c3:
        metric_card("Churn Rate", f"{churn_rate:.2f}%", "Filtered churn percentage")

    with c4:
        metric_card("Average Age", f"{avg_age:.1f}", "Average customer age")

    with c5:
        metric_card("Average Balance", f"{avg_balance:,.0f}", "Average account balance")

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        if "Status" in filtered_df.columns and len(filtered_df) > 0:
            churn_count = filtered_df["Status"].value_counts().reset_index()
            churn_count.columns = ["Status", "Customers"]

            fig = px.pie(
                churn_count,
                names="Status",
                values="Customers",
                hole=0.58,
                title="Customer Churn Distribution",
                color_discrete_sequence=["#5b091c", "#d8ae57"],
            )

            fig = create_plot_layout(fig, height=440)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No records available for churn distribution.")

    with col2:
        if "Exited" in filtered_df.columns and len(filtered_df) > 0:
            geo_churn = (
                filtered_df.groupby("Geography")["Exited"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"Exited": "Churn Rate (%)"})
            )

            fig = px.bar(
                geo_churn,
                x="Geography",
                y="Churn Rate (%)",
                text_auto=".2f",
                title="Churn Rate by Geography",
                color="Churn Rate (%)",
                color_continuous_scale=["#f8dede", "#5b091c"],
            )

            fig = create_plot_layout(fig, height=440)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No records available for geography chart.")

    st.markdown('<div class="section-heading">Business Insights</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="insight-box">
            <b>Insight 1:</b> Churn prediction helps banks identify high-risk customers before they leave.
        </div>
        <div class="insight-box">
            <b>Insight 2:</b> Age, number of products, active membership, geography, and balance are important churn factors.
        </div>
        <div class="insight-box">
            <b>Insight 3:</b> Sidebar filters allow quick analysis by geography, gender, age, credit score, and balance.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# TAB 2: CUSTOMER INSIGHTS
# =========================================================

with tab2:
    st.markdown('<div class="section-heading">Customer EDA Insights</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if "Status" in filtered_df.columns and len(filtered_df) > 0:
            fig = px.histogram(
                filtered_df,
                x="Age",
                color="Status",
                nbins=35,
                title="Age Distribution by Churn Status",
                color_discrete_sequence=["#5b091c", "#d8ae57"],
            )
        else:
            fig = px.histogram(
                filtered_df,
                x="Age",
                nbins=35,
                title="Age Distribution",
                color_discrete_sequence=["#5b091c"],
            )

        fig = create_plot_layout(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Status" in filtered_df.columns and len(filtered_df) > 0:
            fig = px.box(
                filtered_df,
                x="Status",
                y="Balance",
                color="Status",
                title="Balance Distribution by Churn Status",
                color_discrete_sequence=["#5b091c", "#d8ae57"],
            )
        else:
            fig = px.box(
                filtered_df,
                y="Balance",
                title="Balance Distribution",
                color_discrete_sequence=["#5b091c"],
            )

        fig = create_plot_layout(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if "Exited" in filtered_df.columns and len(filtered_df) > 0:
            product_churn = (
                filtered_df.groupby("NumOfProducts")["Exited"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"Exited": "Churn Rate (%)"})
            )

            fig = px.line(
                product_churn,
                x="NumOfProducts",
                y="Churn Rate (%)",
                markers=True,
                title="Churn Rate by Number of Products",
                color_discrete_sequence=["#5b091c"],
            )
            fig.update_traces(line=dict(width=4), marker=dict(size=10))
        else:
            fig = px.histogram(
                filtered_df,
                x="NumOfProducts",
                title="Customer Count by Number of Products",
                color_discrete_sequence=["#5b091c"],
            )

        fig = create_plot_layout(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "Exited" in filtered_df.columns and len(filtered_df) > 0:
            active_churn = (
                filtered_df.groupby("IsActiveMember")["Exited"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"Exited": "Churn Rate (%)"})
            )

            active_churn["Member Status"] = active_churn["IsActiveMember"].map(
                {
                    0: "Inactive Member",
                    1: "Active Member",
                    0.0: "Inactive Member",
                    1.0: "Active Member",
                }
            )

            fig = px.bar(
                active_churn,
                x="Member Status",
                y="Churn Rate (%)",
                text_auto=".2f",
                title="Churn Rate: Active vs Inactive Members",
                color="Churn Rate (%)",
                color_continuous_scale=["#f8dede", "#5b091c"],
            )
        else:
            temp_df = filtered_df.copy()
            temp_df["Member Status"] = temp_df["IsActiveMember"].map(
                {
                    0: "Inactive Member",
                    1: "Active Member",
                    0.0: "Inactive Member",
                    1.0: "Active Member",
                }
            )

            fig = px.histogram(
                temp_df,
                x="Member Status",
                title="Active vs Inactive Customer Count",
                color_discrete_sequence=["#5b091c"],
            )

        fig = create_plot_layout(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-heading">Filtered Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df.head(200), use_container_width=True)


# =========================================================
# TAB 3: SINGLE CUSTOMER PREDICTION
# =========================================================

with tab3:
    st.markdown('<div class="section-heading">Predict Churn for a New Customer</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model not found. Please run `python src/train_model.py` first.")

    else:
        st.markdown(
            f"""
            <div class="insight-box">
                <b>Current Best Model:</b> {best_model_name}<br>
                <b>Optimal Threshold:</b> {optimal_threshold}<br>
                Use the form below to predict churn risk for a new bank customer.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                credit_score = st.slider("Credit Score", 300, 900, 650)
                geography = st.radio(
                    "Geography",
                    sorted(default_df["Geography"].unique()),
                    horizontal=True,
                )
                gender = st.radio(
                    "Gender",
                    sorted(default_df["Gender"].unique()),
                    horizontal=True,
                )

            with col2:
                age = st.slider("Age", 18, 95, 35)
                tenure = st.slider("Tenure", 0, 10, 5)
                balance = st.number_input(
                    "Balance",
                    min_value=0.0,
                    value=50000.0,
                    step=1000.0,
                )

            with col3:
                num_products = st.radio(
                    "Number of Products",
                    sorted(default_df["NumOfProducts"].unique()),
                    horizontal=True,
                )

                has_card_label = st.radio(
                    "Has Credit Card?",
                    ["No", "Yes"],
                    horizontal=True,
                )

                active_member_label = st.radio(
                    "Is Active Member?",
                    ["No", "Yes"],
                    horizontal=True,
                )

                salary = st.number_input(
                    "Estimated Salary",
                    min_value=0.0,
                    value=100000.0,
                    step=1000.0,
                )

            submit_prediction = st.form_submit_button("Predict Churn Risk")

        if submit_prediction:
            has_card = 1.0 if has_card_label == "Yes" else 0.0
            active_member = 1.0 if active_member_label == "Yes" else 0.0

            new_customer = create_prediction_input_dataframe(
                credit_score,
                geography,
                gender,
                age,
                tenure,
                balance,
                num_products,
                has_card,
                active_member,
                salary,
            )

            churn_probability = model.predict_proba(new_customer)[0][1]
            churn_percentage = churn_probability * 100
            prediction = 1 if churn_probability >= optimal_threshold else 0

            result_col1, result_col2 = st.columns([1, 1])

            with result_col1:
                st.plotly_chart(
                    create_gauge_chart(churn_percentage, "Churn Probability"),
                    use_container_width=True,
                )

            with result_col2:
                if prediction == 1:
                    st.markdown(
                        """
                        <div class="risk-high">
                            🔴 High Risk Customer<br>
                            This customer is likely to churn.
                        </div>
                        <div class="recommend-card">
                            <b>Recommended Action:</b><br>
                            Offer loyalty rewards, personalized support, reduced service charges,
                            special banking offers, or direct customer engagement.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="risk-low">
                            🟢 Low Risk Customer<br>
                            This customer is likely to stay.
                        </div>
                        <div class="recommend-card">
                            <b>Recommended Action:</b><br>
                            Continue positive engagement, maintain service quality,
                            and offer relevant products to increase long-term retention.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="section-heading">Customer Input Summary</div>', unsafe_allow_html=True)
            st.dataframe(new_customer, use_container_width=True)


# =========================================================
# TAB 4: MODEL PERFORMANCE
# =========================================================

with tab4:
    st.markdown('<div class="section-heading">Model Performance</div>', unsafe_allow_html=True)

    if metrics is None:
        st.error("Metrics file not found. Please run `python src/train_model.py` first.")

    else:
        st.markdown(
            f"""
            <div class="insight-box">
                <b>Best Model:</b> {best_model_name}<br>
                <b>Selection Metric:</b> ROC-AUC<br>
                <b>Optimal Threshold:</b> {optimal_threshold}
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            metric_card("Accuracy", best_model_metrics.get("accuracy", "N/A"))

        with c2:
            metric_card("Precision", best_model_metrics.get("precision", "N/A"))

        with c3:
            metric_card("Recall", best_model_metrics.get("recall", "N/A"))

        with c4:
            metric_card("F1 Score", best_model_metrics.get("f1_score", "N/A"))

        with c5:
            metric_card("ROC-AUC", best_model_metrics.get("roc_auc", "N/A"))

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            all_model_metrics = metrics.get("all_model_metrics", {})

            comparison_data = []

            for model_name, model_metric in all_model_metrics.items():
                comparison_data.append(
                    {
                        "Model": model_name,
                        "Accuracy": model_metric.get("accuracy"),
                        "Precision": model_metric.get("precision"),
                        "Recall": model_metric.get("recall"),
                        "F1 Score": model_metric.get("f1_score"),
                        "ROC-AUC": model_metric.get("roc_auc"),
                    }
                )

            comparison_df = pd.DataFrame(comparison_data)

            fig = px.bar(
                comparison_df,
                x="ROC-AUC",
                y="Model",
                orientation="h",
                title="Model Comparison by ROC-AUC",
                color="ROC-AUC",
                color_continuous_scale=["#f8dede", "#5b091c"],
            )

            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            fig = create_plot_layout(fig, height=460)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cm = np.array(best_model_metrics.get("confusion_matrix", [[0, 0], [0, 0]]))

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Not Churned", "Actual Churned"],
                columns=["Predicted Not Churned", "Predicted Churned"],
            )

            fig = px.imshow(
                cm_df,
                text_auto=True,
                title=f"Confusion Matrix - {best_model_name}",
                color_continuous_scale=["#fff4ef", "#5b091c"],
            )

            fig = create_plot_layout(fig, height=460)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-heading">Feature Importance</div>', unsafe_allow_html=True)

        if importance_df is not None:
            fig = px.bar(
                importance_df.head(12),
                x="importance",
                y="feature",
                orientation="h",
                title="Top Features Influencing Customer Churn",
                color="importance",
                color_continuous_scale=["#f8dede", "#5b091c"],
            )

            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            fig = create_plot_layout(fig, height=540)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(importance_df, use_container_width=True)
        else:
            st.warning("Feature importance file not found.")


# =========================================================
# TAB 5: BATCH PREDICTION
# =========================================================

with tab5:
    st.markdown('<div class="section-heading">Batch Customer Churn Prediction</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="insight-box">
            Upload a customer CSV file here. The dashboard will predict churn probability for every customer
            and allow you to download the prediction results as a CSV file.
        </div>
        """,
        unsafe_allow_html=True,
    )

    batch_file = st.file_uploader(
        "Upload Customer CSV for Batch Prediction",
        type=["csv"],
        key="batch_prediction_file",
    )

    if model is None:
        st.error("Model not found. Please run `python src/train_model.py` first.")

    elif batch_file is not None:
        batch_df = pd.read_csv(batch_file)

        missing_batch_cols = [col for col in required_columns if col not in batch_df.columns]

        if missing_batch_cols:
            st.error(f"Uploaded batch file is missing columns: {missing_batch_cols}")

        else:
            batch_proba = model.predict_proba(batch_df)[:, 1]
            batch_prediction = (batch_proba >= optimal_threshold).astype(int)

            result_df = batch_df.copy()
            result_df["Churn_Probability"] = batch_proba
            result_df["Churn_Percentage"] = batch_proba * 100
            result_df["Predicted_Churn"] = batch_prediction
            result_df["Risk_Label"] = result_df["Predicted_Churn"].map(
                {
                    0: "Low Risk",
                    1: "High Risk",
                }
            )

            result_df.to_csv(BATCH_OUTPUT_PATH, index=False)

            high_risk_count = int(result_df["Predicted_Churn"].sum())
            total_batch = len(result_df)
            high_risk_rate = high_risk_count / total_batch * 100 if total_batch > 0 else 0

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card("Batch Customers", f"{total_batch:,}", "Uploaded customer records")

            with c2:
                metric_card("High Risk Customers", f"{high_risk_count:,}", "Predicted churn customers")

            with c3:
                metric_card("High Risk Rate", f"{high_risk_rate:.2f}%", "Batch churn risk")

            fig = px.histogram(
                result_df,
                x="Churn_Percentage",
                color="Risk_Label",
                nbins=30,
                title="Batch Prediction: Churn Probability Distribution",
                color_discrete_sequence=["#2e7d32", "#b71c1c"],
            )

            fig = create_plot_layout(fig, height=460)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-heading">Prediction Results</div>', unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Batch Predictions CSV",
                data=csv_data,
                file_name="customer_churn_batch_predictions.csv",
                mime="text/csv",
            )

    else:
        st.info("Upload a customer CSV file to generate batch predictions.")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    """
    <div class="footer-text">
        Developed by <b>Wajiha Babar</b> | Customer Churn Prediction | Machine Learning + Streamlit Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)