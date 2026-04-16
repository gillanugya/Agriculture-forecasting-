import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import shap

# ================= PAGE CONFIG =================


# ================= UI THEME & LAYOUT =================
st.markdown("""
<style>

/* ===== ROOT COLORS ===== */
:root {
    --bg-main: #0383535;
    --bg-panel: #111827;
    --bg-card: #0f172a;
    --accent-green: #22c55e;
    --accent-yellow: #facc15;
    --accent-red: #ef4444;
    --text-main: #e5e7eb;
    --text-muted: #9ca3af;
}

/* ===== APP BACKGROUND ===== */
.stApp {
    background: linear-gradient(180deg, #383535 0%, #020617 100%);
    color: var(--text-main);
}

/* ===== MAIN CONTENT ===== */
.block-container {
    background-color: transparent;
    padding-top: 1.5rem;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #827C7C, #020617);
    border-right: 1px solid #1f2933;
}

section[data-testid="stSidebar"] * {
    color: var(--text-main);
}

/* ===== SECTION PANELS ===== */
.section-box {
    background-color: var(--bg-panel);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.55);
    border-left: 5px solid var(--accent-green);
    margin-bottom: 32px;
}

/* ===== KPI CARDS ===== */
.kpi-card {
    background-color: var(--bg-card);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.05);
    text-align: center;
}

.kpi-card-title {
    font-size: 13px;
    color: var(--text-muted);
}

.kpi-card-value {
    font-size: 30px;
    font-weight: 600;
    color: var(--accent-green);
}

.kpi-card-sub {
    font-size: 12px;
    color: var(--text-muted);
}

/* ===== HEADINGS ===== */
h1, h2, h3, h4 {
    color: #f9fafb;
}

/* ===== DIVIDERS ===== */
hr {
    border-color: rgba(255,255,255,0.08);
}

/* ===== MATPLOTLIB ===== */
figure {
    background-color: transparent !important;
}

/* ===================================================== */
/* ===== SELECTBOX: WHITE BOX + BLACK TEXT (FINAL FIX) == */
/* ===================================================== */

/* Entire select control */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 14px !important;
}

/* Selected value wrapper */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #111827 !important;
    opacity: 1 !important;
}

/* Actual selected text */
div[data-baseweb="select"] span {
    color: #111827 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Input fallback */
div[data-baseweb="select"] input {
    color: #111827 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Placeholder (when empty) */
div[data-baseweb="select"] ::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg {
    fill: #111827 !important;
}

</style>
""", unsafe_allow_html=True)





# ================= KPI CARD =================
def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-title">{title}</div>
            <div class="kpi-card-value">{value}</div>
            <div class="kpi-card-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("merged_lag_datset.csv")  # <-- make sure this is your rainfall dataset

df = load_data()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_assets():
    model = joblib.load("final_time_forecast_model.pkl")
    scaler = joblib.load("final_time_scaler.pkl")
    feature_names = joblib.load("final_time_features.pkl")
    return model, scaler, feature_names


# Call AFTER definition
model, scaler, FEATURE_NAMES = load_model_assets()

# ================= HEADER =================
st.title("🌾 AI Crop Yield Intelligence System")
st.subheader("District-Level Agricultural Decision Support Dashboard")
st.divider()

# ================= SIDEBAR =================
st.sidebar.header("🔍 User Input Panel")

state = st.sidebar.selectbox(
    "Select State",
    sorted(df["State"].unique())
)

district = st.sidebar.selectbox(
    "Select District",
    sorted(df[df["State"] == state]["District"].unique())
)

# Filter crops with sufficient data (>= 5 years)
valid_crops = []

district_data = df[
    (df["State"] == state) &
    (df["District"] == district)
]

for crop_name in district_data["Crop"].unique():
    crop_count = district_data[district_data["Crop"] == crop_name].shape[0]
    if crop_count >= 5:
        valid_crops.append(crop_name)

if len(valid_crops) == 0:
    st.sidebar.warning("No crops with sufficient data available.")
    st.stop()

crop = st.sidebar.selectbox(
    "Select Crop",
    sorted(valid_crops)
)

analysis_mode = st.sidebar.radio(
    "Select Analysis Mode",
    [
        "📈 Historical Analysis",
        "🤖 AI Prediction",
        "📉 Scenario Simulation",
        "⚠️ Risk & Shock Analysis",
        "🔍 Explainable AI (Next)"
    ]
)

# ================= FILTER DATA =================
data = df[
    (df["State"] == state) &
    (df["District"] == district) &
    (df["Crop"] == crop)
].sort_values("Year")

st.divider()

# ================= HELPERS =================
def prepare_features(latest_row):
    # Select ONLY training features
    X = latest_row[FEATURE_NAMES].copy()

    # Ensure numeric dtype (CRITICAL)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Handle any missing values safely
    X = X.fillna(0)

    # Scale using training scaler
    X_scaled = scaler.transform(X)

    return X_scaled



def predict_with_scenario(latest_row, multiplier):
    X = latest_row[FEATURE_NAMES].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    X_scenario = X * multiplier
    X_scaled = scaler.transform(X_scenario)

    return model.predict(X_scaled)[0]



# ================= MAIN CONTENT =================

# ---- HISTORICAL ANALYSIS ----
if analysis_mode == "📈 Historical Analysis":
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("📈 Historical Yield Analysis")

    trend = "Increasing 📈" if data["Yield"].iloc[-1] > data["Yield"].iloc[0] else "Decreasing 📉"

    col1, col2, col3 = st.columns(3)

    with col1:
        kpi_card(
            "Average Yield",
            f"{data['Yield'].mean():.2f} Kg/ha",
            "Historical Mean"
        )

    with col2:
        kpi_card(
            "Yield Volatility",
            f"{data['Yield'].std():.2f}",
            "Standard Deviation"
        )

    with col3:
        kpi_card(
            "Trend",
            trend,
            "Long-term Pattern"
        )

    # ---- PLOT ----
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        data["Year"],
        data["Yield"],
        marker="o",
        linewidth=2,
        color="#22c55e"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (Kg/ha)")
    ax.set_title("Historical Yield Trend")
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- AI PREDICTION ----
elif analysis_mode == "🤖 AI Prediction":

    st.markdown("### 🤖 Next-Year Yield Forecast")
    st.markdown("<hr>", unsafe_allow_html=True)

    if data.shape[0] < 2:
        st.warning("Not enough data available for forecasting.")
    else:
        # Get latest available year
        latest = data.sort_values("Year").iloc[-1:]

        # Prepare features EXACTLY as trained
        X_latest = latest[FEATURE_NAMES].copy()
        X_latest = X_latest.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Scale using saved scaler
        X_scaled = scaler.transform(X_latest)

        # Predict next year yield
        prediction = model.predict(X_scaled)[0]

        # Get context values
        last_actual = latest["Yield"].values[0]
        next_year = int(latest["Year"].values[0]) + 1
        avg_yield = data["Yield"].mean()

    
        # ---- KPI CARDS ----
        col1, col2 = st.columns(2)

        with col1:
            kpi_card(
                "Predicted Yield",
                f"{prediction:.2f} Kg/ha",
                f"Forecast for {next_year}"
            )

        with col2:
            kpi_card(
                "Last Observed Yield",
                f"{last_actual:.2f} Kg/ha",
                f"{int(latest['Year'].values[0])}"
            )

        # ---- VISUALIZATION ----
        st.subheader("Forecast Visualization")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(data["Year"], data["Yield"], marker="o", label="Historical Yield")
        ax.scatter(next_year, prediction, color="red", s=120, label="Forecast")
        ax.plot(
            [data["Year"].iloc[-1], next_year],
            [last_actual, prediction],
            linestyle="--",
            color="red"
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Yield (Kg/ha)")
        ax.set_title("Next-Year Yield Forecast")
        ax.legend()

        st.pyplot(fig)

        st.success("Next-year yield forecast generated successfully.")

        # ===== HISTORICAL + PREDICTION LINE GRAPH =====
        st.subheader("Historical vs Predicted Yield")

        years = data["Year"].tolist() + [data["Year"].iloc[-1] + 1]
        yields = data["Yield"].tolist() + [prediction]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(years[:-1], yields[:-1], marker="o", label="Historical Yield")
        ax.scatter(years[-1], yields[-1], color="red", s=100, label="Predicted Yield")
        ax.plot(years[-2:], yields[-2:], linestyle="--", color="red")

        ax.set_xlabel("Year")
        ax.set_ylabel("Yield (Kg/ha)")
        ax.set_title("Historical Trend with AI Prediction")
        ax.legend()

        st.pyplot(fig)

        # ===== CONTEXT BAR CHART =====
        st.subheader("Prediction Context")

        fig2, ax2 = plt.subplots()
        ax2.bar(
            ["Last Yield", "Predicted Yield", "Historical Avg"],
            [last_actual, prediction, avg_yield]
           
        )
        ax2.set_ylabel("Yield (Kg/ha)")
        ax2.set_title("Yield Comparison")

        st.pyplot(fig2)

        st.success("AI prediction and visual comparison generated successfully.")


# ---- SCENARIO SIMULATION (ML-BASED) ----
elif analysis_mode == "📉 Scenario Simulation":
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("📉 ML-Based Scenario Simulation")

    latest = data.iloc[-1:].copy()

    scenarios = {
        "Normal": 1.0,
        "−10% Stress": 0.9,
        "−20% Stress": 0.8,
        "+10% Improvement": 1.1
    }

    preds = {
        name: predict_with_scenario(latest, mult)
        for name, mult in scenarios.items()
    }

    # ===== KPI SUMMARY =====
    col1, col2, col3, col4 = st.columns(4)

    scenario_items = list(preds.items())

    with col1:
        kpi_card("Normal Scenario", f"{scenario_items[0][1]:.2f} Kg/ha", "Baseline")

    with col2:
        kpi_card("−10% Stress", f"{scenario_items[1][1]:.2f} Kg/ha", "Moderate Stress")

    with col3:
        kpi_card("−20% Stress", f"{scenario_items[2][1]:.2f} Kg/ha", "Severe Stress")

    with col4:
        kpi_card("+10% Improvement", f"{scenario_items[3][1]:.2f} Kg/ha", "Optimistic")

    # ===== BAR CHART =====
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        preds.keys(),
        preds.values(),
        color="#FFB6C1"
    )

    ax.set_ylabel("Predicted Yield (Kg/ha)")
    ax.set_title("Scenario Comparison (ML-Based)")

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RISK & SHOCK ANALYSIS ----
elif analysis_mode == "⚠️ Risk & Shock Analysis":
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("⚠️ Yield Risk & Shock Analysis")

    df_risk = data.copy()
    df_risk["Yield_Shock"] = (
        (df_risk["Yield"] - df_risk["Yield"].shift(1)) / df_risk["Yield"].shift(1) < -0.2
    ).astype(int)

    shock_rate = df_risk["Yield_Shock"].mean() * 100
    risk = "Low 🟢" if shock_rate < 20 else "Medium 🟠" if shock_rate < 40 else "High 🔴"

    col1, col2 = st.columns(2)

    with col1:
        kpi_card(
            "Shock Probability",
            f"{shock_rate:.1f}%",
            "Severe Yield Drop Events"
        )

    with col2:
        kpi_card(
            "Risk Level",
            risk,
            "Overall Assessment"
        )

    # ---- SHOCK TIMELINE PLOT ----
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        df_risk["Year"],
        df_risk["Yield_Shock"],
        color="#ef4444"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Shock Occurrence")
    ax.set_title("Historical Yield Shock Timeline")

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)


# ---- EXPLAINABLE AI (SHAP) ----
elif analysis_mode == "🔍 Explainable AI (Next)":
    st.header("🔍 Explainable AI — Feature Importance (SHAP)")
    st.caption("Top features influencing the AI prediction")

    if data.shape[0] < 2:
        st.warning("Not enough data for explainability.")
    else:
        latest = data.iloc[-1:].copy()

        # Prepare features
        X_latest = latest[FEATURE_NAMES].copy()
        X_latest = X_latest.apply(pd.to_numeric, errors="coerce").fillna(0)

        # SHAP explainer
        explainer = shap.TreeExplainer(model)

        # 🔑 THIS LINE WAS MISSING
        shap_values = explainer.shap_values(X_latest)

        # ===== WHITE SHAP PLOT ONLY =====
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        shap.summary_plot(
            shap_values,
            X_latest,
            plot_type="bar",
            max_display=10,
            show=False
        )

        # Force readable text
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")

        st.pyplot(fig)

        st.success("Explainability generated for the latest AI prediction.")
