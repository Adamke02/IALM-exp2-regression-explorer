import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes, fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Regression Explorer", page_icon="∿", layout="wide")

# Dark interface styling inspired by the supplied reference.
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #10131f !important;
    color: #e8ecf8 !important;
}
[data-testid="stHeader"] { background: rgba(16,19,31,.96) !important; }
[data-testid="stSidebar"] { background: #171b2b !important; border-right: 1px solid #2b3147; }
[data-testid="stSidebar"] * { color: #e8ecf8; }
h1,h2,h3,h4,p,label,span,div { color: inherit; }
.block-container { padding-top: 1.5rem; max-width: 1600px; }
.hero { display:flex; align-items:center; gap:16px; margin-bottom:22px; }
.logo { width:56px; height:56px; border-radius:16px; display:flex; align-items:center; justify-content:center;
        background:#5145e5; color:white; font-size:34px; font-weight:800; }
.hero-title { font-size:29px; font-weight:800; color:#f5f7ff; letter-spacing:-.7px; }
.hero-subtitle { color:#9ba6c5; margin-top:3px; font-size:14px; }
.top-tag { margin-left:auto; padding:10px 14px; border-radius:9px; background:#25234e; color:#a9a2ff;
           font-weight:800; font-size:12px; letter-spacing:.5px; }
.panel { background:#191e2e; border:1px solid #2b3248; border-radius:16px; padding:20px; }
.metric-card { background:#191e2e; border:1px solid #2b3248; border-radius:15px; padding:18px 20px; min-height:118px; }
.metric-label { color:#9ca8c8; font-size:13px; margin-bottom:13px; }
.metric-value { color:#f6f7ff; font-size:29px; font-weight:800; letter-spacing:-.5px; }
.metric-help { color:#8f9ab8; font-size:12px; margin-top:7px; }
.section-title { color:#f4f6ff; font-weight:750; font-size:18px; }
.section-sub { color:#9ba6c5; font-size:13px; margin-top:4px; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background:#20263a !important; border-color:#3a4360 !important; border-radius:9px !important;
}
div[data-baseweb="select"] *, input { color:#eef2ff !important; }
.stButton > button { border-radius:9px; min-height:43px; font-weight:750; background:#5145e5; color:white; border:1px solid #6258f0; }
.stButton > button:hover { background:#635bfa; color:white; border-color:#8178ff; }
div[data-testid="stTabs"] button { color:#aeb8d4; font-weight:700; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#aaa3ff !important; background:#252849; border-radius:10px; }
div[data-testid="stTabs"] [data-baseweb="tab-list"] { background:#191e2e; border:1px solid #2b3248; padding:8px; border-radius:15px; gap:8px; }
div[data-testid="stDataFrame"], div[data-testid="stTable"] { background:#191e2e; }
hr { border-color:#2b3248; }
small, .stCaption { color:#9ba6c5 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(name):
    if name == "Study hours vs exam score":
        rng = np.random.default_rng(21)
        hours = np.arange(1, 11, dtype=float)
        score = np.array([34, 39, 47, 58, 55, 70, 68, 76, 85, 92], dtype=float)
        return pd.DataFrame({"Study hours": hours, "Exam score": score}), "Exam score"
    if name == "Diabetes":
        d = load_diabetes(as_frame=True)
        df = d.data.copy()
        df["Target"] = d.target
        return df, "Target"
    if name == "California Housing":
        d = fetch_california_housing(as_frame=True)
        # Sampling makes the interactive demo faster while retaining varied records.
        df = d.frame.sample(n=min(5000, len(d.frame)), random_state=42).reset_index(drop=True)
        return df, "MedHouseVal"
    X, y = make_regression(n_samples=1200, n_features=5, n_informative=4, noise=20, random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(5)])
    df["Target"] = y
    return df, "Target"

def make_model(name):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Ridge Regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "Decision Tree":
        return DecisionTreeRegressor(max_depth=8, random_state=42)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10))

MODEL_NAMES = ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest", "Support Vector Regression"]

with st.sidebar:
    st.markdown("### Analysis settings")
    dataset_name = st.selectbox(
    "Select dataset",
    [
        "Diabetes",
        "California Housing",
        "Synthetic Regression"
    ]
)
    method = st.selectbox("Regression Method", ["Best performing model"] + MODEL_NAMES)
    raw_df, target_col = load_data(dataset_name)
    feature_options = [c for c in raw_df.columns if c != target_col]
    default_feature = feature_options[0]
    feature = st.selectbox("Input Feature", feature_options, index=feature_options.index(default_feature))
    split_label = st.selectbox("Test split", ["80% train / 20% test", "75% train / 25% test", "70% train / 30% test"])
    test_size = {"80% train / 20% test": .20, "75% train / 25% test": .25, "70% train / 30% test": .30}[split_label]
    run = st.button("Run regression", use_container_width=True)
    reset = st.button("Reset analysis", use_container_width=True)
    st.markdown("---")
    st.caption("MAE, MSE and RMSE: lower is better")
    st.caption("R² score: higher is generally better")

if reset:
    st.session_state.pop("results_bundle", None)
    st.rerun()

# Fit using the selected input feature to keep the visual regression plot easy to interpret.
settings = (dataset_name, method, feature, split_label)
if run or "results_bundle" not in st.session_state or st.session_state.get("settings") != settings:
    y = raw_df[target_col].astype(float)
    X = raw_df[[feature]].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    chosen = MODEL_NAMES if method == "Best performing model" else [method]
    result_rows, model_objects, pred_objects = [], {}, {}
    for model_name in chosen:
        model = make_model(model_name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        result_rows.append({
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R² Score": r2_score(y_test, pred)
        })
        model_objects[model_name] = model
        pred_objects[model_name] = {"actual": np.asarray(y_test), "predicted": np.asarray(pred),
                                    "residuals": np.asarray(y_test) - np.asarray(pred),
                                    "X_test": X_test, "y_test": np.asarray(y_test)}
    scores = pd.DataFrame(result_rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
    best_name = scores.iloc[0]["Model"]
    st.session_state["results_bundle"] = (scores, pred_objects, best_name)
    st.session_state["settings"] = settings

scores, pred_objects, best_name = st.session_state["results_bundle"]
shown_model = best_name if method == "Best performing model" else method
metrics_row = scores[scores["Model"] == shown_model].iloc[0]
plot_data = pred_objects[shown_model]

st.markdown("""
<div class="hero">
  <div class="logo">∿</div>
  <div><div class="hero-title">Regression Explorer</div>
  <div class="hero-subtitle">Regression analysis &amp; model evaluation</div></div>
  <div class="top-tag">MODEL COMPARISON</div>
</div>
""", unsafe_allow_html=True)

overview_tab, compare_tab, plot_tab, residual_tab, conclusion_tab = st.tabs([
    "📊 Overview", "🏆 Model Comparison", "📈 Regression Plot", "🔎 Residuals", "🎓 Conclusions"
])

def dark_axes(ax):
    ax.set_facecolor("#191e2e")
    ax.tick_params(colors="#aeb8d4")
    ax.xaxis.label.set_color("#aeb8d4")
    ax.yaxis.label.set_color("#aeb8d4")
    ax.title.set_color("#f4f6ff")
    for spine in ax.spines.values():
        spine.set_color("#343c55")
    ax.grid(color="#303750", alpha=.65)

def metric_card(label, value, help_text):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>', unsafe_allow_html=True)

with overview_tab:
    st.markdown(f"### {dataset_name}")
    st.caption(f"Target variable: {target_col} · Input feature: {feature} · Model displayed: {shown_model}")
    cols = st.columns(4)
    with cols[0]: metric_card("Mean Absolute Error", f"{metrics_row['MAE']:.3f}", "MAE · lower is better")
    with cols[1]: metric_card("Mean Squared Error", f"{metrics_row['MSE']:.3f}", "MSE · lower is better")
    with cols[2]: metric_card("Root Mean Squared Error", f"{metrics_row['RMSE']:.3f}", "RMSE · lower is better")
    with cols[3]: metric_card("R² Score", f"{metrics_row['R² Score']:.3f}", "Higher is better")
    st.markdown("")
    left, right = st.columns([1.08, 2.8])
    with left:
        st.markdown('<div class="panel"><div class="section-title">Analysis summary</div>', unsafe_allow_html=True)
        st.write(f"**Dataset:** {dataset_name}")
        st.write(f"**Records:** {len(raw_df):,}")
        st.write(f"**Input feature:** {feature}")
        st.write(f"**Train/test split:** {split_label}")
        st.write(f"**Best model:** {best_name}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="section-title">Regression fit</div>', unsafe_allow_html=True)
        st.caption(f"{feature} vs {target_col} · fitted model: {shown_model}")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.scatter(plot_data["X_test"][feature], plot_data["actual"], color="#8585ff", alpha=.9, label="Observed values")
        order = np.argsort(plot_data["X_test"][feature].to_numpy())
        ax.plot(plot_data["X_test"][feature].to_numpy()[order], plot_data["predicted"][order],
                color="#ff824d", linewidth=2.7, label="Model fit")
        ax.set_xlabel(feature); ax.set_ylabel(target_col); ax.legend(facecolor="#20263a", labelcolor="#e8ecf8")
        dark_axes(ax); fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

with compare_tab:
    st.markdown("### Model comparison")
    st.caption("Models are evaluated on the same held-out test split.")
    st.dataframe(scores.style.format({"MAE":"{:.4f}", "MSE":"{:.4f}", "RMSE":"{:.4f}", "R² Score":"{:.4f}"}),
                 use_container_width=True, hide_index=True)
    metric = st.selectbox("Compare metric", ["R² Score", "MAE", "MSE", "RMSE"], key="metric_compare")
    ascending = metric != "R² Score"
    sorted_scores = scores.sort_values(metric, ascending=ascending)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(sorted_scores["Model"], sorted_scores[metric], color="#6258f0")
    bars[0].set_color("#ff824d")
    ax.set_ylabel(metric); ax.tick_params(axis="x", rotation=20)
    dark_axes(ax); fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.download_button("Download model comparison CSV", scores.to_csv(index=False), "model_comparison.csv", "text/csv")

with plot_tab:
    st.markdown("### Regression plot")
    feature_for_plot = st.selectbox("Plot feature", feature_options, index=feature_options.index(feature), key="plot_feature")
    # Refit selected model using the chosen plotting feature.
    Xp = raw_df[[feature_for_plot]].astype(float); yp = raw_df[target_col].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=test_size, random_state=42)
    plot_model = make_model(shown_model); plot_model.fit(Xtr, ytr); pred = plot_model.predict(Xte)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(Xte[feature_for_plot], yte, color="#8585ff", alpha=.9, label="Observed values")
    order = np.argsort(Xte[feature_for_plot].to_numpy())
    ax.plot(Xte[feature_for_plot].to_numpy()[order], pred[order], color="#ff824d", linewidth=2.7, label="Model fit")
    ax.set_xlabel(feature_for_plot); ax.set_ylabel(target_col); ax.legend(facecolor="#20263a", labelcolor="#e8ecf8")
    dark_axes(ax); fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

with residual_tab:
    st.markdown(f"### Residual analysis · {shown_model}")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(plot_data["predicted"], plot_data["residuals"], color="#8585ff", alpha=.8)
        ax.axhline(0, color="#ff824d", linestyle="--")
        ax.set_xlabel("Predicted values"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs predicted")
        dark_axes(ax); fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(plot_data["residuals"], bins=18, color="#6258f0", edgecolor="#191e2e")
        ax.set_xlabel("Residual"); ax.set_ylabel("Frequency"); ax.set_title("Residual distribution")
        dark_axes(ax); fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.info("Residual = actual value − predicted value. A random scatter around zero is generally desirable.")

with conclusion_tab:
    st.markdown("### Conclusions")
    st.write(f"- The selected dataset is **{dataset_name}**, with **{len(raw_df)} records**.")
    st.write(f"- Based on the current test split, **{best_name}** has the highest R² score among the evaluated models.")
    st.write(f"- The selected model's MAE is **{metrics_row['MAE']:.4f}**, RMSE is **{metrics_row['RMSE']:.4f}**, and R² is **{metrics_row['R² Score']:.4f}**.")
    st.write("- Lower MAE, MSE, and RMSE indicate smaller prediction errors; a higher R² generally indicates a better fit.")
    st.write("- The best model can vary by dataset and input feature. Compare all metrics and residual plots before making a final choice.")
    st.markdown("#### Current metric table")
    st.dataframe(scores, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Experiment No. 2 · Comparative Analysis of Regression Techniques Across Multiple Datasets")
