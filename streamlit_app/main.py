"""Streamlit main application."""

import sys
import os
from pathlib import Path
import pandas as pd

# Add root to sys path for src imports
root_path = str(Path(__file__).parent.parent.absolute())
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st
from src.utils.logger import setup_logging, get_logger
from src.utils.config_loader import ConfigManager
from src.inference.predictor import Predictor

setup_logging()
logger = get_logger("streamlit_app")

st.set_page_config(
    page_title="Diamond Dynamics: Price & Segment Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for premium look
st.markdown(
    """
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        color: #94a3b8;
        padding: 0 20px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(99, 102, 241, 0.15);
        color: #e2e8f0;
        border-color: rgba(99, 102, 241, 0.4);
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4);
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .predict-box {
        background: rgba(30, 41, 59, 0.7);
        padding: 30px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_app_resources():
    try:
        config = ConfigManager()
        predictor = Predictor(config)
        return config, predictor
    except Exception as e:
        logger.error(f"Failed to load app resources: {e}")
        st.error(f"Failed to initialize application: {e}")
        st.stop()


config, predictor = load_app_resources()

st.title("💎 Diamond Dynamics")
st.subheader("Price Prediction & Market Segmentation Dashboard")

tab_predict, tab_eda, tab_performance = st.tabs(
    ["🎯 Predictor Modules", "📊 Exploratory Data Insights", "📈 Model Architectures & Performance"]
)

with tab_predict:
    st.markdown("### Enter Diamond Specifications")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Basic Metrics")
        carat = st.number_input("Carat Weight", 0.1, 5.0, 0.7)
        cut = st.selectbox(
            "Cut Quality", list(config.model_config["mappings"]["cut"].keys()), index=4
        )
        color = st.selectbox(
            "Color Grade", list(config.model_config["mappings"]["color"].keys()), index=3
        )
        clarity = st.selectbox(
            "Clarity Grade", list(config.model_config["mappings"]["clarity"].keys()), index=3
        )

    with col2:
        st.markdown("##### Physical Dimensions")
        x = st.number_input("Length (x) mm", 1.0, 15.0, 5.7)
        y = st.number_input("Width (y) mm", 1.0, 15.0, 5.7)
        z = st.number_input("Depth (z) mm", 1.0, 10.0, 3.5)

    with col3:
        st.markdown("##### Proportions")
        depth = st.number_input("Depth %", 40.0, 80.0, 61.8)
        table = st.number_input("Table %", 40.0, 80.0, 57.0)

    if st.button("🔮 Predict", type="primary"):
        inputs = {
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
        }
        try:
            price_inr, price_usd, cluster_name, cluster_id = predictor.predict(inputs)

            st.markdown("### 🔍 Model Outputs")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='predict-box'>", unsafe_allow_html=True)
                st.markdown("#### Estimated Diamond Price")
                st.markdown(
                    f"<h2 style='color: #4ade80;'>₹ {price_inr:,.2f} INR</h2>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"<p>Equivalent to: ${price_usd:,.2f} USD</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='predict-box'>", unsafe_allow_html=True)
                st.markdown("#### Market Segment")
                st.markdown(
                    f"<h2 style='color: #38bdf8;'>{cluster_name}</h2>", unsafe_allow_html=True
                )
                st.markdown(f"<p>Segment ID: Cluster {cluster_id}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {e}")

with tab_eda:
    st.markdown("### Exploratory Data Insights")
    plots_dir = config.get_path("plots_dir")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(plots_dir, "cluster_plot_2d.png")) if os.path.exists(
            os.path.join(plots_dir, "cluster_plot_2d.png")
        ) else st.warning("Plot missing")
        st.image(os.path.join(plots_dir, "carat_vs_price.png")) if os.path.exists(
            os.path.join(plots_dir, "carat_vs_price.png")
        ) else st.warning("Plot missing")
    with col2:
        st.image(os.path.join(plots_dir, "correlation_heatmap.png")) if os.path.exists(
            os.path.join(plots_dir, "correlation_heatmap.png")
        ) else st.warning("Plot missing")
        st.image(os.path.join(plots_dir, "price_variations_boxplot.png")) if os.path.exists(
            os.path.join(plots_dir, "price_variations_boxplot.png")
        ) else st.warning("Plot missing")

with tab_performance:
    st.markdown("### Model Performance")
    try:
        results_file = os.path.join(
            config.get_path("models_dir"), config.get_model_config("regression_results")
        )
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            st.dataframe(results_df)
    except Exception as e:
        st.error(f"Could not load performance results: {e}")
