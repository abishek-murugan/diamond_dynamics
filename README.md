<div align="center">
  <h1>💎 Diamond Dynamics</h1>
  <p><b>Production-Grade Machine Learning Pipeline for Diamond Valuation & Market Segmentation</b></p>
  
  [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B.svg)](https://streamlit.io/)
  [![RAPIDS cuML](https://img.shields.io/badge/RAPIDS-cuML-76B900.svg)](https://rapids.ai/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-2.1-blue.svg)](https://xgboost.readthedocs.io/)
  [![uv](https://img.shields.io/badge/uv-Fast_Python_Packaging-purple.svg)](https://github.com/astral-sh/uv)
</div>

<br>

Diamond Dynamics is a robust, end-to-end Machine Learning portal designed to predict the market value of diamonds (in INR/USD) and categorize them into distinct commercial segments. Developed with a focus on scalability and production-readiness, this repository serves as a blueprint for bridging high-performance GPU model training (via NVIDIA RAPIDS) with robust CPU-based web inference.

---

## ✨ Key Technical Highlights

* **High-Performance Training**: Models were trained on Google Colab GPUs using **RAPIDS cuML** and **XGBoost** for maximum scalability and speed on large tabular datasets.
* **CPU-Inference Bridge**: Features a custom cross-environment serialization pipeline (`CumlUnpickler`) that gracefully translates CuPy-backed GPU state arrays from RAPIDS into standard NumPy-backed `scikit-learn` objects for lightweight CPU-only inference.
* **Modular MLOps Architecture**: The codebase strictly adheres to software engineering best practices with dependency injection (`ConfigManager`), decoupled logic (`Predictor` class), and central YAML configuration.
* **Interactive Dashboard**: A responsive, dynamically styled **Streamlit** frontend allows stakeholders to input diamond specifications and receive instant predictions alongside exploratory visual insights.
* **Modern Tooling**: Managed entirely by **Astral `uv`**, ensuring lightning-fast dependency resolution and virtual environment synchronization.

---

## 🏗️ Project Architecture

```text
diamond_dynamics/
├── config/
│   └── model.yaml             # Centralized pipeline configurations and feature maps
├── models/                    # Serialized GPU-trained artifacts (Pickle)
│   ├── best_clustering_model.pkl
│   ├── best_regression_model.pkl
│   ├── clustering_scaler.pkl
│   └── regression_scaler.pkl
├── notebooks/
│   └── diamond_dynamics.ipynb # Original EDA and GPU training source of truth
├── src/
│   ├── inference/
│   │   ├── engineering.py     # Log transformations and feature pipelines
│   │   └── predictor.py       # Core inference engine (KMeans & XGBoost wrappers)
│   └── utils/
│       ├── config_loader.py   # YAML singleton manager
│       ├── exceptions.py      # Custom domain exceptions
│       ├── file_handler.py    # Custom unpickler for RAPIDS -> Scikit-Learn conversion
│       └── logger.py          # Centralized logging setup
├── streamlit_app/
│   └── main.py                # Interactive web application frontend
├── pyproject.toml             # UV dependency and build definitions
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have [Astral uv](https://docs.astral.sh/uv/) installed on your system.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Installation
Clone the repository and synchronize the dependencies. Due to our custom serialization handlers, heavy GPU libraries (like `cuml`) are **not** required to run the web application.

```bash
git clone https://github.com/abishek-murugan/diamond_dynamics.git
cd diamond_dynamics

# uv will automatically create a virtual environment and install dependencies
uv sync
```

### 3. Launching the Application
Start the Streamlit inference dashboard locally:

```bash
uv run streamlit run streamlit_app/main.py
```

Navigate to `http://localhost:8501` in your web browser to explore the dashboard.

---

## 🧠 Modeling Pipeline

1. **Clustering (Market Segmentation)**: Utilizes a K-Means algorithm to partition diamonds into tiers (e.g., *Affordable Small*, *Premium Heavy*) based on physical proportions. 
2. **Regression (Valuation)**: Leverages an `XGBRegressor` trained on logarithmically scaled prices and dimensions to capture non-linear market pricing dynamics with high accuracy.

> **Note on Model Serialization**: The `.pkl` files in the `models/` directory were natively trained in a RAPIDS environment. The application utilizes a bespoke `pickle.Unpickler` override to bypass `cupy` dependencies, enabling seamless model loading on edge/CPU hardware without requiring retraining.
