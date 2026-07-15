# Diamond Dynamics

Diamond Dynamics is a machine learning portal that predicts the market value of diamonds in INR and categorizes them into distinct commercial segments.

## Project Structure

This project follows a modular, production-ready ML architecture.

- `src/`: Source code
- `config/`: Configuration files
- `tests/`: Unit and integration tests
- `models/`: Trained model artifacts
- `notebooks/`: Original EDA and model training notebooks
- `streamlit_app/`: Streamlit web application

## Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

## Running the Application

```bash
uv run streamlit run streamlit_app/main.py
```
