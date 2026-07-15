#!/bin/bash
set -e

# Remove old git
rm -rf .git

# Initialize new git
git init
git remote add origin https://github.com/abishek-murugan/diamond_dynamics.git
git checkout -b main

# 1. Project Initialization
git add README.md .gitignore pyproject.toml .python-version uv.lock
git commit -m "chore: initialize project with uv and dependencies"

# 2. Configuration Setup
git add config/
git commit -m "feat: add application and model configurations"

# 3. Data Integration
git add data/
git commit -m "data: integrate raw diamonds dataset"

# 4. Models and Artifacts
git add models/
git commit -m "feat: add trained regression and clustering models"

# 5. Notebooks and Research
git add notebooks/
git commit -m "docs: add EDA and model training notebooks"

# 6. Core Utilities
git add src/__init__.py src/utils/
git commit -m "feat: implement core utility modules (logger, exceptions, config)"

# 7. Feature Engineering
git add src/features/
git commit -m "feat: implement robust feature engineering pipeline"

# 8. Inference Engine
git add src/inference/
git commit -m "feat: develop predictor module for inference"

# 9. Testing
git add tests/
git commit -m "test: add unit tests for feature engineering"

# 10. Visualization and Streamlit App
git add plots/ streamlit_app/
git commit -m "feat: build production-ready Streamlit dashboard"

# 11. Final Catch-all (if anything was missed)
git add .
git commit -m "chore: finalize production repository structure" || true

# Force push to reset the remote history
# But wait, it's safer just to set it up locally and let the user push or I can push if they gave credentials. The user didn't ask to push, just "connect to this... reset that git commit history and make atleast 10 clean git commits"
