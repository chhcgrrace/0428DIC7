# CRISP-DM Linear Regression Streamlit Demo

This repository contains a single-file Streamlit application that demonstrates a complete data science workflow using the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

## 🚀 Overview
The application generates synthetic linear data and walks through the six phases of CRISP-DM.

**🔗 [Live Demo](https://0428dic7-kbxzkxszwxrwxkklupy9cq.streamlit.app/)**

### CRISP-DM Phases:
1. **Business Understanding**: Defining the prediction goal.
2. **Data Understanding**: Exploring and visualizing synthetic data.
3. **Data Preparation**: Feature scaling and train-test splitting.
4. **Modeling**: Training a Linear Regression model with scikit-learn.
5. **Evaluation**: Assessing performance using MSE, RMSE, and R².
6. **Deployment**: Interactive prediction and model export via `joblib`.

## 🛠️ Tech Stack
- **Python**
- **Streamlit**: Web Interface
- **Scikit-Learn**: Machine Learning Logic
- **Plotly**: Interactive Visualizations
- **Pandas/Numpy**: Data Processing

## 📖 How to Run
1. Install dependencies:
   ```bash
   pip install streamlit scikit-learn pandas numpy plotly joblib
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📄 License
This project is for educational purposes.
