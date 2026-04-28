import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Linear Regression CRISP-DM Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #007bff !important; color: white !important; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🛠️ Configuration")
with st.sidebar:
    st.header("Data Parameters")
    n_samples = st.slider("Number of samples (n)", 100, 1000, 500)
    noise_var = st.slider("Noise Variance", 0, 1000, 100)
    seed = st.number_input("Random Seed", value=42)
    
    st.divider()
    st.header("Ground Truth (Hidden from Model)")
    true_a = st.slider("True Slope (a)", -10.0, 10.0, 2.5)
    true_b = st.slider("True Intercept (b)", -50.0, 50.0, 10.0)
    noise_mean = st.slider("Noise Mean", -10.0, 10.0, 0.0)
    
    generate_data_btn = st.button("Generate Data", type="primary", use_container_width=True)

# --- Data Generation Logic ---
@st.cache_data(show_spinner=False)
def generate_data(n, a, b, n_mean, n_var, s):
    np.random.seed(s)
    x = np.random.uniform(-100, 100, (n, 1))
    noise = np.random.normal(n_mean, np.sqrt(n_var), (n, 1))
    y = a * x + b + noise
    df = pd.DataFrame(data=np.hstack([x, y]), columns=['X', 'y'])
    return df, x, y

# Initialize Data
if 'df' not in st.session_state or generate_data_btn:
    df, x, y = generate_data(n_samples, true_a, true_b, noise_mean, noise_var, seed)
    st.session_state.df = df
    st.session_state.x = x
    st.session_state.y = y
    st.session_state.params = {'a': true_a, 'b': true_b}
    # Clear model if data changes
    if 'model' in st.session_state:
        del st.session_state.model

# --- Main App ---
st.title("🚀 Linear Regression Workflow (CRISP-DM)")
st.markdown("---")

tabs = st.tabs([
    "📍 1. Business Understanding",
    "📊 2. Data Understanding",
    "⚙️ 3. Data Preparation",
    "🏗️ 4. Modeling",
    "🧪 5. Evaluation",
    "🚢 6. Deployment"
])

# 1. Business Understanding
with tabs[0]:
    st.header("1. Business Understanding")
    st.info("**Objective:** Predict a continuous target variable 'y' based on input feature 'x' using a linear relationship.")
    st.markdown("""
    **Business Problem:** In many industries, we need to understand how one variable affects another. For instance:
    *   Advertising spend vs Sales
    *   Engine size vs Fuel efficiency
    *   Experience vs Salary
    
    In this demo, we aim to recover the 'true' underlying physical laws (the slope and intercept) from noisy data.
    """)

# 2. Data Understanding
with tabs[1]:
    st.header("2. Data Understanding")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.subheader("Statistics")
        st.write(st.session_state.df.describe())
    
    with col2:
        st.subheader("Visual Exploration")
        fig = px.scatter(st.session_state.df, x='X', y='y', title="Scatter Plot of Raw Data",
                         labels={'X': 'Feature X', 'y': 'Target y'},
                         opacity=0.6, color_discrete_sequence=['#4B90E2'])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# 3. Data Preparation
with tabs[2]:
    st.header("3. Data Preparation")
    st.markdown("Splitting data into training and testing sets, then scaling features.")
    
    test_ratio = st.select_slider("Test Set Split Ratio", options=[0.1, 0.2, 0.3, 0.4, 0.5], value=0.2)
    
    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state.x, st.session_state.y, test_size=test_ratio, random_state=seed
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.session_state.prepared_data = {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler, 'X_test_raw': X_test
    }
    
    st.success(f"✅ Data Prepared! Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    with st.expander("Show Prep Data"):
        st.write("Scaled Training Data (first 5):", X_train_scaled[:5])

# 4. Modeling
with tabs[3]:
    st.header("4. Modeling")
    st.markdown("Fitting a `LinearRegression` model using scikit-learn.")
    
    if st.button("🚀 Train Model", type="primary"):
        prep = st.session_state.prepared_data
        model = LinearRegression()
        model.fit(prep['X_train'], prep['y_train'])
        
        st.session_state.model = model
        st.balloons()
        st.success("Model trained successfully!")

    if 'model' in st.session_state:
        model = st.session_state.model
        scaler = st.session_state.prepared_data['scaler']
        
        # Calculate unscaled parameters
        learned_a = model.coef_[0][0] / scaler.scale_[0]
        learned_b = model.intercept_[0] - (learned_a * scaler.mean_[0])
        
        st.session_state.learned_params = {'a': learned_a, 'b': learned_b}
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Learned Slope (a)", f"{learned_a:.4f}", delta=f"{learned_a - st.session_state.params['a']:.4f}")
        with col2:
            st.metric("Learned Intercept (b)", f"{learned_b:.4f}", delta=f"{learned_b - st.session_state.params['b']:.4f}")

# 5. Evaluation
with tabs[4]:
    st.header("5. Evaluation")
    if 'model' in st.session_state:
        prep = st.session_state.prepared_data
        y_pred = st.session_state.model.predict(prep['X_test'])
        
        mse = mean_squared_error(prep['y_test'], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(prep['y_test'], y_pred)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MSE", f"{mse:.2f}")
        m2.metric("RMSE", f"{rmse:.2f}")
        m3.metric("R² Score", f"{r2:.4f}")
        
        # Plotting Regression Line
        st.subheader("Model Fit")
        line_x = np.linspace(-100, 100, 100).reshape(-1, 1)
        line_x_scaled = prep['scaler'].transform(line_x)
        line_y = st.session_state.model.predict(line_x_scaled)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prep['X_test_raw'].flatten(), y=prep['y_test'].flatten(), mode='markers', name='Actual (Test)', opacity=0.5))
        fig.add_trace(go.Scatter(x=line_x.flatten(), y=line_y.flatten(), mode='lines', name='Predicted Line', line=dict(color='red', width=3)))
        fig.update_layout(template="plotly_white", xaxis_title="X", yaxis_title="y")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please train the model first in the 'Modeling' tab.")

# 6. Deployment
with tabs[5]:
    st.header("6. Deployment")
    if 'model' in st.session_state:
        st.subheader("Interactive Prediction")
        user_x = st.number_input("Enter X value:", value=0.0)
        if st.button("Predict"):
            user_x_scaled = st.session_state.prepared_data['scaler'].transform([[user_x]])
            pred_y = st.session_state.model.predict(user_x_scaled)
            st.success(f"Predicted y: **{pred_y[0][0]:.4f}**")
        
        st.divider()
        st.subheader("Save Model")
        if st.button("💾 Download Model & Scaler"):
            joblib.dump(st.session_state.model, "linear_model.joblib")
            joblib.dump(st.session_state.prepared_data['scaler'], "scaler.joblib")
            st.success("Files 'linear_model.joblib' and 'scaler.joblib' saved to local directory.")
            
            with open("linear_model.joblib", "rb") as f:
                st.download_button("Download model.joblib", f, file_name="linear_model.joblib")
    else:
        st.warning("Model not found. Please follow the steps in order.")

# Footer
st.markdown("---")
st.markdown("<center>Built by Antigravity | Linear Regression Demo</center>", unsafe_allow_html=True)
