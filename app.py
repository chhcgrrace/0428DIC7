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

# --- 頁面配置 ---
st.set_page_config(
    page_title="線性回歸 CRISP-DM 示範程式",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定義樣式 ---
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

# --- 側邊欄 ---
st.sidebar.title("🛠️ 參數設定")
with st.sidebar:
    st.header("1. 資料產生參數")
    n_samples = st.slider("樣本數 (n)", 100, 1000, 500)
    noise_var = st.slider("雜訊變異數 (Variance)", 0, 1000, 100)
    seed = st.number_input("亂數種子 (Seed)", value=42)
    
    st.divider()
    st.header("2. 真實規律 (對模型隱藏)")
    true_a = st.slider("真實斜率 (a)", -10.0, 10.0, 2.5)
    true_b = st.slider("真實截距 (b)", -50.0, 50.0, 10.0)
    noise_mean = st.slider("雜訊平均值", -10.0, 10.0, 0.0)
    
    generate_data_btn = st.button("重新產生資料", type="primary", use_container_width=True)

# --- 資料產生邏輯 ---
@st.cache_data(show_spinner=False)
def generate_data(n, a, b, n_mean, n_var, s):
    np.random.seed(s)
    x = np.random.uniform(-100, 100, (n, 1))
    noise = np.random.normal(n_mean, np.sqrt(n_var), (n, 1))
    y = a * x + b + noise
    df = pd.DataFrame(data=np.hstack([x, y]), columns=['特徵 X', '目標 y'])
    return df, x, y

# 初始化資料
if 'df' not in st.session_state or generate_data_btn:
    df, x, y = generate_data(n_samples, true_a, true_b, noise_mean, noise_var, seed)
    st.session_state.df = df
    st.session_state.x = x
    st.session_state.y = y
    st.session_state.params = {'a': true_a, 'b': true_b}
    # 當資料改變時清除現有模型
    if 'model' in st.session_state:
        del st.session_state.model

# --- 主程式介面 ---
st.title("🚀 線性回歸工作流 (CRISP-DM 範例)")
st.markdown("---")

tabs = st.tabs([
    "📍 1. 商業理解",
    "📊 2. 資料理解",
    "⚙️ 3. 資料準備",
    "🏗️ 4. 建模",
    "🧪 5. 評估",
    "🚢 6. 部署"
])

# 1. 商業理解
with tabs[0]:
    st.header("🎯 第一階段：商業理解 (Business Understanding)")
    st.info("**目標：** 訓練一個線性回歸模型，根據輸入特徵 'x' 預測連續型目標變數 'y'。")
    st.markdown("""
    **商業問題：** 在各行各業中，我們常需要建立變數間的關聯。例如：
    *   廣告支出 vs 銷售額
    *   引擎排氣量 vs 油耗
    *   工作年資 vs 薪資
    
    在本示範中，我們的目標是從含有雜訊的觀測資料中，還原出背後的「真實物理規律」（斜率與截距）。
    """)

# 2. 資料理解
with tabs[1]:
    st.header("🔍 第二階段：資料理解 (Data Understanding)")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("資料預覽 (前 10 筆)")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.subheader("統計描述")
        st.write(st.session_state.df.describe())
    
    with col2:
        st.subheader("探索性資料分析 (EDA)")
        fig = px.scatter(st.session_state.df, x='特徵 X', y='目標 y', title="原始資料散點圖",
                         labels={'特徵 X': '特徵值 (X)', '目標 y': '目標值 (y)'},
                         opacity=0.6, color_discrete_sequence=['#4B90E2'])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# 3. 資料準備
with tabs[2]:
    st.header("🛠️ 第三階段：資料準備 (Data Preparation)")
    st.markdown("此階段包含資料集分割（訓練與測試集）以及特徵標準化（Standard Scaling）。")
    
    test_ratio = st.select_slider("測試集分割比例 (Test Ratio)", options=[0.1, 0.2, 0.3, 0.4, 0.5], value=0.2)
    
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
    
    st.success(f"✅ 資料準備完成！訓練集：{len(X_train)} 筆，測試集：{len(X_test)} 筆")
    
    with st.expander("查看處理後的資料"):
        st.write("標準化後的訓練特徵 (前 5 筆)：", X_train_scaled[:5])

# 4. 建模
with tabs[3]:
    st.header("🏗️ 第四階段：建模 (Modeling)")
    st.markdown("使用 scikit-learn 的 `LinearRegression` 演算法進行模型訓練。")
    
    if st.button("🚀 開始訓練模型", type="primary"):
        prep = st.session_state.prepared_data
        model = LinearRegression()
        model.fit(prep['X_train'], prep['y_train'])
        
        st.session_state.model = model
        st.balloons()
        st.success("✨ 模型訓練成功！")

    if 'model' in st.session_state:
        model = st.session_state.model
        scaler = st.session_state.prepared_data['scaler']
        
        # 還原標準化後的參數
        learned_a = model.coef_[0][0] / scaler.scale_[0]
        learned_b = model.intercept_[0] - (learned_a * scaler.mean_[0])
        
        st.session_state.learned_params = {'a': learned_a, 'b': learned_b}
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("模型學到的斜率 (a')", f"{learned_a:.4f}", 
                      delta=f"{learned_a - st.session_state.params['a']:.4f}",
                      delta_color="normal")
        with col2:
            st.metric("模型學到的截距 (b')", f"{learned_b:.4f}", 
                      delta=f"{learned_b - st.session_state.params['b']:.4f}",
                      delta_color="normal")

# 5. 評估
with tabs[4]:
    st.header("🧪 第五階段：評估 (Evaluation)")
    if 'model' in st.session_state:
        prep = st.session_state.prepared_data
        y_pred = st.session_state.model.predict(prep['X_test'])
        
        mse = mean_squared_error(prep['y_test'], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(prep['y_test'], y_pred)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("均方誤差 (MSE)", f"{mse:.2f}")
        m2.metric("均方根誤差 (RMSE)", f"{rmse:.2f}")
        m3.metric("判定係數 (R² Score)", f"{r2:.4f}")
        
        # 繪製回歸線
        st.subheader("模型擬合視覺化")
        line_x = np.linspace(-100, 100, 100).reshape(-1, 1)
        line_x_scaled = prep['scaler'].transform(line_x)
        line_y = st.session_state.model.predict(line_x_scaled)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prep['X_test_raw'].flatten(), y=prep['y_test'].flatten(), 
                                 mode='markers', name='測試集實際值', opacity=0.5))
        fig.add_trace(go.Scatter(x=line_x.flatten(), y=line_y.flatten(), 
                                 mode='lines', name='模型預測線', line=dict(color='red', width=3)))
        fig.update_layout(template="plotly_white", xaxis_title="特徵 X", yaxis_title="目標 y")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("請先在『建模』分頁中訓練模型。")

# 6. 部署
with tabs[5]:
    st.header("🚢 第六階段：部署 (Deployment)")
    if 'model' in st.session_state:
        st.subheader("即時預測功能")
        user_x = st.number_input("輸入 X 值進行預測：", value=0.0)
        if st.button("進行預測"):
            user_x_scaled = st.session_state.prepared_data['scaler'].transform([[user_x]])
            pred_y = st.session_state.model.predict(user_x_scaled)
            st.success(f"結果：當 X = **{user_x}** 時，預測之 y 值約為 **{pred_y[0][0]:.4f}**")
        
        st.divider()
        st.subheader("模型保存與匯出")
        if st.button("💾 下載模型與標準化工具 (joblib)"):
            joblib.dump(st.session_state.model, "linear_model.joblib")
            joblib.dump(st.session_state.prepared_data['scaler'], "scaler.joblib")
            st.success("成功在本地端生成 'linear_model.joblib' 與 'scaler.joblib'！")
            
            with open("linear_model.joblib", "rb") as f:
                st.download_button("點此下載模型 (Model)", f, file_name="linear_model.joblib")
    else:
        st.warning("尚未偵測到訓練好的模型。請依序執行 CRISP-DM 步驟。")

# 頁尾
st.markdown("---")
st.markdown("<center>由 Antigravity 建立 | 線性回歸教學展示</center>", unsafe_allow_html=True)
