# CRISP-DM 線性回歸 Streamlit 示範程式

本專案是一個單一檔案的 Streamlit 應用程式，旨在展示使用 **CRISP-DM**（跨行業資料探勘標準流程）方法論的完整數據科學工作流。

## 🚀 專案概覽
該應用程式會自動產生模擬的線性資料，並引導使用者走過 CRISP-DM 的六個階段。

**🔗 [線上展示 (Live Demo)](https://0428dic7-kbxzkxszwxrwxkklupy9cq.streamlit.app/)**

### CRISP-DM 階段詳解：
1. **商業理解 (Business Understanding)**：定義預測目標與商業價值。
2. **資料理解 (Data Understanding)**：探索資料特徵並進行初步視覺化。
3. **資料前處理 (Data Preparation)**：進行特徵縮放（Scaling）與資料集分割（Train-Test Split）。
4. **建模 (Modeling)**：使用 scikit-learn 訓練線性回歸模型。
5. **評估 (Evaluation)**：透過 MSE、RMSE 與 R² 指標評估模型表現。
6. **部署 (Deployment)**：提供即時分析與模型匯出功能（使用 `joblib`）。

## 🛠️ 技術棧
- **Python**
- **Streamlit**：網頁介面開發
- **Scikit-Learn**：機器學習核心邏輯
- **Plotly**：互動式視覺化圖表
- **Pandas/Numpy**：資料處理與運算

## 📖 如何在本地執行
1. 安裝必要套件：
   ```bash
   pip install streamlit scikit-learn pandas numpy plotly joblib matplotlib
   ```
2. 執行應用程式：
   ```bash
   streamlit run app.py
   ```

## 📄 授權說明
本專案僅供教學與示範用途。
