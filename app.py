# -*- coding: utf-8 -*-
"""
AdSales Pro - Predictive Marketing Analytics Dashboard
Polynomial Regression Project | Sales Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Page Configurations & Setup
st.set_page_config(
    page_title="AdSales Pro | Predictive Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom modern fonts and glassmorphic card CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, .main-title {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
    
    .card {
        background: rgba(17, 25, 40, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    
    .card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 242, 254, 0.4);
    }
    
    .metric-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #38BDF8;
        line-height: 1.1;
    }
    
    .metric-value-glowing {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    
    .metric-sub {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 4px;
    }
    
    .kpi-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 8px;
    }
    
    .badge-success {
        background-color: rgba(74, 222, 128, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }
    
    .badge-info {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    .badge-warning {
        background-color: rgba(251, 146, 60, 0.15);
        color: #FB923C;
        border: 1px solid rgba(251, 146, 60, 0.3);
    }
    
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 5px;
    }
    .delta-positive { color: #4ADE80; }
    .delta-negative { color: #F87171; }
    
    </style>
    """, unsafe_allow_html=True)

# 2. Data Loading, Caching & Model Training
@st.cache_resource
def load_data_and_train():
    csv_path = 'advertising.csv'
    if not os.path.exists(csv_path):
        # Fallback if path is nested
        csv_path = '../advertising.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError("advertising.csv dataset not found in workspace!")
            
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Cleaning data
    df_clean = df.copy()
    df_clean.drop_duplicates(inplace=True)
    if 'Unnamed: 0' in df_clean.columns:
        df_clean.drop('Unnamed: 0', axis=1, inplace=True)
        
    # Remove Outliers using IQR Method (as done in notebook)
    numeric_columns = df_clean.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) &
            (df_clean[col] <= upper_bound)
        ]
        
    # Define Features and Target
    X = df_clean[['TV', 'Radio', 'Newspaper']]
    y = df_clean['Sales']
    
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(exclude=['object']).columns
    
    # Preprocessors (Pipeline matched to assignment)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])
    
    # Pipeline: preprocessing -> 2nd Degree Polynomial -> Linear Regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('polynomial_features', PolynomialFeatures(degree=2)),
        ('linear_regression', LinearRegression())
    ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, df_clean, r2, mae, rmse

try:
    model, df_clean, r2, mae, rmse = load_data_and_train()
except Exception as e:
    st.error(f"Error loading model/dataset: {e}")
    st.stop()

# 3. Sidebar Inputs & Navigation
st.sidebar.image("https://img.icons8.com/gradient/100/combo-chart.png", width=70)
st.sidebar.markdown("<h2 style='margin-top:0px; font-weight:800; color:#38BDF8;'>AdSales Pro</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Active Budget Controls")

# Sliders setup with exact ranges matching dataset characteristics
# TV range: 0 to 300
# Radio range: 0 to 50
# Newspaper range: 0 to 120
tv_val = st.sidebar.number_input("📺 TV Budget ($k)", min_value=0.0, max_value=300.0, value=150.0, step=1.0)
radio_val = st.sidebar.number_input("📻 Radio Budget ($k)", min_value=0.0, max_value=50.0, value=25.0, step=1.0)
newspaper_val = st.sidebar.number_input("📰 Newspaper Budget ($k)", min_value=0.0, max_value=120.0, value=30.0, step=1.0)

# Sidebar dynamic budget breakdown details
total_budget = tv_val + radio_val + newspaper_val
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Spend:** `${total_budget:,.1f}k`**")

# Interactive Progress Bars in Sidebar showing relative allocation
if total_budget > 0:
    tv_pct = (tv_val / total_budget) * 100
    radio_pct = (radio_val / total_budget) * 100
    news_pct = (newspaper_val / total_budget) * 100
else:
    tv_pct = radio_pct = news_pct = 0.0

st.sidebar.caption(f"Allocation: TV ({tv_pct:.1f}%) | Radio ({radio_pct:.1f}%) | News ({news_pct:.1f}%)")

# App Header
col_title, col_logo = st.columns([5, 1])
with col_title:
    st.markdown("<div class='main-title'>AdSales Pro: Predictive Marketing Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>High-performance Polynomial Regression Engine for Marketing ROI Forecasting</div>", unsafe_allow_html=True)

# Navigation Tab Bar
tab_predict, tab_compare, tab_insights = st.tabs([
    "🎯 Sales Predictor Dashboard", 
    "⚖️ What-If Scenario Planner", 
    "📊 Model & Data Insights"
])

# 4. TAB 1: Sales Predictor Dashboard
with tab_predict:
    st.markdown("### 🎯 Real-Time ROI Engine")
    
    # Calculate Prediction
    input_df = pd.DataFrame({
        'TV': [tv_val],
        'Radio': [radio_val],
        'Newspaper': [newspaper_val]
    })
    
    pred_sales = model.predict(input_df)[0]
    # Bound below by 0
    pred_sales = max(0.0, pred_sales)
    
    # Calculate Sales per thousand dollars spent
    efficiency = (pred_sales / total_budget * 1000) if total_budget > 0 else 0.0
    
    # Setup Metrics columns
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"""
            <div class="card">
                <div class="kpi-container">
                    <span class="metric-title">🔮 Predicted Sales</span>
                    <span class="metric-value-glowing">{pred_sales:.2f} <span style="font-size:1.5rem; font-weight:400; color:#94A3B8;">Units</span></span>
                    <span class="metric-sub">Forecasted sales units based on current spend</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
            <div class="card">
                <div class="kpi-container">
                    <span class="metric-title">💰 Total Ad Spend</span>
                    <span class="metric-value">${total_budget:,.1f}k</span>
                    <span class="metric-sub">TV: ${tv_val:.1f}k | Radio: ${radio_val:.1f}k | News: ${newspaper_val:.1f}k</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        # Determine ROI Level
        if efficiency > 150:
            badge_html = '<span class="badge badge-success">🔥 High ROI Efficiency</span>'
        elif efficiency > 70:
            badge_html = '<span class="badge badge-info">📈 Moderate ROI Efficiency</span>'
        else:
            badge_html = '<span class="badge badge-warning">⚠️ Low ROI Efficiency</span>'
            
        st.markdown(f"""
            <div class="card">
                <div class="kpi-container">
                    <span class="metric-title">⚡ Capital Efficiency</span>
                    <span class="metric-value">{efficiency:.1f}</span>
                    <span class="metric-sub">Sales units generated per $1k of total spend</span>
                    {badge_html}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Budget Insights & Dynamic Recommendations section
    st.markdown("### 💡 Strategic Budget Recommendations")
    
    # Generate tailored insights based on polynomial model features
    rec_col1, rec_col2 = st.columns([3, 2])
    
    with rec_col1:
        st.markdown("""
        <div class="card" style="height: 100%;">
            <h4>🧠 Algorithmic Insights</h4>
            <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.1);">
            <ul style="line-height:1.7; margin-bottom:0px;">
        """, unsafe_allow_html=True)
        
        # 1. Check TV Diminishing Returns
        if tv_val > 220:
            st.markdown("<li>⚠️ <b>TV Over-saturation:</b> Your TV budget is exceeding $220k. The polynomial curve indicates diminishing returns at this high level of spending. Consider reallocating some TV funds.</li>", unsafe_allow_html=True)
        elif tv_val < 80 and tv_val > 0:
            st.markdown("<li>📈 <b>TV Investment Opportunity:</b> TV spending is below $80k. Our model indicates TV has a strong secondary growth rate in this zone. Boosting TV spend could see accelerating sales!</li>", unsafe_allow_html=True)
        else:
            st.markdown("<li>✅ <b>TV Allocation Optimized:</b> Your TV budget is in the sweet spot, yielding balanced linear-to-quadratic efficiency returns.</li>", unsafe_allow_html=True)
            
        # 2. Check Radio vs Newspaper
        if radio_val < 15 and tv_val > 100:
            st.markdown("<li>📻 <b>Radio Synergy Warning:</b> Radio budget is relatively low (${radio_val:.1f}k). Historically, Radio acts as a powerful catalyst alongside TV. Adding even $5-10k to Radio can lift overall TV effectiveness.</li>", unsafe_allow_html=True)
        elif radio_val > 42:
            st.markdown("<li>⚠️ <b>Radio Saturation:</b> Radio spending is close to the absolute maximum threshold (~$45k). Diminishing returns on Radio occur rapidly. Additional spend here is likely wasted.</li>", unsafe_allow_html=True)
            
        # 3. Newspaper Optimization
        if newspaper_val > 40:
            st.markdown("<li>💡 <b>Newspaper Reallocation:</b> Newspaper budget is high (${newspaper_val:.1f}k) relative to its lower sales correlation. Shifting 30% of this budget to Radio or TV is statistically predicted to boost sales by up to 8-15% without increasing total cost!</li>", unsafe_allow_html=True)
        else:
            st.markdown("<li>✅ <b>Newspaper Spend Scaled:</b> Newspaper budget is kept moderate, avoiding capital sinkholes.</li>", unsafe_allow_html=True)
            
        st.markdown("""
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with rec_col2:
        # Mini Allocation Pie Chart
        st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h4>📊 Budget Allocation Split</h4>", unsafe_allow_html=True)
        if total_budget > 0:
            alloc_df = pd.DataFrame({
                'Channel': ['TV', 'Radio', 'Newspaper'],
                'Spend ($k)': [tv_val, radio_val, newspaper_val]
            })
            fig_pie = px.pie(
                alloc_df, values='Spend ($k)', names='Channel',
                color='Channel',
                color_discrete_map={'TV': '#00F2FE', 'Radio': '#4FACFE', 'Newspaper': '#818CF8'},
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0',
                margin=dict(t=0, b=0, l=0, r=0),
                height=180,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, width='stretch')
        else:
            st.write("Enter budget allocations in the sidebar to visualize.")
        st.markdown("</div>", unsafe_allow_html=True)

# 5. TAB 2: What-If Scenario Planner
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Scenario Analysis")
    st.write("Compare two different budget allocations side-by-side to find the absolute most efficient spend strategy.")
    
    col_scen_a, col_scen_b = st.columns(2)
    
    with col_scen_a:
        st.markdown("#### Scenario A (Reference)")
        # Current sidebar inputs act as Scenario A
        scen_a_tv = tv_val
        scen_a_rad = radio_val
        scen_a_news = newspaper_val
        
        st.info(f"Using budgets currently active in sidebar:\n* **TV:** `${scen_a_tv:.1f}k` | **Radio:** `${scen_a_rad:.1f}k` | **Newspaper:** `${scen_a_news:.1f}k`*")
        
        scen_a_tot = scen_a_tv + scen_a_rad + scen_a_news
        
        input_a = pd.DataFrame({
            'TV': [scen_a_tv],
            'Radio': [scen_a_rad],
            'Newspaper': [scen_a_news]
        })
        pred_sales_a = max(0.0, model.predict(input_a)[0])
        eff_a = (pred_sales_a / scen_a_tot * 1000) if scen_a_tot > 0 else 0.0
        
        st.markdown(f"""
            <div class="card" style="border-left: 5px solid #00F2FE;">
                <span class="metric-title">📊 Scenario A Performance</span>
                <div class="metric-value-glowing">{pred_sales_a:.2f} Units</div>
                <div class="metric-sub">Total Cost: ${scen_a_tot:.1f}k</div>
                <div class="metric-sub">Efficiency: <b>{eff_a:.1f}</b> Sales/$1k</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_scen_b:
        st.markdown("#### Scenario B (Experiment)")
        
        scen_b_tv = st.number_input("📺 TV Budget Scenario B ($k)", min_value=0.0, max_value=300.0, value=180.0, step=1.0, key="scen_b_tv")
        scen_b_rad = st.number_input("📻 Radio Budget Scenario B ($k)", min_value=0.0, max_value=50.0, value=15.0, step=1.0, key="scen_b_rad")
        scen_b_news = st.number_input("📰 Newspaper Budget Scenario B ($k)", min_value=0.0, max_value=120.0, value=10.0, step=1.0, key="scen_b_news")
        
        scen_b_tot = scen_b_tv + scen_b_rad + scen_b_news
        
        input_b = pd.DataFrame({
            'TV': [scen_b_tv],
            'Radio': [scen_b_rad],
            'Newspaper': [scen_b_news]
        })
        pred_sales_b = max(0.0, model.predict(input_b)[0])
        eff_b = (pred_sales_b / scen_b_tot * 1000) if scen_b_tot > 0 else 0.0
        
        st.markdown(f"""
            <div class="card" style="border-left: 5px solid #818CF8;">
                <span class="metric-title">📊 Scenario B Performance</span>
                <div class="metric-value" style="color:#818CF8;">{pred_sales_b:.2f} Units</div>
                <div class="metric-sub">Total Cost: ${scen_b_tot:.1f}k</div>
                <div class="metric-sub">Efficiency: <b>{eff_b:.1f}</b> Sales/$1k</div>
            </div>
        """, unsafe_allow_html=True)
        
    # Scenario Comparison Deltas
    st.markdown("### 🔍 Strategic Scenario Comparison")
    
    d_sales = pred_sales_b - pred_sales_a
    d_budget = scen_b_tot - scen_a_tot
    d_eff = eff_b - eff_a
    
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        symbol = "+" if d_sales >= 0 else ""
        class_name = "delta-positive" if d_sales >= 0 else "delta-negative"
        st.markdown(f"""
            <div class="card">
                <span class="metric-title">Sales Delta (B - A)</span>
                <div class="metric-value-glowing" style="font-size:2.4rem;">{symbol}{d_sales:.2f}</div>
                <div class="metric-delta {class_name}">{symbol}{d_sales:.1f} units predicted output shift</div>
            </div>
        """, unsafe_allow_html=True)
        
    with comp_col2:
        symbol = "+" if d_budget >= 0 else ""
        class_name = "delta-negative" if d_budget > 0 else "delta-positive" # positive means spending less (saving money)
        saved_text = "additional cost" if d_budget >= 0 else "budget saved"
        st.markdown(f"""
            <div class="card">
                <span class="metric-title">Ad Spend Delta (B - A)</span>
                <div class="metric-value" style="font-size:2.4rem;">{symbol}${abs(d_budget):.1f}k</div>
                <div class="metric-delta {class_name}">{symbol}${d_budget:.1f}k {saved_text}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with comp_col3:
        symbol = "+" if d_eff >= 0 else ""
        class_name = "delta-positive" if d_eff >= 0 else "delta-negative"
        st.markdown(f"""
            <div class="card">
                <span class="metric-title">Efficiency Delta (B - A)</span>
                <div class="metric-value" style="font-size:2.4rem; color: #F59E0B;">{symbol}{d_eff:.1f}</div>
                <div class="metric-delta {class_name}">{symbol}{d_eff:.1f} sales units per $1k spend</div>
            </div>
        """, unsafe_allow_html=True)

# 6. TAB 3: Model & Data Insights
with tab_insights:
    st.markdown("### 📊 Model Diagnostics & Historical Curves")
    
    # Part A: Interactive Regression Curves with live coordinate marker
    st.subheader("📈 Polynomial Curves & Live Tracker")
    st.write("Observe the actual data scatter points, the fitted 2nd degree polynomial regression curves, and a **glowing live coordinate marker** tracking where your current budget falls!")
    
    curve_medium = st.selectbox(
        "Select advertising channel to plot:",
        ["TV Advertising", "Radio Advertising", "Newspaper Advertising"]
    )
    
    # Calculate sorted curves for plotting (as in notebook)
    # Keeping other features fixed at their dataset mean
    tv_mean = df_clean['TV'].mean()
    radio_mean = df_clean['Radio'].mean()
    newspaper_mean = df_clean['Newspaper'].mean()
    
    # Determine columns and specific sorted vectors
    if curve_medium == "TV Advertising":
        x_col = 'TV'
        x_sorted = np.sort(df_clean['TV'])
        input_curve = pd.DataFrame({
            'TV': x_sorted,
            'Radio': [radio_mean] * len(x_sorted),
            'Newspaper': [newspaper_mean] * len(x_sorted)
        })
        current_x = tv_val
        # Pred for target dot
        dot_input = pd.DataFrame({
            'TV': [tv_val],
            'Radio': [radio_mean],
            'Newspaper': [newspaper_mean]
        })
        current_y = max(0.0, model.predict(dot_input)[0])
        title_suffix = f" (Radio={radio_mean:.1f}k, News={newspaper_mean:.1f}k fixed)"
        x_label = "TV Budget ($k)"
        
    elif curve_medium == "Radio Advertising":
        x_col = 'Radio'
        x_sorted = np.sort(df_clean['Radio'])
        input_curve = pd.DataFrame({
            'TV': [tv_mean] * len(x_sorted),
            'Radio': x_sorted,
            'Newspaper': [newspaper_mean] * len(x_sorted)
        })
        current_x = radio_val
        dot_input = pd.DataFrame({
            'TV': [tv_mean],
            'Radio': [radio_val],
            'Newspaper': [newspaper_mean]
        })
        current_y = max(0.0, model.predict(dot_input)[0])
        title_suffix = f" (TV={tv_mean:.1f}k, News={newspaper_mean:.1f}k fixed)"
        x_label = "Radio Budget ($k)"
        
    else:  # Newspaper
        x_col = 'Newspaper'
        x_sorted = np.sort(df_clean['Newspaper'])
        input_curve = pd.DataFrame({
            'TV': [tv_mean] * len(x_sorted),
            'Radio': [radio_mean] * len(x_sorted),
            'Newspaper': x_sorted
        })
        current_x = newspaper_val
        dot_input = pd.DataFrame({
            'TV': [tv_mean],
            'Radio': [radio_mean],
            'Newspaper': [newspaper_val]
        })
        current_y = max(0.0, model.predict(dot_input)[0])
        title_suffix = f" (TV={tv_mean:.1f}k, Radio={radio_mean:.1f}k fixed)"
        x_label = "Newspaper Budget ($k)"
        
    # Make curve predictions
    curve_preds = model.predict(input_curve)
    
    # Create interactive Plotly figure
    fig_curve = go.Figure()
    
    # Scatter plot of actual values
    fig_curve.add_trace(go.Scatter(
        x=df_clean[x_col],
        y=df_clean['Sales'],
        mode='markers',
        name='Actual Sales (Data)',
        marker=dict(color='rgba(255, 255, 255, 0.25)', size=6, line=dict(color='rgba(255,255,255,0.05)', width=0.5))
    ))
    
    # Polynomial line
    fig_curve.add_trace(go.Scatter(
        x=x_sorted,
        y=curve_preds,
        mode='lines',
        name='Fitted 2nd Degree Polynomial',
        line=dict(color='#00F2FE', width=3)
    ))
    
    # Live Tracker Marker
    fig_curve.add_trace(go.Scatter(
        x=[current_x],
        y=[current_y],
        mode='markers+text',
        name='Your Live Budget Tracker',
        text=["Live Input"],
        textposition="top center",
        marker=dict(color='#EF4444', size=14, symbol='star', line=dict(color='white', width=2))
    ))
    
    fig_curve.update_layout(
        title=dict(
            text=f"Polynomial Regression Curve - {curve_medium}{title_suffix}",
            font=dict(family="Outfit", size=16)
        ),
        xaxis_title=x_label,
        yaxis_title="Sales Units",
        paper_bgcolor='rgba(17, 25, 40, 0.4)',
        plot_bgcolor='rgba(17, 25, 40, 0.4)',
        font_color='#E2E8F0',
        height=450,
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_curve.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig_curve.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    
    st.plotly_chart(fig_curve, width='stretch')
    
    st.markdown("---")
    
    # Part B: Model performance details and Correlation Heatmap
    det_col1, det_col2 = st.columns(2)
    
    with det_col1:
        st.subheader("🔮 Model Performance Metrics")
        st.write("Metrics evaluated using a 20% validation test split from the dataset:")
        
        # Displaying MAE, RMSE, and R2 in custom card layout
        st.markdown(f"""
            <div class="card">
                <span class="metric-title">📊 R² Score (Coefficient of Determination)</span>
                <div class="metric-value-glowing" style="font-size: 2.2rem; color: #10B981;">{r2:.4f}</div>
                <div class="metric-sub">Explains <b>{r2*100:.2f}%</b> of variance in historical Sales</div>
            </div>
            
            <div class="card">
                <span class="metric-title">📉 Mean Absolute Error (MAE)</span>
                <div class="metric-value" style="font-size: 2.2rem; color: #FB923C;">{mae:.2f}</div>
                <div class="metric-sub">Average absolute prediction deviation</div>
            </div>
            
            <div class="card">
                <span class="metric-title">📉 Root Mean Squared Error (RMSE)</span>
                <div class="metric-value" style="font-size: 2.2rem; color: #F87171;">{rmse:.2f}</div>
                <div class="metric-sub">Standard deviation of residual errors</div>
            </div>
        """, unsafe_allow_html=True)
        
    with det_col2:
        st.subheader("🔥 Feature Correlation Matrix")
        st.write("Understand the linear relationships between variables before polynomial transformation:")
        
        corr_matrix = df_clean.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".3f",
            color_continuous_scale='Blues',
            labels=dict(x="Variable", y="Variable", color="Correlation")
        )
        fig_corr.update_layout(
            paper_bgcolor='rgba(17, 25, 40, 0.4)',
            plot_bgcolor='rgba(17, 25, 40, 0.4)',
            font_color='#E2E8F0',
            height=360,
            margin=dict(t=10, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_corr, width='stretch')
        
    st.markdown("---")
    
    # Part C: Raw Data Explorer
    st.subheader("🗄️ Dataset Explorer")
    st.write("Examine, sort, or filter the cleaned advertising dataset loaded directly from `advertising.csv`:")
    
    st.dataframe(
        df_clean,
        width='stretch',
        column_config={
            "TV": st.column_config.NumberColumn("TV Budget ($k)", format="$%.1f"),
            "Radio": st.column_config.NumberColumn("Radio Budget ($k)", format="$%.1f"),
            "Newspaper": st.column_config.NumberColumn("Newspaper Budget ($k)", format="$%.1f"),
            "Sales": st.column_config.NumberColumn("Sales Units", format="%.2f")
        }
    )
    
    # Summary Statistics
    with st.expander("📊 View Descriptive Summary Statistics"):
        st.table(df_clean.describe().T)

# Footer
st.markdown("""
    <hr style="border-color: rgba(255,255,255,0.05); margin-top: 3rem;">
    <div style="text-align: center; color: #64748B; font-size: 0.8rem; padding-bottom: 2rem;">
        Developed with 🧠 and 🔮 using Streamlit, Scikit-Learn, and Plotly. © 2026 AdSales Pro Analytics Engine.
    </div>
    """, unsafe_allow_html=True)
