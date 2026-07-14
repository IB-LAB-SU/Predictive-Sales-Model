import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="Retail Insights ML Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Retail Revenue & Profit Predictive Engine")
st.markdown("""
This interactive dashboard demonstrates a dual-objective machine learning pipeline built using **scikit-learn**. 
Instead of optimizing purely for top-line revenue, this system isolates and compares the structural features driving gross **Sales Amount** versus bottom-line **Net Profit**.
""")
st.markdown("---")


@st.cache_data
def run_analytics_pipeline():
    # Load raw asset
    df = pd.read_csv('retail_sales_cleaned(1).csv')
    

    for col in ['gender', 'region', 'product_category', 'payment_method', 'order_status']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            

    df['days_to_ship_imputed'] = df['days_to_ship'].isna().astype(int)
    df['days_to_ship'] = df['days_to_ship'].fillna(df['days_to_ship'].median())

    drop_cols = [
        'order_id', 'order_date', 'customer_id', 'customer_name', 'city', 'product_name',
        'sales_amount', 'profit', 'customer_satisfaction', 'age', 'quantity'
    ]
    X = df.drop(columns=drop_cols)
    

    X_encoded = pd.get_dummies(X, columns=['gender', 'region', 'product_category', 'payment_method', 'order_status'], drop_first=True)
    
    return df, X_encoded, df['sales_amount'], df['profit']


df, X_encoded, y_sales, y_profit = run_analytics_pipeline()


X_train, X_test, y_sales_train, y_sales_test = train_test_split(X_encoded, y_sales, test_size=0.2, random_state=42)
_, _, y_profit_train, y_profit_test = train_test_split(X_encoded, y_profit, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


sales_model = LinearRegression().fit(X_train_scaled, y_sales_train)
profit_model = LinearRegression().fit(X_train_scaled, y_profit_train)


sales_r2 = sales_model.score(X_test_scaled, y_sales_test)
profit_r2 = profit_model.score(X_test_scaled, y_profit_test)




st.sidebar.header("🕹️ Model Parameters")
target_selection = st.sidebar.radio(
    "Select Prediction Objective:",
    ("Sales Volume Optimization", "Profit Margin Optimization")
)

st.sidebar.markdown("""
### Model Methodology
* **Algorithm:** Multi-variable OLS Linear Regression
* **Scaling:** Z-score Standard Normalization
* **Encoding:** K-1 Binary Dummy Variates
""")


metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.metric(label="Sales Model Variance Explained ($R^2$)", value=f"{sales_r2 * 100:.2f}%")
with metric_col2:
    st.metric(label="Profit Model Variance Explained ($R^2$)", value=f"{profit_r2 * 100:.2f}%")

st.markdown("---")


trends_df = pd.DataFrame({
    'Feature Space': X_encoded.columns,
    'Sales Weight': sales_model.coef_,
    'Profit Weight': profit_model.coef_
})


if target_selection == "Sales Volume Optimization":
    st.subheader("📈 Core Feature Impact Matrix: Sales Target")
    st.write("Bars extending right represent features driving upward revenue trends; bars extending left denote structural volume drags.")
    
    sorted_sales = trends_df.sort_values(by='Sales Weight', ascending=True)
    fig_sales = px.bar(
        sorted_sales, 
        x='Sales Weight', 
        y='Feature Space', 
        orientation='h',
        color='Sales Weight',
        color_continuous_scale=px.colors.sequential.Bluered,
        labels={'Sales Weight': 'Coefficient Magnitude (Impact Variance)'}
    )
    fig_sales.update_layout(height=650, margin=dict(l=50, r=50, t=30, b=50))
    st.plotly_chart(fig_sales, use_container_width=True)

else:
    st.subheader("💵 Core Feature Impact Matrix: Profit Target")
    st.write("Isolates which parameters protect financial margins vs variables that diminish actual net operational profitability.")
    
    sorted_profit = trends_df.sort_values(by='Profit Weight', ascending=True)
    fig_profit = px.bar(
        sorted_profit, 
        x='Profit Weight', 
        y='Feature Space', 
        orientation='h',
        color='Profit Weight',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'Profit Weight': 'Coefficient Magnitude (Impact Variance)'}
    )
    fig_profit.update_layout(height=650, margin=dict(l=50, r=50, t=30, b=50))
    st.plotly_chart(fig_profit, use_container_width=True)