import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import os

# Page configuration
st.set_page_config(
    page_title="Market Maker Bot Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🤖 Quant Execution Bot - Market Making Dashboard")
st.markdown("### Real-time visualization of market-making simulation results")

# Load data
@st.cache_data
def load_simulation_report():
    csv_path = 'simulation_report.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_trace_data():
    trace_path = 'trace.json'
    if os.path.exists(trace_path):
        with open(trace_path, 'r') as f:
            return json.load(f)
    return None

# Load the data
df_report = load_simulation_report()
trace_data = load_trace_data()

if df_report is None:
    st.error("⚠️ No simulation data found. Please run `python3 main.py` first.")
    st.stop()

# Sidebar - Market selector
st.sidebar.header("📈 Controls")
selected_market = st.sidebar.selectbox(
    "Select Market",
    df_report['market'].tolist(),
    index=0
)

# Main metrics row
st.header("📊 Summary Metrics")
cols = st.columns(4)

with cols[0]:
    st.metric(
        label="Total Markets",
        value=len(df_report),
        delta=None
    )

with cols[1]:
    total_pnl = df_report['pnl'].sum()
    st.metric(
        label="Total PnL",
        value=f"${total_pnl:.2f}",
        delta=f"{total_pnl:.2f}" if total_pnl > 0 else None
    )

with cols[2]:
    total_fills = df_report['fill_count'].sum()
    st.metric(
        label="Total Fills",
        value=f"{int(total_fills)}",
        delta=None
    )

with cols[3]:
    total_notional = df_report['notional'].sum()
    st.metric(
        label="Total Notional",
        value=f"${total_notional:.2f}",
        delta=None
    )

# Market comparison section
st.header("🏪 Market Comparison")

col1, col2 = st.columns(2)

with col1:
    # PnL by market
    fig_pnl = px.bar(
        df_report,
        x='market',
        y='pnl',
        title='PnL by Market',
        color='pnl',
        color_continuous_scale=['red', 'yellow', 'green'],
        labels={'pnl': 'PnL ($)', 'market': 'Market'}
    )
    fig_pnl.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_pnl, use_container_width=True)

with col2:
    # Inventory by market
    fig_inv = px.bar(
        df_report,
        x='market',
        y='inventory',
        title='Final Inventory by Market',
        color='inventory',
        color_continuous_scale=['blue', 'lightblue', 'orange'],
        labels={'inventory': 'Inventory', 'market': 'Market'}
    )
    fig_inv.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_inv, use_container_width=True)

# Market details table
st.header("📋 Detailed Market Metrics")

# Format the dataframe with color coding
def color_pnl(val):
    if val > 0:
        return 'background-color: #d4edda; color: #155724'
    elif val < 0:
        return 'background-color: #f8d7da; color: #721c24'
    return ''

st.dataframe(
    df_report.style.format({
        'mid': '{:.4f}',
        'spread': '{:.4f}',
        'inventory': '{:.2f}',
        'pnl': '${:.2f}',
        'notional': '${:.2f}',
        'max_drawdown': '${:.2f}'
    }).applymap(color_pnl, subset=['pnl']),
    use_container_width=True,
    height=200
)

# Time series analysis for selected market
if trace_data:
    st.header(f"📈 Time Series Analysis - {selected_market}")
    
    # Extract time series for selected market
    steps = []
    mids = []
    inventories = []
    pnls = []
    spreads = []
    
    for step_idx, step in enumerate(trace_data):
        if selected_market in step:
            steps.append(step_idx)
            market_data = step[selected_market]
            mids.append(market_data['mid'])
            inventories.append(market_data['inventory'])
            pnls.append(market_data['pnl'])
            spreads.append(market_data['spread'])
    
    # Create subplots
    col1, col2 = st.columns(2)
    
    with col1:
        # Mid price evolution
        fig_mid = go.Figure()
        fig_mid.add_trace(go.Scatter(
            x=steps,
            y=mids,
            mode='lines',
            name='Mid Price',
            line=dict(color='blue', width=2)
        ))
        fig_mid.update_layout(
            title='Mid Price Evolution',
            xaxis_title='Step',
            yaxis_title='Mid (Probability)',
            height=350
        )
        st.plotly_chart(fig_mid, use_container_width=True)
        
        # PnL evolution
        fig_pnl_ts = go.Figure()
        fig_pnl_ts.add_trace(go.Scatter(
            x=steps,
            y=pnls,
            mode='lines',
            name='PnL',
            line=dict(color='green', width=2),
            fill='tozeroy'
        ))
        fig_pnl_ts.update_layout(
            title='PnL Evolution',
            xaxis_title='Step',
            yaxis_title='PnL ($)',
            height=350
        )
        st.plotly_chart(fig_pnl_ts, use_container_width=True)
    
    with col2:
        # Inventory evolution
        fig_inv_ts = go.Figure()
        fig_inv_ts.add_trace(go.Scatter(
            x=steps,
            y=inventories,
            mode='lines',
            name='Inventory',
            line=dict(color='orange', width=2)
        ))
        fig_inv_ts.update_layout(
            title='Inventory Evolution',
            xaxis_title='Step',
            yaxis_title='Inventory',
            height=350
        )
        st.plotly_chart(fig_inv_ts, use_container_width=True)
        
        # Spread evolution
        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(
            x=steps,
            y=spreads,
            mode='lines',
            name='Spread',
            line=dict(color='purple', width=2)
        ))
        fig_spread.update_layout(
            title='Spread Evolution',
            xaxis_title='Step',
            yaxis_title='Spread',
            height=350
        )
        st.plotly_chart(fig_spread, use_container_width=True)
    
    # Combined view
    st.subheader("🔄 Combined Multi-Metric View")
    
    # Create figure with secondary y-axis using make_subplots
    fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig_combined.add_trace(
        go.Scatter(x=steps, y=mids, mode='lines', name='Mid Price', line=dict(color='blue')),
        secondary_y=False
    )
    
    fig_combined.add_trace(
        go.Scatter(x=steps, y=inventories, mode='lines', name='Inventory', line=dict(color='orange')),
        secondary_y=True
    )
    
    # Update axes labels
    fig_combined.update_xaxes(title_text="Step")
    fig_combined.update_yaxes(title_text="Mid Price", secondary_y=False)
    fig_combined.update_yaxes(title_text="Inventory", secondary_y=True)
    
    # Update layout
    fig_combined.update_layout(
        title_text='Multi-Metric Overview',
        height=400
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)

# Performance metrics
st.header("🎯 Performance Metrics")
selected_row = df_report[df_report['market'] == selected_market].iloc[0]

metrics_cols = st.columns(5)
with metrics_cols[0]:
    st.metric("Mid Price", f"{selected_row['mid']:.4f}")
with metrics_cols[1]:
    st.metric("Spread", f"{selected_row['spread']:.4f}")
with metrics_cols[2]:
    st.metric("PnL", f"${selected_row['pnl']:.2f}")
with metrics_cols[3]:
    st.metric("Max Drawdown", f"${selected_row['max_drawdown']:.2f}")
with metrics_cols[4]:
    st.metric("Fill Count", f"{int(selected_row['fill_count'])}")

# Footer
st.markdown("---")
st.markdown("**Quant Execution Bot Dashboard** | Market Making Simulation | Built with Streamlit & Plotly")
