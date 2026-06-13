import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline

# Page Configuration
st.set_page_config(page_title="Custom RRG Dashboard", layout="wide")
st.title("📈 Custom Relative Rotation Graph (RRG) Generator")
st.markdown("Track momentum and relative strength trends mapped smoothly across market quadrants.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration Settings")

# Text input for Custom Tickers
ticker_input = st.sidebar.text_input(
    "Asset Tickers (Comma separated)", 
    value="AAPL, MSFT, GOOGL, AMZN, NVDA"
)

# Text input for Benchmark
benchmark_input = st.sidebar.text_input(
    "Benchmark Ticker (e.g., SPY, QQQ)", 
    value="SPY"
)

# Combo Box for Interval
interval_choice = st.sidebar.selectbox(
    "Data Time Interval",
    options=["1 Day", "1 Week"],
    index=0
)

# Tail points input 
tail_points = st.sidebar.number_input(
    "Number of Tail Points (History)", 
    min_value=3, 
    max_value=100, 
    value=14, 
    step=1
)

# Go Button
trigger_go = st.sidebar.button("🚀 Render RRG Chart")

# --- CALCULATION HELPER FUNCTIONS ---
def calculate_rrg_metrics(tickers, benchmark, interval_str, history_needed):
    """Fetches stock data and extracts standard RS-Ratio and RS-Momentum lines."""
    interval_map = {"1 Day": "1d", "1 Week": "1wk"}
    yf_interval = interval_map[interval_str]
    
    # Clean list of all unique symbols needed
    all_tickers = list(set(tickers + [benchmark]))
    
    # Download raw data
    raw_data = yf.download(all_tickers, period=history_needed, interval=yf_interval)
    
    if raw_data.empty:
        return None
        
    # BULLETPROOF FIX FOR NEW YFINANCE MULTI-INDEX: Flatten the columns explicitly
    # yfinance returns MultiIndex like ('Close', 'AAPL') or ('Price', 'Close', 'Ticker', 'AAPL')
    df_close = pd.DataFrame(index=raw_data.index)
    
    for t in all_tickers:
        # Check all possible column variations in recent yfinance updates
        found = False
        for col in raw_data.columns:
            if isinstance(col, tuple):
                # Matches ('Close', 'AAPL') or similar structures
                if 'Close' in col and t in col:
                    df_close[t] = raw_data[col]
                    found = True
                    break
            else:
                # Fallback for single ticker strings
                if col == 'Close' and len(all_tickers) == 1:
                    df_close[t] = raw_data[col]
                    found = True
                    break
                    
    df_close = df_close.dropna()
    if benchmark not in df_close.columns:
        return None

    rrg_results = {}
    
    # Calculate RS-Ratio and RS-Momentum properties
    for t in tickers:
        if t not in df_close.columns or t == benchmark:
            continue
            
        # 1. Base Relative Strength Ratio vs the Benchmark
        rs_ratio_raw = df_close[t] / df_close[benchmark]
        
        # 2. Normalize via moving averages (14-period standard baseline)
        rs_ratio_ma = rs_ratio_raw.rolling(window=14).mean()
        rs_ratio_std = rs_ratio_raw.rolling(window=14).std()
        
        # JdK RS-Ratio index center proxy (scaled around 100 base)
        rs_ratio_index = 100 + ((rs_ratio_raw - rs_ratio_ma) / (rs_ratio_std + 1e-8)) * 5
        
        # JdK RS-Momentum rate of change proxy (scaled around 100 base)
        rs_mom_index = 100 + (rs_ratio_index.pct_change(periods=5) * 100)
        
        # Combine metrics alongside their true timestamp index
        ticker_df = pd.DataFrame({
            'RS_Ratio': rs_ratio_index, 
            'RS_Momentum': rs_mom_index
        }).dropna()
        
        rrg_results[t] = ticker_df
        
    return rrg_results

def smooth_trajectory(x_coords, y_coords, steps=200):
    """Uses a Cubic Spline to smoothly fill spaces between jagged data nodes."""
    t_original = np.linspace(0, 1, len(x_coords))
    t_smooth = np.linspace(0, 1, steps)
    
    cs_x = CubicSpline(t_original, x_coords)
    cs_y = CubicSpline(t_original, y_coords)
    
    return cs_x(t_smooth), cs_y(t_smooth)

# --- APPLICATION LOGIC EXECUTIVE BRANCH ---
if trigger_go:
    # Clean string inputs into isolated tokens
    parsed_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    bench_ticker = benchmark_input.strip().upper()
    
    if not parsed_tickers or not bench_ticker:
        st.error("Please provide both valid asset symbols and a benchmark tracker.")
    else:
        with st.spinner("Analyzing market momentum fields and generating clean vectors..."):
            raw_rrg_data = calculate_rrg_metrics(
                tickers=parsed_tickers, 
                benchmark=bench_ticker, 
                interval_str=interval_choice, 
                history_needed="2y"
            )
            
            if not raw_rrg_data:
                st.error("Data tracking failed. Could not find or isolate the specified tickers on Yahoo Finance.")
            else:
                # Initialize custom interactive canvas configuration
                fig = go.Figure()
                all_x, all_y = [], []
                
                # Setup Plotly color loop to match line paths and dot intervals explicitly
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                
                # Process each item vector independently 
                for idx, (ticker, df) in enumerate(raw_rrg_data.items()):
                    # Slice tail elements requested by user inputs
                    tail_df = df.tail(int(tail_points))
                    if len(tail_df) < 3:
                        continue
                    
                    # Choose a consistent unique color for this specific ticker track
                    color = colors[idx % len(colors)]
                    
                    # Save raw underlying nodes (with dates) for the periodic hover dots
                    x_raw = tail_df['RS_Ratio'].values
                    y_raw = tail_df['RS_Momentum'].values
                    dates_raw = tail_df.index.strftime('%Y-%m-%d').tolist()
                    
                    # Apply fine mathematical spline curve smoothings for the track line
                    x_smooth, y_smooth = smooth_trajectory(x_raw, y_raw, steps=200)
                    
                    # Accumulate boundaries tracking pointers
                    all_x.extend(x_raw)
                    all_y.extend(y_raw)
                    
                    # Extract the leading frontier point coordinates
                    head_x = x_raw[-1]
                    head_y = y_raw[-1]
                    
                    # 1. Trace Line: The smoothed historic tail path (No hover to prevent crowding)
                    fig.add_trace(go.Scatter(
                        x=x_smooth, y=y_smooth,
                        mode='lines',
                        name=f"{ticker} Path",
                        line=dict(width=3, color=color),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                    
                    # 2. Intermittent Dots: Weekly/Daily spaced real historic intervals with date hover
                    fig.add_trace(go.Scatter(
                        x=x_raw, y=y_raw,
                        mode='markers',
                        name=ticker,
                        marker=dict(size=6, color=color, symbol='circle'),
                        text=[f"<b>{ticker}</b><br>Date: {d}<br>RS-Ratio: {x:.2f}<br>RS-Mom: {y:.2f}" for d, x, y in zip(dates_raw, x_raw, y_raw)],
                        hoverinfo='text',
                        showlegend=True
                    ))
                    
                    # 3. Leading Arrow: Explicit Head Marker identifying current status node
                    fig.add_trace(go.Scatter(
                        x=[head_x], y=[head_y],
                        mode='markers+text',
                        name=f"{ticker} (Latest)",
                        text=[f"<b>{ticker}</b>"],
                        textposition="top center",
                        marker=dict(size=12, symbol='triangle-up', color=color, line=dict(width=2, color='black')),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                
                if not all_x or not all_y:
                    st.error("Not enough historical data found to construct the RRG tail configurations.")
                else:
                    # Compute balanced boundary conditions limits for symmetry 
                    max_dev = max(
                        max(abs(np.array(all_x) - 100)), 
                        max(abs(np.array(all_y) - 100))
                    ) * 1.15
                    
                    x_min, x_max = 100 - max_dev, 100 + max_dev
                    y_min, y_max = 100 - max_dev, 100 + max_dev
                    
                    # --- QUADRANT BACKGROUND SHADING CONFIGURATIONS ---
                    fig.add_vrect(x0=100, x1=x_max, y0=100, y1=y_max, fillcolor="rgba(0, 200, 0, 0.04)", layer="below", line_width=0)  # Leading
                    fig.add_vrect(x0=100, x1=x_max, y0=y_min, y1=100, fillcolor="rgba(200, 200, 0, 0.04)", layer="below", line_width=0)  # Weakening
                    fig.add_vrect(x0=x_min, x1=100, y0=y_min, y1=100, fillcolor="rgba(200, 0, 0, 0.04)", layer="below", line_width=0)  # Lagging
                    fig.add_vrect(x0=x_min, x1=100, y0=100, y1=y_max, fillcolor="rgba(0, 0, 200, 0.04)", layer="below", line_width=0)  # Improving
                    
                    # Thin Crosshair Center Lines Fixed at 100
