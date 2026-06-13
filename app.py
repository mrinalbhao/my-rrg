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
    value="XLE, XLK"
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
    index=1  # Matches your weekly rrgoptima reference chart
)

# Tail points input 
tail_points = st.sidebar.number_input(
    "Number of Tail Points (History)", 
    min_value=3, 
    max_value=100, 
    value=10, 
    step=1
)

# Go Button
trigger_go = st.sidebar.button("🚀 Render RRG Chart")

# --- INSTITIONAL JDK CALCULATIONS ---
def calculate_rrg_metrics(tickers, benchmark, interval_str, history_needed):
    """Calculates canonical JdK RS-Ratio and RS-Momentum tracking indices."""
    interval_map = {"1 Day": "1d", "1 Week": "1wk"}
    yf_interval = interval_map[interval_str]
    
    all_tickers = list(set(tickers + [benchmark]))
    # Download raw assets directly 
    data = yf.download(all_tickers, period=history_needed, interval=yf_interval)
    
    if data.empty:
        return None
        
    # Safely isolate the closing prices, avoiding any alphabetical dictionary shuffles
    df_close = pd.DataFrame()
    if 'Close' in data.columns:
        close_df = data['Close']
        for t in all_tickers:
            if t in close_df.columns:
                df_close[t] = close_df[t]
                
    df_close = df_close.dropna()
    if benchmark not in df_close.columns:
        return None

    rrg_results = {}
    
    # Process each ticker explicitly against its distinct mapped column
    for t in tickers:
        if t not in df_close.columns or t == benchmark:
            continue
            
        # Step 1: Compute Base Relative Strength (RS)
        rs_raw = df_close[t] / df_close[benchmark]
        
        # Step 2: Compute JdK RS-Ratio via institutional Exponential Smoothing matrix
        # Uses a 14-period benchmark mean and standard deviation framework
        rs_mean = rs_raw.ewm(span=14, adjust=False).mean()
        rs_std = rs_raw.rolling(window=14).std()
        
        # Standardize and center tightly around the baseline center index of 100
        rs_ratio = 100 + ((rs_raw - rs_mean) / (rs_std + 1e-8)) * 5
        
        # Step 3: Compute JdK RS-Momentum using a double smoothed Rate-of-Change (ROC) 
        # Tracks the physical velocity shift of the RS-Ratio track over a 5-period window
        rs_ratio_smoothed = rs_ratio.ewm(span=14, adjust=False).mean()
        rs_mom = 100 + (rs_ratio_smoothed.pct_change(periods=5) * 100) * 3
        
        # Pack results alongside data index rows safely
        ticker_df = pd.DataFrame({'RS_Ratio': rs_ratio, 'RS_Momentum': rs_mom}).dropna()
        rrg_results[t] = ticker_df
        
    return rrg_results

def smooth_trajectory(x_coords, y_coords, steps=100):
    """Uses a Cubic Spline to smoothly fill spaces between jagged data nodes."""
    t_original = np.linspace(0, 1, len(x_coords))
    t_smooth = np.linspace(0, 1, steps)
    
    cs_x = CubicSpline(t_original, x_coords)
    cs_y = CubicSpline(t_original, y_coords)
    
    return cs_x(t_smooth), cs_y(t_smooth)

# --- APPLICATION EXECUTION ---
if trigger_go:
    parsed_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    bench_ticker = benchmark_input.strip().upper()
    
    if not parsed_tickers or not bench_ticker:
        st.error("Please provide both valid asset symbols and a benchmark tracker.")
    else:
        with st.spinner("Processing institutional JdK curves..."):
            raw_rrg_data = calculate_rrg_metrics(
                tickers=parsed_tickers, 
                benchmark=bench_ticker, 
                interval_str=interval_choice, 
                history_needed="3y" # Expanded padding loopback timeline to initialize EMAs correctly
            )
            
            if not raw_rrg_data:
                st.error("Data tracking failed. Please ensure stock tickers exist on Yahoo Finance.")
            else:
                fig = go.Figure()
                all_x, all_y = [], []
                
                # High contrast palette mapping (Energy=Red, Technology=Orange/Gold to match your image)
                color_map = {
                    "XLE": "#D62728", # Deep Red
                    "XLK": "#FF7F0E", # Vivid Orange/Yellow
                }
                fallback_colors = ["#2CA02C", "#1F77B4", "#9467BD", "#8C564B"]
                
                for idx, (ticker, df) in enumerate(raw_rrg_data.items()):
                    tail_df = df.tail(int(tail_points))
                    if len(tail_df) < 3:
                        continue
                        
                    # Pull sequential metrics in exact chronological progression order
                    x_raw = tail_df['RS_Ratio'].values
                    y_raw = tail_df['RS_Momentum'].values
                    
                    # Apply specific color choice or fallback smoothly
                    ticker_color = color_map.get(ticker, fallback_colors[idx % len(fallback_colors)])
                    
                    # Track lines interpolation engine
                    x_smooth, y_smooth = smooth_trajectory(x_raw, y_raw, steps=200)
                    
                    all_x.extend(x_raw)
                    all_y.extend(y_raw)
                    
                    head_x = x_raw[-1]
                    head_y = y_raw[-1]
                    
                    # 1. Curve Track Plot
                    fig.add_trace(go.Scatter(
                        x=x_smooth, y=y_smooth,
                        mode='lines',
                        name=f"{ticker} Path",
                        line=dict(width=3, color=ticker_color),
                        hoverinfo='skip'
                    ))
                    
                    # 2. Historical Context Dots
                    fig.add_trace(go.Scatter(
                        x=x_raw[:-1], y=y_raw[:-1],
                        mode='markers',
                        name=f"{ticker} History",
                        marker=dict(size=6, color=ticker_color, symbol='circle'),
                        hoverinfo='skip'
                    ))
                    
                    # 3. Leading Head Vector Node
                    fig.add_trace(go.Scatter(
                        x=[head_x], y=[head_y],
                        mode='markers+text',
                        name=ticker,
                        text=[f"<b>{ticker}</b>"],
                        textposition="top center",
                        marker=dict(size=12, symbol='triangle-up', color=ticker_color, line=dict(width=2, color='black'))
                    ))
                
                if not all_x or not all_y:
                    st.error("Not enough historical data found to construct the RRG tail configurations.")
                else:
                    # Calculate balanced axis limits symmetric to 100 center points
                    max_dev = max(
                        max(abs(np.array(all_x) - 100)), 
                        max(abs(np.array(all_y) - 100))
                    ) * 1.25
                    
                    if max_dev < 5:
                        max_dev = 5
                        
                    x_min, x_max = 100 - max_dev, 100 + max_dev
                    y_min, y_max = 100 - max_dev, 100 + max_dev
                    
                    # --- QUADRANT BACKGROUND SHADING CONFIGURATIONS ---
                    fig.add_vrect(x0=100, x1=x_max, y0=100, y1=y_max, fillcolor="rgba(0, 200, 0, 0.04)", layer="below", line_width=0)  # Leading
                    fig.add_vrect(x0=100, x1=x_max, y0=y_min, y1=100, fillcolor="rgba(200, 200, 0, 0.04)", layer="below", line_width=0)  # Weakening
                    fig.add_vrect(x0=x_min, x1=100, y0=y_min, y1=100, fillcolor="rgba(200, 0, 0, 0.04)", layer="below", line_width=0)  # Lagging
                    fig.add_vrect(x0=x_min, x1=100, y0=100, y1=y_max, fillcolor="rgba(0, 0, 200, 0.04)", layer="below", line_width=0)  # Improving
                    
                    # Thin Crosshair Center Lines Fixed at 100
                    fig.add_shape(type="line", x0=100, y0=y_min, x1=100, y1=y_max, line=dict(color="rgba(0,0,0,0.5)", width=1.5, dash="dash"))
                    fig.add_shape(type="line", x0=x_min, y0=100, x1=x_max, y1=100, line=dict(color="rgba(0,0,0,0.5)", width=1.5, dash="dash"))
                    
                    # Quadrant Labels
                    fig.add_annotation(x=100 + (max_dev/2), y=100 + (max_dev/2), text="<b>LEADING</b>", font=dict(color="green", size=16), showarrow=False)
                    fig.add_annotation(x=100 + (max_dev/2), y=100 - (max_dev/2), text="<b>WEAKENING</b>", font=dict(color="gold", size=16), showarrow=False)
                    fig.add_annotation(x=100 - (max_dev/2), y=100 - (max_dev/2), text="<b>LAGGING</b>", font=dict(color="red", size=16), showarrow=False)
                    fig.add_annotation(x=100 - (max_dev/2), y=100 + (max_dev/2), text="<b>IMPROVING</b>", font=dict(color="blue", size=16), showarrow=False)
                    
                    # Layout formatting
                    fig.update_layout(
                        width=950,
                        height=780,
