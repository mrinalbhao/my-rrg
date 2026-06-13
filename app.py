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
    
    # Clean list of all symbols
    all_tickers = list(set(tickers + [benchmark]))
    
    # Download using default column grouping to maintain structural consistency
    data = yf.download(all_tickers, period=history_needed, interval=yf_interval, group_by='column')
    
    if data.empty:
        return None
        
    # Standardize data extraction to handle the MultiIndex securely
    df_close = pd.DataFrame()
    
    # Safely isolate the 'Close' prices level mapping
    if 'Close' in data.columns:
        close_data = data['Close']
        for t in all_tickers:
            if t in close_data.columns:
                df_close[t] = close_data[t]
                
    df_close = df_close.dropna()
    if benchmark not in df_close.columns:
        return None

    rrg_results = {}
    
    # Calculate canonical RS-Ratio and RS-Momentum properties
    for t in tickers:
        if t not in df_close.columns or t == benchmark:
            continue
            
        # 1. Base Relative Strength Ratio vs the Benchmark
        rs_ratio_raw = df_close[t] / df_close[benchmark]
        
        # 2. Normalize via moving averages to mimic JdK metrics (14-period standard baseline)
        rs_ratio_ma = rs_ratio_raw.rolling(window=14).mean()
        rs_ratio_std = rs_ratio_raw.rolling(window=14).std()
        
        # JdK RS-Ratio index center proxy (scaled around 100 base)
        rs_ratio_index = 100 + ((rs_ratio_raw - rs_ratio_ma) / (rs_ratio_std + 1e-8)) * 5
        
        # JdK RS-Momentum rate of change proxy (scaled around 100 base)
        rs_mom_index = 100 + (rs_ratio_index.pct_change(periods=5) * 100)
        
        # Combine metrics into a clean dataframe
        ticker_df = pd.DataFrame({'RS_Ratio': rs_ratio_index, 'RS_Momentum': rs_mom_index}).dropna()
        rrg_results[t] = ticker_df
        
    return rrg_results

def smooth_trajectory(x_coords, y_coords, steps=100):
    """Uses a Cubic Spline to smoothly fill spaces between jagged data nodes."""
    t_original = np.linspace(0, 1, len(x_coords))
    t_smooth = np.linspace(0, 1, steps)
    
    cs_x = CubicSpline(t_original, x_coords)
    cs_y = CubicSpline(t_original, y_coords)
    
    return cs_x(t_smooth), cs_y(t_smooth)

# --- APPLICATION LOGIC EXECUTIVE BRANCH ---
if trigger_go:
    # Clean string inputs into isolated programmatic tokens
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
                st.error("Data tracking failed. Please ensure stock tickers exist on Yahoo Finance.")
            else:
                # Initialize custom interactive canvas configuration
                fig = go.Figure()
                
                # Dynamic quadrant axis constraints evaluation setup
                all_x, all_y = [], []
                
                # Process each item vector independently 
                for ticker, df in raw_rrg_data.items():
                    # Slice tail elements requested by user inputs
                    tail_df = df.tail(int(tail_points))
                    if len(tail_df) < 3:
                        continue
                        
                    x_raw = tail_df['RS_Ratio'].values
                    y_raw = tail_df['RS_Momentum'].values
                    
                    # Apply fine mathematical smoothings
                    x_smooth, y_smooth = smooth_trajectory(x_raw, y_raw, steps=200)
                    
                    # Accumulate boundaries tracking pointers
                    all_x.extend(x_raw)
                    all_y.extend(y_raw)
                    
                    # Extract the leading frontier point metadata coordinates
                    head_x = x_raw[-1]
                    head_y = y_raw[-1]
                    
                    # Line Plot for the smoothed historic tail path 
                    fig.add_trace(go.Scatter(
                        x=x_smooth, y=y_smooth,
                        mode='lines',
                        name=f"{ticker} Path",
                        line=dict(width=3),
                        hoverinfo='skip'
                    ))
                    
                    # Explicit Head Marker identifying current status node
                    fig.add_trace(go.Scatter(
                        x=[head_x], y=[head_y],
                        mode='markers+text',
                        name=ticker,
                        text=[f"<b>{ticker}</b>"],
                        textposition="top center",
                        marker=dict(size=12, symbol='triangle-up', line=dict(width=2, color='black'))
                    ))
                
                if not all_x or not all_y:
                    st.error("Not enough historical data found to construct the RRG tail.")
                else:
                    # Compute balanced boundary conditions limits for symmetry 
                    max_dev = max(
                        max(abs(np.array(all_x) - 100)), 
                        max(abs(np.array(all_y) - 100))
                    ) * 1.15
                    
                    x_min, x_max = 100 - max_dev, 100 + max_dev
                    y_min, y_max = 100 - max_dev, 100 + max_dev
                    
                    # --- QUADRANT BACKGROUND SHADING CONFIGURATIONS ---
                    fig.add_vrect(x0=100, x1=x_max, y0=100, y1=y_max, fillcolor="rgba(0, 200, 0, 0.05)", layer="below", line_width=0)  # Leading
                    fig.add_vrect(x0=100, x1=x_max, y0=y_min, y1=100, fillcolor="rgba(200, 200, 0, 0.05)", layer="below", line_width=0)  # Weakening
                    fig.add_vrect(x0=x_min, x1=100, y0=y_min, y1=100, fillcolor="rgba(200, 0, 0, 0.05)", layer="below", line_width=0)  # Lagging
                    fig.add_vrect(x0=x_min, x1=100, y0=100, y1=y_max, fillcolor="rgba(0, 0, 200, 0.05)", layer="below", line_width=0)  # Improving
                    
                    # Thin Crosshair Center Lines
                    fig.add_shape(type="line", x0=100, y0=y_min, x1=100, y1=y_max, line=dict(color="black", width=1, dash="dash"))
                    fig.add_shape(type="line", x0=x_min, y0=100, x1=x_max, y1=100, line=dict(color="black", width=1, dash="dash"))
                    
                    # Quadrant Static Matrix Text Labels
                    fig.add_annotation(x=100 + (max_dev/2), y=100 + (max_dev/2), text="<b>LEADING</b>", font=dict(color="green", size=16), showarrow=False)
                    fig.add_annotation(x=100 + (max_dev/2), y=100 - (max_dev/2), text="<b>WEAKENING</b>", font=dict(color="gold", size=16), showarrow=False)
                    fig.add_annotation(x=100 - (max_dev/2), y=100 - (max_dev/2), text="<b>LAGGING</b>", font=dict(color="red", size=16), showarrow=False)
                    fig.add_annotation(x=100 - (max_dev/2), y=100 + (max_dev/2), text="<b>IMPROVING</b>", font=dict(color="blue", size=16), showarrow=False)
                    
                    # Final layout configurations
                    fig.update_layout(
                        width=900,
                        height=750,
                        xaxis=dict(title="<b>RS-Ratio (Trend)</b>", range=[x_min, x_max], zeroline=False),
                        yaxis=dict(title="<b>RS-Momentum (Velocity)</b>", range=[y_min, y_max], zeroline=False),
                        title=f"Relative Rotation Graph vs {bench_ticker} ({interval_choice} System)",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Configure variables inside left side panel and click 'Render RRG Chart' to track structural transformations.")
