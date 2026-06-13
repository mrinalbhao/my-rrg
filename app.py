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
    index=1  # Default to 1 Week to match your chart
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

# --- CALCULATION HELPER FUNCTIONS ---
def calculate_rrg_metrics(tickers, benchmark, interval_str, history_needed):
    """Fetches stock data and extracts standard RS-Ratio and RS-Momentum lines."""
    interval_map = {"1 Day": "1d", "1 Week": "1wk"}
    yf_interval = interval_map[interval_str]
    
    all_tickers = list(set(tickers + [benchmark]))
    data = yf.download(all_tickers, period=history_needed, interval=yf_interval, group_by='column')
    
    if data.empty:
        return None
        
    df_close = pd.DataFrame()
    if 'Close' in data.columns:
        close_data = data['Close']
        for t in all_tickers:
            if t in close_data.columns:
                df_close[t] = close_data[t]
                
    df_close = df_close.dropna()
    if benchmark not in df_close.columns:
        return None

    rrg_results = {}
    
    for t in tickers:
        if t not in df_close.columns or t == benchmark:
            continue
            
        # 1. Base Price Relative Strength Ratio
        rs_raw = (df_close[t] / df_close[benchmark]) * 100
        
        # 2. CANONICAL JdK RS-RATIO FORMULA
        # Uses a 14-period rolling average window based strictly on selected time interval
        rs_mean = rs_raw.rolling(window=14).mean()
        rs_std = rs_raw.rolling(window=14).std(ddof=0)
        
        # Canonical scaling index adjustment to match professional platforms
        rs_ratio = 100 + ((rs_raw - rs_mean) / (rs_std + 1e-8)) * 5
        
        # 3. CANONICAL JdK RS-MOMENTUM FORMULA
        # Calculates Rate-of-Change (ROC) over 1 period to maintain direction accuracy
        rs_ratio_roc = 100 * ((rs_ratio / rs_ratio.shift(1)) - 1)
        
        roc_mean = rs_ratio_roc.rolling(window=14).mean()
        roc_std = rs_ratio_roc.rolling(window=14).std(ddof=0)
        
        rs_mom = 100 + ((rs_ratio_roc - roc_mean) / (roc_std + 1e-8)) * 5
        
        # Combine metrics into a clean dataframe
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

# --- APPLICATION LOGIC ---
if trigger_go:
    parsed_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    bench_ticker = benchmark_input.strip().upper()
    
    if not parsed_tickers or not bench_ticker:
        st.error("Please provide both valid asset symbols and a benchmark tracker.")
    else:
        with st.spinner("Analyzing market momentum fields and generating clean vectors..."):
            # Pull 3 years of history to allow weekly windows to calculate accurately
            raw_rrg_data = calculate_rrg_metrics(
                tickers=parsed_tickers, 
                benchmark=bench_ticker, 
                interval_str=interval_choice, 
                history_needed="3y"
            )
            
            if not raw_rrg_data:
                st.error("Data tracking failed. Please ensure stock tickers exist on Yahoo Finance.")
            else:
                fig = go.Figure()
                all_x, all_y = [], []
                
                # Distinct color palette sequence matching institutional charts (XLE Dark Red, XLK Gold)
                color_palette = ["#B22222", "#FFBF00", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#e377c2"]
                
                for idx, (ticker, df) in enumerate(raw_rrg_data.items()):
                    tail_df = df.tail(int(tail_points))
                    if len(tail_df) < 3:
                        continue
                        
                    x_raw = tail_df['RS_Ratio'].values
                    y_raw = tail_df['RS_Momentum'].values
                    
                    ticker_color = color_palette[idx % len(color_palette)]
                    
                    # Smooth out the lines seamlessly
                    x_smooth, y_smooth = smooth_trajectory(x_raw, y_raw, steps=200)
                    
                    all_x.extend(x_raw)
                    all_y.extend(y_raw)
                    
                    head_x = x_raw[-1]
                    head_y = y_raw[-1]
                    
                    # Line Plot for the smoothed historic tail path 
                    fig.add_trace(go.Scatter(
                        x=x_smooth, y=y_smooth,
                        mode='lines',
                        name=f"{ticker} Path",
                        line=dict(width=3, color=ticker_color),
                        hoverinfo='skip'
                    ))
                    
                    # Add simple structural checkpoint dots along the trail history nodes
                    fig.add_trace(go.Scatter(
                        x=x_raw[:-1], y=y_raw[:-1],
                        mode='markers',
                        name=f"{ticker} History",
                        marker=dict(size=6, color=ticker_color, symbol='circle'),
                        hoverinfo='skip'
                    ))
                    
                    # Explicit Head Marker identifying current status node
                    fig.add_trace(go.Scatter(
                        x=[head_x], y=[head_y],
                        mode='markers+text',
                        name=ticker,
                        text=[f"<b>{ticker}</b>"],
                        textposition="top center",
                        marker=dict(size=12, symbol='triangle-up', color=ticker_color, line=dict(width=2, color='black'))
                    ))
                
                if not all_x or not all_y:
                    st.error("Not enough historical data found to construct the RRG tail.")
                else:
                    # Enforce wider standard axis bounds so XLE can expand out to the 116 area
                    max_dev = max(
                        max(abs(np.array(all_x) - 100)), 
                        max(abs(np.array(all_y) - 100))
                    ) * 1.15
                    
                    # Set minimum axis scale to 18 units from center to capture high-volatility shifts perfectly
                    if max_dev < 18:
                        max_dev = 18
                        
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
                    
                    # Quadrant Labels
                    fig.add_annotation(x=100 + (max_dev/2), y=100 + (max_dev/2), text="<b>LEADING</b>", font=dict(color="green", size=16), showarrow=False)
                    fig.add_annotation(x=100 + (max_dev/2), y=100 - (max_dev/2), text="<b>WEAKENING</b>", font=dict(color="gold", size=16), showarrow=False)
                    fig.add_annotation(x=100 - (max_dev/2), y=100 - (max_dev/2), text="<b>LAGGING</b>", font=dict(color="red", size=16), showarrow=False)
                    fig.add_annotation(x=100 - (max_dev/2), y=100 + (max_dev/2), text="<b>IMPROVING</b>", font=dict(color="blue", size=16), showarrow=False)
                    
                    # Final layout configurations
                    fig.update_layout(
                        width=950,
                        height=780,
                        xaxis=dict(title="<b>RS-Ratio (Trend)</b>", range=[x_min, x_max], zeroline=False),
