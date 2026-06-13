import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from alpha_vantage.timeseries import TimeSeries

# Page Configuration
st.set_page_config(page_title="Custom RRG Dashboard", layout="wide")
st.title("📈 Custom Cloud-API Relative Rotation Graph")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration Settings")

# ADD AN API KEY TEXTBOX TO THE SIDEBAR
api_key = st.sidebar.text_input("Enter Free Alpha Vantage API Key", type="password")

ticker_input = st.sidebar.text_input("Asset Tickers", value="XLE, XLK, IGV, SMH, EUAD")
benchmark_input = st.sidebar.text_input("Benchmark Ticker", value="SPY")

interval_choice = st.sidebar.selectbox("Interval", options=["1 Day", "1 Week"], index=1)
tail_points = st.sidebar.number_input("Tail Points", min_value=3, max_value=100, value=10)
trigger_go = st.sidebar.button("🚀 Render RRG Chart")

def calculate_rrg_metrics_av(tickers, benchmark, interval_str, key):
    """Fetches high-accuracy historical arrays from Alpha Vantage API."""
    ts = TimeSeries(key=key, output_format='pandas')
    df_close = pd.DataFrame()
    
    all_tickers = list(set(tickers + [benchmark]))
    
    # Download each asset file cleanly from the cloud data server
    for t in all_tickers:
        try:
            if interval_str == "1 Week":
                data, _ = ts.get_weekly(symbol=t)
                # Alpha Vantage returns columns named: '4. close'
                df_close[t] = data['4. close']
            else:
                data, _ = ts.get_daily(symbol=t, outputsize='full')
                df_close[t] = data['4. close']
        except Exception:
            continue
            
    if df_close.empty or benchmark not in df_close.columns:
        return None
        
    # Sort chronologically (Past to Present)
    df_close = df_close.sort_index().dropna()
    
    rrg_results = {}
    for t in tickers:
        if t not in df_close.columns or t == benchmark:
            continue
            
        # Base RS calculation
        rs_raw = (df_close[t] / df_close[benchmark]) * 100
        
        # JdK Style trend line mapping approximation loops
        rs_mean = rs_raw.rolling(window=14).mean()
        rs_std = rs_raw.rolling(window=14).std()
        rs_ratio = 100 + ((rs_raw - rs_mean) / (rs_std + 1e-8)) * 5
        rs_mom = 100 + (rs_ratio.pct_change(periods=5) * 100)
        
        rrg_results[t] = pd.DataFrame({'RS_Ratio': rs_ratio, 'RS_Momentum': rs_mom}).dropna()
        
    return rrg_results

def smooth_trajectory(x_coords, y_coords, steps=100):
    t_original = np.linspace(0, 1, len(x_coords))
    t_smooth = np.linspace(0, 1, steps)
    cs_x = CubicSpline(t_original, x_coords)
    cs_y = CubicSpline(t_original, y_coords)
    return cs_x(t_smooth), cs_y(t_smooth)

if trigger_go:
    if not api_key:
        st.error("Please enter your free Alpha Vantage API Key in the sidebar.")
    else:
        parsed_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        bench_ticker = benchmark_input.strip().upper()
        
        with st.spinner("Streaming high-accuracy node coordinates..."):
            raw_rrg_data = calculate_rrg_metrics_av(parsed_tickers, bench_ticker, interval_choice, api_key)
            
            if not raw_rrg_data:
                st.error("Failed to fetch cloud data. Check your API key or ticker symbols.")
            else:
                fig = go.Figure()
                all_x, all_y = [], []
                color_palette = ["#ff7f0e", "#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#e377c2"]
                
                for idx, (ticker, df) in enumerate(raw_rrg_data.items()):
                    tail_df = df.tail(int(tail_points))
                    if len(tail_df) < 3:
                        continue
                        
                    x_raw = tail_df['RS_Ratio'].values
                    y_raw = tail_df['RS_Momentum'].values
                    ticker_color = color_palette[idx % len(color_palette)]
                    
                    x_smooth, y_smooth = smooth_trajectory(x_raw, y_raw, steps=200)
                    all_x.extend(x_raw)
                    all_y.extend(y_raw)
                    
                    # Line Path
                    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', line=dict(width=3, color=ticker_color), hoverinfo='skip'))
                    # Historic Dots
                    fig.add_trace(go.Scatter(x=x_raw[:-1], y=y_raw[:-1], mode='markers', marker=dict(size=6, color=ticker_color, symbol='circle'), hoverinfo='skip'))
                    # Head Marker
                    fig.add_trace(go.Scatter(x=[x_raw[-1]], y=[y_raw[-1]], mode='markers+text', text=[f"<b>{ticker}</b>"], textposition="top center", marker=dict(size=12, symbol='circle', color=ticker_color, line=dict(width=2, color='black'))))
                
                max_dev = max(max(abs(np.array(all_x) - 100)), max(abs(np.array(all_y) - 100))) * 1.15
                if max_dev < 3: max_dev = 3
                x_min, x_max = 100 - max_dev, 100 + max_dev
                y_min, y_max = 100 - max_dev, 100 + max_dev
                
                fig.add_vrect(x0=100, x1=x_max, y0=100, y1=y_max, fillcolor="rgba(0, 200, 0, 0.05)", layer="below", line_width=0)
                fig.add_vrect(x0=100, x1=x_max, y0=y_min, y1=100, fillcolor="rgba(200, 200, 0, 0.05)", layer="below", line_width=0)
                fig.add_vrect(x0=x_min, x1=100, y0=y_min, y1=100, fillcolor="rgba(200, 0, 0, 0.05)", layer="below", line_width=0)
                fig.add_vrect(x0=x_min, x1=100, y0=100, y1=y_max, fillcolor="rgba(0, 0, 200, 0.05)", layer="below", line_width=0)
                
                fig.add_shape(type="line", x0=100, y0=y_min, x1=100, y1=y_max, line=dict(color="black", width=1, dash="dash"))
                fig.add_shape(type="line", x0=x_min, y0=100, x1=x_max, y1=100, line=dict(color="black", width=1, dash="dash"))
                
                fig.update_layout(width=950, height=780, xaxis=dict(range=[x_min, x_max], zeroline=False), yaxis=dict(range=[y_min, y_max], zeroline=False), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter your free API Key, configure your panel settings, and click 'Render'.")
