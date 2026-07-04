import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
TICKER_OPTIONS = [
    "IGV", "SMH", "DRAM", "BOTZ", "PPA", "LIT", "LYTE", "LAZR", "BOTT", 
    "DTCR", "QTUM", "HACK", "SKYY", "NLR", "SHLD", "WARP", "IBB", "DRIV", 
    "REMX", "FINX", "HERO"
]

st.set_page_config(layout="wide", page_title="Custom Sub-Sector RRG Matrix")

# ---------------------------------------------------------
# DYNAMIC STATE MANAGEMENT (Dropdown & Comma-Separated Box)
# ---------------------------------------------------------
# Step 1: Initialise the unified state string if it doesn't exist
if "ticker_string" not in st.session_state:
    st.session_state.ticker_string = "IGV, SMH, DRAM, BOTZ, PPA"

# Callback: Fired whenever user alters the checklist dropdown
def update_text_from_dropdown():
    selected_from_dropdown = st.session_state.dropdown_select
    
    # Extract any manually typed tickers that are NOT in our pre-defined options array
    current_tokens = [t.strip().upper() for t in st.session_state.text_box_input.split(",") if t.strip()]
    custom_manual_tokens = [t for t in current_tokens if t not in TICKER_OPTIONS]
    
    # Maintain manual user entries and append newly checked options cleanly
    combined_list = custom_manual_tokens + selected_from_dropdown
    st.session_state.ticker_string = ", ".join(combined_list)

# Callback: Fired if a user modifies the plain text field manually
def update_dropdown_from_text():
    current_tokens = [t.strip().upper() for t in st.session_state.text_box_input.split(",") if t.strip()]
    # Update the select box state to match what's currently written in the box
    matched_dropdown_tokens = [t for t in current_tokens if t in TICKER_OPTIONS]
    st.session_state.dropdown_select = matched_dropdown_tokens
    st.session_state.ticker_string = st.session_state.text_box_input

# ---------------------------------------------------------
# APP UI LAYOUT
# ---------------------------------------------------------
st.title("📊 Custom Thematic Relative Rotation Graph (RRG)")
st.caption("Track rotational momentum configurations across specialized technical sub-sectors.")

with st.sidebar:
    st.header("⚙️ Configuration Engine")
    
    # 2. Main Comma-Separated Text Box
    ticker_input_string = st.text_input(
        "Tickers to Plot (Comma-Separated Grid):",
        key="text_box_input",
        value=st.session_state.ticker_string,
        on_change=update_dropdown_from_text,
        help="Type symbols manually or let checkboxes populate this field dynamically."
    )
    
    # Read the current string to determine defaults for multi-select
    current_text_tokens = [t.strip().upper() for t in ticker_input_string.split(",") if t.strip()]
    dropdown_default = [t for t in current_text_tokens if t in TICKER_OPTIONS]

    # 3. Checkbox Multi-Select Tool (Synchronized)
    selected_dropdown = st.multiselect(
        "Select Sub-Sector Tickers to Add:",
        options=TICKER_OPTIONS,
        default=dropdown_default,
        key="dropdown_select",
        on_change=update_text_from_dropdown,
        help="Checking these boxes adds them instantly into the manual input box above."
    )
    
    st.markdown("---")
    
    # Benchmark and Time Controls
    benchmark_input = st.text_input("Benchmark Ticker (RRG Axis Center):", value="RSP", help="Use equal weight index (e.g. RSP) to balance cross-industry signals.")
    tail_length = st.slider("Historical Tail Length (Periods):", min_value=2, max_value=30, value=5)
    history_window = st.selectbox("Lookback History Period:", options=["1y", "2y", "6mo", "3y"], index=0)
    rolling_window = st.slider("Smoothing Window (Standard JdK is 14):", min_value=5, max_value=30, value=14)
    
    plot_trigger = st.button("Generate RRG Matrix Plot", type="primary")

# Final clean array compilation
final_ticker_list = [ticker.strip().upper() for ticker in st.session_state.ticker_string.split(",") if ticker.strip()]

# ---------------------------------------------------------
# MATHEMATICAL RRG COMPONENT ENGINE
# ---------------------------------------------------------
def calculate_rrg_metrics(tickers, benchmark, period, window):
    all_symbols = list(set(tickers + [benchmark]))
    # Fetch historical adjusted close data
    data = yf.download(all_symbols, period=period, progress=False)["Adj Close"]
    
    if data.empty or benchmark not in data.columns:
        return None, None
        
    df_assets = data[tickers]
    df_bench = data[benchmark]
    
    # Step 1: Base Relative Strength Calculation
    rs_ratio_df = pd.DataFrame()
    for col in df_assets.columns:
        # Relative Strength base line
        rs_ratio_df[col] = (df_assets[col] / df_bench) * 100
        
    # Step 2: Extract Moving Averages and compute standard tracking deviation
    rs_mean = rs_ratio_df.rolling(window=window).mean()
    rs_std = rs_ratio_df.rolling(window=window).std()
    
    # Step 3: Compute standardized Z-Scores representing the RS-Ratio Value
    rs_ratio_normalized = ((rs_ratio_df - rs_mean) / rs_std) + 100
    
    # Step 4: Calculate Velocity Rate-Of-Change representing RS-Momentum Value
    rs_momentum_raw = rs_ratio_normalized.pct_change(periods=1) * 100
    mom_mean = rs_momentum_raw.rolling(window=window).mean()
    mom_std = rs_momentum_raw.rolling(window=window).std()
    rs_momentum_normalized = ((rs_momentum_raw - mom_mean) / mom_std) + 100
    
    return rs_ratio_normalized.dropna(), rs_momentum_normalized.dropna()

# ---------------------------------------------------------
# CHART PLOTTING EXECUTION
# ---------------------------------------------------------
if plot_trigger or len(final_ticker_list) > 0:
    if not final_ticker_list:
        st.warning("Please supply or check at least one sub-sector symbol vector to construct the canvas.")
    else:
        with st.spinner("Downloading financial data models and calculating indicators..."):
            ratio_df, momentum_df = calculate_rrg_metrics(
                tickers=final_ticker_list,
                benchmark=benchmark_input.strip().upper(),
                period=history_window,
                window=rolling_window
            )
            
        if ratio_df is None or ratio_df.empty:
            st.error("Error retrieving asset data from provider. Verify symbol spelling.")
        else:
            # Initialize complete Plotly figure canvas
            fig = go.Figure()
            
            # Map structural quadrant background layouts
            fig.add_shape(type="rect", x0=100, y0=100, x1=105, y1=105, fillcolor="rgba(0, 200, 0, 0.08)", line_width=0) # Leading
            fig.add_shape(type="rect", x0=100, y0=95, x1=105, y1=100, fillcolor="rgba(200, 200, 0, 0.08)", line_width=0) # Weakening
            fig.add_shape(type="rect", x0=95, y0=95, x1=100, y1=100, fillcolor="rgba(200, 0, 0, 0.08)", line_width=0)   # Lagging
            fig.add_shape(type="rect", x0=95, y0=100, x1=100, y1=105, fillcolor="rgba(0, 0, 200, 0.08)", line_width=0)  # Improving
            
            # Draw baseline center grid crosshairs
            fig.add_hline(y=100, line_dash="dash", line_color="rgba(100,100,100,0.5)", line_width=1.5)
            fig.add_vline(x=100, line_dash="dash", line_color="rgba(100,100,100,0.5)", line_width=1.5)
            
            # Extract and draw lines for each verified ticker asset 
            available_tickers = ratio_df.columns
            
            for ticker in available_tickers:
                # Grab ending slices matching historical tail limit parameter
                x_trail = ratio_df[ticker].tail(tail_length).values
                y_trail = momentum_df[ticker].tail(tail_length).values
                
                if len(x_trail) == 0:
                    continue
                
                # Draw the historical rotation momentum lines (Tail)
                fig.add_trace(go.Scatter(
                    x=x_trail, y=y_trail,
                    mode="lines",
                    name=f"{ticker} Trail",
                    line=dict(width=2),
                    hoverinfo="skip",
                    showlegend=False
                ))
                
                # Plot the current endpoint node marker
                fig.add_trace(go.Scatter(
                    x=[x_trail[-1]], y=[y_trail[-1]],
                    mode="markers+text",
                    name=ticker,
                    text=[ticker],
                    textposition="top center",
                    marker=dict(size=10, line=dict(width=1, color="white")),
                    hovertemplate=f"<b>{ticker}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Mom: %{{y:.2f}}<extra></extra>"
                ))
            
            # Quadrant Text Anchors
            annotations = [
                dict(x=102.5, y=104.5, text="<b>LEADING</b>", showarrow=False, font=dict(color="green", size=14)),
                dict(x=102.5, y=95.5, text="<b>WEAKENING</b>", showarrow=False, font=dict(color="darkgoldenrod", size=14)),
                dict(x=97.5, y=95.5, text="<b>LAGGING</b>", showarrow=False, font=dict(color="red", size=14)),
                dict(x=97.5, y=104.5, text="<b>IMPROVING</b>", showarrow=False, font=dict(color="blue", size=14))
            ]
            
            fig.update_layout(
                title=dict(text=f"Relative Rotation Graph vs {benchmark_input.upper()}", font=dict(size=18)),
                xaxis=dict(title="RS-Ratio", range=[95, 105]),
                yaxis=dict(title="RS-Momentum", range=[95, 105]),
                annotations=annotations,
                height=700,
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
