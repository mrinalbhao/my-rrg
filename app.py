import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as objects

# Dictionary to map tickers to readable chart labels
TICKER_MAP = {
    "XLE": "XLE (energy)",
    "XLB": "XLB (materials)",
    "XLI": "XLI (industrials)",
    "XLY": "XLY (discretionary)",
    "XLP": "XLP (staples)",
    "XLV": "XLV (healthcare)",
    "XLF": "XLF (financials)",
    "XLK": "XLK (technology)",
    "XLC": "XLC (communications)",
    "XLU": "XLU (utilities)",
    "XLRE": "XLRE (real estate)"
}

def calculate_rrg_metrics(prices_df, benchmark_df, window=14):
    # Calculate relative price ratio
    relative_ratio = (prices_df.div(benchmark_df, axis=0)) * 100
    
    # Calculate RS-Ratio using moving average of momentum
    ratio_ma = relative_ratio.rolling(window=window).mean()
    rs_ratio = (relative_ratio / ratio_ma) * 100
    
    # Calculate RS-Momentum as rate of change of RS-Ratio
    rs_momentum = (rs_ratio / rs_ratio.shift(1)) * 100
    
    return rs_ratio.dropna(), rs_momentum.dropna()

st.title("Relative Rotation Graph (RRG) Generator")

# User inputs
tickers_input = st.text_input("Enter Asset Tickers (comma-separated):", "XLE,XLB,XLI,XLY,XLP,XLV,XLF,XLK,XLC,XLU,XLRE")
benchmark_ticker = st.text_input("Enter Benchmark Ticker:", "SPY")
history_period = st.selectbox("Select History Period:", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
tail_length = st.slider("Select Tail Length (trailing periods):", min_value=1, max_value=30, value=5)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.button("Generate RRG"):
    all_tickers = tickers + [benchmark_ticker]
    
    with st.spinner("Fetching historical data from Yahoo Finance..."):
        data = yf.download(all_tickers, period=history_period)["Adj Close"]
        
    if not data.empty and benchmark_ticker in data.columns:
        benchmark_prices = data[benchmark_ticker]
        asset_prices = data[tickers]
        
        rs_ratio, rs_momentum = calculate_rrg_metrics(asset_prices, benchmark_prices)
        
        # Take the trailing slices based on selected tail length
        ratio_tails = rs_ratio.tail(tail_length)
        momentum_tails = rs_momentum.tail(tail_length)
        
        fig = objects.Figure()
        
        # Plot data strings loop
        for ticker in tickers:
            x_vals = ratio_tails[ticker].values
            y_vals = momentum_tails[ticker].values
            
            # Map ticker name if a readable name exists in our custom dictionary
            display_label = TICKER_MAP.get(ticker, ticker)
            
            # Draw tracking tail
            fig.add_trace(objects.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=display_label,
                line=dict(width=2),
                marker=dict(size=4)
            ))
            
            # Place prominent label exactly on the latest data coordinate point
            fig.add_annotation(
                x=x_vals[-1],
                y=y_vals[-1],
                text=display_label,
                showarrow=True,
                arrowhead=1,
                ax=10,
                ay=-10
            )
            
        # Chart layout quadrants (Leading, Weakening, Lagging, Improving)
        fig.update_layout(
            title="Relative Rotation Graph (RRG)",
            xaxis_title="RS-Ratio",
            yaxis_title="RS-Momentum",
            shapes=[
                # Quadrant background grid markers centered at baseline 100
                dict(type="line", x0=100, y0=min(rs_momentum.min()), x1=100, y1=max(rs_momentum.max()), line=dict(color="gray", dash="dash")),
                dict(type="line", x0=min(rs_ratio.min()), y0=100, x1=max(rs_ratio.max()), y1=100, line=dict(color="gray", dash="dash"))
            ],
            hovermode="closest"
        )
        
        # Label quadrant spaces explicitly
        fig.add_annotation(x=101, y=101, text="Leading", showarrow=False, font=dict(color="green", size=14))
        fig.add_annotation(x=101, y=99, text="Weakening", showarrow=False, font=dict(color="orange", size=14))
        fig.add_annotation(x=99, y=99, text="Lagging", showarrow=False, font=dict(color="red", size=14))
        fig.add_annotation(x=99, y=101, text="Improving", showarrow=False, font=dict(color="blue", size=14))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch adequate historical data metrics. Please check data alignment or target inputs.")
