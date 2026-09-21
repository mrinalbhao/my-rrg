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
# Tighten up default Streamlit spacing so the whole panel fits one screen.
st.markdown("""
<style>
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.6rem; }
section[data-testid="stSidebar"] .stCheckbox {
    margin-bottom: 2px;
    padding: 1px 0;
}
section[data-testid="stSidebar"] .stCheckbox label p { font-size: 0.9rem; }
section[data-testid="stSidebar"] hr { margin: 0.6rem 0; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    margin-top: 0; margin-bottom: 0.4rem; padding-top: 0;
}
section[data-testid="stSidebar"] .stButton button { padding: 0.35rem 0.6rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.subheader("Configuration Settings")

# The permanent checklist universe — independent of the text box below.
# These 17 always show up as checkboxes; checking/unchecking toggles them
# on the chart. Nothing you type in the text box ever removes these.
DEFAULT_TICKERS = [
    "IGV", "SMH", "BOTZ", "PPA", "LIT", "BOTT", "DTCR", "QTUM", "HACK",
    "SKYY", "NLR", "SHLD", "IBB", "DRIV", "REMX", "FINX", "HERO"
]

# Text input for ADDITIONAL tickers — these always render, no checkbox needed,
# and are layered on top of whichever DEFAULT_TICKERS are checked.
ticker_input = st.sidebar.text_input(
    "Additional Tickers (Comma separated, always rendered)",
    value=""
)

# Parse the text box into an ordered, de-duplicated extra list
extra_tickers = []
for t in ticker_input.split(","):
    t = t.strip().upper()
    if t and t not in extra_tickers:
        extra_tickers.append(t)

# Benchmark, interval and tail points share two rows instead of three
bench_col, interval_col = st.sidebar.columns(2)
benchmark_input = bench_col.text_input("Benchmark", value="SPY")
interval_choice = interval_col.selectbox(
    "Interval", options=["1 Day", "1 Week"], index=1  # Default to 1 Week to match your chart
)

tail_points = st.sidebar.number_input(
    "Number of Tail Points (History)",
    min_value=3,
    max_value=100,
    value=10,
    step=1
)

# --- VISIBILITY CHECKBOXES ---
# Fixed set of DEFAULT_TICKERS. Unchecking one hides its trail from the
# chart without refetching anything and without ever removing it from view here.
st.sidebar.markdown("---")

CHK_PREFIX = "chk_"

header_col, all_col, none_col = st.sidebar.columns([2, 1, 1])
header_col.markdown("**Visible Tickers**")
select_all = all_col.button("All", use_container_width=True)
clear_all = none_col.button("None", use_container_width=True)

# Seed state for the default tickers (default: visible), then honour the
# bulk buttons. Both happen before the widgets are instantiated.
for t in DEFAULT_TICKERS:
    key = CHK_PREFIX + t
    if key not in st.session_state:
        st.session_state[key] = True
    if select_all:
        st.session_state[key] = True
    if clear_all:
        st.session_state[key] = False

# 3 columns keeps 17 tickers to 6 short rows instead of 17.
chk_cols = st.sidebar.columns(3)
for i, t in enumerate(DEFAULT_TICKERS):
    chk_cols[i % 3].checkbox(t, key=CHK_PREFIX + t)

checked_defaults = [t for t in DEFAULT_TICKERS if st.session_state.get(CHK_PREFIX + t, True)]

# Final render list: checked defaults + always-on extras, de-duplicated,
# preserving default order first then any new extras.
parsed_tickers = list(dict.fromkeys(DEFAULT_TICKERS + extra_tickers))  # full universe, for colour stability
selected_tickers = list(dict.fromkeys(checked_defaults + extra_tickers))

st.sidebar.caption(f"{len(selected_tickers)} shown ({len(checked_defaults)} checked + {len(extra_tickers)} extra)")

st.sidebar.markdown("---")

# Go Button
trigger_go = st.sidebar.button("🚀 Render RRG Chart", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def calculate_rrg_metrics(tickers, benchmark, interval_str, history_needed):
    """Fetches stock data and extracts standard RS-Ratio and RS-Momentum lines.

    Cached on its arguments, so toggling checkboxes re-renders instantly
    instead of hitting Yahoo Finance again. `tickers` must be a tuple.
    """
    tickers = list(tickers)

    # Force daily downloads if 1 Week is requested to manually resample data points
    yf_interval = "1d" if interval_str == "1 Week" else {"1 Day": "1d"}[interval_str]

    all_tickers = list(set(tickers + [benchmark]))
    data = yf.download(all_tickers, period=history_needed, interval=yf_interval, group_by='column')

    if data.empty:
        return None

    df_close = data['Close'] if 'Close' in data.columns else pd.DataFrame()
    df_close = df_close.dropna()
    if benchmark not in df_close.columns:
        return None

    # --- ADVANCED TIME-RESAMPLING FOR ACCURATE TAIL-END NODES ---
    if interval_str == "1 Week":
        # Aggregate daily to weekly and adjust index to end on Friday
        df_close = df_close.resample('W').last().dropna()
        df_close.index = df_close.index - pd.to_timedelta(df_close.index.dayofweek - 4, unit='D')
        df_close = df_close.dropna()

    rrg_results = {}
    for t in tickers:
        if t not in df_close.columns or t == benchmark:
            continue

        rs_raw = (df_close[t] / df_close[benchmark]) * 100
        rs_ema1 = rs_raw.ewm(span=14, adjust=False).mean()
        rs_ema2 = rs_ema1.ewm(span=14, adjust=False).mean()
        rs_std = rs_raw.rolling(window=14).std()
        rs_ratio = 100 + ((rs_ema2 - rs_ema2.rolling(window=14).mean()) / (rs_std + 1e-8)) * 10

        rs_mom_ema = rs_ratio.ewm(span=14, adjust=False).mean()
        rs_mom = 100 + ((rs_ratio - rs_mom_ema) / (rs_ratio.rolling(window=14).std() + 1e-8)) * 10

        rrg_results[t] = pd.DataFrame({'RS_Ratio': rs_ratio, 'RS_Momentum': rs_mom}).dropna()

    return rrg_results


def smooth_trajectory(x_coords, y_coords, steps=100):
    """Uses a Cubic Spline to smoothly fill spaces between jagged data nodes."""
    t_original = np.linspace(0, 1, len(x_coords))
    t_smooth = np.linspace(0, 1, steps)

    cs_x = CubicSpline(t_original, x_coords)
    cs_y = CubicSpline(t_original, y_coords)

    return cs_x(t_smooth), cs_y(t_smooth)


# --- APPLICATION LOGIC ---
# Once rendered the first time, keep rendering on every rerun so checkbox
# toggles take effect immediately without another button press.
if trigger_go:
    st.session_state["rrg_rendered"] = True

if st.session_state.get("rrg_rendered"):
    bench_ticker = benchmark_input.strip().upper()

    if not parsed_tickers or not bench_ticker:
        st.error("Please provide both valid asset symbols and a benchmark tracker.")
    elif not selected_tickers:
        st.warning("No tickers are checked. Tick at least one in the sidebar to draw the chart.")
    else:
        with st.spinner("Analyzing market momentum fields and generating clean vectors..."):
            raw_rrg_data = calculate_rrg_metrics(
                tickers=tuple(parsed_tickers),
                benchmark=bench_ticker,
                interval_str=interval_choice,
                history_needed="2y"
            )

            if not raw_rrg_data:
                st.error("Data tracking failed. Please ensure stock tickers exist on Yahoo Finance.")
            else:
                fig = go.Figure()
                all_x, all_y = [], []

                # Distinct color palette sequence matching institutional charts
                color_palette = [
                    "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd",
                    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
                    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                    "#c49c94", "#f7b6d2",
                ]
                # Colour is pinned to the ticker's slot in the full text-box list,
                # so hiding one ticker never recolours the others.
                color_map = {
                    t: color_palette[i % len(color_palette)]
                    for i, t in enumerate(parsed_tickers)
                }

                missing = [t for t in selected_tickers if t not in raw_rrg_data]

                for ticker in selected_tickers:
                    df = raw_rrg_data.get(ticker)
                    if df is None:
                        continue

                    tail_df = df.tail(int(tail_points))
                    if len(tail_df) < 3:
                        continue

                    x_raw = tail_df['RS_Ratio'].values
                    y_raw = tail_df['RS_Momentum'].values

                    ticker_color = color_map.get(ticker, "#1f77b4")

                    # Smooth out the lines seamlessly
                    x_smooth, y_smooth = smooth_trajectory(x_raw, y_raw, steps=200)

                    all_x.extend(x_raw)
                    all_y.extend(y_raw)

                    head_x = x_raw[-1]
                    head_y = y_raw[-1]

                    # Extract and format the actual dates matching the historical nodes
                    dates_raw = tail_df.index.strftime('%Y-%m-%d').tolist()

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
                        hovertext=dates_raw[:-1],
                        hovertemplate=(
                            f"<b>{ticker}</b><br>" +
                            "Date: %{hovertext}<br>" +
                            "RS-Ratio: %{x:.2f}<br>" +
                            "RS-Momentum: %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    ))

                    # Explicit Head Marker identifying current status node
                    fig.add_trace(go.Scatter(
                        x=[head_x], y=[head_y],
                        mode='markers+text',
                        name=ticker,
                        text=[f"<b>{ticker}</b>"],
                        textposition="top center",
                        marker=dict(size=12, symbol='circle', color=ticker_color, line=dict(width=2, color='black')),
                        hovertext=[dates_raw[-1]],
                        hovertemplate=(
                            f"<b>{ticker}</b><br>" +
                            "Date: %{hovertext}<br>" +
                            "RS-Ratio: %{x:.2f}<br>" +
                            "RS-Momentum: %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    ))

                if not all_x or not all_y:
                    st.error("Not enough historical data found to construct the RRG tail.")
                else:
                    max_dev = max(
                        max(abs(np.array(all_x) - 100)),
                        max(abs(np.array(all_y) - 100))
                    ) * 1.15

                    if max_dev < 3:
                        max_dev = 3

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
                    fig.add_annotation(x=100 + (max_dev / 2), y=100 + (max_dev / 2), text="<b>LEADING</b>", font=dict(color="green", size=16), showarrow=False)
                    fig.add_annotation(x=100 + (max_dev / 2), y=100 - (max_dev / 2), text="<b>WEAKENING</b>", font=dict(color="gold", size=16), showarrow=False)
                    fig.add_annotation(x=100 - (max_dev / 2), y=100 - (max_dev / 2), text="<b>LAGGING</b>", font=dict(color="red", size=16), showarrow=False)
                    fig.add_annotation(x=100 - (max_dev / 2), y=100 + (max_dev / 2), text="<b>IMPROVING</b>", font=dict(color="blue", size=16), showarrow=False)

                    # Final layout configurations
                    fig.update_layout(
                        width=950,
                        height=780,
                        xaxis=dict(title="<b>RS-Ratio (Trend)</b>", range=[x_min, x_max], zeroline=False),
                        yaxis=dict(title="<b>RS-Momentum (Velocity)</b>", range=[y_min, y_max], zeroline=False),
                        title=f"Relative Rotation Graph vs {bench_ticker} ({interval_choice} System)",
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    if missing:
                        st.caption("No data returned for: " + ", ".join(missing))
else:
    st.info("Configure variables inside left side panel and click 'Render RRG Chart' to track structural transformations.")
