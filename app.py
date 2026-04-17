import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MACD Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main { background-color: #0d1117; color: #e6edf3; }
  .stApp { background-color: #0d1117; }

  h1, h2, h3 { font-family: 'Space Mono', monospace; }

  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
  }
  .positive { color: #3fb950; }
  .negative { color: #f85149; }
  .neutral  { color: #58a6ff; }

  .trade-table th {
    background: #161b22 !important;
    color: #8b949e !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
  }
  section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

  .stSlider > div { color: #e6edf3; }
  .header-tag {
    display: inline-block;
    background: #1f3a5f;
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
    letter-spacing: 0.1em;
  }
</style>
""", unsafe_allow_html=True)


# ── MACD logic ────────────────────────────────────────────────────────────────
def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def run_backtest(df: pd.DataFrame, capital: float):
    """Simple MACD crossover: buy on bullish cross, sell on bearish cross."""
    macd, signal, hist = compute_macd(df["Close"])
    df = df.copy()
    df["macd"]     = macd
    df["signal"]   = signal
    df["hist"]     = hist

    position  = 0          # shares held
    cash      = capital
    equity    = []
    trades    = []
    entry_price = 0.0
    entry_date  = None

    for i in range(1, len(df)):
        prev_diff = df["macd"].iloc[i-1] - df["signal"].iloc[i-1]
        curr_diff = df["macd"].iloc[i]   - df["signal"].iloc[i]
        price     = df["Close"].iloc[i]
        date      = df.index[i]

        # Bullish crossover → BUY
        if prev_diff < 0 and curr_diff >= 0 and position == 0:
            shares      = int(cash // price)
            if shares > 0:
                position    = shares
                cash       -= shares * price
                entry_price = price
                entry_date  = date

        # Bearish crossover → SELL
        elif prev_diff > 0 and curr_diff <= 0 and position > 0:
            cash += position * price
            pnl   = (price - entry_price) * position
            pct   = (price - entry_price) / entry_price * 100
            trades.append({
                "Entrada": entry_date.strftime("%Y-%m-%d"),
                "Salida":  date.strftime("%Y-%m-%d"),
                "Precio entrada": round(entry_price, 2),
                "Precio salida":  round(price, 2),
                "Shares": position,
                "P&L ($)": round(pnl, 2),
                "Retorno (%)": round(pct, 2),
            })
            position = 0

        equity.append(cash + position * price)

    df_eq = df.iloc[1:].copy()
    df_eq["equity"] = equity
    return df_eq, pd.DataFrame(trades), df


# ── Stats ─────────────────────────────────────────────────────────────────────
def compute_stats(df_eq, trades_df, capital):
    final_equity = df_eq["equity"].iloc[-1]
    total_return = (final_equity - capital) / capital * 100

    # Buy & hold
    bh_return = (df_eq["Close"].iloc[-1] - df_eq["Close"].iloc[0]) / df_eq["Close"].iloc[0] * 100

    # Max drawdown
    roll_max   = df_eq["equity"].cummax()
    drawdown   = (df_eq["equity"] - roll_max) / roll_max * 100
    max_dd     = drawdown.min()

    # Win rate
    win_rate = 0.0
    avg_win = avg_loss = 0.0
    if not trades_df.empty:
        wins     = trades_df[trades_df["P&L ($)"] > 0]
        losses   = trades_df[trades_df["P&L ($)"] <= 0]
        win_rate = len(wins) / len(trades_df) * 100
        avg_win  = wins["P&L ($)"].mean() if not wins.empty else 0
        avg_loss = losses["P&L ($)"].mean() if not losses.empty else 0

    return {
        "Capital final":    round(final_equity, 2),
        "Retorno total":    round(total_return, 2),
        "Buy & Hold":       round(bh_return, 2),
        "Max Drawdown":     round(max_dd, 2),
        "Nº operaciones":   len(trades_df),
        "Win rate":         round(win_rate, 2),
        "Avg ganancia":     round(avg_win, 2),
        "Avg pérdida":      round(avg_loss, 2),
    }


# ── Charts ────────────────────────────────────────────────────────────────────
def build_chart(df_eq, trades_df):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.03,
        subplot_titles=("Precio + Señales", "MACD", "Equity Curve"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_eq.index,
        open=df_eq["Open"], high=df_eq["High"],
        low=df_eq["Low"],   close=df_eq["Close"],
        name="Precio",
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
    ), row=1, col=1)

    # Trade markers
    if not trades_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(trades_df["Entrada"]),
            y=trades_df["Precio entrada"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#58a6ff"),
            name="Compra",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(trades_df["Salida"]),
            y=trades_df["Precio salida"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#f0883e"),
            name="Venta",
        ), row=1, col=1)

    # MACD
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in df_eq["hist"]]
    fig.add_trace(go.Bar(x=df_eq.index, y=df_eq["hist"], name="Histograma",
                         marker_color=colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq["macd"],
                             line=dict(color="#58a6ff", width=1.5), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq["signal"],
                             line=dict(color="#f0883e", width=1.5), name="Signal"), row=2, col=1)

    # Equity
    fig.add_trace(go.Scatter(
        x=df_eq.index, y=df_eq["equity"],
        fill="tozeroy",
        line=dict(color="#58a6ff", width=2),
        fillcolor="rgba(88,166,255,0.08)",
        name="Equity",
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="DM Sans"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        height=750,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", row=i, col=1)

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="header-tag">CONFIGURACIÓN</div>', unsafe_allow_html=True)
    st.markdown("## ⚙️ Parámetros")

    ticker  = st.text_input("Ticker", value="AAPL").upper().strip()
    capital = st.number_input("Capital inicial ($)", value=10_000, step=500, min_value=100)

    st.markdown("---")
    st.markdown("**Período histórico**")
    end_date   = datetime.today()
    start_date = st.date_input("Desde", value=end_date - timedelta(days=3*365))
    end_date   = st.date_input("Hasta", value=end_date)

    st.markdown("---")
    st.markdown("**Parámetros MACD**")
    fast   = st.slider("EMA rápida",   5,  50, 12)
    slow   = st.slider("EMA lenta",   10, 100, 26)
    signal = st.slider("Señal",        3,  20,  9)

    run_btn = st.button("▶ Ejecutar backtest", use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="header-tag">BACKTESTER</div>', unsafe_allow_html=True)
st.markdown("# 📈 MACD Strategy Backtester")
st.markdown("Estrategia de cruce de MACD sobre datos históricos reales · Coste 0 · `yfinance` + `Streamlit`")
st.markdown("---")

if run_btn:
    with st.spinner(f"Descargando datos de {ticker}..."):
        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if raw.empty:
        st.error(f"No se encontraron datos para **{ticker}**. Revisa el ticker.")
    else:
        # Flatten multi-index columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df_eq, trades_df, df_full = run_backtest(raw, capital)
        stats = compute_stats(df_eq, trades_df, capital)

        # ── KPI row ──
        cols = st.columns(4)
        kpis = [
            ("Capital final",  f"${stats['Capital final']:,.0f}",
             "positive" if stats["Retorno total"] >= 0 else "negative"),
            ("Retorno estrategia", f"{stats['Retorno total']:+.1f}%",
             "positive" if stats["Retorno total"] >= 0 else "negative"),
            ("Buy & Hold",     f"{stats['Buy & Hold']:+.1f}%",
             "positive" if stats["Buy & Hold"] >= 0 else "negative"),
            ("Max Drawdown",   f"{stats['Max Drawdown']:.1f}%", "negative"),
        ]
        for col, (label, value, cls) in zip(cols, kpis):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        cols2 = st.columns(4)
        kpis2 = [
            ("Operaciones",   str(stats["Nº operaciones"]),   "neutral"),
            ("Win Rate",      f"{stats['Win rate']:.1f}%",
             "positive" if stats["Win rate"] >= 50 else "negative"),
            ("Avg ganancia",  f"${stats['Avg ganancia']:,.0f}", "positive"),
            ("Avg pérdida",   f"${stats['Avg pérdida']:,.0f}", "negative"),
        ]
        for col, (label, value, cls) in zip(cols2, kpis2):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Chart ──
        st.plotly_chart(build_chart(df_eq, trades_df), use_container_width=True)

        # ── Trade log ──
        if not trades_df.empty:
            st.markdown("### 📋 Registro de operaciones")
            st.dataframe(
                trades_df.style.applymap(
                    lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                    else ("color: #f85149" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["P&L ($)", "Retorno (%)"]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No se generaron operaciones en este período. Prueba a ampliar el rango de fechas.")

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #8b949e;">
      <div style="font-size: 4rem;">📊</div>
      <div style="font-family: 'Space Mono', monospace; font-size: 1.2rem; margin-top: 1rem;">
        Configura los parámetros en el panel izquierdo<br>y pulsa <strong style="color:#58a6ff">▶ Ejecutar backtest</strong>
      </div>
      <div style="margin-top: 1rem; font-size: 0.85rem;">
        Prueba con tickers como <code>AAPL</code>, <code>TSLA</code>, <code>MSFT</code>, <code>NVDA</code>, <code>BTC-USD</code>
      </div>
    </div>
    """, unsafe_allow_html=True)
