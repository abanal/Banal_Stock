import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Configuració de pàgina ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backtester MACD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    -webkit-text-size-adjust: 100%;
  }
  .stApp { background: #080c14; color: #dde3ed; }
  .block-container { padding: 1.5rem 1.25rem 3rem !important; }

  /* ── Tipografia ── */
  h1 {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 1.65rem;
    letter-spacing: -0.03em;
    color: #f0f4ff;
    margin-bottom: 0.1rem;
  }
  h2, h3 {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
  }

  /* ── Capçalera ── */
  .cap-subtitol {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #5a6680;
    letter-spacing: 0.02em;
    margin-bottom: 1.5rem;
  }
  .cap-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(99,160,255,0.1);
    border: 1px solid rgba(99,160,255,0.2);
    color: #63a0ff;
    font-family: 'Outfit', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }

  /* ── Títols de secció ── */
  .titol-seccio {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #4a5568;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 1.8rem 0 0.9rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #111827;
  }

  /* ── Targetes mètriques ── */
  .metric-card {
    background: #0d1321;
    border: 1px solid #1a2235;
    border-radius: 14px;
    padding: 1rem 0.85rem 0.85rem;
    text-align: center;
    margin-bottom: 0.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
  }
  .metric-card.positiu::before { background: linear-gradient(90deg, #00d084, #00b36b); }
  .metric-card.negatiu::before { background: linear-gradient(90deg, #ff4d6d, #e63950); }
  .metric-card.neutral::before  { background: linear-gradient(90deg, #63a0ff, #4080e8); }
  .metric-card.avis::before     { background: linear-gradient(90deg, #ffaa00, #e09000); }

  .metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.63rem;
    font-weight: 500;
    color: #3d4f6e;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    line-height: 1.4;
  }
  .metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.02em;
  }
  .metric-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    color: #2d3a52;
    margin-top: 0.3rem;
    font-weight: 400;
  }

  /* ── Colors valors ── */
  .positiu .metric-value { color: #00d084; }
  .negatiu .metric-value { color: #ff4d6d; }
  .neutral  .metric-value { color: #63a0ff; }
  .avis     .metric-value { color: #ffaa00; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #0a0f1c;
    border-right: 1px solid #111827;
  }
  section[data-testid="stSidebar"] * { color: #c8d3e8 !important; }
  section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1a3a6b, #0f2548) !important;
    border: 1px solid #1e3d73 !important;
    color: #63a0ff !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem !important;
  }

  /* ── Divisor ── */
  hr { border-color: #111827 !important; margin: 1rem 0 !important; }

  /* ── Taula ── */
  .dataframe thead th {
    background: #0d1321 !important;
    color: #3d4f6e !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    white-space: nowrap;
  }
  .dataframe tbody td {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    white-space: nowrap;
    background: #080c14 !important;
    color: #8a9bb8 !important;
  }

  /* ── Mòbil ── */
  @media (max-width: 640px) {
    h1 { font-size: 1.35rem !important; }
    .metric-value { font-size: 1.15rem; }
    .block-container { padding: 0.9rem 0.7rem 2.5rem !important; }
  }
</style>
""", unsafe_allow_html=True)


# ── MACD ──────────────────────────────────────────────────────────────────────
def calcular_macd(close, rapid=12, lent=26, senyal=9):
    ema_rap  = close.ewm(span=rapid, adjust=False).mean()
    ema_len  = close.ewm(span=lent,  adjust=False).mean()
    linia    = ema_rap - ema_len
    senyal_l = linia.ewm(span=senyal, adjust=False).mean()
    histog   = linia - senyal_l
    return linia, senyal_l, histog


# ── Backtest ──────────────────────────────────────────────────────────────────
def executar_backtest(df, capital, import_op, rapid, lent, senyal):
    macd, sig, hist = calcular_macd(df["Close"], rapid, lent, senyal)
    df = df.copy()
    df["macd"]   = macd
    df["senyal"] = sig
    df["hist"]   = hist

    posicio      = 0
    efectiu      = capital
    equitat      = []
    operacions   = []
    preu_entrada = 0.0
    data_entrada = None

    for i in range(1, len(df)):
        dif_prev = df["macd"].iloc[i-1] - df["senyal"].iloc[i-1]
        dif_act  = df["macd"].iloc[i]   - df["senyal"].iloc[i]
        preu     = float(df["Close"].iloc[i])
        data     = df.index[i]

        if dif_prev < 0 and dif_act >= 0 and posicio == 0:
            pressupost = min(import_op, efectiu) if import_op > 0 else efectiu
            accions    = int(pressupost // preu)
            if accions > 0:
                posicio      = accions
                efectiu     -= accions * preu
                preu_entrada = preu
                data_entrada = data

        elif dif_prev > 0 and dif_act <= 0 and posicio > 0:
            efectiu += posicio * preu
            ganancia = (preu - preu_entrada) * posicio
            pct      = (preu - preu_entrada) / preu_entrada * 100
            dies     = (data - data_entrada).days
            operacions.append({
                "Entrada":        data_entrada.strftime("%Y-%m-%d"),
                "Sortida":        data.strftime("%Y-%m-%d"),
                "Dies":           dies,
                "Preu entrada":   round(preu_entrada, 2),
                "Preu sortida":   round(preu, 2),
                "Accions":        posicio,
                "Import inv.($)": round(preu_entrada * posicio, 2),
                "G/P ($)":        round(ganancia, 2),
                "Retorn (%)":     round(pct, 2),
            })
            posicio = 0

        equitat.append(efectiu + posicio * preu)

    df_eq = df.iloc[1:].copy()
    df_eq["equitat"] = equitat
    return df_eq, pd.DataFrame(operacions), df


# ── Estadístiques ─────────────────────────────────────────────────────────────
def calcular_stats(df_eq, ops_df, capital):
    equitat_final = df_eq["equitat"].iloc[-1]
    retorn_total  = (equitat_final - capital) / capital * 100
    retorn_bh     = (df_eq["Close"].iloc[-1] - df_eq["Close"].iloc[0]) / df_eq["Close"].iloc[0] * 100

    max_acum = df_eq["equitat"].cummax()
    drawdown = (df_eq["equitat"] - max_acum) / max_acum * 100
    max_dd   = drawdown.min()

    saldo_max      = df_eq["equitat"].max()
    saldo_min      = df_eq["equitat"].min()
    data_saldo_max = df_eq["equitat"].idxmax().strftime("%Y-%m-%d")
    data_saldo_min = df_eq["equitat"].idxmin().strftime("%Y-%m-%d")

    ret_diaris = df_eq["equitat"].pct_change().dropna()
    sharpe = (ret_diaris.mean() / ret_diaris.std() * np.sqrt(252)
              if ret_diaris.std() > 0 else 0.0)

    taxa_enc = ganancia_m = perdua_m = millor = pitjor = dies_m = 0.0
    ratxa_g = ratxa_p = 0

    if not ops_df.empty:
        guanys  = ops_df[ops_df["G/P ($)"] > 0]
        perdues = ops_df[ops_df["G/P ($)"] <= 0]
        taxa_enc   = len(guanys) / len(ops_df) * 100
        ganancia_m = guanys["G/P ($)"].mean()  if not guanys.empty  else 0
        perdua_m   = perdues["G/P ($)"].mean() if not perdues.empty else 0
        millor     = ops_df["G/P ($)"].max()
        pitjor     = ops_df["G/P ($)"].min()
        dies_m     = ops_df["Dies"].mean()

        cur = 0
        for gp in ops_df["G/P ($)"]:
            cur = (cur + 1 if cur >= 0 else 1) if gp > 0 else (cur - 1 if cur <= 0 else -1)
            if cur > 0: ratxa_g = max(ratxa_g, cur)
            else:       ratxa_p = max(ratxa_p, abs(cur))

    return {
        "Equitat final":    round(equitat_final, 2),
        "Retorn total":     round(retorn_total, 2),
        "Buy & Hold":       round(retorn_bh, 2),
        "Max Drawdown":     round(max_dd, 2),
        "Sharpe":           round(sharpe, 2),
        "Num ops":          len(ops_df),
        "Taxa encert":      round(taxa_enc, 2),
        "Ganancia mitjana": round(ganancia_m, 2),
        "Perdua mitjana":   round(perdua_m, 2),
        "Millor op":        round(millor, 2),
        "Pitjor op":        round(pitjor, 2),
        "Saldo max":        round(saldo_max, 2),
        "Data saldo max":   data_saldo_max,
        "Saldo min":        round(saldo_min, 2),
        "Data saldo min":   data_saldo_min,
        "Ratxa guanys":     ratxa_g,
        "Ratxa perdues":    ratxa_p,
        "Dies mitjana":     round(dies_m, 1),
    }


# ── Gràfic ────────────────────────────────────────────────────────────────────
def construir_grafic(df_eq, ops_df, stats):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.03,
        subplot_titles=("Preu + Senyals", "MACD", "Corba d'equitat"),
    )

    fig.add_trace(go.Candlestick(
        x=df_eq.index,
        open=df_eq["Open"], high=df_eq["High"],
        low=df_eq["Low"],   close=df_eq["Close"],
        name="Preu",
        increasing_line_color="#00d084",
        decreasing_line_color="#ff4d6d",
    ), row=1, col=1)

    if not ops_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(ops_df["Entrada"]),
            y=ops_df["Preu entrada"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="#63a0ff",
                        line=dict(width=1, color="#3070d0")),
            name="Compra",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(ops_df["Sortida"]),
            y=ops_df["Preu sortida"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=11, color="#ffaa00",
                        line=dict(width=1, color="#cc8800")),
            name="Venda",
        ), row=1, col=1)

    bar_colors = ["#00d084" if v >= 0 else "#ff4d6d" for v in df_eq["hist"]]
    fig.add_trace(go.Bar(x=df_eq.index, y=df_eq["hist"],
                         marker_color=bar_colors, opacity=0.6, name="Histograma"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq["macd"],
                             line=dict(color="#63a0ff", width=1.5), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq["senyal"],
                             line=dict(color="#ffaa00", width=1.5), name="Senyal"), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_eq.index, y=df_eq["equitat"],
        fill="tozeroy",
        line=dict(color="#63a0ff", width=2),
        fillcolor="rgba(99,160,255,0.07)",
        name="Equitat",
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(stats["Data saldo max"])],
        y=[stats["Saldo max"]],
        mode="markers+text",
        marker=dict(symbol="star", size=13, color="#00d084"),
        text=["MÀX"], textposition="top center",
        textfont=dict(color="#00d084", size=9, family="Outfit"),
        name="Saldo màxim",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(stats["Data saldo min"])],
        y=[stats["Saldo min"]],
        mode="markers+text",
        marker=dict(symbol="x", size=11, color="#ff4d6d"),
        text=["MÍN"], textposition="bottom center",
        textfont=dict(color="#ff4d6d", size=9, family="Outfit"),
        name="Saldo mínim",
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#080c14",
        font=dict(color="#3d4f6e", family="Inter"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(13,19,33,0.9)",
            bordercolor="#1a2235",
            borderwidth=1,
            font=dict(family="Inter", size=11),
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        height=700,
        margin=dict(l=4, r=4, t=44, b=4),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#0d1321", row=i, col=1, showline=False)
        fig.update_yaxes(gridcolor="#0d1321", row=i, col=1, showline=False)
        fig.layout.annotations[i-1].font.family = "Outfit"
        fig.layout.annotations[i-1].font.size   = 11
        fig.layout.annotations[i-1].font.color  = "#3d4f6e"

    return fig


# ── Helper targeta ────────────────────────────────────────────────────────────
def targeta(col, etiqueta, valor, cls, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    col.markdown(f"""
    <div class="metric-card {cls}">
      <div class="metric-label">{etiqueta}</div>
      <div class="metric-value">{valor}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paràmetres")

    ticker = st.text_input("Ticker", value="AAPL").upper().strip()

    st.markdown("**Capital i mida de l'operació**")
    capital   = st.number_input("Capital inicial ($)", value=10_000, step=500, min_value=100)
    import_op = st.number_input(
        "Import per operació ($)", value=0, step=500, min_value=0,
        help="Import fix per cada compra. 0 = tot l'efectiu disponible."
    )
    if import_op == 0:
        st.caption("🔄 Tot el capital disponible per operació")
    else:
        st.caption(f"🎯 Màxim **${import_op:,}** per operació")

    st.markdown("---")
    st.markdown("**Període històric**")
    fi_dt      = datetime.today()
    data_inici = st.date_input("Des de", value=fi_dt - timedelta(days=3*365))
    data_fi    = st.date_input("Fins a", value=fi_dt)

    st.markdown("---")
    st.markdown("**Paràmetres MACD**")
    rapid_p  = st.slider("EMA ràpida",  5,  50, 12)
    lent_p   = st.slider("EMA lenta",  10, 100, 26)
    senyal_p = st.slider("Senyal",      3,  20,  9)

    executar_btn = st.button("▶ Executar backtest", use_container_width=True)


# ── Capçalera ─────────────────────────────────────────────────────────────────
st.markdown('<div class="cap-badge">📈 BACKTESTER</div>', unsafe_allow_html=True)
st.markdown("# Estratègia MACD")
st.markdown('<div class="cap-subtitol">Creuament MACD · Dades històriques reals · Cost 0</div>',
            unsafe_allow_html=True)

if executar_btn:
    with st.spinner(f"Descarregant dades de {ticker}..."):
        raw = yf.download(ticker, start=data_inici, end=data_fi, auto_adjust=True, progress=False)

    if raw.empty:
        st.error(f"No s'han trobat dades per a **{ticker}**. Comprova el ticker.")
    else:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df_eq, ops_df, _ = executar_backtest(raw, capital, import_op, rapid_p, lent_p, senyal_p)
        s = calcular_stats(df_eq, ops_df, capital)

        # ── Rendiment general ────────────────────────────────────────────────
        st.markdown('<div class="titol-seccio">📊 Rendiment general</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        targeta(c1, "Capital final",     f"${s['Equitat final']:,.0f}",
                "positiu" if s["Retorn total"] >= 0 else "negatiu")
        targeta(c2, "Retorn estratègia", f"{s['Retorn total']:+.1f}%",
                "positiu" if s["Retorn total"] >= 0 else "negatiu")
        targeta(c3, "Buy & Hold",        f"{s['Buy & Hold']:+.1f}%",
                "positiu" if s["Buy & Hold"] >= 0 else "negatiu")

        c4, c5 = st.columns(2)
        targeta(c4, "Màx. Drawdown",    f"{s['Max Drawdown']:.1f}%", "negatiu")
        targeta(c5, "Ràtio de Sharpe",  f"{s['Sharpe']:.2f}",
                "positiu" if s["Sharpe"] >= 1 else "avis" if s["Sharpe"] >= 0 else "negatiu")

        # ── Saldo del compte ─────────────────────────────────────────────────
        st.markdown('<div class="titol-seccio">💰 Saldo del compte</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        targeta(c1, "Saldo màxim assolit",   f"${s['Saldo max']:,.0f}", "positiu",
                sub=f"📅 {s['Data saldo max']}")
        targeta(c2, "Saldo mínim registrat",  f"${s['Saldo min']:,.0f}", "negatiu",
                sub=f"📅 {s['Data saldo min']}")
        c3, = st.columns(1)
        targeta(c3, "Capital inicial", f"${capital:,.0f}", "neutral",
                sub=f"Import/op: {'Tot l\'efectiu' if import_op == 0 else f'${import_op:,}'}")

        # ── Operacions ───────────────────────────────────────────────────────
        st.markdown('<div class="titol-seccio">🎯 Estadístiques d\'operacions</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        targeta(c1, "Nº operacions",   str(s["Num ops"]),            "neutral")
        targeta(c2, "Taxa d'encert",   f"{s['Taxa encert']:.1f}%",
                "positiu" if s["Taxa encert"] >= 50 else "negatiu")
        targeta(c3, "Dies mitjana/op", f"{s['Dies mitjana']} dies",  "neutral")

        c4, c5, c6 = st.columns(3)
        targeta(c4, "Guany mitjà",     f"${s['Ganancia mitjana']:,.0f}", "positiu")
        targeta(c5, "Millor operació", f"${s['Millor op']:,.0f}",        "positiu")
        targeta(c6, "Pèrdua mitjana",  f"${s['Perdua mitjana']:,.0f}",   "negatiu")

        c7, c8, c9 = st.columns(3)
        targeta(c7, "Pitjor operació",       f"${s['Pitjor op']:,.0f}",    "negatiu")
        targeta(c8, "Ratxa guanyadora màx",  f"{s['Ratxa guanys']} ops",   "positiu")
        targeta(c9, "Ratxa perdedora màx",   f"{s['Ratxa perdues']} ops",  "negatiu")

        st.markdown("---")

        # ── Gràfic ───────────────────────────────────────────────────────────
        st.plotly_chart(construir_grafic(df_eq, ops_df, s), use_container_width=True)

        # ── Registre ─────────────────────────────────────────────────────────
        if not ops_df.empty:
            st.markdown('<div class="titol-seccio">📋 Registre d\'operacions</div>',
                        unsafe_allow_html=True)

            def color_gp(v):
                if isinstance(v, (int, float)) and v > 0: return "color: #00d084"
                if isinstance(v, (int, float)) and v < 0: return "color: #ff4d6d"
                return ""

            try:
                styled = ops_df.style.map(color_gp, subset=["G/P ($)", "Retorn (%)"])
            except AttributeError:
                styled = ops_df.style.applymap(color_gp, subset=["G/P ($)", "Retorn (%)"])

            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("No s'han generat operacions. Prova a ampliar el rang de dates.")

else:
    st.markdown("""
    <div style="text-align:center; padding:3.5rem 1.5rem; color:#1e2d45;">
      <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
      <div style="font-family:'Outfit',sans-serif; font-size:1.05rem; font-weight:600;
                  color:#2a3d5e; line-height:1.7;">
        Configura els paràmetres al panell<br>
        i prem <span style="color:#63a0ff;">▶ Executar backtest</span>
      </div>
      <div style="margin-top:1.4rem; font-size:0.78rem; color:#1a2840;
                  line-height:2.6; font-family:'Inter',sans-serif;">
        <code style="background:#0d1321;color:#3d6099;padding:2px 7px;border-radius:5px;">AAPL</code>
        &nbsp;·&nbsp;
        <code style="background:#0d1321;color:#3d6099;padding:2px 7px;border-radius:5px;">TSLA</code>
        &nbsp;·&nbsp;
        <code style="background:#0d1321;color:#3d6099;padding:2px 7px;border-radius:5px;">NVDA</code>
        <br>
        <code style="background:#0d1321;color:#3d6099;padding:2px 7px;border-radius:5px;">BTC-USD</code>
        &nbsp;·&nbsp;
        <code style="background:#0d1321;color:#3d6099;padding:2px 7px;border-radius:5px;">EURUSD=X</code>
        &nbsp;·&nbsp;
        <code style="background:#0d1321;color:#3d6099;padding:2px 7px;border-radius:5px;">ITX.MC</code>
      </div>
    </div>
    """, unsafe_allow_html=True)
