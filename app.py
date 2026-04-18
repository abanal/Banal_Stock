import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

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
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; -webkit-text-size-adjust: 100%; }
  .stApp { background: #080c14; color: #dde3ed; }
  .block-container { padding: 1.5rem 1.25rem 3rem !important; }
  h1 { font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.65rem;
       letter-spacing: -0.03em; color: #f0f4ff; margin-bottom: 0.1rem; }
  h2, h3 { font-family: 'Outfit', sans-serif; font-weight: 600; }
  .cap-subtitol { font-family: 'Inter', sans-serif; font-size: 0.8rem; color: #5a6680;
                  letter-spacing: 0.02em; margin-bottom: 1.5rem; }
  .cap-badge { display: inline-flex; align-items: center; gap: 5px;
               background: rgba(99,160,255,0.1); border: 1px solid rgba(99,160,255,0.2);
               color: #63a0ff; font-family: 'Outfit', sans-serif; font-size: 0.65rem;
               font-weight: 600; padding: 3px 10px; border-radius: 999px;
               letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.6rem; }
  .titol-seccio { display: flex; align-items: center; gap: 0.5rem;
                  font-family: 'Outfit', sans-serif; font-size: 0.7rem; font-weight: 600;
                  color: #4a5568; letter-spacing: 0.14em; text-transform: uppercase;
                  margin: 1.8rem 0 0.9rem 0; padding-bottom: 0.5rem;
                  border-bottom: 1px solid #111827; }
  .metric-card { background: #0d1321; border: 1px solid #1a2235; border-radius: 14px;
                 padding: 1rem 0.85rem 0.85rem; text-align: center; margin-bottom: 0.6rem;
                 position: relative; overflow: hidden; }
  .metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0;
                          height: 2px; border-radius: 14px 14px 0 0; }
  .metric-card.positiu::before { background: linear-gradient(90deg, #00d084, #00b36b); }
  .metric-card.negatiu::before { background: linear-gradient(90deg, #ff4d6d, #e63950); }
  .metric-card.neutral::before  { background: linear-gradient(90deg, #63a0ff, #4080e8); }
  .metric-card.avis::before     { background: linear-gradient(90deg, #ffaa00, #e09000); }
  .metric-label { font-family: 'Inter', sans-serif; font-size: 0.63rem; font-weight: 500;
                  color: #3d4f6e; letter-spacing: 0.06em; text-transform: uppercase;
                  margin-bottom: 0.4rem; line-height: 1.4; }
  .metric-value { font-family: 'Outfit', sans-serif; font-size: 1.35rem; font-weight: 700;
                  line-height: 1.1; letter-spacing: -0.02em; }
  .metric-sub { font-family: 'Inter', sans-serif; font-size: 0.6rem; color: #2d3a52;
                margin-top: 0.3rem; font-weight: 400; }
  .positiu .metric-value { color: #00d084; }
  .negatiu .metric-value { color: #ff4d6d; }
  .neutral  .metric-value { color: #63a0ff; }
  .avis     .metric-value { color: #ffaa00; }
  section[data-testid="stSidebar"] { background: #0a0f1c; border-right: 1px solid #111827; }
  section[data-testid="stSidebar"] * { color: #c8d3e8 !important; }
  section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1a3a6b, #0f2548) !important;
    border: 1px solid #1e3d73 !important; color: #63a0ff !important;
    border-radius: 10px !important; font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    letter-spacing: 0.04em !important; padding: 0.6rem !important; }
  hr { border-color: #111827 !important; margin: 1rem 0 !important; }
  .dataframe thead th { background: #0d1321 !important; color: #3d4f6e !important;
    font-family: 'Outfit', sans-serif !important; font-size: 0.68rem !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important; white-space: nowrap; }
  .dataframe tbody td { font-family: 'Inter', sans-serif; font-size: 0.78rem;
    white-space: nowrap; background: #080c14 !important; color: #8a9bb8 !important; }
  .info-box { background: #0d1321; border: 1px solid #1a2235; border-left: 3px solid #63a0ff;
              border-radius: 8px; padding: 0.9rem 1rem; margin: 0.5rem 0 1rem;
              font-size: 0.78rem; color: #7a8fa8; line-height: 1.6; }
  .info-box strong { color: #63a0ff; }
  @media (max-width: 640px) {
    h1 { font-size: 1.35rem !important; }
    .metric-value { font-size: 1.15rem; }
    .block-container { padding: 0.9rem 0.7rem 2.5rem !important; }
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INDICADORS COMUNS
# ══════════════════════════════════════════════════════════════════════════════
def calcular_macd(close, rapid=12, lent=26, senyal=9):
    ema_rap  = close.ewm(span=rapid, adjust=False).mean()
    ema_len  = close.ewm(span=lent,  adjust=False).mean()
    linia    = ema_rap - ema_len
    senyal_l = linia.ewm(span=senyal, adjust=False).mean()
    histog   = linia - senyal_l
    return linia, senyal_l, histog


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÈGIA 1: CREUAMENT MACD
# ══════════════════════════════════════════════════════════════════════════════
def executar_creuament(df, capital, import_op, rapid, lent, senyal):
    macd, sig, hist = calcular_macd(df["Close"], rapid, lent, senyal)
    df = df.copy()
    df["macd"]   = macd
    df["senyal"] = sig
    df["hist"]   = hist

    posicio = 0; efectiu = capital
    equitat = []; operacions = []
    preu_entrada = 0.0; data_entrada = None

    for i in range(1, len(df)):
        dif_prev = df["macd"].iloc[i-1] - df["senyal"].iloc[i-1]
        dif_act  = df["macd"].iloc[i]   - df["senyal"].iloc[i]
        preu     = float(df["Close"].iloc[i])
        data     = df.index[i]

        if dif_prev < 0 and dif_act >= 0 and posicio == 0:
            pressupost = min(import_op, efectiu) if import_op > 0 else efectiu
            accions    = int(pressupost // preu)
            if accions > 0:
                posicio = accions; efectiu -= accions * preu
                preu_entrada = preu; data_entrada = data

        elif dif_prev > 0 and dif_act <= 0 and posicio > 0:
            efectiu += posicio * preu
            ganancia = (preu - preu_entrada) * posicio
            pct      = (preu - preu_entrada) / preu_entrada * 100
            dies     = (data - data_entrada).days
            operacions.append({
                "Entrada": data_entrada.strftime("%Y-%m-%d"),
                "Sortida": data.strftime("%Y-%m-%d"),
                "Dies": dies, "Preu entrada": round(preu_entrada, 2),
                "Preu sortida": round(preu, 2), "Accions": posicio,
                "Import inv.($)": round(preu_entrada * posicio, 2),
                "G/P ($)": round(ganancia, 2), "Retorn (%)": round(pct, 2),
                "Senyal": "Creuament",
            })
            posicio = 0

        equitat.append(efectiu + posicio * preu)

    df_eq = df.iloc[1:].copy()
    df_eq["equitat"] = equitat
    return df_eq, pd.DataFrame(operacions)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÈGIA 2: DIVERGÈNCIA OCULTA + ENDING DIAGONAL
# ══════════════════════════════════════════════════════════════════════════════

def trobar_pivots(serie, finestra=5):
    """Troba pics (1) i valls (-1) locals en una sèrie."""
    pivots = pd.Series(0, index=serie.index)
    vals   = serie.values
    for i in range(finestra, len(vals) - finestra):
        seg = vals[i - finestra: i + finestra + 1]
        if vals[i] == max(seg): pivots.iloc[i] =  1   # pic
        if vals[i] == min(seg): pivots.iloc[i] = -1   # vall
    return pivots


def detectar_divergencia_oculta(close, macd_hist, pivots, i, lookback=40):
    """
    Divergència oculta BAJISTA (per venda/sortida):
      Preu: nou màxim (preu[i] > preu anterior)
      MACD: màxim inferior (histograma[i] < histograma anterior)
    Divergència oculta ALCISTA (per compra/entrada):
      Preu: nou mínim (preu[i] < preu anterior)
      MACD: mínim superior (histograma[i] > histograma anterior)
    Retorna: 'alcista', 'bajista' o None
    """
    inici = max(0, i - lookback)

    # Busca pivots anteriors rellevants
    pics_prev  = [j for j in range(inici, i) if pivots.iloc[j] ==  1]
    valls_prev = [j for j in range(inici, i) if pivots.iloc[j] == -1]

    # ── Divergència oculta bajista (senyal de venda) ──
    if pivots.iloc[i] == 1 and len(pics_prev) >= 1:
        j = pics_prev[-1]   # pic anterior més proper
        preu_actual  = float(close.iloc[i])
        preu_prev    = float(close.iloc[j])
        macd_actual  = float(macd_hist.iloc[i])
        macd_prev    = float(macd_hist.iloc[j])
        # Preu: màxim superior | MACD: màxim inferior → debilitat oculta
        if preu_actual > preu_prev and macd_actual < macd_prev:
            return "bajista"

    # ── Divergència oculta alcista (senyal de compra) ──
    if pivots.iloc[i] == -1 and len(valls_prev) >= 1:
        j = valls_prev[-1]  # vall anterior més propera
        preu_actual  = float(close.iloc[i])
        preu_prev    = float(close.iloc[j])
        macd_actual  = float(macd_hist.iloc[i])
        macd_prev    = float(macd_hist.iloc[j])
        # Preu: mínim inferior | MACD: mínim superior → força oculta
        if preu_actual < preu_prev and macd_actual > macd_prev:
            return "alcista"

    return None


def detectar_canal_convergent(close, i, finestra=20, min_r2=0.65):
    """
    Detecta si els últims N preus formen un canal convergent (wedge/ending diagonal).
    Fa regressió lineal dels màxims i mínims separadament.
    Retorna True si les dues línies convergeixen (pendent oposada i s'apropen).
    """
    if i < finestra + 5:
        return False, 0.0

    seg     = close.iloc[i - finestra: i + 1].values
    x       = np.arange(len(seg))
    maxims  = []
    minims  = []

    # Separa màxims locals i mínims locals dins el segment
    for k in range(1, len(seg) - 1):
        if seg[k] >= seg[k-1] and seg[k] >= seg[k+1]: maxims.append((k, seg[k]))
        if seg[k] <= seg[k-1] and seg[k] <= seg[k+1]: minims.append((k, seg[k]))

    if len(maxims) < 2 or len(minims) < 2:
        return False, 0.0

    # Regressió lineal dels màxims
    xm, ym = zip(*maxims)
    pm     = np.polyfit(xm, ym, 1)   # pm[0] = pendent màxims

    # Regressió lineal dels mínims
    xv, yv = zip(*minims)
    pv     = np.polyfit(xv, yv, 1)   # pv[0] = pendent mínims

    # Convergència: pendents de signe oposat O les dues convergint cap al centre
    amplada_inici = abs(np.polyval(pm, 0)    - np.polyval(pv, 0))
    amplada_final = abs(np.polyval(pm, finestra) - np.polyval(pv, finestra))
    convergent    = amplada_final < amplada_inici * 0.75   # 25% menys ample al final

    return convergent, amplada_final / max(amplada_inici, 1e-9)


def executar_divergencia_diagonal(df, capital, import_op, rapid, lent, senyal,
                                   finestra_pivot=5, finestra_canal=20,
                                   lookback_div=40, confirmar_creuament=True):
    """
    Estratègia: Divergència Oculta + Ending Diagonal (contratendència)

    COMPRA quan:
      1. Divergència oculta ALCISTA detectada en una vall
      2. Canal convergent (wedge descendent)
      3. (Opcional) MACD histograma comença a girar cap amunt

    VENDA quan:
      1. Divergència oculta BAJISTA detectada en un pic
      2. Canal convergent (wedge ascendent)
      3. (Opcional) MACD histograma comença a girar cap avall
    """
    macd_l, sig_l, hist = calcular_macd(df["Close"], rapid, lent, senyal)
    df = df.copy()
    df["macd"]   = macd_l
    df["senyal"] = sig_l
    df["hist"]   = hist

    pivots = trobar_pivots(df["Close"], finestra=finestra_pivot)

    posicio = 0; efectiu = capital
    equitat = []; operacions = []
    preu_entrada = 0.0; data_entrada = None
    tipus_entrada = None
    senyals_detectats = []   # per pintar al gràfic

    for i in range(finestra_canal + finestra_pivot + 5, len(df)):
        preu = float(df["Close"].iloc[i])
        data = df.index[i]

        div = detectar_divergencia_oculta(
            df["Close"], df["hist"], pivots, i, lookback=lookback_div
        )
        canal_ok, ratio_conv = detectar_canal_convergent(
            df["Close"], i, finestra=finestra_canal
        )

        hist_gira_amunt = (float(df["hist"].iloc[i]) > float(df["hist"].iloc[i-1])
                           and float(df["hist"].iloc[i-1]) < 0)
        hist_gira_avall = (float(df["hist"].iloc[i]) < float(df["hist"].iloc[i-1])
                           and float(df["hist"].iloc[i-1]) > 0)

        confirma_compra = hist_gira_amunt if confirmar_creuament else True
        confirma_venda  = hist_gira_avall if confirmar_creuament else True

        # ── SENYAL DE COMPRA ──────────────────────────────────────────────────
        if div == "alcista" and canal_ok and confirma_compra and posicio == 0:
            pressupost = min(import_op, efectiu) if import_op > 0 else efectiu
            accions    = int(pressupost // preu)
            if accions > 0:
                posicio = accions; efectiu -= accions * preu
                preu_entrada = preu; data_entrada = data
                tipus_entrada = "Div. Oculta Alcista + Diagonal"
                senyals_detectats.append({
                    "data": data, "preu": preu, "tipus": "compra",
                    "ratio_conv": round(ratio_conv, 2)
                })

        # ── SORTIDA per divergència bajista ──────────────────────────────────
        elif div == "bajista" and canal_ok and confirma_venda and posicio > 0:
            efectiu += posicio * preu
            ganancia = (preu - preu_entrada) * posicio
            pct      = (preu - preu_entrada) / preu_entrada * 100
            dies     = (data - data_entrada).days
            operacions.append({
                "Entrada":       data_entrada.strftime("%Y-%m-%d"),
                "Sortida":       data.strftime("%Y-%m-%d"),
                "Dies":          dies,
                "Preu entrada":  round(preu_entrada, 2),
                "Preu sortida":  round(preu, 2),
                "Accions":       posicio,
                "Import inv.($)":round(preu_entrada * posicio, 2),
                "G/P ($)":       round(ganancia, 2),
                "Retorn (%)":    round(pct, 2),
                "Senyal":        tipus_entrada,
            })
            senyals_detectats.append({
                "data": data, "preu": preu, "tipus": "venda",
                "ratio_conv": round(ratio_conv, 2)
            })
            posicio = 0; tipus_entrada = None

        equitat.append(efectiu + posicio * preu)

    df_eq = df.iloc[finestra_canal + finestra_pivot + 5:].copy()
    min_len = min(len(df_eq), len(equitat))
    df_eq = df_eq.iloc[:min_len]
    df_eq["equitat"] = equitat[:min_len]

    return df_eq, pd.DataFrame(operacions), pd.DataFrame(senyals_detectats)


# ══════════════════════════════════════════════════════════════════════════════
# ESTADÍSTIQUES (compartides)
# ══════════════════════════════════════════════════════════════════════════════
def calcular_stats(df_eq, ops_df, capital):
    equitat_final = df_eq["equitat"].iloc[-1]
    retorn_total  = (equitat_final - capital) / capital * 100
    retorn_bh     = (df_eq["Close"].iloc[-1] - df_eq["Close"].iloc[0]) / df_eq["Close"].iloc[0] * 100

    max_acum = df_eq["equitat"].cummax()
    drawdown = (df_eq["equitat"] - max_acum) / max_acum * 100
    max_dd   = drawdown.min()
    saldo_max = df_eq["equitat"].max(); saldo_min = df_eq["equitat"].min()
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
        "Equitat final": round(equitat_final, 2), "Retorn total": round(retorn_total, 2),
        "Buy & Hold": round(retorn_bh, 2), "Max Drawdown": round(max_dd, 2),
        "Sharpe": round(sharpe, 2), "Num ops": len(ops_df),
        "Taxa encert": round(taxa_enc, 2), "Ganancia mitjana": round(ganancia_m, 2),
        "Perdua mitjana": round(perdua_m, 2), "Millor op": round(millor, 2),
        "Pitjor op": round(pitjor, 2), "Saldo max": round(saldo_max, 2),
        "Data saldo max": data_saldo_max, "Saldo min": round(saldo_min, 2),
        "Data saldo min": data_saldo_min, "Ratxa guanys": ratxa_g,
        "Ratxa perdues": ratxa_p, "Dies mitjana": round(dies_m, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRÀFIC
# ══════════════════════════════════════════════════════════════════════════════
def construir_grafic(df_eq, ops_df, stats, senyals_df=None):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.03,
        subplot_titles=("Preu + Senyals", "MACD", "Corba d'equitat"),
    )

    # Espelmes
    fig.add_trace(go.Candlestick(
        x=df_eq.index, open=df_eq["Open"], high=df_eq["High"],
        low=df_eq["Low"], close=df_eq["Close"], name="Preu",
        increasing_line_color="#00d084", decreasing_line_color="#ff4d6d",
    ), row=1, col=1)

    # Marcadors d'operacions de la taula
    if not ops_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(ops_df["Entrada"]), y=ops_df["Preu entrada"],
            mode="markers", marker=dict(symbol="triangle-up", size=11, color="#63a0ff",
                                        line=dict(width=1, color="#3070d0")),
            name="Compra",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(ops_df["Sortida"]), y=ops_df["Preu sortida"],
            mode="markers", marker=dict(symbol="triangle-down", size=11, color="#ffaa00",
                                        line=dict(width=1, color="#cc8800")),
            name="Venda",
        ), row=1, col=1)

    # Senyals de divergència detectats (cercles)
    if senyals_df is not None and not senyals_df.empty:
        comp = senyals_df[senyals_df["tipus"] == "compra"]
        vend = senyals_df[senyals_df["tipus"] == "venda"]
        if not comp.empty:
            fig.add_trace(go.Scatter(
                x=comp["data"], y=comp["preu"],
                mode="markers",
                marker=dict(symbol="circle-open", size=16, color="#00d084",
                            line=dict(width=2)),
                name="Div. alcista", hovertemplate="Div. oculta alcista<br>Conv: %{customdata:.0%}",
                customdata=comp["ratio_conv"],
            ), row=1, col=1)
        if not vend.empty:
            fig.add_trace(go.Scatter(
                x=vend["data"], y=vend["preu"],
                mode="markers",
                marker=dict(symbol="circle-open", size=16, color="#ff4d6d",
                            line=dict(width=2)),
                name="Div. bajista", hovertemplate="Div. oculta bajista<br>Conv: %{customdata:.0%}",
                customdata=vend["ratio_conv"],
            ), row=1, col=1)

    # MACD
    bar_colors = ["#00d084" if v >= 0 else "#ff4d6d" for v in df_eq["hist"]]
    fig.add_trace(go.Bar(x=df_eq.index, y=df_eq["hist"],
                         marker_color=bar_colors, opacity=0.6, name="Histograma"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq["macd"],
                             line=dict(color="#63a0ff", width=1.5), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq["senyal"],
                             line=dict(color="#ffaa00", width=1.5), name="Senyal"), row=2, col=1)

    # Corba d'equitat
    fig.add_trace(go.Scatter(
        x=df_eq.index, y=df_eq["equitat"], fill="tozeroy",
        line=dict(color="#63a0ff", width=2), fillcolor="rgba(99,160,255,0.07)",
        name="Equitat",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(stats["Data saldo max"])], y=[stats["Saldo max"]],
        mode="markers+text", marker=dict(symbol="star", size=13, color="#00d084"),
        text=["MÀX"], textposition="top center",
        textfont=dict(color="#00d084", size=9, family="Outfit"), name="Saldo màxim",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(stats["Data saldo min"])], y=[stats["Saldo min"]],
        mode="markers+text", marker=dict(symbol="x", size=11, color="#ff4d6d"),
        text=["MÍN"], textposition="bottom center",
        textfont=dict(color="#ff4d6d", size=9, family="Outfit"), name="Saldo mínim",
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#080c14", plot_bgcolor="#080c14",
        font=dict(color="#3d4f6e", family="Inter"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(13,19,33,0.9)", bordercolor="#1a2235", borderwidth=1,
                    font=dict(family="Inter", size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=700, margin=dict(l=4, r=4, t=44, b=4),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#0d1321", row=i, col=1, showline=False)
        fig.update_yaxes(gridcolor="#0d1321", row=i, col=1, showline=False)
        fig.layout.annotations[i-1].font.family = "Outfit"
        fig.layout.annotations[i-1].font.size   = 11
        fig.layout.annotations[i-1].font.color  = "#3d4f6e"
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS UI
# ══════════════════════════════════════════════════════════════════════════════
def targeta(col, etiqueta, valor, cls, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    col.markdown(f"""
    <div class="metric-card {cls}">
      <div class="metric-label">{etiqueta}</div>
      <div class="metric-value">{valor}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)


def mostrar_kpis(s, capital, import_op):
    st.markdown('<div class="titol-seccio">📊 Rendiment general</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    targeta(c1, "Capital final",     f"${s['Equitat final']:,.0f}",
            "positiu" if s["Retorn total"] >= 0 else "negatiu")
    targeta(c2, "Retorn estratègia", f"{s['Retorn total']:+.1f}%",
            "positiu" if s["Retorn total"] >= 0 else "negatiu")
    targeta(c3, "Buy & Hold",        f"{s['Buy & Hold']:+.1f}%",
            "positiu" if s["Buy & Hold"] >= 0 else "negatiu")
    c4, c5 = st.columns(2)
    targeta(c4, "Màx. Drawdown",   f"{s['Max Drawdown']:.1f}%", "negatiu")
    targeta(c5, "Ràtio de Sharpe", f"{s['Sharpe']:.2f}",
            "positiu" if s["Sharpe"] >= 1 else "avis" if s["Sharpe"] >= 0 else "negatiu")

    st.markdown('<div class="titol-seccio">💰 Saldo del compte</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    targeta(c1, "Saldo màxim assolit",  f"${s['Saldo max']:,.0f}", "positiu",
            sub=f"📅 {s['Data saldo max']}")
    targeta(c2, "Saldo mínim registrat", f"${s['Saldo min']:,.0f}", "negatiu",
            sub=f"📅 {s['Data saldo min']}")
    c3, = st.columns(1)
    targeta(c3, "Capital inicial", f"${capital:,.0f}", "neutral",
            sub=f"Import/op: {'Tot l\'efectiu' if import_op == 0 else f'${import_op:,}'}")

    st.markdown('<div class="titol-seccio">🎯 Estadístiques d\'operacions</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    targeta(c1, "Nº operacions",   str(s["Num ops"]),           "neutral")
    targeta(c2, "Taxa d'encert",   f"{s['Taxa encert']:.1f}%",
            "positiu" if s["Taxa encert"] >= 50 else "negatiu")
    targeta(c3, "Dies mitjana/op", f"{s['Dies mitjana']} dies", "neutral")
    c4, c5, c6 = st.columns(3)
    targeta(c4, "Guany mitjà",     f"${s['Ganancia mitjana']:,.0f}", "positiu")
    targeta(c5, "Millor operació", f"${s['Millor op']:,.0f}",        "positiu")
    targeta(c6, "Pèrdua mitjana",  f"${s['Perdua mitjana']:,.0f}",   "negatiu")
    c7, c8, c9 = st.columns(3)
    targeta(c7, "Pitjor operació",      f"${s['Pitjor op']:,.0f}",   "negatiu")
    targeta(c8, "Ratxa guanyadora màx", f"{s['Ratxa guanys']} ops",  "positiu")
    targeta(c9, "Ratxa perdedora màx",  f"{s['Ratxa perdues']} ops", "negatiu")


def mostrar_taula(ops_df):
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
        st.info("No s'han generat operacions. Prova a ampliar el rang de dates o ajustar els paràmetres.")


# ══════════════════════════════════════════════════════════════════════════════
# DESCÀRREGA AMB REINTENTS
# ══════════════════════════════════════════════════════════════════════════════
def descarregar_dades(ticker, data_inici, data_fi, intents=3, espera=5):
    for intent in range(intents):
        try:
            raw = yf.download(ticker, start=data_inici, end=data_fi,
                              auto_adjust=True, progress=False)
            return raw, None
        except Exception as e:
            missatge = str(e)
            if "Rate" in missatge or "Too Many" in missatge or "429" in missatge:
                if intent < intents - 1:
                    time.sleep(espera * (intent + 1))
                    continue
                return None, "rate_limit"
            return None, missatge
    return None, "rate_limit"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Paràmetres")

    ticker = st.text_input("Ticker", value="AAPL").upper().strip()

    estrategia = st.selectbox(
        "Estratègia",
        ["🔀 Creuament MACD", "🔭 Divergència Oculta + Ending Diagonal"],
        help="Tria l'estratègia a fer el backtest"
    )

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

    # Paràmetres extra per Estratègia 2
    if "Divergència" in estrategia:
        st.markdown("---")
        st.markdown("**Paràmetres Divergència / Diagonal**")
        finestra_pivot = st.slider("Finestra pivots",    3, 15, 5,
            help="Períodes per detectar pics i valls locals")
        finestra_canal = st.slider("Finestra canal",    10, 50, 20,
            help="Períodes per detectar el canal convergent")
        lookback_div   = st.slider("Lookback divergència", 10, 80, 40,
            help="Períodes enrere per buscar el pivot anterior")
        confirmar_macd = st.checkbox("Confirmar amb gir MACD", value=True,
            help="Exigeix que l'histograma MACD giri en la mateixa direcció")
    else:
        finestra_pivot = 5; finestra_canal = 20; lookback_div = 40; confirmar_macd = True

    executar_btn = st.button("▶ Executar backtest", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÀGINA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="cap-badge">📈 BACKTESTER</div>', unsafe_allow_html=True)

if "Divergència" in estrategia:
    st.markdown("# Divergència Oculta + Ending Diagonal")
    st.markdown('<div class="cap-subtitol">Estratègia contratendència · Divergència oculta MACD · Canal convergent · Cost 0</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <strong>Com funciona aquesta estratègia:</strong><br>
      Detecta el possible final d'una tendència combinant tres filtres simultanis:<br>
      &nbsp;&nbsp;① <strong>Divergència oculta alcista</strong> — preu fa un nou mínim però l'histograma MACD no → força oculta<br>
      &nbsp;&nbsp;② <strong>Canal convergent (wedge)</strong> — els màxims i mínims s'apropen → energia comprimida<br>
      &nbsp;&nbsp;③ <strong>Gir de l'histograma MACD</strong> (opcional) — confirma el canvi de momentum<br>
      La sortida es produeix quan apareix una <strong>divergència oculta bajista</strong> + canal convergent.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("# Estratègia MACD")
    st.markdown('<div class="cap-subtitol">Creuament MACD · Dades històriques reals · Cost 0</div>',
                unsafe_allow_html=True)

if executar_btn:
    with st.spinner(f"Descarregant dades de {ticker}..."):
        raw, error = descarregar_dades(ticker, data_inici, data_fi)

    if error == "rate_limit":
        st.warning("⏱️ **Yahoo Finance ha limitat les peticions.** Espera 1-2 minuts i torna a provar.")
        st.stop()
    elif error:
        st.error(f"Error en descarregar dades: {error}")
        st.stop()

    if raw is None or raw.empty:
        st.error(f"No s'han trobat dades per a **{ticker}**. Comprova el ticker.")
    else:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        senyals_df = None

        if "Divergència" in estrategia:
            with st.spinner("Detectant divergències i canals convergents..."):
                df_eq, ops_df, senyals_df = executar_divergencia_diagonal(
                    raw, capital, import_op, rapid_p, lent_p, senyal_p,
                    finestra_pivot=finestra_pivot, finestra_canal=finestra_canal,
                    lookback_div=lookback_div, confirmar_creuament=confirmar_macd
                )
            if not senyals_df.empty:
                n_comp = len(senyals_df[senyals_df["tipus"] == "compra"])
                n_vend = len(senyals_df[senyals_df["tipus"] == "venda"])
                st.info(f"🔍 S'han detectat **{n_comp}** senyals de compra i **{n_vend}** de venda. "
                        f"Les operacions executades són {len(ops_df)} (requereixen els tres filtres actius).")
        else:
            df_eq, ops_df = executar_creuament(
                raw, capital, import_op, rapid_p, lent_p, senyal_p
            )

        s = calcular_stats(df_eq, ops_df, capital)
        mostrar_kpis(s, capital, import_op)
        st.markdown("---")
        st.plotly_chart(construir_grafic(df_eq, ops_df, s, senyals_df),
                        use_container_width=True)
        mostrar_taula(ops_df)

else:
    st.markdown("""
    <div style="text-align:center; padding:3.5rem 1.5rem; color:#1e2d45;">
      <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
      <div style="font-family:'Outfit',sans-serif; font-size:1.05rem; font-weight:600;
                  color:#2a3d5e; line-height:1.7;">
        Tria l'estratègia al panell lateral<br>
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
