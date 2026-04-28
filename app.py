"""
Agente de Avaliação de Empresas Brasileiras com IA
Disciplina: Inteligência Artificial Aplicada ao Mercado Financeiro — Prof. Chela
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import sys
import os
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CHELA · Equity Research IA",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System ────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ───────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Fundo geral */
.stApp { background: #080c10; }
section[data-testid="stSidebar"] { background: #0c1018 !important; border-right: 1px solid #1a2332; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] * { color: #8899aa !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
    color: #c8a96e !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 1.4rem !important;
    margin-bottom: 0.4rem !important;
}
section[data-testid="stSidebar"] hr { border-color: #1a2332 !important; margin: 0.8rem 0 !important; }

/* Multiselect chips */
span[data-baseweb="tag"] {
    background: #1a2d1a !important;
    border: 1px solid #2d5a2d !important;
}
span[data-baseweb="tag"] span { color: #6fcf6f !important; }

/* Radio buttons */
div[data-testid="stRadio"] label { gap: 0.5rem; }
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem !important;
}

/* Date inputs */
div[data-baseweb="input"] input {
    background: #0c1018 !important;
    border: 1px solid #1a2332 !important;
    color: #c8a96e !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Botão principal ─────────────────────────────────────────────────────── */
div[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: #c8a96e !important;
    color: #080c10 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease;
    box-shadow: 0 0 20px rgba(200,169,110,0.15);
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #e2c47e !important;
    box-shadow: 0 0 30px rgba(200,169,110,0.35);
    transform: translateY(-1px);
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
div[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #445566 !important;
    padding: 0.6rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #c8a96e !important;
    border-bottom-color: #c8a96e !important;
    background: transparent !important;
}
div[data-testid="stTabsPanel"] { padding-top: 1.5rem !important; }

/* ── Progress bar ────────────────────────────────────────────────────────── */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #c8a96e, #e2c47e) !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: #0c1018 !important;
    border: 1px solid #1a2332 !important;
    border-radius: 4px !important;
}
div[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #667788 !important;
}

/* ── Download button ─────────────────────────────────────────────────────── */
div[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid #1a2332 !important;
    color: #667788 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 3px !important;
    transition: all 0.2s;
}
div[data-testid="stDownloadButton"] > button:hover {
    border-color: #c8a96e !important;
    color: #c8a96e !important;
}

/* ── Selectbox ───────────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div {
    background: #0c1018 !important;
    border: 1px solid #1a2332 !important;
    color: #aabbcc !important;
}

/* ── Dataframe / table ───────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] { border-radius: 4px; overflow: hidden; }

/* ── Scrollbar global ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0c1018; }
::-webkit-scrollbar-thumb { background: #1a2d3a; border-radius: 2px; }

/* ── Componentes customizados ────────────────────────────────────────────── */
.sys-header {
    border-bottom: 1px solid #1a2332;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: baseline;
    gap: 1rem;
}
.sys-header .logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #c8a96e;
    letter-spacing: -0.02em;
}
.sys-header .sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #334455;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.sys-header .live-badge {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3d8b5e;
    background: #0d2018;
    border: 1px solid #1a4030;
    padding: 0.25rem 0.6rem;
    border-radius: 2px;
    letter-spacing: 0.1em;
}

.ticker-card {
    background: #0c1018;
    border: 1px solid #1a2332;
    border-top: 2px solid var(--accent);
    padding: 1.2rem 1.4rem;
    position: relative;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.ticker-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(200,169,110,0.07);
}
.ticker-card .tc-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #334455;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.ticker-card .tc-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #ccd8e4;
    margin-bottom: 0.2rem;
}
.ticker-card .tc-ticker {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #445566;
    margin-bottom: 0.8rem;
}
.ticker-card .tc-score {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.ticker-card .tc-rec {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: var(--accent);
    margin-top: 0.3rem;
}
.ticker-card .tc-bar-wrap {
    margin-top: 0.8rem;
    height: 3px;
    background: #1a2332;
    border-radius: 1px;
}
.ticker-card .tc-bar {
    height: 100%;
    background: var(--accent);
    border-radius: 1px;
    transition: width 0.6s ease;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: #1a2332;
    border: 1px solid #1a2332;
    margin: 1rem 0;
}
.kpi-cell {
    background: #0c1018;
    padding: 1rem 1.2rem;
}
.kpi-cell .kc-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #445566;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.kpi-cell .kc-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #ccd8e4;
}
.kpi-cell .kc-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #334455;
    margin-top: 0.2rem;
}

.rec-banner {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.2rem 1.6rem;
    border: 1px solid var(--rec-border);
    background: var(--rec-bg);
    margin: 1rem 0 1.5rem;
}
.rec-banner .rb-signal {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--rec-color);
    letter-spacing: -0.02em;
}
.rec-banner .rb-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #445566;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.rec-banner .rb-desc {
    font-size: 0.82rem;
    color: #667788;
    margin-top: 0.2rem;
}
.rec-banner .rb-score {
    margin-left: auto;
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--rec-color);
    opacity: 0.25;
}

.score-bar-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
}
.score-bar-row .sbr-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #556677;
    width: 100px;
    flex-shrink: 0;
}
.score-bar-row .sbr-track {
    flex: 1;
    height: 4px;
    background: #1a2332;
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-row .sbr-fill {
    height: 100%;
    background: #c8a96e;
    border-radius: 2px;
}
.score-bar-row .sbr-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #c8a96e;
    width: 32px;
    text-align: right;
    flex-shrink: 0;
}

.dossie-terminal {
    background: #060a0d;
    border: 1px solid #1a2332;
    padding: 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #5d8a6a;
    white-space: pre-wrap;
    max-height: 480px;
    overflow-y: auto;
    line-height: 1.7;
}
.dossie-terminal .dt-header { color: #c8a96e; }
.dossie-terminal .dt-sep    { color: #1a3322; }
.dossie-terminal .dt-label  { color: #3d6655; }
.dossie-terminal .dt-value  { color: #8dbfa0; }

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #334455;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a2332;
}

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: #c8a96e !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem;
}
.sidebar-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem !important;
    color: #2a3a4a !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.pipeline-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #2a4a3a !important;
    background: #0d1810;
    border: 1px solid #1a3328;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    margin: 0.1rem;
}
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
}
.empty-state .es-icon {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #1a2332;
    letter-spacing: -0.05em;
    margin-bottom: 1.5rem;
}
.empty-state .es-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #2a3a4a;
    margin-bottom: 0.8rem;
}
.empty-state .es-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: #1e2e3e;
    line-height: 1.7;
    max-width: 420px;
    margin: 0 auto;
}
.rank-row {
    display: flex;
    align-items: center;
    padding: 0.85rem 1.2rem;
    border-bottom: 1px solid #0f1820;
    gap: 1.2rem;
    transition: background 0.15s;
}
.rank-row:hover { background: #0c1520; }
.rank-row:last-child { border-bottom: none; }
.rr-pos { font-family:'DM Mono',monospace; font-size:0.7rem; color:#2a3a4a; width:20px; }
.rr-name { font-family:'Syne',sans-serif; font-size:0.88rem; font-weight:600; color:#aabbcc; flex:1; }
.rr-tick { font-family:'DM Mono',monospace; font-size:0.65rem; color:#334455; width:80px; }
.rr-bar-wrap { flex:2; height:3px; background:#1a2332; border-radius:1px; }
.rr-bar { height:100%; border-radius:1px; }
.rr-score { font-family:'DM Mono',monospace; font-size:0.78rem; color:#c8a96e; width:36px; text-align:right; }
.rr-rec { font-family:'DM Mono',monospace; font-size:0.65rem; font-weight:500; width:36px; text-align:right; letter-spacing:0.06em; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%d %b %Y  %H:%M")
st.markdown(f"""
<div class="sys-header">
    <div class="logo">CHELA<span style="color:#1a3344;font-weight:400">·AI</span></div>
    <div class="sub">Equity Research · Commodities & Siderurgia BR</div>
    <div class="live-badge">◉ LIVE DATA · {now_str}</div>
</div>
""", unsafe_allow_html=True)

# ─── Importações pesadas (com cache e feedback) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def carregar_dependencias():
    """Importa bibliotecas pesadas uma única vez."""
    try:
        import yfinance as yf
        from unidecode import unidecode
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.cluster import KMeans
        return True, None
    except ImportError as e:
        return False, str(e)

ok, err = carregar_dependencias()
if not ok:
    st.error(f"❌ Dependência ausente: {err}\n\nRode: `pip install -r requirements.txt`")
    st.stop()

import yfinance as yf
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans

# ─── Constantes do projeto ────────────────────────────────────────────────────
UNIVERSO = {
    'VALE3.SA':  {'nome': 'Vale',               'setor': 'Mineração',        'macro': 'Commodities'},
    'SUZB3.SA':  {'nome': 'Suzano',             'setor': 'Papel e Celulose', 'macro': 'Commodities'},
    'KLBN11.SA': {'nome': 'Klabin',             'setor': 'Papel e Celulose', 'macro': 'Commodities'},
    'GGBR4.SA':  {'nome': 'Gerdau',             'setor': 'Siderurgia',       'macro': 'Siderurgia'},
    'GOAU4.SA':  {'nome': 'Metalúrgica Gerdau', 'setor': 'Siderurgia',       'macro': 'Siderurgia'},
    'CSNA3.SA':  {'nome': 'CSN',                'setor': 'Siderurgia',       'macro': 'Siderurgia'},
    'USIM5.SA':  {'nome': 'Usiminas',           'setor': 'Siderurgia',       'macro': 'Siderurgia'},
}
BENCHMARK   = '^BVSP'
DIAS_UTEIS  = 252
RANDOM_STATE = 42

STOPWORDS_PT = {
    'a','o','as','os','um','uma','de','do','da','dos','das','em','no','na',
    'nos','nas','por','pelo','pela','para','com','sem','sobre','e','ou','mas',
    'que','se','foi','foram','ja','tambem','ainda','apenas','mais','menos',
    'cia','sa','ltda','sao','eh',
}

VOCAB_RISK = {
    'risco','inadimplencia','passivo','litigio','multa','autuacao','incerteza',
    'queda','prejuizo','negativo','reducao','perda','divida','calote','default',
    'impacto','adverso','penalidade','rescisao','regulatorio',
}
VOCAB_QUALITY = {
    'crescimento','lucro','expansao','investimento','inovacao','eficiencia',
    'resultado','positivo','aumento','superavit','aprovado','ganho','sucesso',
    'oportunidade','estrategia','melhora','sustentavel','dividendo','record',
}

# ─── Cache de info do Yahoo (evita rate limit) ───────────────────────────────
import time

@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_info(ticker: str) -> dict:
    """Busca yf.Ticker.info com retry exponencial — evita rate limit."""
    for tentativa in range(4):
        try:
            info = yf.Ticker(ticker).info
            if info and len(info) > 5:
                return info
        except Exception:
            pass
        time.sleep(2 ** tentativa)   # 1s → 2s → 4s → 8s
    return {}

# ─── Funções auxiliares ───────────────────────────────────────────────────────
def safe_div(a, b):
    if b == 0 or pd.isna(b) or pd.isna(a):
        return np.nan
    return a / b

def zscore(x):
    s = pd.Series(x)
    return float((s.iloc[-1] - s.mean()) / s.std()) if s.std() > 0 else 0.0

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ''
    t = unidecode(texto.lower())
    t = __import__('re').sub(r'[^a-z0-9\s]', ' ', t)
    return __import__('re').sub(r'\s+', ' ', t).strip()

def tokenizar(texto):
    return [t for t in normalizar_texto(texto).split()
            if t not in STOPWORDS_PT and len(t) >= 3]

# ─── Pipeline de Análise ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fase1_mercado(ticker: str, data_inicio: str, data_fim: str) -> dict:
    """Fase 1: Ingestão de preços e features de mercado."""
    try:
        tickers_dl = [ticker, BENCHMARK]
        raw = yf.download(tickers=tickers_dl, start=data_inicio, end=data_fim,
                          auto_adjust=False, progress=False, group_by='ticker')
        if raw.empty:
            return {'erro': 'Yahoo Finance retornou vazio'}

        def extrair(t):
            try:
                col = 'Adj Close' if 'Adj Close' in raw[t].columns else 'Close'
                return raw[t][col].dropna()
            except Exception:
                return pd.Series(dtype=float)

        preco   = extrair(ticker)
        bench   = extrair(BENCHMARK)

        if preco.empty:
            return {'erro': f'Sem dados de preço para {ticker}'}

        rets    = preco.pct_change().dropna()
        vol_20  = float(rets.rolling(20).std().iloc[-1] * np.sqrt(DIAS_UTEIS))
        dd      = float((preco / preco.cummax() - 1).min())
        mom_6   = float(preco.pct_change(6 * 21).iloc[-1])
        mom_12  = float(preco.pct_change(12 * 21).iloc[-1])
        ret_tot = float(preco.iloc[-1] / preco.iloc[0] - 1)

        # Beta (60 dias)
        beta = np.nan
        if not bench.empty:
            r_bench = bench.pct_change().dropna()
            comb = pd.concat([rets, r_bench], axis=1).dropna()
            if len(comb) > 30:
                cov   = np.cov(comb.iloc[:, 0], comb.iloc[:, 1])
                beta  = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan

        # Score de mercado (0-25)
        score_comp = []
        if not pd.isna(mom_6):
            score_comp.append(min(25, max(0, 12.5 + mom_6 * 50)))
        if not pd.isna(vol_20):
            score_comp.append(max(0, 25 - vol_20 * 20))
        score_mercado = float(np.mean(score_comp)) if score_comp else 12.5

        return {
            'preco_atual': float(preco.iloc[-1]),
            'retorno_total': ret_tot,
            'vol_20d': vol_20,
            'max_drawdown': dd,
            'mom_6m': mom_6,
            'mom_12m': mom_12,
            'beta': beta,
            'score_mercado': score_mercado,
            'serie_precos': preco.to_dict(),
        }
    except Exception as e:
        return {'erro': str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def fase3_kpis_sinteticos(ticker: str) -> dict:
    """Fase 3: KPIs sintéticos via Yahoo Finance Info (substitui CVM para UI)."""
    try:
        info = get_ticker_info(ticker)
        if not info:
            return {'erro': 'Sem dados fundamentalistas'}

        pl   = info.get('trailingPE')
        pvp  = info.get('priceToBook')
        roe  = info.get('returnOnEquity')
        ml   = info.get('profitMargins')
        div  = info.get('dividendYield')
        mc   = info.get('marketCap')
        rec  = info.get('recommendationMean')  # 1=Strong Buy … 5=Strong Sell

        # Score KPI (0-25): quanto mais barato e lucrativo, melhor
        parcelas = []
        if pl and 0 < pl < 50:
            parcelas.append(max(0, 25 - pl * 0.5))
        if roe and roe > 0:
            parcelas.append(min(25, roe * 150))
        if ml and ml > 0:
            parcelas.append(min(25, ml * 100))
        score_kpi = float(np.mean(parcelas)) if parcelas else 12.5

        return {
            'pl': pl, 'pvp': pvp, 'roe': roe,
            'margem_liquida': ml, 'dividend_yield': div,
            'market_cap': mc, 'rec_analistas': rec,
            'score_kpi': score_kpi,
        }
    except Exception as e:
        return {'erro': str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def fase4_nlp_sintetico(ticker: str) -> dict:
    """Fase 4: NLP simplificado — simula score textual via dados do Yahoo."""
    try:
        info = get_ticker_info(ticker)
        # Usa longBusinessSummary como proxy do texto corporativo
        texto = info.get('longBusinessSummary', '')
        tokens = tokenizar(texto)
        n_risk    = sum(1 for tok in tokens if tok in VOCAB_RISK)
        n_quality = sum(1 for tok in tokens if tok in VOCAB_QUALITY)
        total = n_risk + n_quality if (n_risk + n_quality) > 0 else 1
        irq   = float((n_quality - n_risk) / total)
        score_nlp = float(12.5 + irq * 12.5)  # 0-25
        return {
            'n_risk': n_risk, 'n_quality': n_quality,
            'irq': irq, 'score_nlp': score_nlp,
            'tokens_analisados': len(tokens),
        }
    except Exception as e:
        return {'erro': str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def fase5_ml_score(ticker: str, score_mercado: float, score_kpi: float) -> dict:
    """Fase 5: Score ML sintético combinando features disponíveis."""
    try:
        info = get_ticker_info(ticker)
        beta     = info.get('beta', 1.0) or 1.0
        eps_fwd  = info.get('forwardEps', 0.0) or 0.0
        eps_trail = info.get('trailingEps', 0.0) or 0.0
        eps_growth = safe_div(eps_fwd - eps_trail, abs(eps_trail)) if eps_trail else 0.0

        # Feature vector simples
        features = np.array([[
            score_mercado / 25,
            score_kpi / 25,
            min(1.0, max(-1.0, beta - 1)),
            min(1.0, max(-1.0, eps_growth if not pd.isna(eps_growth) else 0.0)),
        ]])

        # Score heurístico (0-100) que simula a saída do GBM treinado
        base = (score_mercado + score_kpi) * 2
        ajuste_beta = max(-10, min(10, (1 - beta) * 5))
        ajuste_eps  = max(-10, min(10, (eps_growth or 0) * 10))
        score_ml = float(np.clip(base + ajuste_beta + ajuste_eps, 0, 100))

        return {
            'score_ml': score_ml,
            'beta_usado': beta,
            'eps_growth': eps_growth,
        }
    except Exception as e:
        return {'score_ml': (score_mercado + score_kpi) * 2, 'erro': str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def fase6_valuation(ticker: str) -> dict:
    """Fase 6: Valuation por múltiplos + análise de cenários."""
    try:
        info = get_ticker_info(ticker)
        pl   = info.get('trailingPE')
        pvp  = info.get('priceToBook')
        ev_r = info.get('enterpriseToRevenue')
        ev_e = info.get('enterpriseToEbitda')

        # Score valuation (0-25): múltiplos baixos = score alto
        parcelas = []
        if pl and 0 < pl < 30:
            parcelas.append(max(0, 25 - pl * 0.8))
        elif pl and pl >= 30:
            parcelas.append(0)
        if pvp and 0 < pvp < 5:
            parcelas.append(max(0, 25 - pvp * 5))
        score_val = float(np.mean(parcelas)) if parcelas else 12.5

        # Cenários macro (simulação conforme metodologia)
        cenarios = {
            'Otimista':   round(score_val * 1.15, 1),
            'Base':       round(score_val, 1),
            'Pessimista': round(score_val * 0.85, 1),
        }
        return {
            'pl': pl, 'pvp': pvp,
            'ev_receita': ev_r, 'ev_ebitda': ev_e,
            'score_valuation': score_val,
            'cenarios': cenarios,
        }
    except Exception as e:
        return {'score_valuation': 12.5, 'erro': str(e)}


def score_final_e_recomendacao(scores: dict, perfil: str) -> dict:
    """Fase 7: Consolida scores e gera recomendação."""
    s_merc = scores.get('score_mercado', 12.5)
    s_kpi  = scores.get('score_kpi', 12.5)
    s_nlp  = scores.get('score_nlp', 12.5)
    s_val  = scores.get('score_valuation', 12.5)
    s_ml   = scores.get('score_ml', 50.0)

    # Pesos iguais entre dimensões (conforme metodologia)
    score_comp = (s_merc + s_kpi + s_nlp + s_val) / 4  # 0-25

    if perfil == 'conservador':
        score_100 = score_comp * 4 * 0.6 + s_ml * 0.4
    else:  # agressivo
        score_100 = score_comp * 4 * 0.4 + s_ml * 0.6

    score_100 = float(np.clip(score_100, 0, 100))

    if perfil == 'conservador':
        thr_buy, thr_sell = 65, 40
    else:
        thr_buy, thr_sell = 55, 35

    if score_100 >= thr_buy:
        rec, cor = 'BUY', '#27ae60'
    elif score_100 <= thr_sell:
        rec, cor = 'SELL', '#e74c3c'
    else:
        rec, cor = 'HOLD', '#f39c12'

    return {
        'score_final': score_100,
        'recomendacao': rec,
        'cor': cor,
        'componentes': {
            'Mercado (F1)': round(s_merc, 1),
            'KPIs (F3)':    round(s_kpi, 1),
            'NLP (F4)':     round(s_nlp, 1),
            'Valuation (F6)': round(s_val, 1),
        },
    }


def gerar_dossie(ticker: str, dados: dict, rec: dict) -> str:
    """Gera dossiê formatado — equivalente ao build_dossie() do notebook."""
    nome   = UNIVERSO.get(ticker, {}).get('nome', ticker)
    setor  = UNIVERSO.get(ticker, {}).get('setor', '—')
    macro  = UNIVERSO.get(ticker, {}).get('macro', '—')
    ts     = datetime.now().strftime('%Y-%m-%d %H:%M')

    m  = dados.get('mercado', {})
    k  = dados.get('kpis', {})
    v  = dados.get('valuation', {})
    nl = dados.get('nlp', {})
    ml = dados.get('ml', {})

    def fmt(x, pct=False, prefix=''):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 'N/D'
        if pct:
            return f'{prefix}{x*100:.1f}%'
        return f'{prefix}{x:.2f}'

    linhas = [
        '═' * 60,
        f'  DOSSIÊ DE ANÁLISE — {nome} ({ticker})',
        f'  Gerado em: {ts}',
        '═' * 60,
        f'  Setor: {setor} | Macro-setor: {macro}',
        '',
        f'  ► RECOMENDAÇÃO: {rec["recomendacao"]}',
        f'  ► SCORE FINAL:  {rec["score_final"]:.1f} / 100',
        '',
        '─' * 60,
        '  FASE 1 — MERCADO',
        f'  Preço atual:      R$ {fmt(m.get("preco_atual"))}',
        f'  Retorno total:    {fmt(m.get("retorno_total"), pct=True)}',
        f'  Vol. anualizada:  {fmt(m.get("vol_20d"), pct=True)}',
        f'  Max Drawdown:     {fmt(m.get("max_drawdown"), pct=True)}',
        f'  Momentum 6m:      {fmt(m.get("mom_6m"), pct=True)}',
        f'  Beta:             {fmt(m.get("beta"))}',
        '',
        '─' * 60,
        '  FASE 3 — KPIs FUNDAMENTALISTAS',
        f'  P/L:              {fmt(k.get("pl"))}x',
        f'  P/VP:             {fmt(k.get("pvp"))}x',
        f'  ROE:              {fmt(k.get("roe"), pct=True)}',
        f'  Margem Líquida:   {fmt(k.get("margem_liquida"), pct=True)}',
        f'  Dividend Yield:   {fmt(k.get("dividend_yield"), pct=True)}',
        '',
        '─' * 60,
        '  FASE 4 — NLP / SENTIMENTO TEXTUAL',
        f'  Tokens risco:     {nl.get("n_risk", "N/D")}',
        f'  Tokens qualidade: {nl.get("n_quality", "N/D")}',
        f'  IRQ (índice):     {fmt(nl.get("irq"))}',
        '',
        '─' * 60,
        '  FASE 6 — VALUATION & CENÁRIOS',
        f'  EV/EBITDA:        {fmt(v.get("ev_ebitda"))}x',
        f'  Cenário Otimista: {v.get("cenarios", {}).get("Otimista", "N/D")} pts',
        f'  Cenário Base:     {v.get("cenarios", {}).get("Base", "N/D")} pts',
        f'  Cenário Pessimista:{v.get("cenarios", {}).get("Pessimista", "N/D")} pts',
        '',
        '─' * 60,
        '  SCORES POR COMPONENTE (0-25):',
    ]
    for comp, val in rec.get('componentes', {}).items():
        barra = '█' * int(val) + '░' * (25 - int(val))
        linhas.append(f'  {comp:<18} {val:>4.1f}  |{barra}|')
    linhas += ['', '═' * 60, '']
    return '\n'.join(linhas)


def executar_agente(tickers: list, perfil: str, data_inicio: str, data_fim: str,
                    progress_bar=None) -> dict:
    """Orquestrador principal — equivalente ao agente_mock() do notebook."""
    resultados = {}
    total = len(tickers)

    # Pré-aquece o cache de info do Yahoo com delay entre tickers
    # para evitar rate limit antes de iniciar o pipeline
    for i, ticker in enumerate(tickers):
        if progress_bar:
            progress_bar.progress(i / (total * 2), text=f"🔄 Buscando dados: {ticker}…")
        get_ticker_info(ticker)          # cacheia aqui
        if i < total - 1:
            time.sleep(1.5)              # 1.5s entre tickers = safe para Yahoo

    for i, ticker in enumerate(tickers):
        if progress_bar:
            progress_bar.progress(0.5 + i / (total * 2), text=f"🤖 Calculando scores: {ticker}…")

        dados = {}

        # Fases — info já está em cache, sem novas requisições
        dados['mercado']   = fase1_mercado(ticker, data_inicio, data_fim)
        dados['kpis']      = fase3_kpis_sinteticos(ticker)
        dados['nlp']       = fase4_nlp_sintetico(ticker)
        dados['valuation'] = fase6_valuation(ticker)

        s_merc = dados['mercado'].get('score_mercado', 12.5)
        s_kpi  = dados['kpis'].get('score_kpi', 12.5)
        dados['ml'] = fase5_ml_score(ticker, s_merc, s_kpi)

        scores = {
            'score_mercado':   s_merc,
            'score_kpi':       s_kpi,
            'score_nlp':       dados['nlp'].get('score_nlp', 12.5),
            'score_valuation': dados['valuation'].get('score_valuation', 12.5),
            'score_ml':        dados['ml'].get('score_ml', 50.0),
        }

        rec    = score_final_e_recomendacao(scores, perfil)
        dossie = gerar_dossie(ticker, dados, rec)

        resultados[ticker] = {'dados': dados, 'recomendacao': rec, 'dossie': dossie}

    if progress_bar:
        progress_bar.progress(1.0, text="✅ Análise concluída!")

    return resultados


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Inputs do usuário
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">CHELA·AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Prof. João Luiz Chela · FGV EAESP</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### Universo")
    opcoes = {f"{v['nome']} ({k})": k for k, v in UNIVERSO.items()}
    tickers_sel = st.multiselect(
        "Empresas:",
        options=list(opcoes.keys()),
        default=list(opcoes.keys())[:3],
        label_visibility="collapsed",
    )

    st.markdown("### Perfil")
    perfil = st.radio(
        "Perfil:",
        options=['conservador', 'agressivo'],
        format_func=lambda x: '▣  Conservador' if x == 'conservador' else '▲  Agressivo',
        label_visibility="collapsed",
    )

    st.markdown("### Horizonte")
    data_inicio = st.date_input(
        "Início", value=datetime(2022, 1, 1),
        min_value=datetime(2020, 1, 1),
        max_value=datetime.now() - timedelta(days=90),
        label_visibility="visible",
    ).strftime('%Y-%m-%d')

    data_fim = st.date_input(
        "Fim", value=datetime.now() - timedelta(days=1),
        min_value=datetime(2020, 6, 1),
        max_value=datetime.now(),
        label_visibility="visible",
    ).strftime('%Y-%m-%d')

    st.markdown("<hr>", unsafe_allow_html=True)
    executar = st.button("EXECUTAR AGENTE", use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:0.5rem;'>
    <div class='pipeline-pill'>F1 MERCADO</div>
    <div class='pipeline-pill'>F2 CVM*</div>
    <div class='pipeline-pill'>F3 KPIs</div>
    <div class='pipeline-pill'>F4 NLP</div>
    <div class='pipeline-pill'>F5 ML</div>
    <div class='pipeline-pill'>F6 VALUATION</div>
    <div class='pipeline-pill'>F7 AGENTE</div>
    <div style='font-family:DM Mono,monospace;font-size:0.58rem;color:#1a2a1a;margin-top:0.8rem;line-height:1.6;'>
    *F2 usa Yahoo Finance como proxy para CVM nesta interface.
    </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ÁREA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
if not tickers_sel:
    st.markdown("""
    <div class="empty-state">
        <div class="es-icon">◈</div>
        <div class="es-title">Sistema em espera</div>
        <div class="es-desc">
            Selecione as empresas, o perfil do investidor e o horizonte temporal
            na barra lateral. Depois clique em <strong>EXECUTAR AGENTE</strong>
            para iniciar o pipeline de 7 fases.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

tickers_para_analisar = [opcoes[t] for t in tickers_sel]

# Estado de sessão
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'ultima_config' not in st.session_state:
    st.session_state.ultima_config = None

config_atual = (tuple(tickers_para_analisar), perfil, data_inicio, data_fim)

if executar or (st.session_state.resultados is not None and
                st.session_state.ultima_config == config_atual):

    if executar or st.session_state.resultados is None:
        progress = st.progress(0, text="Iniciando análise…")
        st.session_state.resultados = executar_agente(
            tickers_para_analisar, perfil, data_inicio, data_fim, progress
        )
        st.session_state.ultima_config = config_atual

    resultados = st.session_state.resultados

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_ranking, tab_dossie, tab_graficos, tab_dados = st.tabs(
        ["RANKING", "DOSSIÊ", "GRÁFICOS", "DADOS BRUTOS"]
    )

    # ── TAB 1: Ranking ───────────────────────────────────────────────────────
    with tab_ranking:
        st.markdown('<div class="section-label">Ranking consolidado — Score 0→100</div>', unsafe_allow_html=True)

        # Montar lista ordenada
        linhas = []
        for tk, res in resultados.items():
            rec  = res['recomendacao']
            linhas.append({
                'tk': tk,
                'nome': UNIVERSO[tk]['nome'],
                'setor': UNIVERSO[tk]['setor'],
                'score': rec['score_final'],
                'rec': rec['recomendacao'],
                'cor': rec['cor'],
                'comp': rec['componentes'],
            })
        linhas.sort(key=lambda x: x['score'], reverse=True)

        # Cards top-3
        n_cards = min(3, len(linhas))
        if n_cards > 0:
            cols_top = st.columns(n_cards)
            rank_labels = ['01', '02', '03']
            rec_colors = {
                'BUY':  ('--accent:#3d8b5e', '#0d2018', '#1a4030', '#3d8b5e'),
                'HOLD': ('--accent:#c8a96e', '#1a1500', '#3a2e00', '#c8a96e'),
                'SELL': ('--accent:#8b3d3d', '#1a0808', '#401818', '#8b3d3d'),
            }
            for i, col in enumerate(cols_top):
                if i >= len(linhas):
                    break
                row = linhas[i]
                _, bg, border, color = rec_colors.get(row['rec'], rec_colors['HOLD'])
                col.markdown(f"""
                <div class="ticker-card" style="--accent:{color}; border-top-color:{color};">
                    <div class="tc-rank">#{rank_labels[i]} · {row['setor']}</div>
                    <div class="tc-name">{row['nome']}</div>
                    <div class="tc-ticker">{row['tk']}</div>
                    <div class="tc-score">{row['score']:.1f}</div>
                    <div class="tc-rec">{row['rec']}</div>
                    <div class="tc-bar-wrap">
                        <div class="tc-bar" style="width:{row['score']}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # Lista completa
        rank_html = '<div style="background:#0c1018;border:1px solid #1a2332;">'
        for i, row in enumerate(linhas):
            rec_color = {'BUY':'#3d8b5e','HOLD':'#c8a96e','SELL':'#8b3d3d'}.get(row['rec'],'#556677')
            rank_html += f"""
            <div class="rank-row">
                <div class="rr-pos">{str(i+1).zfill(2)}</div>
                <div class="rr-name">{row['nome']}</div>
                <div class="rr-tick">{row['tk']}</div>
                <div class="rr-bar-wrap">
                    <div class="rr-bar" style="width:{row['score']}%;background:{rec_color};"></div>
                </div>
                <div class="rr-score">{row['score']:.1f}</div>
                <div class="rr-rec" style="color:{rec_color};">{row['rec']}</div>
            </div>"""
        rank_html += '</div>'
        st.markdown(rank_html, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Download CSV discreto
        df_export = pd.DataFrame([{
            'Pos': i+1, 'Empresa': r['nome'], 'Ticker': r['tk'],
            'Score': round(r['score'],1), 'Recomendação': r['rec'],
            'Mercado': r['comp'].get('Mercado (F1)',np.nan),
            'KPIs': r['comp'].get('KPIs (F3)',np.nan),
            'NLP': r['comp'].get('NLP (F4)',np.nan),
            'Valuation': r['comp'].get('Valuation (F6)',np.nan),
        } for i, r in enumerate(linhas)])
        st.download_button(
            "↓ exportar ranking (.csv)",
            data=df_export.to_csv(index=False).encode('utf-8'),
            file_name=f"ranking_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv',
        )

    # ── TAB 2: Dossiê ────────────────────────────────────────────────────────
    with tab_dossie:
        st.markdown('<div class="section-label">Análise detalhada por empresa</div>', unsafe_allow_html=True)

        ticker_escolhido = st.selectbox(
            "Empresa",
            options=list(resultados.keys()),
            format_func=lambda t: f"{UNIVERSO[t]['nome']}  ·  {t}",
            label_visibility="collapsed",
        )

        if ticker_escolhido:
            res = resultados[ticker_escolhido]
            rec = res['recomendacao']
            dados = res['dados']

            # Cores por recomendação
            rec_styles = {
                'BUY':  ('#3d8b5e', '#0d2018', '#1a4030'),
                'HOLD': ('#c8a96e', '#1a1500', '#3a2e00'),
                'SELL': ('#8b3d3d', '#1a0808', '#401818'),
            }
            rc, rbg, rborder = rec_styles.get(rec['recomendacao'], rec_styles['HOLD'])

            # Banner de recomendação
            perfil_label = 'CONSERVADOR' if perfil == 'conservador' else 'AGRESSIVO'
            st.markdown(f"""
            <div class="rec-banner" style="--rec-color:{rc};--rec-bg:{rbg};--rec-border:{rborder};">
                <div>
                    <div class="rb-label">Sinal do Agente · Perfil {perfil_label}</div>
                    <div class="rb-signal">{rec['recomendacao']}</div>
                    <div class="rb-desc">{UNIVERSO[ticker_escolhido]['nome']} · {UNIVERSO[ticker_escolhido]['setor']}</div>
                </div>
                <div class="rb-score">{rec['score_final']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            # KPI grid
            m = dados.get('mercado', {})
            k = dados.get('kpis', {})
            v = dados.get('valuation', {})

            def fv(x, pct=False, prefix='R$'):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return '—'
                if pct:
                    return f'{x*100:+.1f}%'
                return f'{x:.2f}'

            mc_bi = k.get('market_cap')
            mc_str = f"R$ {mc_bi/1e9:.1f}B" if mc_bi and not np.isnan(mc_bi) else '—'

            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-cell">
                    <div class="kc-label">Preço Atual</div>
                    <div class="kc-value">R$ {fv(m.get('preco_atual'))}</div>
                    <div class="kc-sub">última cotação</div>
                </div>
                <div class="kpi-cell">
                    <div class="kc-label">Retorno Total</div>
                    <div class="kc-value" style="color:{'#3d8b5e' if (m.get('retorno_total') or 0)>0 else '#8b3d3d'}">
                        {fv(m.get('retorno_total'), pct=True)}
                    </div>
                    <div class="kc-sub">no período analisado</div>
                </div>
                <div class="kpi-cell">
                    <div class="kc-label">Vol. Anualizada</div>
                    <div class="kc-value">{fv(m.get('vol_20d'), pct=True)}</div>
                    <div class="kc-sub">20 dias anualizado</div>
                </div>
                <div class="kpi-cell">
                    <div class="kc-label">P/L</div>
                    <div class="kc-value">{fv(k.get('pl'))}×</div>
                    <div class="kc-sub">trailing</div>
                </div>
                <div class="kpi-cell">
                    <div class="kc-label">ROE</div>
                    <div class="kc-value">{fv(k.get('roe'), pct=True)}</div>
                    <div class="kc-sub">retorno s/ PL</div>
                </div>
                <div class="kpi-cell">
                    <div class="kc-label">Market Cap</div>
                    <div class="kc-value">{mc_str}</div>
                    <div class="kc-sub">valor de mercado</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Score bars
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Decomposição do Score</div>', unsafe_allow_html=True)
            comp = rec['componentes']
            bars_html = ''
            for label, val in comp.items():
                pct = min(100, val / 25 * 100)
                bars_html += f"""
                <div class="score-bar-row">
                    <div class="sbr-label">{label}</div>
                    <div class="sbr-track"><div class="sbr-fill" style="width:{pct}%"></div></div>
                    <div class="sbr-val">{val:.1f}</div>
                </div>"""
            # Score ML
            ml_val = dados.get('ml', {}).get('score_ml', 50)
            ml_pct = min(100, ml_val)
            bars_html += f"""
            <div class="score-bar-row">
                <div class="sbr-label" style="color:#c8a96e;">ML (F5)</div>
                <div class="sbr-track"><div class="sbr-fill" style="width:{ml_pct}%;background:#c8a96e;"></div></div>
                <div class="sbr-val">{ml_val:.1f}</div>
            </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

            # Terminal dossiê
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Dossiê Completo</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="dossie-terminal">{res["dossie"]}</div>', unsafe_allow_html=True)

            todos_dossies = '\n'.join(r['dossie'] for r in resultados.values())
            st.download_button(
                "↓ exportar todos os dossiês (.txt)",
                data=todos_dossies.encode('utf-8'),
                file_name=f"dossies_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime='text/plain',
            )

    # ── TAB 3: Gráficos ──────────────────────────────────────────────────────
    with tab_graficos:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker

        PLT_BG   = '#080c10'
        PLT_AX   = '#0c1018'
        PLT_GRID = '#1a2332'
        PLT_TEXT = '#556677'
        GOLD     = '#c8a96e'

        rec_cores = {'BUY':'#3d8b5e','HOLD':'#c8a96e','SELL':'#8b3d3d'}
        palette   = [GOLD,'#4a9bbe','#6abf7a','#c87e5e','#9e6abf','#be4a7a','#6abebe']

        st.markdown('<div class="section-label">Score final por empresa</div>', unsafe_allow_html=True)

        # Gráfico 1: barras horizontais elegantes
        sorted_res = sorted(resultados.items(), key=lambda x: x[1]['recomendacao']['score_final'])
        nomes_s  = [UNIVERSO[t]['nome'] for t,_ in sorted_res]
        scores_s = [r['recomendacao']['score_final'] for _,r in sorted_res]
        recs_s   = [r['recomendacao']['recomendacao'] for _,r in sorted_res]
        cores_s  = [rec_cores.get(r,'#556677') for r in recs_s]

        fig, ax = plt.subplots(figsize=(10, max(3, len(nomes_s) * 0.75)))
        fig.patch.set_facecolor(PLT_BG)
        ax.set_facecolor(PLT_AX)

        y_pos = range(len(nomes_s))
        # Barra de fundo (track)
        ax.barh(y_pos, [100]*len(nomes_s), color=PLT_GRID, height=0.45, zorder=1)
        # Barra de valor
        bars = ax.barh(y_pos, scores_s, color=cores_s, height=0.45, zorder=2, alpha=0.9)

        for i, (bar, s, r) in enumerate(zip(bars, scores_s, recs_s)):
            ax.text(s + 1.5, i, f'{s:.1f}', va='center',
                    color=rec_cores.get(r,'#556677'), fontsize=9,
                    fontfamily='monospace', fontweight='bold')
            ax.text(-1, i, r, va='center', ha='right',
                    color=PLT_TEXT, fontsize=7.5, fontfamily='monospace')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(nomes_s, color='#889aaa', fontsize=9)
        ax.set_xlim(-12, 108)
        ax.set_xlabel('Score (0–100)', color=PLT_TEXT, fontsize=8, fontfamily='monospace')
        ax.tick_params(colors=PLT_TEXT, labelsize=8)
        ax.axvline(35, color=PLT_GRID, lw=1, zorder=3)
        ax.axvline(65, color=PLT_GRID, lw=1, zorder=3)
        ax.text(35, len(nomes_s)-0.1, 'SELL│HOLD', color=PLT_TEXT,
                fontsize=6.5, fontfamily='monospace', ha='center')
        ax.text(65, len(nomes_s)-0.1, 'HOLD│BUY', color=PLT_TEXT,
                fontsize=6.5, fontfamily='monospace', ha='center')
        for sp in ax.spines.values():
            sp.set_edgecolor(PLT_GRID)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        legend_patches = [mpatches.Patch(color=c, label=l) for l,c in rec_cores.items()]
        ax.legend(handles=legend_patches, loc='lower right', facecolor=PLT_AX,
                  labelcolor='#889aaa', edgecolor=PLT_GRID, fontsize=8)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Gráfico 2: score components heatmap style
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Decomposição por componente</div>', unsafe_allow_html=True)

        comp_labels = ['Mercado\n(F1)', 'KPIs\n(F3)', 'NLP\n(F4)', 'Valuation\n(F6)']
        comp_keys   = ['Mercado (F1)', 'KPIs (F3)', 'NLP (F4)', 'Valuation (F6)']

        fig2, ax2 = plt.subplots(figsize=(10, max(2.5, len(resultados)*0.7)))
        fig2.patch.set_facecolor(PLT_BG)
        ax2.set_facecolor(PLT_BG)

        tickers_list = list(resultados.keys())
        matrix = np.array([
            [resultados[t]['recomendacao']['componentes'].get(k, 12.5) for k in comp_keys]
            for t in tickers_list
        ])
        nomes_list = [UNIVERSO[t]['nome'] for t in tickers_list]

        im = ax2.imshow(matrix, aspect='auto', cmap='YlOrBr', vmin=0, vmax=25)
        ax2.set_xticks(range(len(comp_labels)))
        ax2.set_xticklabels(comp_labels, color='#889aaa', fontsize=8, fontfamily='monospace')
        ax2.set_yticks(range(len(nomes_list)))
        ax2.set_yticklabels(nomes_list, color='#889aaa', fontsize=9)
        ax2.tick_params(colors=PLT_TEXT)
        for spine in ax2.spines.values():
            spine.set_edgecolor(PLT_GRID)
        for i in range(len(tickers_list)):
            for j in range(len(comp_keys)):
                val = matrix[i,j]
                ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                         color='#080c10' if val > 15 else '#889aaa',
                         fontsize=9, fontfamily='monospace', fontweight='bold')
        cb = plt.colorbar(im, ax=ax2, pad=0.01)
        cb.ax.tick_params(colors=PLT_TEXT, labelsize=7)
        cb.outline.set_edgecolor(PLT_GRID)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # Gráfico 3: performance de preços
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Performance relativa (base 100)</div>', unsafe_allow_html=True)
        try:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            fig3.patch.set_facecolor(PLT_BG)
            ax3.set_facecolor(PLT_AX)
            has_data = False
            for (tk, res), cor in zip(resultados.items(), palette):
                serie = res['dados']['mercado'].get('serie_precos', {})
                if not serie:
                    continue
                s = pd.Series(serie).dropna()
                if s.empty or s.iloc[0] == 0:
                    continue
                s_norm = s / s.iloc[0] * 100
                ax3.plot(s_norm.index, s_norm.values, label=UNIVERSO[tk]['nome'],
                         color=cor, lw=1.5, alpha=0.9)
                # dot final
                ax3.scatter([s_norm.index[-1]], [s_norm.iloc[-1]],
                            color=cor, s=30, zorder=5)
                has_data = True
            if has_data:
                ax3.axhline(100, color=PLT_GRID, lw=0.8, linestyle='--', alpha=0.6)
                ax3.fill_between(ax3.get_lines()[0].get_xdata(),
                                 100, alpha=0.03, color=GOLD)
                ax3.set_ylabel('Índice (base 100)', color=PLT_TEXT, fontsize=8, fontfamily='monospace')
                ax3.tick_params(colors=PLT_TEXT, labelsize=8)
                ax3.legend(facecolor=PLT_AX, labelcolor='#889aaa',
                           edgecolor=PLT_GRID, fontsize=8, loc='upper left')
                ax3.grid(axis='y', color=PLT_GRID, alpha=0.4, lw=0.5)
                for sp in ax3.spines.values():
                    sp.set_edgecolor(PLT_GRID)
                plt.tight_layout(pad=1.2)
                st.pyplot(fig3, use_container_width=True)
            else:
                st.markdown('<div style="color:#334455;font-family:monospace;font-size:0.8rem;padding:2rem;">Dados de preços indisponíveis.</div>', unsafe_allow_html=True)
            plt.close()
        except Exception as e:
            st.markdown(f'<div style="color:#553333;font-family:monospace;font-size:0.75rem;">erro: {e}</div>', unsafe_allow_html=True)

    # ── TAB 4: Dados Brutos ──────────────────────────────────────────────────
    with tab_dados:
        st.markdown('<div class="section-label">Dados brutos de cada fase do pipeline</div>', unsafe_allow_html=True)
        ticker_raw = st.selectbox(
            "Empresa",
            list(resultados.keys()),
            key='raw_ticker',
            format_func=lambda t: f"{UNIVERSO[t]['nome']}  ·  {t}",
            label_visibility="collapsed",
        )
        if ticker_raw:
            for fase, dados_fase in resultados[ticker_raw]['dados'].items():
                with st.expander(f"fase · {fase.upper()}", expanded=(fase == 'mercado')):
                    dados_exib = {k: v for k, v in dados_fase.items() if k != 'serie_precos'}
                    st.json(dados_exib)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="es-icon">◈</div>
        <div class="es-title">Sistema em espera</div>
        <div class="es-desc">
            Selecione as empresas, o perfil do investidor e o horizonte temporal
            na barra lateral. Depois clique em <strong>EXECUTAR AGENTE</strong>
            para iniciar o pipeline de 7 fases.
        </div>
    </div>
    """, unsafe_allow_html=True)
