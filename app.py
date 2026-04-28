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
    page_title="Agente IA — Equity Research BR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS customizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: #e2b96f; margin: 0; font-size: 2rem; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 0.95rem; }

    .metric-card {
        background: #1e2a3a;
        border: 1px solid #2d3f55;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .label { color: #a8b2d8; font-size: 0.8rem; text-transform: uppercase; }
    .metric-card .value { color: #e2b96f; font-size: 1.8rem; font-weight: 700; }

    .rec-buy  { background: #0d2e1a; border-left: 4px solid #27ae60; padding: 0.8rem 1.2rem; border-radius: 6px; }
    .rec-hold { background: #2e2510; border-left: 4px solid #f39c12; padding: 0.8rem 1.2rem; border-radius: 6px; }
    .rec-sell { background: #2e0d0d; border-left: 4px solid #e74c3c; padding: 0.8rem 1.2rem; border-radius: 6px; }

    .phase-badge {
        display: inline-block; background: #0f3460; color: #e2b96f;
        padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin: 0.15rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #e2b96f, #c8962a);
        color: #1a1a2e; font-weight: 700; border: none;
        border-radius: 8px; padding: 0.6rem 2rem;
        font-size: 1rem; width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; }

    div[data-testid="stSidebar"] { background: #0f1724; }
    div[data-testid="stSidebar"] label { color: #a8b2d8 !important; }

    .dossie-box {
        background: #0d1b2a; border: 1px solid #2d3f55;
        border-radius: 10px; padding: 1.5rem;
        font-family: 'Courier New', monospace; font-size: 0.82rem;
        color: #cdd6f4; white-space: pre-wrap; max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Agente de Equity Research com IA</h1>
    <p>Avaliação automatizada · Commodities & Siderurgia brasileiras · Pipeline de 7 fases</p>
    <p style="font-size:0.8rem; color:#6b7fa3;">Disciplina: IA Aplicada ao Mercado Financeiro — Prof. João Luiz Chela</p>
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
        info = yf.Ticker(ticker).info
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
        t    = yf.Ticker(ticker)
        info = t.info
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
        info = yf.Ticker(ticker).info
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
        info = yf.Ticker(ticker).info
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

    for i, ticker in enumerate(tickers):
        if progress_bar:
            progress_bar.progress((i) / total, text=f"Analisando {ticker}…")

        dados = {}

        # Fases paralelas
        dados['mercado']  = fase1_mercado(ticker, data_inicio, data_fim)
        dados['kpis']     = fase3_kpis_sinteticos(ticker)
        dados['nlp']      = fase4_nlp_sintetico(ticker)
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
    st.markdown("## ⚙️ Configuração")
    st.markdown("---")

    # Seleção de empresas
    st.markdown("### 🏢 Universo de Análise")
    opcoes = {f"{v['nome']} ({k})": k for k, v in UNIVERSO.items()}
    tickers_sel = st.multiselect(
        "Selecione as empresas:",
        options=list(opcoes.keys()),
        default=list(opcoes.keys())[:3],
        help="Mínimo 1, máximo 7 empresas",
    )

    # Perfil do investidor
    st.markdown("### 👤 Perfil do Investidor")
    perfil = st.radio(
        "Perfil:",
        options=['conservador', 'agressivo'],
        format_func=lambda x: '🛡️ Conservador' if x == 'conservador' else '🚀 Agressivo',
        help="Conservador: pesos maiores para KPIs. Agressivo: pesos maiores para ML.",
    )

    # Horizonte temporal
    st.markdown("### 📅 Horizonte Temporal")
    data_inicio = st.date_input(
        "Data início:", value=datetime(2022, 1, 1),
        min_value=datetime(2020, 1, 1),
        max_value=datetime.now() - timedelta(days=90),
    ).strftime('%Y-%m-%d')

    data_fim = st.date_input(
        "Data fim:", value=datetime.now() - timedelta(days=1),
        min_value=datetime(2020, 6, 1),
        max_value=datetime.now(),
    ).strftime('%Y-%m-%d')

    st.markdown("---")

    # Botão de execução
    executar = st.button("🤖 Executar Agente", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#6b7fa3;'>
    <b>Pipeline de 7 Fases:</b><br>
    <span class='phase-badge'>F1 Mercado</span>
    <span class='phase-badge'>F2 CVM*</span>
    <span class='phase-badge'>F3 KPIs</span>
    <span class='phase-badge'>F4 NLP</span>
    <span class='phase-badge'>F5 ML</span>
    <span class='phase-badge'>F6 Valuation</span>
    <span class='phase-badge'>F7 Agente</span><br><br>
    <i>*F2 usa dados Yahoo Finance como proxy para CVM nesta interface.</i>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ÁREA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
if not tickers_sel:
    st.info("👈 Selecione pelo menos uma empresa na barra lateral para começar.")
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

    # ── Tabs principais ──────────────────────────────────────────────────────
    tab_ranking, tab_dossie, tab_graficos, tab_dados = st.tabs(
        ["🏆 Ranking", "📋 Dossiê", "📈 Gráficos", "🔢 Dados Brutos"]
    )

    # ── TAB 1: Ranking ───────────────────────────────────────────────────────
    with tab_ranking:
        st.markdown("### 🏆 Ranking Final — Score Consolidado (0-100)")

        linhas = []
        for tk, res in resultados.items():
            rec  = res['recomendacao']
            nome = UNIVERSO[tk]['nome']
            setor = UNIVERSO[tk]['setor']
            linhas.append({
                'Empresa': nome,
                'Ticker': tk,
                'Setor': setor,
                'Score Final': rec['score_final'],
                'Recomendação': rec['recomendacao'],
                'Mercado': rec['componentes'].get('Mercado (F1)', np.nan),
                'KPIs': rec['componentes'].get('KPIs (F3)', np.nan),
                'NLP': rec['componentes'].get('NLP (F4)', np.nan),
                'Valuation': rec['componentes'].get('Valuation (F6)', np.nan),
            })

        df_rank = pd.DataFrame(linhas).sort_values('Score Final', ascending=False).reset_index(drop=True)
        df_rank.index = df_rank.index + 1

        # Cards de destaque
        cols_k = st.columns(min(3, len(df_rank)))
        labels_pos = ['🥇 Melhor', '🥈 2º lugar', '🥉 3º lugar']
        for i, col in enumerate(cols_k):
            if i < len(df_rank):
                row = df_rank.iloc[i]
                cor_rec = {'BUY': '#27ae60', 'HOLD': '#f39c12', 'SELL': '#e74c3c'}.get(row['Recomendação'], '#888')
                col.markdown(f"""
                <div class="metric-card">
                    <div class="label">{labels_pos[i]}</div>
                    <div class="value">{row['Score Final']:.1f}</div>
                    <div style="color:#a8b2d8; font-size:0.9rem;">{row['Empresa']}</div>
                    <div style="color:{cor_rec}; font-weight:700; font-size:1.1rem;">{row['Recomendação']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # Tabela completa com formatação
        def colorir_rec(val):
            cores = {'BUY': '#0d2e1a', 'HOLD': '#2e2510', 'SELL': '#2e0d0d'}
            return f'background-color: {cores.get(val, "")}; color: white; font-weight: bold'

        def colorir_score(val):
            if pd.isna(val):
                return ''
            if val >= 65:
                return 'color: #27ae60; font-weight: bold'
            elif val <= 35:
                return 'color: #e74c3c; font-weight: bold'
            return 'color: #f39c12'

        st.dataframe(
            df_rank.style
                .map(colorir_rec, subset=['Recomendação'])
                .map(colorir_score, subset=['Score Final'])
                .format({'Score Final': '{:.1f}', 'Mercado': '{:.1f}', 'KPIs': '{:.1f}',
                         'NLP': '{:.1f}', 'Valuation': '{:.1f}'}),
            use_container_width=True, height=300,
        )

        # Download
        csv = df_rank.to_csv(index=True).encode('utf-8')
        st.download_button(
            "⬇️ Baixar Ranking (CSV)",
            data=csv, file_name=f"ranking_agente_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv',
        )

    # ── TAB 2: Dossiês ───────────────────────────────────────────────────────
    with tab_dossie:
        st.markdown("### 📋 Dossiê Completo por Empresa")

        ticker_escolhido = st.selectbox(
            "Selecione a empresa:",
            options=list(resultados.keys()),
            format_func=lambda t: f"{UNIVERSO[t]['nome']} ({t})",
        )

        if ticker_escolhido:
            res  = resultados[ticker_escolhido]
            rec  = res['recomendacao']
            cor  = rec['cor']
            css_class = {'BUY': 'rec-buy', 'HOLD': 'rec-hold', 'SELL': 'rec-sell'}.get(rec['recomendacao'], 'rec-hold')

            c1, c2, c3 = st.columns(3)
            c1.metric("Score Final", f"{rec['score_final']:.1f} / 100")
            c2.markdown(f"""
            <div class='{css_class}'>
                <b style='font-size:1.3rem; color:{cor};'>{rec['recomendacao']}</b><br>
                <span style='color:#a8b2d8; font-size:0.8rem;'>Perfil {perfil}</span>
            </div>
            """, unsafe_allow_html=True)
            c3.metric("Preço Atual", f"R$ {res['dados']['mercado'].get('preco_atual', 0):.2f}")

            st.markdown("")
            st.markdown(f"""
            <div class="dossie-box">{res['dossie']}</div>
            """, unsafe_allow_html=True)

            # Download dossiê
            todos_dossies = '\n'.join(r['dossie'] for r in resultados.values())
            st.download_button(
                "⬇️ Baixar Todos os Dossiês (.txt)",
                data=todos_dossies.encode('utf-8'),
                file_name=f"agente_dossies_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime='text/plain',
            )

    # ── TAB 3: Gráficos ──────────────────────────────────────────────────────
    with tab_graficos:
        st.markdown("### 📈 Visualizações")

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        cores_rec = {'BUY': '#27ae60', 'HOLD': '#f39c12', 'SELL': '#e74c3c'}

        # Gráfico 1: Barras horizontais — score final
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0d1b2a')

        nomes = [UNIVERSO[t]['nome'] for t in resultados]
        scores_f = [resultados[t]['recomendacao']['score_final'] for t in resultados]
        recs_f   = [resultados[t]['recomendacao']['recomendacao'] for t in resultados]
        cores_f  = [cores_rec.get(r, '#888') for r in recs_f]

        idx = np.argsort(scores_f)
        nomes_s  = [nomes[i] for i in idx]
        scores_s = [scores_f[i] for i in idx]
        cores_s  = [cores_f[i] for i in idx]
        recs_s   = [recs_f[i] for i in idx]

        bars = ax.barh(nomes_s, scores_s, color=cores_s, edgecolor='#2d3f55', height=0.6)
        ax.axvline(55, color='gray', lw=0.8, linestyle='--', alpha=0.5)
        ax.axvline(35, color='gray', lw=0.8, linestyle='--', alpha=0.5)
        for bar, s, r in zip(bars, scores_s, recs_s):
            ax.text(s + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{s:.1f} ({r})', va='center', color='white', fontsize=9)
        ax.set_xlim(0, 105)
        ax.set_xlabel('Score Final (0-100)', color='#a8b2d8')
        ax.set_title('Ranking do Agente por Score Final', color='#e2b96f', pad=12)
        ax.tick_params(colors='#a8b2d8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3f55')
        legend = [mpatches.Patch(color=c, label=l) for l, c in cores_rec.items()]
        ax.legend(handles=legend, loc='lower right', facecolor='#1e2a3a',
                  labelcolor='white', edgecolor='#2d3f55')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Gráfico 2: Radar de componentes
        st.markdown("#### Decomposição por Componente")
        tickers_vis = list(resultados.keys())[:4]  # máx 4 para legibilidade
        categorias  = ['Mercado', 'KPIs', 'NLP', 'Valuation']
        fig2, axes  = plt.subplots(1, len(tickers_vis), figsize=(14, 4))
        fig2.patch.set_facecolor('#0d1b2a')
        if len(tickers_vis) == 1:
            axes = [axes]

        cores_list = ['#e2b96f', '#4fc3f7', '#81c784', '#ff8a65']
        for ax2, tk, cor_bar in zip(axes, tickers_vis, cores_list):
            ax2.set_facecolor('#1e2a3a')
            rec = resultados[tk]['recomendacao']
            vals = [rec['componentes'].get(f'{c} (F{i})', 12.5)
                    for c, i in zip(['Mercado','KPIs','NLP','Valuation'], [1,3,4,6])]
            bars2 = ax2.bar(categorias, vals, color=cor_bar, alpha=0.85, edgecolor='#2d3f55')
            ax2.set_ylim(0, 25)
            ax2.set_title(UNIVERSO[tk]['nome'], color='#e2b96f', fontsize=10)
            ax2.tick_params(colors='#a8b2d8', labelsize=7)
            for spine in ax2.spines.values():
                spine.set_edgecolor('#2d3f55')
            for bar, v in zip(bars2, vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                         f'{v:.1f}', ha='center', va='bottom', color='white', fontsize=8)
        plt.suptitle('Score por Componente (máx 25)', color='#a8b2d8', fontsize=11, y=1.02)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Gráfico 3: Séries de preços
        st.markdown("#### Performance de Preços")
        try:
            fig3, ax3 = plt.subplots(figsize=(11, 4))
            fig3.patch.set_facecolor('#0d1b2a')
            ax3.set_facecolor('#0d1b2a')
            colors_line = ['#e2b96f','#4fc3f7','#81c784','#ff8a65','#ce93d8','#f48fb1','#80cbc4']
            for (tk, res), c in zip(resultados.items(), colors_line):
                serie = res['dados']['mercado'].get('serie_precos', {})
                if serie:
                    s = pd.Series(serie)
                    s_norm = s / s.iloc[0] * 100
                    ax3.plot(s_norm.index, s_norm.values, label=UNIVERSO[tk]['nome'],
                             color=c, alpha=0.9)
            ax3.axhline(100, color='gray', lw=0.8, linestyle='--', alpha=0.5)
            ax3.set_ylabel('Índice (base 100)', color='#a8b2d8')
            ax3.set_title('Performance Relativa (base 100)', color='#e2b96f')
            ax3.tick_params(colors='#a8b2d8')
            ax3.legend(facecolor='#1e2a3a', labelcolor='white', edgecolor='#2d3f55',
                       fontsize=8, loc='upper left')
            ax3.grid(True, alpha=0.15, color='#2d3f55')
            for spine in ax3.spines.values():
                spine.set_edgecolor('#2d3f55')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
        except Exception as e:
            st.warning(f"Gráfico de preços indisponível: {e}")

    # ── TAB 4: Dados Brutos ──────────────────────────────────────────────────
    with tab_dados:
        st.markdown("### 🔢 Dados Brutos por Empresa")
        ticker_raw = st.selectbox(
            "Empresa:", list(resultados.keys()),
            key='raw_ticker',
            format_func=lambda t: f"{UNIVERSO[t]['nome']} ({t})",
        )
        if ticker_raw:
            res_raw = resultados[ticker_raw]
            for fase, dados_fase in res_raw['dados'].items():
                with st.expander(f"📂 {fase.upper()}", expanded=(fase == 'mercado')):
                    dados_exib = {k: v for k, v in dados_fase.items() if k != 'serie_precos'}
                    st.json(dados_exib)

else:
    # Estado inicial
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#6b7fa3;'>
        <div style='font-size:4rem;'>🤖</div>
        <h3 style='color:#a8b2d8;'>Configure e execute o agente</h3>
        <p>Selecione as empresas, o perfil do investidor e o horizonte temporal na barra lateral.<br>
        Depois clique em <b style='color:#e2b96f;'>Executar Agente</b>.</p>
        <br>
        <p style='font-size:0.85rem;'>O pipeline cobre as 7 fases da metodologia:<br>
        Mercado → CVM → KPIs → NLP → ML → Valuation → Agente</p>
    </div>
    """, unsafe_allow_html=True)
