# 📊 Agente de Equity Research com IA

**Avaliação automatizada de empresas brasileiras** — Commodities & Siderurgia  
Disciplina: IA Aplicada ao Mercado Financeiro · Prof. João Luiz Chela · FGV EAESP

---

## O que é

Interface Streamlit para o pipeline de 7 fases desenvolvido no projeto PP1.  
O agente aceita inputs (tickers, perfil, horizonte) e gera:

- 🏆 **Ranking** com score 0-100 e recomendação Buy/Hold/Sell
- 📋 **Dossiê** completo por empresa (equivalente ao `build_dossie()` do notebook)
- 📈 **Gráficos** de performance, decomposição por componente e preços
- ⬇️ **Export** em CSV e TXT

---

## Estrutura de arquivos

```
chela_app/
├── app.py               ← Aplicação principal (este arquivo é o único que o Streamlit roda)
├── requirements.txt     ← Dependências Python
├── README.md            ← Este arquivo
└── .gitignore           ← Arquivos a ignorar no Git
```

---

## Rodar localmente

### 1. Clone o repositório
```bash
git clone https://github.com/SEU_USUARIO/chela-equity-agent.git
cd chela-equity-agent
```

### 2. Crie um ambiente virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute
```bash
streamlit run app.py
```

O app abre automaticamente em `http://localhost:8501`

---

## Deploy no Streamlit Community Cloud (gratuito)

### Passo 1 — Criar conta GitHub (se não tiver)
1. Acesse [github.com](https://github.com) → **Sign up**
2. Escolha um username (ex: `beto-fgv`)
3. Confirme o e-mail

### Passo 2 — Criar o repositório no GitHub
1. Clique em **➕ New repository** (botão verde no canto superior direito)
2. Nome: `chela-equity-agent`
3. Visibilidade: **Public** (obrigatório para o Streamlit Cloud gratuito)
4. Marque **Add a README file**
5. Clique em **Create repository**

### Passo 3 — Subir os arquivos

**Opção A — Via interface web (mais simples):**
1. No repositório criado, clique em **Add file → Upload files**
2. Arraste `app.py`, `requirements.txt` e `README.md`
3. Clique em **Commit changes**

**Opção B — Via terminal (Git):**
```bash
# Configure seu Git (só na primeira vez)
git config --global user.name "Seu Nome"
git config --global user.email "seu@email.com"

# Clone o repo que você criou no GitHub
git clone https://github.com/SEU_USUARIO/chela-equity-agent.git
cd chela-equity-agent

# Copie os arquivos do projeto para esta pasta
# (app.py, requirements.txt, README.md)

# Adicione, commite e envie
git add .
git commit -m "feat: app streamlit do agente de equity research"
git push origin main
```

### Passo 4 — Deploy no Streamlit Cloud
1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Faça login com sua conta **GitHub**
3. Clique em **New app**
4. Preencha:
   - **Repository:** `SEU_USUARIO/chela-equity-agent`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Clique em **Deploy!**
6. Aguarde ~2 minutos → seu app estará em:  
   `https://chela-equity-agent-XXXXX.streamlit.app`

---

## Correspondência com o notebook (PP1)

| Fase do Notebook | Função no app.py | Dados usados |
|---|---|---|
| Fase 1 — Mercado | `fase1_mercado()` | Yahoo Finance (preços reais) |
| Fase 2 — CVM | — | *Yahoo Finance como proxy na UI* |
| Fase 3 — KPIs | `fase3_kpis_sinteticos()` | Yahoo Finance Info |
| Fase 4 — NLP | `fase4_nlp_sintetico()` | `longBusinessSummary` do Yahoo |
| Fase 5 — ML | `fase5_ml_score()` | Features derivadas |
| Fase 6 — Valuation | `fase6_valuation()` | Múltiplos do Yahoo Finance |
| Fase 7 — Agente | `executar_agente()` | Orquestra todas as fases |

> **Nota:** A Fase 2 (ingestão CVM) envolve downloads de ZIPs pesados (~15 min no Colab).  
> Na interface Streamlit, usamos o Yahoo Finance Info como proxy para os dados fundamentalistas,  
> mantendo a mesma lógica de score. Para integração completa com CVM, os DataFrames  
> `fundamentals_wide` e `kpis_trimestrais` gerados pelo notebook podem ser carregados via  
> `st.file_uploader()` — adicione essa funcionalidade como extensão futura.

---

## Extensões sugeridas

- [ ] Upload dos CSVs/Parquets gerados pelo notebook (dados reais da CVM)
- [ ] Integração com API Anthropic para dossiês gerados por LLM (`claude-sonnet-4-5`)
- [ ] Alertas automáticos por e-mail quando recomendação mudar
- [ ] Histórico de análises salvo em banco de dados

---

## Tecnologias

`Python 3.11` · `Streamlit` · `yfinance` · `scikit-learn` · `pandas` · `matplotlib`

---

*Projeto acadêmico — FGV EAESP · 2025*
