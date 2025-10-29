# Narrative Alpha Detector

LLM-based signal system for detecting retail mispricings in prediction markets. 

## Code Availability Notice

The source code for Narrative Alpha Detector is **private** as it’s currently deployed in live trading operations.  
If you’re interested in the methodology, results, or collaboration, **please contact me directly** — I am happy to discuss details or share select components as appropriate.  

## Overview 

Narrative Alpha Detector is a pipeline that scans live prediction markets (via Polymarket) and detects potential pricing dislocations by comparing retail odds to AI-generated probability priors.

By querying large language models (Perplexity Sonar, Grok, GPT & Claude) for calibrated forecasts with built-in confidence, the system ranks trade opportunities where crowd sentiment may diverge from expert-like reasoning.

This is a research-aligned tool for exploring narrative dislocations, sentiment overpricing, and LLM-based alpha surfacing in real time. All artifacts, prompts, and logs are versioned for reproducibility.

### Key Capabilities
- **Ensemble Priors:** Blend Perplexity, GPT, Grok, Claude, and heuristic forecasters with category-aware adaptive weights.
- **Resilient Data Pipeline:** Fault-tolerant Polymarket ingestion with exponential backoff plus automatic model disablement on repeated provider failures.
- **Signal Gatekeeping:** Config-driven evaluator enforces confidence, freshness, and market-status guardrails before opportunities flow downstream.
- **Risk & Execution:** Centralized risk manager enforces bankroll/category/regime limits while a stateful order agent tracks live order lifecycles (fills, cancels, errors).
- **Shadow Trading & CI:** Shadow mode logs intended orders without transmitting them; automated validation, Dockerized services, and CI keep deployments safe.
- **Operational Health:** Health monitor tracks latency, slippage, drawdown, and LLM anomalies, tripping a kill switch when limits are breached.
- **Opportunity Intelligence:** Regime-aware tagging, NLP-driven sector labels, and Streamlit dashboards for real-time decisioning.
- **Outcome Calibration:** Daily resolved-market joins, reliability curves, and stability drift alerts (Brier, KL, KS).
- **Portfolio Simulation:** Kelly-based bet sizing, slippage modeling, drawdown tracking, and factor attribution.
- **Audit Trail:** JSONL lineage, automated HTML/PDF reports, and reproducible artifacts for every run.

### Documentation
- **Signal Filtering Config:** `config/signal_filters.json` centralizes evaluation and opportunity-filter thresholds (confidence, freshness, liquidity, regimes).
- **Risk Limits Config:** `config/risk_limits.json` defines bankroll, category, and per-trade guardrails consumed by the risk manager and order agent.
- **Health Thresholds Config:** `config/health_thresholds.json` sets latency, slippage, drawdown, and anomaly limits for the health monitor and kill switch.

## Architecture

| Module | Description                                                                                         |
|--------|-----------------------------------------------------------------------------------------------------|
| `src/polymarket_api_scraper.py` | Fetches live prediction markets and odds from Polymarket (Gamma API)                                |
| `src/score_mispricing.py` | Compares market vs prior, computes mispricing and expected value                                    |
| `src/signal_evaluator.py` | Applies confidence/freshness gates and annotates pass/fail status for each signal                   |
| `src/log_outcomes.py` | Pulls resolved markets, versioned snapshots, and joins outcomes with predictions                    |
| `src/generate_priors.py` | Multi-model prior generation with adaptive ensemble weighting                                       |
| `src/simulate_portfolio.py` | Backtests automated bet placement with risk and slippage modeling                                   |
| `src/risk_manager.py` | Validates proposed trades against bankroll/category/regime constraints                              |
| `src/execute_order.py` | Stateful order agent with transport abstraction, risk checks, and lifecycle logging                 |
| `src/analyze_opportunities.py` | Analyzes and filters trading opportunities with tagging/meta features                               |
| `src/visualize_opportunities.py` | Creates beautiful charts, dashboards, and reports of trading opportunities                          |
| `src/utils/paths.py` | Centralized project paths and directory scaffolding helpers                                         |
| `src/utils/loggers.py` | JSONL logging utilities for prompts and pipeline events                                             |
| `src/utils/llm_models.py` | Provider-specific LLM adapters (Perplexity, OpenAI, Anthropic, Grok, local heuristic)               |
| `src/utils/ensembles.py` | Adaptive ensemble weighting and history persistence                                                 |
| `src/utils/regimes.py` | Regime classification heuristics and history tracking                                               |
| `src/utils/stability.py` | Distribution drift and calibration monitoring helpers                                               |
| `src/utils/tagging.py` | Keyword + optional LLM tagging helpers (sector, liquidity, horizon)                                 |
| `src/dashboard.py` | Streamlit interactive dashboard (top mispricings, regimes, calibration, PnL)                        |
| `src/stability_monitor.py` | CLI to compare priors vs baseline, log drift alerts                                                 |
| `src/health_monitor.py` | Evaluates latency/slippage/anomaly thresholds and manages the kill switch                           |
| `src/audit_trail.py` | Consolidates predictions/outcomes/simulations into timestamped JSONL                                |
| `src/generate_audit_report.py` | Builds HTML/PDF audit reports summarising the latest run                                            |
| `main.py` | Full pipeline runner                                                                                |
| `results/` | Structured outputs (`raw/`, `priors/`, `scored/`, `figures/`, `reports/`, `logs/`, `prompts/`, etc.) |

## Methodology

**Market Odds**: Polymarket odds are fetched in real time via the Gamma API.

**AI Prior Estimation**: LLMs are queried with strict prompting and confidence weighting:

```
probability | confidence (e.g., 0.72 | 0.65)
```

Final priors are adjusted via:

```
adjusted = (1 - confidence) * 0.5 + confidence * raw_prob
```

Predictions fan out asynchronously across enabled providers with per-model timeouts; any provider that times out or raises an API error is dropped from the run and logged for auditability.

**Signal Evaluation**: `src/signal_evaluator.py` reads `config/signal_filters.json` and labels each scored signal as `passed`/`failed` based on confidence floors, market status, and signal staleness before downstream filtering.

**Risk Management & Execution**:
- `src/risk_manager.py` loads `config/risk_limits.json` and enforces portfolio, category, regime, and per-trade caps.
- `src/simulate_portfolio.py` calls the risk manager before sizing any position and records approval metadata for every simulated trade.
- `src/execute_order.py` wraps live submission with a state machine (`SENT → ACKNOWLEDGED → PARTIAL/FULL_FILL`, `CANCEL_SENT → CANCELLED`, `ERROR`) and gracefully handles API rejections without crashing.

**Operational Monitoring & Kill Switch**:
- `main.py` logs per-stage latencies and persists end-to-end runtime metrics under `results/health/`.
- `src/health_monitor.py` compares those metrics, portfolio drawdowns, slippage, and LLM anomalies to `config/health_thresholds.json`, raising or clearing a kill switch (`results/health/kill_switch.json`).
- The Streamlit dashboard (`src/dashboard.py`) surfaces health status, active breaches, and kill switch state in real time.

**Scoring**: Mispricing = |Market - Prior|; Expected value = directionally adjusted mispricing

**Filtering**: Expired markets and extreme hallucinations are clipped or skipped

**Regime Scoring & Structure**:
- Each market is scored into `trending`, `news_shock`, `meme`, or `low_attention` regimes using NLP heuristics, volume, and price velocity.
- Regime scores are appended to scored opportunities, logged per run (`results/regimes/`), and rolled up to track transitions where alpha comes/goes.
- Tags incorporate `regime:<label>` so analysts can filter reports, dashboards, and simulations by structural state.

**Stability Monitoring**:
- Distributional drift (KL, KS, mean/std shift) for priors is compared to the baseline snapshot in `results/stability/baseline.json`.
- Calibration decay is monitored using Brier history from resolved outcomes; alerts fire when deterioration exceeds thresholds.
- Reports land in `results/stability/*_stability.json` for audit-ready summaries and daily checks.

**Auditability & Reporting**:
- `src/audit_trail.py` appends snapshot metadata for priors, scored signals, outcomes, and portfolio trades into `results/logs/audit_trail.jsonl`.
- `src/generate_audit_report.py` renders HTML (and optional PDF when WeasyPrint is installed) summaries combining regime stats, calibration, stability alerts, and portfolio attribution.
- Each run emits a per-trade trace (`results/audit/{run_id}_trade_trace.jsonl`) linking raw snapshots → priors → evaluation → risk/execute events, and the audit report is refreshed automatically.
- Streamlit dashboard (`src/dashboard.py`) provides interactive insights across mispricings, regimes, calibration curves, and simulated PnL.

**Ensemble Modeling**:
- Multiple models (Perplexity/GPT/Claude/local heuristic) generate priors; each prediction is logged with per-model confidence.
- Historical outcome performance (Brier/log-loss) is tracked by `src/log_outcomes.py` and stored in `results/resolved/model_performance*.csv`.
- `src/generate_priors.py` uses those metrics to adapt ensemble weights per category via inverse-Brier weighting and records usage in `results/priors/ensemble_history.jsonl`.

**Versioning & Auditability**:
- Every Polymarket pull is captured as `results/raw/YYYYMMDD_hhmmss_markets.json` plus a `latest_markets.csv`.
- Prior runs persist to `results/priors/` (timestamped + latest) and legacy compatibility files remain in `results/`.
- JSONL prompt logs land in `results/prompts/`, while pipeline components emit structured event logs to `results/logs/`.
- CLI invocations set a run ID (`NAD_RUN_ID`) that stitches all artifacts together.
- Regime tags and confidence scores are archived in `results/regimes/` for transition analysis.


## Setup

```bash
git clone https://github.com/lucaskemper/narrative-alpha-detector.git
cd narrative-alpha-detector
pip install -r requirements.txt
``` 

Create a `.env` file with your API key:

```
PERPLEXITY_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here       
ANTHROPIC_API_KEY=your_key_here
GROK_API_KEY=your_key_here
```
If a key is missing the model simply drops out of the ensemble (and is noted in the pipeline logs).

The first run will automatically scaffold the `results/` hierarchy (raw snapshots, priors, scored data, figures, reports, logs, prompts, etc.) and a `notebooks/` folder for analysis notebooks.

## Usage

```bash
# Full pipeline
python main.py

# Or run individual steps
python src/polymarket_api_scraper.py
python src/generate_priors.py
python src/score_mispricing.py
python src/signal_evaluator.py             # Apply quality gates + annotate evaluation status
python src/execute_order.py                # Submit live/dry-run orders with lifecycle tracking
python src/health_monitor.py               # Evaluate latency/slippage/anomaly thresholds + kill switch
python src/log_outcomes.py              # Daily resolved market snapshots + joins
python scripts/validate_changes.py         # Quick validation run (main → sim → health, optional shadow order)

# Backtesting & factor analysis (requires scored + resolved data)
python src/simulate_portfolio.py --bet-sizing fractional_kelly --bankroll 15000

# Stability monitoring & drift detection
python src/stability_monitor.py --update-baseline

# Update consolidated audit log + produce HTML/PDF report
python src/audit_trail.py
python src/generate_audit_report.py --output-html results/reports/audit_report.html

# Launch interactive Streamlit dashboard
streamlit run src/dashboard.py

# Regime summaries (already produced via tagging step)
ls results/regimes/
```

Helpful switches:

```bash
# Reproduce a run with a fixed seed
python main.py --seed 42

# Only regenerate visuals (expects prior & scored data)
python main.py --visualize-only

# Inspect scored markets interactively with tagging overlays
python src/analyze_opportunities.py --quick --tagging-mode keyword

# Blend keyword + LLM tags for the top 15 markets (requires PERPLEXITY_API_KEY)
python src/analyze_opportunities.py --tagging-mode hybrid --llm-sample-size 15

# Re-run priors with OpenAI/Claude enabled and inspect ensemble history
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python src/generate_priors.py
```

### Daily Research Loop

1. `python main.py` — fetch markets, generate ensemble priors, rank mispricings, build visuals.
2. `python src/log_outcomes.py --days-back 30` — refresh the resolved/outcome dataset and model scorecards.
3. `python src/generate_priors.py` — re-run priors with updated weights (skips markets already scored).
4. `python src/score_mispricing.py` → `python src/signal_evaluator.py` — refresh mispricing ranks and gate them through the evaluator.
5. `python src/simulate_portfolio.py --bet-sizing fractional_kelly` — run a what-if portfolio pass with risk validation and inspect outputs in `results/portfolio/`.
6. `python src/execute_order.py --order-file orders/pending.json --shadow-trade` — submit validated trades via the order agent while remaining in shadow mode.
7. `python src/stability_monitor.py --update-baseline` — log calibration/distribution drift and refresh alert baselines.
8. `python src/health_monitor.py` — evaluate latency/slippage/anomaly thresholds and engage/reset the kill switch as needed.
9. `python scripts/validate_changes.py --run-shadow-order` — optional validation pass that re-runs the pipeline, simulation, health monitor, and logs a sample shadow order.
10. Review `results/regimes/` for regime counts + transition stats before making allocation decisions.
11. `python src/audit_trail.py` → `python src/generate_audit_report.py` to archive the day's work and shareable artifact.

Artifacts to inspect after a run:
- `results/raw/` — raw Polymarket snapshots (JSON + latest CSV)
- `results/priors/` — prior estimates (`*_priors.csv`, `latest_priors.csv`)
- `results/priors/ensemble_history.jsonl` — per-run ensemble weights by category
- `results/scored/` — scored mispricings (`*_scored_markets.csv`, `latest_scored_markets.csv`)
- `results/figures/` — charts, dashboards, and other visuals
- `results/reports/` — text summaries and exports
- `results/logs/` — machine-readable pipeline logs (`{run_id}_*.jsonl`)
- `results/prompts/` — LLM prompt/response archives (`{run_id}_prompts.jsonl`)
- `results/resolved/` — resolved market snapshots + joined calibration datasets
- `results/portfolio/` — portfolio trades, equity curves, summaries, and factor attribution
- `results/regimes/` — regime counts & transition history (`regime_history.csv`, per-run summaries)
- `results/stability/` — drift & calibration reports (`*_stability.json`, `baseline.json`)
- `results/health/` — pipeline latency logs, health status snapshots, and kill-switch state
- `results/audit/` — per-run trade traces and archived audit reports
- `notebooks/calibration_analysis.ipynb` — Brier/ECE/log-loss diagnostics & meta-analysis starter

## Containerization & CI/CD

- Build service images (scraper, priors generator, order agent, dashboard):

  ```bash
  docker compose build
  docker compose up dashboard   # launches Streamlit on http://localhost:8501
  ```

- The shared `results/` directory is mounted into each container to persist artifacts across services.
- GitHub Actions workflow `.github/workflows/ci.yml` runs linting (ruff), bytecode compilation, pytest, and Docker builds for every push/pull request.

## Shadow Trading Mode

Enable shadow trading to log intended orders without hitting venue APIs:

```bash
export NAD_SHADOW_TRADE=1
python src/execute_order.py --slug example-market --direction buy_yes --stake 250 --price 0.45
```

The order agent records a `shadow` lifecycle entry in `results/logs/<run>_order_agent.jsonl`, enabling live-vs-sim reconciliation without transmitting orders.

## Literature & Reference Foundation

### Prediction Markets & Market Inefficiency

- **Becker, H. & Challet, D. (2025). "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets." Proceedings of AFT 2025.**  
  *Defines sources and persistence of arbitrage in prediction markets, with empirical Polymarket data analysis.*

- **Seth, Cowgill, et al. (2020). "Forecasting Skill of a Crowd-Prediction Platform." Journal of Futures Markets 40(6), 2020.**  
  *Benchmarks accuracy and calibration of crowd-driven markets, highlighting where consensus mispricings emerge.*

- **AINVEST (2025). "Prediction Markets 2025: From Speculation to Financial Infrastructure."**  
  *Industry whitepaper covering evolution of retail and institutional prediction market structure.*

---

### LLM-Based Forecasting & Calibration

- **Sonar, GPT, Claude & Prophet Arena Teams (2025). "LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena." arXiv:2510.17638v1.**  
  *Large-scale, cross-model calibration and outcome validation for AI forecasting vs. real-world market prices.*

- **Anonymous (2024). "Evaluating LLMs on Real-World Forecasting Against Financial and Political Markets." arXiv:2507.04562v3.**  
  *Quantitative analysis of top LLMs as market forecasters, including calibration and reliability diagrams.*

- **SpyLab.ai (2025). "LLM Forecasting Evaluations Need Fixing."**  
  *Analysis detailing best practices and pitfalls in out-of-sample LLM forecast validation.*

- **AI Realist (2025). "If You Bet Like an LLM, Would You Make Money?"**  
  *Blog meta-review of LLM-based betting strategies and calibration error risks.*

---

### Statistical & Ensemble Methods for Forecasting

- **DeGroot, M. & Fienberg, S. (1983). "Calibration of Probability Assessments in Multinomial Problems." Journal of the American Statistical Association, 78(383), 1983.**  
  *Seminal paper on forecast calibration metrics and proper scoring rules.*

- **Farrow, T. & Shipley, B. (2023). "Ensemble Forecasting Methods in Decomposable Event Spaces." Statistical Computing 33(2), 2023.**  
  *Modern techniques for combining and calibrating multiple probabilistic signals.*

---

### Narrative Economics & Sentiment-driven Anomalies

- **Shiller, R. (2022). "Economic Narratives and Market Outcomes." American Economic Association Annual Meeting.**  
  *Defines mechanism for narrative contagion and its impact on trading/pricing regimes.*

---

### Recent Technical & Open-Source Extensions

- **HuggingFace Blog (2025). "PrediBench: Testing AI Models on Prediction Markets."**  
  *Benchmarking open-source, LLM-based prediction engines across diverse event types.*

---

#### Additional Resources

- **Polymarket Gamma API Docs**  
  *For technical reference on market data structure and reliability.*

- **MIT Prediction Market Database**  
  *For historical, resolved outcomes across research and live markets.*



## Why It Matters

- Prediction markets reflect retail consensus.
- LLMs, when prompted carefully, represent a composite of expert knowledge + internet priors.
- This project surfaces where those diverge — and highlights where narrative-driven mispricings might occur.

## Notes

- LLMs are not reliable oracles. Priors are clipped and adjusted for realism.
- This is not a trading system, but a research tool for sentiment monitoring and signal experimentation.
- Certain categories (e.g., celebrity gossip, meme ETFs) are prone to hallucinated priors — these are flagged in postprocessing.

## Future Improvements

- Expand calibration notebook with bootstrapped confidence intervals
- Adaptive ensemble priors (GPT/Claude/local models) with category-aware weighting
- Portfolio simulation with execution/slippage modeling
- Regime change detection and stability monitoring for alpha decay
- Online weight updating during intraday runs and reinforcement for active learning
