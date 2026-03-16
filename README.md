<p align="center">
  <h1 align="center">Non-Equilibrium Thermodynamics of<br>Cryptocurrency Perpetual Futures</h1>
  <p align="center">
    <strong>Jarzynski Equality · Regime Classification · Physics-Informed Trading</strong>
  </p>
  <p align="center">
    <a href="#key-results">Results</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#paper">Paper</a> ·
    <a href="#citation">Citation</a>
  </p>
</p>

---
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19046564.svg)](https://doi.org/10.5281/zenodo.19046564)

> **First-ever empirical test of the Jarzynski equality on financial market data.** We map the perpetual futures funding rate mechanism to non-equilibrium statistical mechanics, validate on 6.2 years of Binance data (19,697 funding cycles), and derive physics-informed trading signals that significantly outperform naive funding arbitrage.

**Author:** Ethan Lee Khoo Chuen · Independent Researcher · [ElvianElvyWork@gmail.com](mailto:ElvianElvyWork@gmail.com)

---

## The Idea in 60 Seconds

Perpetual futures use a **funding rate** every 8 hours to anchor the contract price to spot. We discovered this mechanism is mathematically identical to a **driven particle in a harmonic potential** from statistical physics:

| Market Quantity | Physics Analogue | Definition |
|:---|:---|:---|
| Percentage basis | Particle position | $(p_{\text{perp}} - p_{\text{spot}}) / p_{\text{spot}} \times 100\%$ |
| Funding rate | External driving force | Applied every 8h |
| Spot market | Thermal heat bath | Absorbs fluctuations |
| Basis volatility | Temperature | $T = \kappa \cdot \text{Var}(b)$ |
| Mean-reversion rate | Spring constant | $\kappa$ from AR(1) |
| Entropy production | Irreversibility | $\sigma = W - \Delta F$ |

This isn't a metaphor — the equations are exact. And the physics makes **testable predictions** about when funding arbitrage is safe vs. dangerous.

---

## Key Results

### 1. The Jarzynski Equality Holds for Bitcoin

The JE ($\langle e^{-\beta\sigma}\rangle = 1$) holds to within **2% for BTC** and **3% for ETH** — the first empirical confirmation on financial data.

| Asset | JE (raw) | JE (winsorized) | Deviation |
|:---:|:---:|:---:|:---:|
| **BTC** | 1.020 | 1.016 | 2.0% |
| **ETH** | 1.032 | 1.020 | 3.2% |
| **SOL** | overflow | 1.002 | 0.18% |

### 2. Relaxation Time Predicts JE Validity (Novel Finding)

We discovered that the basis relaxation time $\tau = 1/\kappa$ determines whether the JE holds:

| Asset | $\tau_{\text{relax}}$ | Funding Period | Ratio | JE Valid? |
|:---:|:---:|:---:|:---:|:---:|
| **BTC** | 7.6h | 8h | 0.96 | ✅ Yes |
| **ETH** | 7.1h | 8h | 0.89 | ✅ Yes |
| **SOL** | 12.3h | 8h | 1.53 | ❌ No |

**When $\tau < 8$h:** basis equilibrates between cycles → JE holds → funding arb is safe.  
**When $\tau > 8$h:** basis persists → JE violated → market in non-equilibrium steady state.

### 3. Second Law of Thermodynamics

| Asset | $\langle\sigma\rangle$ | $p$-value | Verdict |
|:---:|:---:|:---:|:---:|
| **BTC** | +0.135 | < 0.001 | **HOLDS** |
| **ETH** | +0.035 | 0.016 | **HOLDS** |
| **SOL** | −0.135 | 1.000 | **VIOLATED** |

SOL's violation confirms it operates in a **non-equilibrium steady state** — exactly what the relaxation time analysis predicts.

### 4. Physics-Informed Trading Beats Naive Arbitrage

Four thermodynamic signals → three-state regime classifier → dynamic position sizing:

| Asset | V2 Physics Sharpe | Naive Sharpe | Improvement | $p$-value |
|:---:|:---:|:---:|:---:|:---:|
| **BTC** | **5.89** | 4.73 | +1.16 | < 0.001 |
| **ETH** | **5.07** | 4.53 | +0.53 | 0.013 |
| **SOL** | **0.73** | −0.25 | +0.98 | < 0.001 |

**The SOL result is the star:** naive funding arb *loses* 4.2%. The physics-informed strategy *gains* 4.3% by correctly identifying and reducing exposure during NESS periods.

> **Note:** High Sharpe ratios reflect the high-frequency nature of the strategy (1,095 settlements/year), not extraordinary alpha. Per-cycle Sharpe is ~0.18; the annualization factor $\sqrt{1095} \approx 33$ amplifies it. Absolute returns are 2–3% annualized.

---

## Quick Start

### Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Installation

```bash
git clone https://github.com/ElvianElvy/fluctuation-theorem-perps.git
cd fluctuation-theorem-perps
uv pip install -e ".[dev]"
```

### Run the Full Pipeline

```bash
# Step 1: Synthetic validation (instant, no internet needed)
python -m scripts.run_analysis

# Step 2: Collect 6+ years of Binance data (~20 min, public API, no keys)
python -m scripts.collect_data

# Step 3: Full analysis — physics + strategy + figures
python -m scripts.run_analysis
```

**Windows users:** Use Windows Terminal (not cmd.exe) for Unicode output.

### Output

```
output/
├── figures/          # 43 publication-quality figures (PDF + PNG)
│   ├── fig_synthetic_panel.png
│   ├── fig7_relaxation.png
│   ├── fig10_strategy_comparison_*.png
│   └── ...
└── results/
data/
└── ftp.duckdb        # DuckDB database with all market data
```

---

## Project Structure

```
fluctuation-theorem-perps/
│
├── config/
│   └── settings.py              # Pydantic configuration
│
├── data/
│   ├── collectors/
│   │   └── binance.py           # Async Binance data collector
│   └── storage/
│       └── db.py                # DuckDB with ASOF JOIN
│
├── theory/                      # Core physics
│   ├── thermodynamic_quantities.py   # E, W, ΔF, σ — all from market data
│   ├── synthetic.py             # Ornstein-Uhlenbeck simulator
│   ├── market_temperature.py    # β estimation from basis volatility
│   ├── jarzynski.py             # 3 JE estimators (numerically stable)
│   ├── crooks.py                # Detailed Fluctuation Theorem (KDE)
│   └── entropy_production.py    # Rolling σ(t), second law tests
│
├── strategy/                    # Trading application
│   ├── signals.py               # 4 physics-derived signals
│   ├── regime.py                # 3-state thermodynamic classifier
│   ├── backtest.py              # V1 entropy-threshold backtester
│   └── backtest_v2.py           # V2 physics-informed (signed funding + basis risk)
│
├── analysis/
│   ├── validation.py            # Full pipeline: synthetic → real → backtest
│   ├── visualizations.py        # 43 publication figures
│   └── strategy_analysis.py     # Strategy comparison + significance tests
│
├── scripts/
│   ├── collect_data.py          # Data collection CLI
│   └── run_analysis.py          # Full pipeline runner
│
├── paper/
│   ├── main.tex                 # LaTeX source (18 pages)
│   └── outline.md               # Paper structure
│
└── pyproject.toml               # Dependencies and build config
```

---

## Technical Details

### Data

- **Source:** Binance public API (no authentication required)
- **Assets:** BTCUSDT, ETHUSDT, SOLUSDT
- **Period:** January 2020 – March 2026 (6.2 years)
- **Volume:** 19,702 funding rates · 314,693 klines · 156,896 basis observations · 19,697 funding cycles
- **Storage:** DuckDB columnar database with ASOF JOIN for time alignment

### Physics Framework

- **Model:** Overdamped Langevin equation: $\dot{b} = -\kappa b - F + \xi$
- **Work protocol:** Sudden quench — $W = -F \cdot b_{\text{start}}$
- **Free energy:** $\Delta F = -F^2 / (2\kappa)$
- **Temperature:** $T = \kappa \cdot \text{Var}(b)$ from fluctuation-dissipation relation
- **Validation:** Exact OU discretization with known analytics ($R^2 = 0.985$ on DFT)

### Trading Strategy

- **Architecture:** Always-on delta-neutral with physics-based position sizing
- **Regimes:** EQUILIBRIUM (100%) · WARM (60%) · NESS (25%)
- **Fees:** 16 bps round-trip (maker/taker blend + 1 bps slippage)
- **Protocol:** Walk-forward (360-cycle train, 90-cycle test, out-of-sample only)
- **PnL model:** Signed funding + basis risk (no `abs()` — realistic)

---

## Four Novel Signals

| # | Signal | Source | What it measures |
|:---:|:---|:---|:---|
| 1 | **Relaxation Ratio** | $r = 1/(8\kappa)$ | Basis mean-reversion speed vs funding period |
| 2 | **Temperature Z-score** | $z = (\beta - \bar\beta)/\sigma_\beta$ | Market stability regime (cold = safe) |
| 3 | **Entropy Rate** | $\dot\sigma = \beta(W - \Delta F)$ | Funding mechanism dissipation strength |
| 4 | **JE Health** | $\|\langle e^{-\beta\sigma}\rangle - 1\|$ | How thermodynamic the market is |

---

## Paper

The full paper (18 pages, 11 figures, 8 tables) is available in this repository:

- **LaTeX source:** [`paper/main.tex`](paper/main.tex)
- **Compiled PDF:** See releases

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{khoochuen2026fluctuation,
  title={Non-Equilibrium Thermodynamics of Cryptocurrency Perpetual Futures: 
         Jarzynski Equality, Regime Classification, and Physics-Informed Trading},
  author={Khoo Chuen, Ethan Lee},
  year={2026},
  note={Independent Research}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>All trading results in this repository use virtual money only. No real capital was traded.</em>
</p>
