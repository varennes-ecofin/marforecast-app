# MAR Forecasting App 🔮

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)]()

**Interactive web application for forecasting extreme trajectories using Mixed Autoregressive (MAR) models with alpha-stable innovations.**

This application implements the methodology described in the paper *"Forecasting extreme trajectories using seminorm representations"* and provides an intuitive interface for analyzing time series with heavy-tailed distributions and extreme events.

<div align="center">
  <img src="docs/images/app_screenshot.png" alt="MAR Forecasting App Interface" width="800"/>
</div>

## 🌟 Features

### 📊 Data Analysis
- **Multiple datasets**: Climate data (SOI, NAO), FRED macro indicators, financial time series, cryptocurrency data
- **Interactive visualization**: Real-time plotting with quantile highlighting
- **Flexible data import**: Easy integration of custom datasets via Excel files

### 🔬 Model Estimation
- **MAR(r,s) models**: Mixed autoregressive models with causal (r) and noncausal (s) components
- **Generalized Covariance (GCoV)**: Robust estimation method for stable processes
- **Alpha-stable innovations**: Automatic parameter estimation (α, β, σ)
- **Root diagnostics**: Automatic stability checks for estimated polynomials

### 🎯 Pattern Recognition
- **Seminorm-based matching**: Identify patterns in extreme trajectories
- **Optimal k₀ search**: Automatic detection of the optimal pattern index
- **Flexible configuration**: Adjustable temporal depth (m) and direction (positive/negative bubbles)

### 📈 Forecasting Methods

#### Crash Dates (MAR with s ≥ 2)
- Pattern-based trajectory forecasts
- Seminorm representation for extreme events
- Visual comparison with historical data

#### Crash Probabilities (MAR with s = 1)
- Probability-based forecasts using survival theory
- Customizable probability cutoffs
- Crash probability tables at different horizons

### 🔄 Backtesting
- Out-of-sample validation
- Visual comparison of forecasts vs. actual values
- Performance metrics (MAE, MedAE, etc.)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/varennes-ecofin/marforecast-app.git
cd marforecast-app

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run marforecast-app.py
```

### Requirements

```
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
openpyxl>=3.0.0
```

### First Run

1. The app will open in your default browser at `http://localhost:8501`
2. Select "Climate data" and "SOI" series
3. Set in-sample date to December 1, 1991
4. Choose r=0, s=2 and click "Estimate"
5. Enable "Flip dk upside down" and click "Search k₀"
6. Set k₀=1, m=10, h*=1 and click "Forecast"

This reproduces Figure 5 from the paper! 🎉

## 📖 Usage Guide

### 1. Data Selection

<details>
<summary>Click to expand</summary>

```python
# Available datasets:
- Climate data (SOI, NAO indices)
- FRED Macro data (economic indicators)
- Financial data (stock prices, returns)
- Artificial data (simulated examples)
- Crypto data (cryptocurrency prices)
```

Select your dataset and time series from the dropdown menus. Use the quantile slider to highlight extreme values in the visualization.

</details>

### 2. Model Estimation

<details>
<summary>Click to expand</summary>

**Model Specification:**
- `r`: Number of backward lags (causal component)
- `s`: Number of forward lags (noncausal component)

**Estimation Process:**
1. Click "Estimate" to fit the model using GCoV
2. If you see "root in unit circle", click "Estimate" again
3. Repeat until you see "Estimators successfully converged"
4. View estimated parameters: ψᵢ (causal), φⱼ (noncausal), α, β, σ (stable)

**Note:** The estimation may converge to local optima. Multiple runs with random initial values are recommended.

</details>

### 3. Pattern Recognition

<details>
<summary>Click to expand</summary>

**Configuration:**
- **Flip dk**: Enable for negative bubbles, disable for positive bubbles
- **Parameter m**: Temporal depth of the segment (typically 5-10)
- **Neighborhood A**: Pattern matching tolerance (default: -0.1 to 0.1)

**Process:**
1. Click "Search k₀" to find optimal pattern indices
2. Examine the k values for different m values
3. Select k₀ and corresponding m based on the results
4. Choose forecast horizon h*

</details>

### 4. Forecasting

<details>
<summary>Click to expand</summary>

**Crash Dates (for s ≥ 2):**
- Uses seminorm-based pattern matching
- Extrapolates the identified pattern forward
- Provides point forecasts for extreme trajectories

**Crash Probabilities (for s = 1):**
- Computes survival probabilities at each horizon
- Uses theoretical crash probability formulas
- Provides probability thresholds for crash detection

**Backtesting:**
- Available when out-of-sample data exists
- Visual comparison of forecasts with actual values
- Useful for validating forecast accuracy

</details>

## 🧮 Mathematical Background

### MAR(r,s) Model

The Mixed Autoregressive model is defined as:

```
Ψ(F)Φ(B)Xₜ = εₜ
```

where:
- `Φ(B) = 1 - Σψᵢ·Bⁱ` is the causal polynomial (backward operator)
- `Ψ(F) = 1 - Σφⱼ·Fʲ` is the noncausal polynomial (forward operator)
- `εₜ ~ S(α, β, σ, 0)` follows an α-stable distribution

### MA(∞) Representation

The model admits an infinite moving average representation:

```
Xₜ = Σδₖ·εₜ₋ₖ  (k from -∞ to +∞)
```

The MA coefficients δₖ are computed via partial fraction decomposition of the transfer function.

### Estimation Methods

#### 1. Generalized Covariance (GCoV)

Minimizes the loss function:

```
L(θ) = Σ Tr(Γ(h)Γ(0)⁻¹Γ(h)ᵀΓ(0)⁻¹)
```

where Γ(h) are autocovariance matrices of nonlinear transformations of pseudo-residuals.

**Reference:** Gourieroux & Jasiak (2023)

#### 2. Alpha-Stable Parameters

Estimated using characteristic function regression on residuals:

```
log(-log|φ̂(u)|) ≈ α·log|u| + log(σᵃ)
```

**Reference:** Kogon & Williams (1998), Nolan (2020)

### Forecasting Methods

#### Seminorm-Based Forecasting (s ≥ 2)

For a pattern at index k₀:

```
X̂ₜ₊ₕ = ‖Xₜ₋ₘ:ₜ‖ · (δₖ₀₊ₕ / ‖δₖ₀:ₖ₀₊ₘ‖)
```

#### Probability-Based Forecasting (s = 1)

Crash probability at horizon h:

```
P(crash at h) = 1 - |φ|^(αh)
```

## 📊 Example: El Niño Forecasting

### Setup
- **Dataset**: Climate data (Southern Oscillation Index)
- **Period**: 1951-1991 (in-sample) / 1991-1996 (out-of-sample)
- **Model**: MAR(0,2) - purely noncausal
- **Pattern**: Negative bubble (anticipating El Niño warming)

### Results
- Successfully identifies the 1991-1992 El Niño event
- k₀ = 1 indicates imminent transition
- Forecast accuracy validated by subsequent observations

<div align="center">
  <img src="docs/images/elnino_forecast.png" alt="El Niño Forecast Example" width="600"/>
</div>

## 🏗️ Project Structure

```
marforecast-app/
│
├── marforecast-app.py      # Main Streamlit application
├── streamfct.py            # Helper functions for Streamlit
├── stablemar.py            # StableMAR class implementation
├── AssetsMAREstim.xlsx     # Data file with multiple datasets
│
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
└── deTruchis_Fries_Thomas_WP_2025.pdf
```

## 🔬 Technical Details

### StableMAR Class

The core `StableMAR` class provides:

```python
class StableMAR:
    """
    Mixed Autoregressive models with alpha-stable innovations
    """
    
    def __init__(self, order: Tuple[int, int]):
        """Initialize MAR(r,s) model"""
        
    def fit(self, data, start, method='gcov', K=2, H=2):
        """Estimate model parameters"""
        
    def forecast(self, x, tau, m, h, k0, vartheta):
        """Generate pattern-based forecasts"""
        
    def foreprob(self, x, k0, h, maxp, t, vartheta):
        """Generate probability-based forecasts"""
        
    def pathfinder(self, x, tau, m, kmax, vartheta):
        """Find optimal pattern index k₀"""
```

### Key Algorithms

**GCoV Estimation:**
1. Compute pseudo-residuals: `uₜ = Xₜ - ΣψᵢXₜ₋ᵢ - ΣφⱼXₜ₊ⱼ + ΣψᵢφⱼXₜ₋ᵢ₊ⱼ`
2. Apply nonlinear transformations: `εₜ,ₖ = uₜᵏ`
3. Compute autocovariance matrices Γ(h)
4. Minimize loss function L(θ)

**Pattern Recognition:**
1. Normalize observed segment: `x̃ = x / ‖x‖`
2. Normalize theoretical patterns: `d̃ₖ = δₖ / ‖δₖ‖`
3. Find k₀ minimizing: `‖x̃ - d̃ₖ‖₁`


## 🐛 Known Issues and Solutions

### Issue: "Root in unit circle"
**Solution:** Click "Estimate" multiple times until convergence

### Issue: "Model order changed"
**Solution:** Re-estimate after changing r or s values

### Issue: Forecast returns NaN
**Solution:** Ensure model is estimated and k₀ is properly selected

For more troubleshooting, see [QUICKSTART.md](docs/QUICKSTART.md).

## 📚 References

### Papers

1. **Fries, S. (2022)**  
   *Conditional Moments of Noncausal Alpha-Stable Processes and the Prediction of Bubble Crash Odds*  
   Journal of Business & Economic Statistics, 40(4), 1596–1616.

2. **Gourieroux, C., & Jasiak, J. (2023)**  
   *Generalized Covariance Estimator*  
   Journal of Business & Economic Statistics, 41(4), 1315–1327.

3. **Nolan, J. P. (2020)**  
   *Univariate Stable Distributions: Models for Heavy Tailed Data*  
   Springer Series in Operations Research and Financial Engineering.

4. **Velasco, C. (2022)**  
   *Estimation of time series models using residuals dependence measures*  
   Ann. Statist. 50(5): 3039-3063 (October 2022).

### Software

- **Streamlit**: https://streamlit.io
- **NumPy**: https://numpy.org
- **SciPy**: https://scipy.org
- **Pandas**: https://pandas.pydata.org


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Gilles de Truchis** - *Initial work* - [varennes-ecofin](https://github.com/varennes-ecofin)



## 📈 Citation

If you use this software in your research, please cite:

```bibtex
@software{marforecast2025,
  title = {Forecasting Extreme Trajectories Using Seminorm Representations},
  author = {de Truchis, Gilles and Thomas, Arthur},
  year = {2025},
  url = {https://github.com/varennes-ecofin/marforecast-app},
  version = {1.2}
}
```

---

<div align="center">
  
**Made with ❤️ for the econometrics and time series community**

[Report Bug](https://github.com/varennes-ecofin/marforecast-app/issues) · 
[Request Feature](https://github.com/varennes-ecofin/marforecast-app/issues) · 
[Documentation](docs/)

</div>
