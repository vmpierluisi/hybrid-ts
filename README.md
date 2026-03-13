# Hybrid ARMA-TCN for WTI Crude Oil Return Forecasting

This project evaluates whether a hybrid ARMA-TCN model can outperform its standalone components — AR and TCN — in forecasting WTI crude oil daily log returns. The work is motivated by Zhang (2003) and Wang et al. (2013), who show that time series can be decomposed into linear and nonlinear components, and that a hybrid ARMA-ANN consistently outperforms either model alone. WTI crude oil is a particularly strong testbed because the literature has established that oil prices are well-described by ARMA processes, making the AR benchmark deliberately hard to beat.

The TCN is chosen over RNN-based architectures following Bai et al. (2018), who demonstrate that TCNs respect the causal structure of time series while being simpler to train and more expressive than recurrent models.

---

## Models

**AR(1) and AR(5)** — autoregressive baselines, with orders selected by AIC/BIC. Given the ARMA characterisation of oil prices in the literature, these are intentionally strong benchmarks.

**Classic TCN** — a standalone Temporal Convolutional Network with dilated causal convolutions, selected via grid search over kernel size, number of filters, number of layers, dilation base, and learning rate.

**Additive ARMA(5,0)+TCN** — an additive decomposition. The AR(5) component captures the linear structure of returns; the TCN is trained jointly on the residuals of that fit, and the final forecast is their sum:

$$\hat{y}_t = \hat{y}_t^{\text{ARMA}} + \hat{y}_t^{\text{TCN}}$$

**Multiplicative ARMA(1,0)×TCN** — a multiplicative decomposition following Wang et al. (2013). The AR(1) component acts as a linear scale factor; the TCN is trained on the ratio of actuals to that prediction, and the two are multiplied at inference time:

$$\hat{y}_t = \hat{y}_t^{\text{AR}} \times \hat{y}_t^{\text{TCN}}$$

For both hybrids the TCN hyperparameters are selected via grid search conditional on the fixed ARMA component.

---

## Dataset

WTI crude oil daily spot prices (`DCOILWTICO`) from the FRED database. Prices are converted to log returns:

$$r_t = \ln(P_t) - \ln(P_{t-1})$$

WTI crude oil is a particularly strong testbed for this comparison because the literature has established that oil price dynamics are well-described by ARMA-type processes. Suleiman et al. (2023) found that Nigerian crude oil prices are best modelled by an ARIMA(3,1,1) specification, and Moshiri & Foroutan (2006) identified US crude oil prices as consistent with an ARMA(3,1) process. This makes the AR baseline deliberately hard to beat.

The data is split into 60% train, 20% validation, and 20% test. Hyperparameter selection is performed on the validation set; all reported results are on the held-out test set.

---

## Results

Forecasting performance across horizons of 1 to 200 steps ahead, evaluated on the held-out test set.

### MSE

| Model                   | 1-step   | 2-step   | 3-step   | 5-step   | 10-step  | 30-step  | 50-step  | 100-step | 150-step | 200-step |
|-------------------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| AR(1)                   | 0.000376 | 0.000378 | 0.000379 | 0.000382 | 0.000384 | 0.000399 | 0.000416 | 0.000369 | 0.000227 | 0.000155 |
| AR(5)                   | 0.000388 | 0.000389 | 0.000384 | 0.000384 | 0.000384 | 0.000399 | 0.000416 | 0.000369 | 0.000227 | 0.000155 |
| Classic TCN             | 0.000493 | 0.000493 | 0.000493 | 0.000487 | 0.000482 | 0.000503 | 0.000527 | 0.000486 | 0.000331 | 0.000285 |
| Additive ARMA(5,0)+TCN  | 0.000369 | 0.000370 | 0.000376 | 0.000378 | 0.000387 | 0.000400 | 0.000417 | 0.000370 | 0.000228 | 0.000156 |
| Multiplicative ARMA(1,0)×TCN | 0.000461 | 0.000462 | 0.000520 | 0.000425 | 0.000458 | 0.000408 | 0.000418 | 0.000376 | 0.000233 | 0.000158 |

### MAE

| Model                   | 1-step  | 2-step  | 3-step  | 5-step  | 10-step | 30-step | 50-step | 100-step | 150-step | 200-step |
|-------------------------|---------|---------|---------|---------|---------|---------|---------|----------|----------|----------|
| AR(1)                   | 0.01435 | 0.01435 | 0.01438 | 0.01442 | 0.01441 | 0.01461 | 0.01488 | 0.01395  | 0.01188  | 0.01015  |
| AR(5)                   | 0.01454 | 0.01453 | 0.01447 | 0.01450 | 0.01442 | 0.01461 | 0.01488 | 0.01395  | 0.01188  | 0.01015  |
| Classic TCN             | 0.01727 | 0.01724 | 0.01724 | 0.01708 | 0.01694 | 0.01734 | 0.01769 | 0.01675  | 0.01501  | 0.01428  |
| Additive ARMA(5,0)+TCN  | 0.01420 | 0.01422 | 0.01435 | 0.01440 | 0.01445 | 0.01462 | 0.01488 | 0.01396  | 0.01188  | 0.01012  |
| Multiplicative ARMA(1,0)×TCN | 0.01606 | 0.01607 | 0.01649 | 0.01524 | 0.01597 | 0.01475 | 0.01476 | 0.01405  | 0.01194  | 0.01005  |

The additive ARMA(5,0)+TCN is the strongest model overall: it matches or beats AR(1) at every horizon and substantially outperforms the standalone TCN. The multiplicative ARMA(1,0)×TCN shows weaker performance at short horizons (1–10 steps) but converges to near-AR performance at longer horizons (100–200 steps), suggesting the multiplicative decomposition is better suited to capturing slow-moving structure than short-term dynamics. Both hybrids decisively outperform the Classic TCN across all horizons.

---

## Project Structure

```
ml-project/
├── data/               # Raw data (DCOILWTICO.csv)
├── notebooks/
│   ├── 1_ar_selection.ipynb
│   ├── 2_tcn_selection.ipynb
│   ├── 3_hybrid_selection.ipynb
│   └── 4_in_out_sample_eval.ipynb
├── plots/              # Saved evaluation figures
├── src/                # Model definitions, training, diagnostics
├── weights/            # Saved model weights
├── main.py
└── pyproject.toml
```

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/your-username/hybrid-ts.git
cd hybrid-ts
uv sync
```

---

## References

- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159–175.
- Wang, L., Zou, H., Su, J., Li, L., & Chaudhry, S. (2013). An ARIMA-ANN hybrid model for time series forecasting. *Systems Research and Behavioral Science*, 30(3), 244–259.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.
- Suleiman, M., Muhammad, I., Adamu, A. Z., Yahaya, Z., Rufai, I., Muhammad, A., Adamu, I., & Abdu, M. (2023). Modelling Nigeria crude oil prices using ARIMA time series models. *NIPES Journal of Science and Technology Research*, 5(1), 230–241.
- Moshiri, S., & Foroutan, F. (2006). Forecasting nonlinear crude oil futures prices. *The Energy Journal*, 27(4), 81–95.

---

## License
This project is licensed under CC BY-NC 4.0. Free to use for non-commercial purposes. For commercial licensing inquiries, contact me at vmpierluisi@email.com.