# Hybrid ARMA-TCN for WTI Crude Oil Return Forecasting

This project evaluates whether a hybrid ARMA-TCN model can outperform its standalone components — AR and TCN — in forecasting WTI crude oil daily log returns. The work is motivated by Zhang (2003) and Wang et al. (2013), who show that time series can be decomposed into linear and nonlinear components, and that a hybrid ARMA-ANN consistently outperforms either model alone. WTI crude oil is a particularly strong testbed because the literature has established that oil prices are well-described by ARMA processes, making the AR benchmark deliberately hard to beat.

The TCN is chosen over RNN-based architectures following Bai et al. (2018), who demonstrate that TCNs respect the causal structure of time series while being simpler to train and more expressive than recurrent models.

---

## Models

**AR(1) and AR(5)** — autoregressive baselines, with orders selected by AIC/BIC. Given the ARMA characterisation of oil prices in the literature, these are intentionally strong benchmarks.

**Classic TCN** — a standalone Temporal Convolutional Network with dilated causal convolutions, selected via grid search over kernel size, number of filters, number of layers, dilation base, and learning rate.

**ARMA+TCN (Additive Hybrid)** — an additive decomposition model. The optimal ARMA order is first selected by AIC/BIC and its parameters are fixed. The TCN is then trained on the residuals of the ARMA fit, and the final forecast is the sum of both components:

$$\hat{y}_t = \hat{y}_t^{\text{ARMA}} + \hat{y}_t^{\text{TCN}}$$

The TCN hyperparameters are selected via grid search conditional on the fixed ARMA component.

---

## Dataset

WTI crude oil daily spot prices (`DCOILWTICO`) from the FRED database. Prices are converted to log returns:

$$r_t = \ln(P_t) - \ln(P_{t-1})$$

WTI crude oil is a particularly strong testbed for this comparison because the literature has established that oil price dynamics are well-described by ARMA-type processes. Suleiman et al. (2023) found that Nigerian crude oil prices are best modelled by an ARIMA(3,1,1) specification, and Moshiri & Foroutan (2006) identified US crude oil prices as consistent with an ARMA(3,1) process. This makes the AR baseline deliberately hard to beat.

The data is split into 60% train, 20% validation, and 20% test. Hyperparameter selection is performed on the validation set; all reported results are on the held-out test set.

---

## Results

Forecasting performance across horizons of 1 to 200 steps ahead, evaluated on the test set.

### MSE

| Model       | 1-step | 2-step | 3-step | 5-step | 10-step | 30-step | 50-step | 100-step | 150-step | 200-step |
|-------------|--------|--------|--------|--------|---------|---------|---------|----------|----------|----------|
| AR(1)       | 0.000376 | 0.000378 | 0.000379 | 0.000382 | 0.000384 | 0.000399 | 0.000416 | 0.000369 | 0.000227 | 0.000155 |
| AR(5)       | 0.000388 | 0.000389 | 0.000384 | 0.000384 | 0.000384 | 0.000399 | 0.000416 | 0.000369 | 0.000227 | 0.000155 |
| Classic TCN | 0.000467 | 0.000467 | 0.000471 | 0.000463 | 0.000451 | 0.000473 | 0.000496 | 0.000454 | 0.000301 | 0.000252 |
| ARMA+TCN    | 0.000383 | 0.000384 | 0.000390 | 0.000380 | 0.000380 | 0.000393 | 0.000412 | 0.000369 | 0.000225 | 0.000158 |

### MAE

| Model       | 1-step | 2-step | 3-step | 5-step | 10-step | 30-step | 50-step | 100-step | 150-step | 200-step |
|-------------|--------|--------|--------|--------|---------|---------|---------|----------|----------|----------|
| AR(1)       | 0.01435 | 0.01435 | 0.01438 | 0.01442 | 0.01441 | 0.01461 | 0.01488 | 0.01395 | 0.01188 | 0.01015 |
| AR(5)       | 0.01454 | 0.01453 | 0.01447 | 0.01450 | 0.01442 | 0.01461 | 0.01488 | 0.01395 | 0.01188 | 0.01015 |
| Classic TCN | 0.01669 | 0.01666 | 0.01675 | 0.01649 | 0.01621 | 0.01663 | 0.01697 | 0.01602 | 0.01428 | 0.01349 |
| ARMA+TCN    | 0.01452 | 0.01453 | 0.01468 | 0.01443 | 0.01443 | 0.01456 | 0.01485 | 0.01403 | 0.01204 | 0.01043 |

The ARMA+TCN hybrid consistently outperforms the standalone TCN across all horizons. Against the AR benchmarks the picture is more mixed — the hybrid matches or improves on AR(1) at multiple horizons, particularly beyond 100 steps, but does not dominate across the board.

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