# Satellite Time-Series Crop Yield & Phenology Prediction

Machine learning pipeline that extracts temporal phenological signatures from multispectral satellite time-series (Sentinel-2 / Landsat) to forecast crop yield with spatial autocorrelation safeguards.

## Key Features
- **Curve-Fitting Phenology Engine:** Parametric double-logistic algorithm extracting peak greenness, seasonal slope, and maturity DOY.
- **Spatial Block Cross-Validation:** Grid-blocked spatial CV avoiding data leakage between adjacent agricultural parcels.
- **Ensemble Regression:** Out-of-fold spatial evaluation reporting spatially honest RMSE and R² metrics.

## Quickstart
```bash
pip install -r requirements.txt
python -m src.pipeline.train_yield
