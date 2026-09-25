import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple

def double_logistic_model(t: np.ndarray, base: float, amp: float, m_up: float, s_up: float, m_down: float, s_down: float) -> np.ndarray:
    """Parametric curve modeling vegetation growth and senescence cycles over day-of-year."""
    growth = 1.0 / (1.0 + np.exp(-s_up * (t - m_up)))
    senescence = 1.0 / (1.0 + np.exp(s_down * (t - m_down)))
    return base + amp * (growth - senescence)

def extract_phenology_metrics(days: np.ndarray, ndvi_series: np.ndarray) -> np.ndarray:
    """Fits phenological function to a time-series and returns extracted functional features.
    
    Returns:
        np.ndarray: [base, amplitude, greenup_doy, greenup_slope, maturity_doy, senescence_slope]
    """
    p0 = [0.1, 0.7, 110, 0.05, 260, 0.05]
    bounds = ([0.0, 0.0, 40, 0.001, 150, 0.001], [0.4, 1.0, 200, 0.5, 365, 0.5])
    try:
        popt, _ = curve_fit(double_logistic_model, days, ndvi_series, p0=p0, bounds=bounds, maxfev=3000)
        return popt
    except (RuntimeError, ValueError):
        return np.array([np.nanmin(ndvi_series), np.nanmax(ndvi_series) - np.nanmin(ndvi_series), 120, 0.05, 270, 0.05])
