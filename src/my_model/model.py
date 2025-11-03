# src/my_model/model.py
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

EARLY_PATH = "/Users/binish/Desktop/Arrhythmia_LVAD/Notebooks_wo engin/xgbearlyVA_survival_imb_ex.pkl"
LATE_PATH  = "/Users/binish/Desktop/Arrhythmia_LVAD/Notebooks_wo engin/xgblateVA_survival_imb.pkl"

def _require_exists(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return p

def _ensure_df(sample) -> pd.DataFrame:
    # Accept dict (single row), pandas Series, or DataFrame
    if isinstance(sample, pd.DataFrame):
        return sample
    if isinstance(sample, pd.Series):
        return sample.to_frame().T
    if isinstance(sample, dict):
        return pd.DataFrame([sample])
    raise TypeError("sample must be a dict, pandas Series, or DataFrame")

@lru_cache(maxsize=1)
def load_early():
    return joblib.load(_require_exists(EARLY_PATH))

@lru_cache(maxsize=1)
def load_late():
    return joblib.load(_require_exists(LATE_PATH))

def feature_names(which: str = "early"):
    m = load_early() if which == "early" else load_late()
    return getattr(m, "feature_names_in_", None)

def predict(sample, which: str = "early"):
    """
    sample: dict/Series/DataFrame with the SAME columns used at training.
    which:  'early' or 'late'
    Returns: numpy array of predictions or probabilities.
    """
    m = load_early() if which == "early" else load_late()
    X = _ensure_df(sample)

    # Align columns if model exposes feature_names_in_
    feats = getattr(m, "feature_names_in_", None)
    if feats is not None:
        X = X.reindex(columns=feats)  # will introduce NaN if missing

    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)      # classifiers
    return m.predict(X)                # regressors
