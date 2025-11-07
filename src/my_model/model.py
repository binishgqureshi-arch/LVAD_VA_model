# src/my_model/model.py
from pathlib import Path
import joblib
import pandas as pd

__all__ = ["feature_names", "predict", "load_early", "load_late"]

# model files live inside the repo under models/
EARLY_PATH = Path(__file__).resolve().parents[2] / "models" / "xgbearlyVA_survival_imb_ex.pkl"
LATE_PATH  = Path(__file__).resolve().parents[2] / "models" / "xgblateVA_survival_imb.pkl"

def load_early():
    return joblib.load(EARLY_PATH)

def load_late():
    return joblib.load(LATE_PATH)

def feature_names(which: str = "early"):
    m = load_early() if which == "early" else load_late()
    feats = getattr(m, "feature_names_in_", None)
    if feats is None:
        return None
    return list(feats)

def _to_dataframe(sample) -> pd.DataFrame:
    if isinstance(sample, pd.DataFrame):
        return sample
    if hasattr(sample, "to_frame"):
        return sample.to_frame().T
    if isinstance(sample, dict):
        return pd.DataFrame([sample])
    raise TypeError("sample must be a dict/Series/DataFrame")

def predict(sample, which: str = "early"):
    """Auto-align columns to training schema and fill missing with 0."""
    m = load_early() if which == "early" else load_late()
    X = _to_dataframe(sample)
    feats = feature_names(which)
    if feats is not None:
        X = X.reindex(columns=feats, fill_value=0)
    return m.predict(X)
