"""Prediction interface for brand loyalty.

Exports:
    predict_loyalty(input_dict)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .config import CFG
from .data import canonicalize_features


def _validate_and_build_input_df(input_dict: Dict[str, Any]) -> pd.DataFrame:
    expected = set(CFG.all_feature_columns)
    missing = expected.difference(set(input_dict.keys()))
    extra = set(input_dict.keys()).difference(expected)
    if missing:
        raise ValueError(f"input_dict missing required keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"input_dict has unexpected keys: {sorted(extra)}")
    df = pd.DataFrame([input_dict], columns=CFG.all_feature_columns)
    return df


def predict_loyalty(
    input_dict: Dict[str, Any],
    *,
    model_path: str | Path = Path("artifacts") / "models" / "random_forest_tuned.joblib",
) -> str:
    """Predict brand loyalty for a single college student profile.

    Args:
        input_dict: feature values keyed by the dataset column names:
            - Brand
            - Usage Duration
            - Experience
            - Next Purchase Decision
            - Discount Influence
            - Peer Influence
            - Decision Factor
            - Social Engagement
            - Price Importance
          (Timestamp/Email are not required.)
        model_path: path to the saved RF pipeline artifact.

    Returns:
        One of: "High", "Medium", "Low"
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    pipeline = joblib.load(model_path)
    df_raw = _validate_and_build_input_df(input_dict)
    df_canonical = canonicalize_features(df_raw)
    X = df_canonical[CFG.all_feature_columns]

    pred_id = int(pipeline.predict(X)[0])
    if set(getattr(pipeline.named_steps["model"], "classes_", [])) <= {0, 1}:
        return CFG.loyalty_binary_id_to_name[pred_id]
    return CFG.loyalty_id_to_name[pred_id]

