"""
Within-fold Random Forest feature selection for risk and protective factors.

Feature selection is performed strictly on training data within each CV fold.
Selected factor indices are then applied to the corresponding validation and
test splits, ensuring no information leakage from held-out data.
"""

import logging
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def select_factors_rf(
    train_df,
    label_col: str,
    n_risk: int = 4,
    n_protective: int = 4,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Select top risk and protective factors using Random Forest feature importance.

    Performed strictly on training fold data to prevent information leakage
    into validation or test splits.

    Args:
        train_df: Training fold DataFrame with 'cur_bp_y' and 'cur_bp_res'.
        label_col: Column name of the target label.
        n_risk: Number of top risk factors to select.
        n_protective: Number of top protective factors to select.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (risk_indices, protective_indices) — sorted lists of selected
        column indices into the original factor arrays.
    """
    # Aggregate per-post factor arrays by mean across the time-window dimension.
    # cur_bp_y per sample: list/array of shape (window_size, n_risk_factors)
    risk_matrix = np.stack([
        np.mean(np.asarray(x), axis=0) for x in train_df['cur_bp_y'].values
    ])  # (n_samples, n_risk_factors)

    protective_matrix = np.stack([
        np.mean(np.asarray(x), axis=0) for x in train_df['cur_bp_res'].values
    ])  # (n_samples, n_protective_factors)

    labels = train_df[label_col].values

    risk_indices = _top_rf_indices(risk_matrix, labels, n_risk, seed)
    protective_indices = _top_rf_indices(protective_matrix, labels, n_protective, seed)

    logger.info(
        "RF feature selection (fold): risk_indices=%s, protective_indices=%s",
        risk_indices,
        protective_indices,
    )
    return risk_indices, protective_indices


def _top_rf_indices(
    features: np.ndarray,
    labels: np.ndarray,
    n_top: int,
    seed: int,
) -> List[int]:
    """Return sorted indices of the top-n_top features by RF importance."""
    n_total = features.shape[1]
    if n_top >= n_total:
        return list(range(n_total))

    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(features, labels)
    ranked = np.argsort(rf.feature_importances_)[::-1]
    return sorted(ranked[:n_top].tolist())


def apply_factor_selection(
    df,
    risk_indices: List[int],
    protective_indices: List[int],
):
    """
    Subset factor arrays to the selected column indices.

    Applies the factor indices learned from the training fold to any split
    (train, val, or test) without re-fitting.
    """
    df = df.copy()
    df['cur_bp_y'] = df['cur_bp_y'].apply(
        lambda x: np.asarray(x)[:, risk_indices]
    )
    df['cur_bp_res'] = df['cur_bp_res'].apply(
        lambda x: np.asarray(x)[:, protective_indices]
    )
    return df
