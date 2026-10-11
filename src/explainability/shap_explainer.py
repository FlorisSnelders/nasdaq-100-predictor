"""
src/explainability/shap_explainer.py
====================================

Model-agnostic SHAP explainer for tree-based bakeoff models.

Design goal (per the week-3 plan): the same code that explains the
LightGBM entry this week should explain the bakeoff winner next week
with a one-line change. To make that possible, this module takes a
*fitted model* and a *feature DataFrame* and does not care which
library produced the model, as long as shap.TreeExplainer accepts it
(LightGBM, XGBoost, CatBoost, sklearn GBM/RF all work).

Public API
----------
explain_model(model, X, feature_names=None, output_dir, ...)
    One-call entry point. Produces:
      - global_importance.png  (bar chart, mean |SHAP|)
      - beeswarm.png           (per-feature impact distribution)
      - local_<i>.png          (one or two waterfall plots)
    Returns the computed shap.Explanation object so callers can do
    further analysis without recomputing.

Lower-level helpers (compute_shap_values, plot_*, etc.) are also
exposed so notebooks can do custom analysis.

Why no class
------------
The whole job is "given a model and data, write figures to disk."
A function with keyword args is simpler than a class and easier
to drop into different notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib
# Use a non-interactive backend so figures save cleanly even when this
# runs from a headless context (CI, batch jobs).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ----------------------- core computation --------------------------------

def compute_shap_values(
    model,
    X: pd.DataFrame,
    *,
    max_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> shap.Explanation:
    """
    Compute SHAP values for a tree-based model.

    Parameters
    ----------
    model : any tree model accepted by shap.TreeExplainer
        (LightGBM, XGBoost, CatBoost, sklearn GBM/RF).
    X : pd.DataFrame
        Feature matrix. Column names are used as feature names in plots.
        Must have the same columns the model was trained on, in the
        same order.
    max_samples : int or None
        For large X, sub-sample to keep beeswarm/global plots tractable.
        Set to None to use all rows. Beeswarm on >10k points is slow
        and visually mostly noise; 5k is a reasonable default.
    random_state : int
        Used only for the sub-sample.

    Returns
    -------
    shap.Explanation
        Has .values, .base_values, .data, .feature_names. Pass this
        to the plot_* helpers below or to shap's own plotting API.

    Notes
    -----
    For binary classifiers, LightGBM/XGBoost typically return SHAP values
    for the positive class directly. For multi-class, shap returns a list
    of arrays (one per class) and the helpers below pick class 1 by
    default — see _select_class_for_binary.
    """
    if max_samples is not None and len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=random_state)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)

    # Normalize the binary-classification case: some model types return
    # an Explanation of shape (n, n_features, 2) where the last axis is
    # the class. Collapse to the positive class so downstream code is
    # uniform.
    explanation = _select_class_for_binary(explanation)
    return explanation


def _select_class_for_binary(explanation: shap.Explanation) -> shap.Explanation:
    """If explanation has a class axis with size 2, keep only class 1."""
    values = explanation.values
    if values.ndim == 3 and values.shape[-1] == 2:
        # Build a new Explanation with the class axis sliced out
        return shap.Explanation(
            values=values[..., 1],
            base_values=(
                explanation.base_values[..., 1]
                if np.ndim(explanation.base_values) > 1
                else explanation.base_values
            ),
            data=explanation.data,
            feature_names=explanation.feature_names,
        )
    return explanation


# ----------------------- plotting helpers --------------------------------

def plot_global_importance(
    explanation: shap.Explanation,
    *,
    output_path: Path,
    max_display: int = 20,
    title: str = "Global feature importance (mean |SHAP|)",
) -> Path:
    """Bar chart of mean(|SHAP value|) per feature. Saved to output_path."""
    plt.figure(figsize=(8, max(4, 0.35 * min(max_display, len(explanation.feature_names)))))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_beeswarm(
    explanation: shap.Explanation,
    *,
    output_path: Path,
    max_display: int = 20,
    title: str = "SHAP beeswarm: per-feature impact distribution",
) -> Path:
    """
    Beeswarm plot: each dot is one prediction, colour is feature value.
    Shows distribution of SHAP impact + the value-to-impact relationship.
    """
    plt.figure(figsize=(8, max(4, 0.35 * min(max_display, len(explanation.feature_names)))))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_local_explanation(
    explanation: shap.Explanation,
    *,
    row_index: int,
    output_path: Path,
    max_display: int = 15,
    title: Optional[str] = None,
) -> Path:
    """
    Waterfall plot for a single row, showing which features pushed
    the prediction up or down from the base value.

    row_index is the positional index into the Explanation, not the
    original DataFrame index.
    """
    plt.figure(figsize=(8, max(4, 0.4 * max_display)))
    shap.plots.waterfall(explanation[row_index], max_display=max_display, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ----------------------- one-call entry point ----------------------------

def explain_model(
    model,
    X: pd.DataFrame,
    *,
    output_dir: Union[str, Path],
    model_name: str = "model",
    local_indices: Sequence[int] = (0, 1),
    max_samples: Optional[int] = 5000,
    max_display: int = 20,
    random_state: int = 42,
) -> dict:
    """
    Run the full SHAP analysis and write all figures to output_dir.

    This is the function notebooks should call. Next week, when the
    bakeoff winner is chosen, the only change needed is the `model`
    argument and (optionally) `model_name`.

    Parameters
    ----------
    model : fitted tree model (LightGBM, XGBoost, CatBoost, sklearn).
    X : feature matrix the SHAP analysis runs on. Typically the test
        fold from the bakeoff, or all rows pooled.
    output_dir : where to write figures. Created if missing.
    model_name : used as a filename prefix, e.g. "lightgbm" produces
        lightgbm_global_importance.png, lightgbm_beeswarm.png, etc.
    local_indices : which rows of X to produce waterfall plots for.
        Defaults to the first two rows. Pass concrete indices to
        showcase a specific positive and negative prediction.
    max_samples : sub-sample size for global plots. See compute_shap_values.
    max_display : how many features to show on each plot.
    random_state : sub-sample seed.

    Returns
    -------
    dict with keys:
        'explanation' : the shap.Explanation object
        'figures'     : list of Path objects, one per saved figure
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    explanation = compute_shap_values(
        model, X, max_samples=max_samples, random_state=random_state,
    )

    figures = []

    figures.append(plot_global_importance(
        explanation,
        output_path=out / f"{model_name}_global_importance.png",
        max_display=max_display,
        title=f"{model_name}: global feature importance (mean |SHAP|)",
    ))

    figures.append(plot_beeswarm(
        explanation,
        output_path=out / f"{model_name}_beeswarm.png",
        max_display=max_display,
        title=f"{model_name}: SHAP beeswarm",
    ))

    for i in local_indices:
        if i < 0 or i >= len(explanation):
            # Skip silently if the caller passed an out-of-range index;
            # better than crashing partway through the analysis.
            continue
        figures.append(plot_local_explanation(
            explanation,
            row_index=int(i),
            output_path=out / f"{model_name}_local_row{int(i)}.png",
            max_display=min(max_display, 15),
            title=f"{model_name}: local explanation for row {int(i)}",
        ))

    return {"explanation": explanation, "figures": figures}
