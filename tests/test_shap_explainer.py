# tests/test_shap_explainer.py
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import RandomForestClassifier

from src.explainability.shap_explainer import (
    compute_shap_values,
    _select_class_for_binary,
    plot_global_importance,
    plot_beeswarm,
    plot_local_explanation,
    explain_model,
)


def _model_and_X():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(40, 3), columns=["a", "b", "c"])
    y = (X["a"] + rng.rand(40) * 0.1 > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=8, max_depth=3, random_state=0)
    model.fit(X, y)
    return model, X


def test_compute_shap_values_shape():
    model, X = _model_and_X()
    exp = compute_shap_values(model, X)
    assert exp.values.shape == (40, 3)


def test_compute_shap_values_subsample():
    model, X = _model_and_X()
    exp = compute_shap_values(model, X, max_samples=10)
    assert exp.values.shape[0] == 10


def test_select_class_for_binary_collapses_3d():
    values = np.random.rand(5, 3, 2)
    base = np.random.rand(5, 2)
    data = np.random.rand(5, 3)
    exp = shap.Explanation(values=values, base_values=base, data=data,
                           feature_names=["a", "b", "c"])
    out = _select_class_for_binary(exp)
    assert out.values.shape == (5, 3)
    assert out.base_values.shape == (5,)
    assert np.allclose(out.values, values[..., 1])


def test_select_class_for_binary_1d_base():
    values = np.random.rand(5, 3, 2)
    base = np.zeros(5)
    exp = shap.Explanation(values=values, base_values=base,
                           data=np.random.rand(5, 3), feature_names=["a", "b", "c"])
    out = _select_class_for_binary(exp)
    assert out.values.shape == (5, 3)
    assert out.base_values.shape == (5,)


def test_select_class_for_binary_2d_passthrough():
    values = np.random.rand(5, 3)
    exp = shap.Explanation(values=values, base_values=np.zeros(5),
                           data=np.random.rand(5, 3), feature_names=["a", "b", "c"])
    out = _select_class_for_binary(exp)
    assert out is exp


def test_plot_helpers_write_files(tmp_path):
    model, X = _model_and_X()
    exp = compute_shap_values(model, X)
    g = plot_global_importance(exp, output_path=tmp_path / "g.png")
    b = plot_beeswarm(exp, output_path=tmp_path / "b.png")
    l = plot_local_explanation(exp, row_index=0, output_path=tmp_path / "l.png")
    assert g.exists() and b.exists() and l.exists()


def test_explain_model_writes_figures(tmp_path):
    model, X = _model_and_X()
    result = explain_model(model, X, output_dir=tmp_path, model_name="rf",
                           local_indices=(0, 1))
    assert "explanation" in result
    assert len(result["figures"]) == 4
    assert all(p.exists() for p in result["figures"])


def test_explain_model_skips_out_of_range_index(tmp_path):
    model, X = _model_and_X()
    result = explain_model(model, X, output_dir=tmp_path, model_name="rf",
                           local_indices=(0, 9999))
    assert len(result["figures"]) == 3