"""Tests for AlephPointSynthesisOS (DTA SROC/AUC synthesis engine).

Covers the engine's mathematical invariants and the numpy>=2.0 `trapz`->
`trapezoid` regression (np.trapz was removed in numpy 2.x, which silently
broke `synthesize()` on this shipped repo until fixed).
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import AlephPointSynthesisOS  # noqa: E402


def _good_df(n=6):
    """A clearly high-accuracy diagnostic cluster (high sens, low FPR)."""
    return pd.DataFrame([{"tp": 90, "fp": 8, "fn": 10, "tn": 92}] * n)


def test_synthesize_runs_and_returns_schema():
    r = AlephPointSynthesisOS().synthesize(_good_df())
    assert set(r) >= {"auc", "aleph_points", "manifold", "stability_index"}
    assert isinstance(r["aleph_points"], list)
    assert set(r["manifold"]) == {"x", "y"}


def test_auc_is_a_probability():
    r = AlephPointSynthesisOS().synthesize(_good_df())
    assert 0.0 <= r["auc"] <= 1.0


def test_high_accuracy_data_gives_high_auc():
    # Tight cluster of near-perfect operating points -> AUC should be high.
    r = AlephPointSynthesisOS().synthesize(_good_df())
    assert r["auc"] > 0.8, f"expected high AUC for near-perfect data, got {r['auc']}"


def test_manifold_is_monotone_nondecreasing():
    # The ROC manifold is enforced monotone via np.maximum.accumulate.
    y = AlephPointSynthesisOS().synthesize(_good_df())["manifold"]["y"]
    assert all(y[i] <= y[i + 1] + 1e-12 for i in range(len(y) - 1))


def test_aleph_points_are_in_unit_square():
    pts = AlephPointSynthesisOS().synthesize(_good_df())["aleph_points"]
    for p in pts:
        assert 0.0 <= p["fpr"] <= 1.0
        assert 0.0 <= p["sens"] <= 1.0
        assert p["weight"] > 0


def test_numpy2_trapz_shim_resolves():
    # Regression: np.trapz removed in numpy 2.x. The engine must still expose
    # a working trapezoid integrator.
    import numpy as np

    from engine import _trapz
    assert _trapz is not None
    # area under y=1 over [0,1] is 1.0
    x = np.linspace(0, 1, 50)
    assert abs(float(_trapz(np.ones_like(x), x)) - 1.0) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
