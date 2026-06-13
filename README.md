# Aleph-Point Synthesis (APS)

## Overview
APS is a small diagnostic-test-accuracy (DTA) synthesis engine. Given study-level
2x2 counts (`tp`, `fp`, `fn`, `tn`), it derives ROC operating points, groups them
with density-based clustering, summarises each cluster as a weighted centroid, and
reconstructs a monotone summary ROC (SROC) curve to estimate the area under the
curve (AUC).

## What the code does
1. **Operating points**: Sensitivity and false-positive rate per study, with a
   0.5 continuity correction applied to all cells.
2. **Topological clustering**: DBSCAN (`eps`, `min_samples`) groups operating
   points; noise points (label `-1`) are dropped.
3. **Weighted centroids ("Aleph points")**: Each cluster is summarised by a
   Youden-index-weighted average FPR and sensitivity (weight = `max(J, 0.1) ** 3`).
4. **SROC manifold**: Centroids are interpolated (anchored at (0,0) and (1,1)) and
   forced monotone non-decreasing via `np.maximum.accumulate`.
5. **AUC**: Trapezoidal integration of the monotone manifold (numpy 1.x/2.x safe).

## Usage
```python
import pandas as pd
from engine import AlephPointSynthesisOS

df = pd.DataFrame({"tp": [80, 85], "fp": [10, 12], "fn": [20, 15], "tn": [90, 88]})
result = AlephPointSynthesisOS().synthesize(df)
print(result["auc"])
```

## Notes & limitations
- This is an exploratory clustering-based SROC estimator, not a validated
  bivariate/Reitsma model. It does not report confidence intervals.
- AUC is computed from a monotone-enforced interpolant, not a parametric SROC fit.
- An empty input raises `ValueError` (fails closed rather than producing a
  meaningless curve).

## Tests
```
python -m pytest -q
```
