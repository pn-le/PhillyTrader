# ML / Training Environment Report

Target: Agentic trading-system build on Python 3.14 (PAPER account).
Constraint checked: nothing was pip-installed. This is a snapshot of what is already present and verified working.

## Runtime

| Component | Version | Status |
|-----------|---------|--------|
| Python    | 3.14.3  | OK |
| numpy     | 2.4.6   | imports OK |
| pandas    | 3.0.3   | imports OK |
| alpaca-py | 0.43.4  | installed (per env spec) |

## ML / training libraries

### scikit-learn — WORKING
- `import sklearn` succeeds, version **1.8.0**.
- Live functional test on Python 3.14 (not just import): fit a `LogisticRegression` on tiny dummy data.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([[0.0],[1.0],[2.0],[3.0]])
y = np.array([0,0,1,1])
m = LogisticRegression().fit(X, y)
m.predict([[0.5],[2.5]])  # -> [0, 1]
```
Result: fit succeeded, predictions `[0, 1]` as expected. scikit-learn is fully usable for model training on this interpreter.

### pyarrow — WORKING
- `import pyarrow` succeeds, version **24.0.0**.
- Parquet read/write backend is available, so datasets can be stored as Parquet.

### joblib — WORKING
- `import joblib` succeeds, version **1.5.3**.
- Model persistence (`joblib.dump` / `joblib.load`) is available.

## CONCLUSION

- **MODEL_BACKEND = "sklearn"** — scikit-learn imports AND fits a LogisticRegression on Python 3.14, so the ML trainer should use scikit-learn directly. (The numpy gradient-descent fallback is NOT required.)
- **DATASET_FORMAT = "parquet"** — pyarrow 24.0.0 works, so datasets are stored as Parquet.
- **MODEL_PERSIST = "joblib"** — joblib 1.5.3 is available, so trained models are persisted with joblib (not JSON weight dumps).

All three optional dependencies are present and verified on Python 3.14; no fallback path is needed.
