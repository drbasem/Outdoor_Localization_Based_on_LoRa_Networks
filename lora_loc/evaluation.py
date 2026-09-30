"""Metrics used in the paper (Section V-H).

All distance metrics are computed in **metres**, after converting the scaled predictions
back to degrees:

* **Mean error**: the headline number in Tables 10, 11 and 13. Table 13 labels this
  column "MSE", but the values are mean distances in metres.
* **Median error (ME)** and **standard deviation (SD)** of the per-sample errors. The
  paper's Eq. (20) is the population standard deviation around the mean error.
* **CDF**: the share of test samples whose error is within 50, 100, ..., 300 m (Table 12).
* **R^2** per coordinate on the scaled targets, which is what the grid-search tables report.
* **MSE** on the scaled targets (the training loss).
* **Inference time** per sample (Fig. 5), measured two ways: (a) one batched
  ``predict`` call divided by n, and (b) ``predict`` on one sample at a time, averaged over
  a subset. The paper timed "each sample in the test set", which corresponds to (b).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from . import config
from .geo import distance_errors_m


def error_summary(errors_m: np.ndarray, thresholds=config.CDF_THRESHOLDS_M) -> dict:
    """Mean / median / SD / max of per-sample errors (m), plus CDF points."""
    e = np.asarray(errors_m)
    out = {
        "mean_m": float(e.mean()),
        "median_m": float(np.median(e)),
        "sd_m": float(e.std()),  # population SD, paper Eq. (20)
        "p90_m": float(np.percentile(e, 90)),
        "max_m": float(e.max()),
    }
    for t in thresholds:
        out[f"within_{t}m"] = float((e <= t).mean())
    return out


def evaluate(y_true_scaled, y_pred_scaled, y_true_deg, pipeline) -> tuple[dict, np.ndarray]:
    """Return ``(metrics, per-sample errors in metres)``."""
    y_pred_deg = pipeline.unscale_y(y_pred_scaled)
    err = distance_errors_m(y_true_deg, y_pred_deg)
    m = error_summary(err)
    m["r2_lat"] = float(r2_score(y_true_scaled[:, 0], y_pred_scaled[:, 0]))
    m["r2_lon"] = float(r2_score(y_true_scaled[:, 1], y_pred_scaled[:, 1]))
    m["mse_scaled"] = float(mean_squared_error(y_true_scaled, y_pred_scaled))
    return m, err


def time_inference(model, X: np.ndarray, n_single: int = 200, seed: int = config.SEED) -> dict:
    """Measure per-sample inference time (s): batched, and one sample at a time."""
    t0 = time.perf_counter()
    model.predict(X)
    batched = (time.perf_counter() - t0) / len(X)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_single, len(X)), replace=False)
    model.predict(X[idx[:1]])  # warm-up (graph tracing etc.)
    times = []
    for i in idx:
        t0 = time.perf_counter()
        model.predict(X[i:i + 1])
        times.append(time.perf_counter() - t0)
    return {"infer_batched_s": batched,
            "infer_single_mean_s": float(np.mean(times)),
            "infer_single_sd_s": float(np.std(times))}
