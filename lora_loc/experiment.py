"""Run one (model, feature vector) experiment end to end, with on-disk caching.

Training all seven models on 91k samples takes a while on a CPU (SVR and the CNNs take
the longest). Each result is therefore saved under ``results/runs/<FV>/<model>.npz`` and
``.json``. Running the same experiment again loads the cached result, unless
``force=True``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, evaluation, features, models, preprocessing

RUNS_DIR = config.RESULTS_DIR / "runs"


def _slug(name: str) -> str:
    return name.replace("-", "").replace(" ", "_").lower()


_JSON_CACHE: dict = {}


def load_source(fv: str, source: str = "auto") -> tuple[pd.DataFrame, list[str]]:
    """Return the DataFrame and RSSI columns needed for ``fv``.

    ``source="auto"``: FV1 and FV2 use the CSV release; FV3-FV16 use the JSON release.
    ``source="json"``: every FV uses the JSON release, as the paper did. The JSON has
    130,430 messages (one more than the CSV) in a different row order, so the random split
    differs from the CSV-based one.
    """
    if source == "csv" or (source == "auto" and fv in features.CSV_FEATURE_VECTORS):
        df = data.load_csv()
        return df, data.active_gateways(df)
    if "df" not in _JSON_CACHE:
        _JSON_CACHE["df"] = data.load_json()
    df = _JSON_CACHE["df"]
    return df, [c for c in df.columns if c.startswith("RSSI_")]


def prepare_fv(fv: str, df: pd.DataFrame | None = None, rssi_cols: list[str] | None = None,
               source: str = "auto"):
    """Build the scaled train/val/test splits for feature vector ``fv``."""
    if df is None:
        df, rssi_cols = load_source(fv, source)
    rssi, other = features.columns_for(fv, df, rssi_cols)
    return preprocessing.prepare(df, rssi, other)


def run(model_name: str, fv: str = "FV1", prepared=None, force: bool = False,
        runs_dir: Path = RUNS_DIR, model_kwargs: dict | None = None, time_n_single: int = 200,
        verbose: bool = True) -> dict:
    """Train ``model_name`` on ``fv``, evaluate it on the test split and cache the result.

    Returns a dict with ``metrics`` (see :mod:`lora_loc.evaluation`), per-sample test
    ``errors_m``, the predicted and true coordinates in degrees, and ``train_s``.
    """
    out_dir = Path(runs_dir) / fv
    stem = out_dir / _slug(model_name)
    if not force and stem.with_suffix(".json").exists() and stem.with_suffix(".npz").exists():
        arr = np.load(stem.with_suffix(".npz"))
        meta = json.loads(stem.with_suffix(".json").read_text())
        if verbose:
            print(f"[cache] {fv} {model_name}: mean {meta['metrics']['mean_m']:.2f} m")
        return {**meta, **{k: arr[k] for k in arr.files}}

    d = prepared if prepared is not None else prepare_fv(fv)
    model = models.build(model_name, **(model_kwargs or {}))
    if verbose:
        print(f"[train] {fv} {model_name} on {d.X_train.shape} ...", flush=True)
    t0 = time.perf_counter()
    model.fit(d.X_train, d.y_train, d.X_val, d.y_val)
    train_s = time.perf_counter() - t0

    y_pred = model.predict(d.X_test)
    metrics, err = evaluation.evaluate(d.y_test, y_pred, d.y_test_deg, d.pipeline)
    if time_n_single:
        metrics.update(evaluation.time_inference(model, d.X_test, n_single=time_n_single))
    metrics["train_s"] = train_s
    metrics["n_features"] = int(d.X_train.shape[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred_deg = d.pipeline.unscale_y(y_pred)
    np.savez_compressed(stem.with_suffix(".npz"), errors_m=err, y_pred_deg=y_pred_deg,
                        y_true_deg=d.y_test_deg, idx_test=d.idx_test)
    meta = {"model": model_name, "fv": fv, "metrics": metrics,
            "kwargs": {k: str(v) for k, v in (model_kwargs or {}).items()}}
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"[done]  {fv} {model_name}: mean {metrics['mean_m']:.2f} m, "
              f"median {metrics['median_m']:.2f} m, train {train_s:.0f} s", flush=True)
    return {**meta, "errors_m": err, "y_pred_deg": y_pred_deg, "y_true_deg": d.y_test_deg,
            "idx_test": d.idx_test}


def collect(runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Gather every cached result into one tidy DataFrame (one row per model x FV)."""
    rows = []
    for f in sorted(Path(runs_dir).glob("*/*.json")):
        meta = json.loads(f.read_text())
        rows.append({"fv": meta["fv"], "model": meta["model"], **meta["metrics"]})
    return pd.DataFrame(rows)
