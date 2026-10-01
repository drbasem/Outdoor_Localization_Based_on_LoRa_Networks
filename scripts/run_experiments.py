#!/usr/bin/env python
"""Train and evaluate the paper's models from the command line.

Examples
--------
Reproduce the RSSI-only comparison (paper Table 10, column FV1) for all seven models::

    python scripts/run_experiments.py --fv FV1

Only the fast models, on FV1 and FV2::

    python scripts/run_experiments.py --fv FV1 FV2 --models k-NN XGBoost LightGBM

Quick smoke test (few epochs, SVR on a 5,000-sample subset)::

    python scripts/run_experiments.py --fv FV1 --quick

Results are cached in results/runs/<FV>/<model>.{json,npz}; add --force to retrain.
A summary table is written to results/summary.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_loc import config, experiment, models  # noqa: E402

QUICK_KWARGS = {"CNN": {"epochs": 2}, "ANN": {"max_epochs": 2}, "Hybrid": {"epochs": 2},
                "SVR": {"max_train": 5000}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fv", nargs="+", default=["FV1"], help="feature vectors, e.g. FV1 FV2")
    ap.add_argument("--models", nargs="+", default=list(models.MODEL_NAMES))
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    ap.add_argument("--quick", action="store_true", help="tiny budgets for a smoke test")
    ap.add_argument("--runs-dir", default=None, help="where to cache results")
    ap.add_argument("--source", choices=["auto", "csv", "json"], default="auto",
                    help="auto: CSV for FV1-2, JSON otherwise; json: JSON for every FV (as in the paper)")
    args = ap.parse_args()

    default = "runs_quick" if args.quick else ("runs_json" if args.source == "json" else "runs")
    runs_dir = Path(args.runs_dir) if args.runs_dir else config.RESULTS_DIR / default
    for fv in args.fv:
        prepared = experiment.prepare_fv(fv, source=args.source)
        for name in args.models:
            kw = QUICK_KWARGS.get(name, {}) if args.quick else {}
            experiment.run(name, fv, prepared=prepared, force=args.force, runs_dir=runs_dir,
                           model_kwargs=kw, time_n_single=20 if args.quick else 200)

    table = experiment.collect(runs_dir)
    cols = ["fv", "model", "n_features", "mean_m", "median_m", "sd_m", "r2_lat", "r2_lon",
            "within_100m", "within_300m", "train_s", "infer_single_mean_s"]
    table = table[[c for c in cols if c in table.columns]]
    out = runs_dir.parent / f"summary{runs_dir.name[4:]}.csv"  # summary.csv, summary_json.csv, ...
    table.to_csv(out, index=False)
    print(table.round(3).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
