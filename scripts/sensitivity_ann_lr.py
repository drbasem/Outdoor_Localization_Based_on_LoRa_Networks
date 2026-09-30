#!/usr/bin/env python
"""Sensitivity of the ANN to its learning rate, which the paper does not report.

The main results use lr = 1e-4 (authors' supplementary notebook). This script retrains the
same architecture with other learning rates on FV1 and writes results/sensitivity_ann_lr.csv.
It does not change the main results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from lora_loc import config, experiment  # noqa: E402

rows = []
d = experiment.prepare_fv("FV1")
for lr in (1e-4, 3e-4, 1e-3):
    config.ANN["learning_rate"] = lr
    r = experiment.run("ANN", "FV1", prepared=d, runs_dir=config.RESULTS_DIR / f"sensitivity/ann_lr_{lr:g}",
                       time_n_single=0)
    m = r["metrics"]
    rows.append({"learning_rate": lr, "mean_m": m["mean_m"], "median_m": m["median_m"],
                 "sd_m": m["sd_m"], "train_s": m["train_s"]})
out = pd.DataFrame(rows)
out.to_csv(config.RESULTS_DIR / "sensitivity_ann_lr.csv", index=False)
print(out.round({"mean_m": 2, "median_m": 2, "sd_m": 2, "train_s": 0}).to_string(index=False))
