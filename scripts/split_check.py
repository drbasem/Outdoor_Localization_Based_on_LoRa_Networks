#!/usr/bin/env python
"""How much of the accuracy survives stricter train/test splits?

The paper splits messages at random. Consecutive messages from the same vehicle, sent
seconds apart from almost the same place, then land in both train and test. Features that
identify "when" and "which device" -- above all the frame counter (CNT) -- can exploit this.
This script retrains LightGBM (the model that reproduces the paper most closely on all 16
FVs) under three protocols and writes results/split_check.csv:

* random        -- the paper's protocol (70/15/15, seed 42)
* chronological -- train on the earliest 70 % of messages, test on the latest 15 %
* device        -- test vehicles are never seen in training

Needs the JSON release (data/lorawan_antwerp_2019_dataset.json.txt).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from lora_loc import config, experiment, features, preprocessing as pp  # noqa: E402

FVS = ["FV1", "FV2", "FV4", "FV9", "FV13", "FV15", "FV16"]
MODEL = "LightGBM"

df, rssi = experiment.load_source("FV1", source="json")
splits = {
    "random": pp.split_indices(len(df)),
    "chronological": pp.chronological_split(df["RxTime"]),
    "device": pp.device_split(df["DevEUI"]),
}
print("devices:", df["DevEUI"].nunique(),
      "| test messages per protocol:", {k: len(v[2]) for k, v in splits.items()})
rows = []
for fv in FVS:
    r, o = features.columns_for(fv, df, rssi)
    for name, split in splits.items():
        d = pp.prepare(df, r, o, split=split)
        res = experiment.run(MODEL, fv, prepared=d, runs_dir=config.RESULTS_DIR / f"split_check/{name}",
                             time_n_single=0)
        m = res["metrics"]
        rows.append({"fv": fv, "split": name, "mean_m": m["mean_m"], "median_m": m["median_m"],
                     "within_100m": m["within_100m"]})
out = pd.DataFrame(rows)
out.to_csv(config.RESULTS_DIR / "split_check.csv", index=False)
print(out.pivot(index="fv", columns="split", values="mean_m").loc[FVS].round(1).to_string())
