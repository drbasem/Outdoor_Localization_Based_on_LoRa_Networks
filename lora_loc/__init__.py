"""lora_loc: reproducible ML fingerprint localization on the LoRaWAN Antwerp dataset.

Companion code for N. Keleşoğlu, M. Halama, A. Strzoda, "Enhancing LoRa-Based Outdoor
Localization Accuracy Using Machine Learning," IEEE Access 13 (2025) 129432-129450.

Modules
-------
config         paths, seeds, and every hyperparameter (with its source)
data           CSV / JSON loaders and checksum verification
preprocessing  -200 -> -128 dBm, log10(|x|+1), min-max scaling, 70/15/15 split
features       feature-vector definitions FV1-FV16 (paper Table 9)
models         k-NN, CNN, SVR, ANN, XGBoost, LightGBM, Hybrid CNN+XGBoost
geo            haversine distances in metres
evaluation     mean / median / SD / CDF / R^2 / inference time
experiment     cached train-evaluate runs
paper_results  numbers reported in the paper, for comparison
plotting       figures used in the notebooks
"""
__version__ = "1.0.0"
