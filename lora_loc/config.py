"""Central configuration: paths, random seeds and the hyperparameters reported in the paper.

Every number in this file is traceable to one of two sources:

* **[P]** N. Keleşoğlu, M. Halama, A. Strzoda, "Enhancing LoRa-Based Outdoor Localization
  Accuracy Using Machine Learning," *IEEE Access*, vol. 13, pp. 129432-129450, 2025,
  doi:10.1109/ACCESS.2025.3589032 (Section V, "final" hyperparameters).
* **[S]** The authors' original supplementary notebook (kept in ``legacy/``), used when the
  paper does not state a value (e.g. the ANN learning rate, CNN batch size).

Values chosen by us because neither source specifies them are marked **[ASSUMPTION]**.
Change them here, not in the notebooks, so that every experiment stays consistent.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

#: Zenodo record 3342253 (dataset version 1.2, published 19 July 2019).
ZENODO_RECORD = "3342253"
ZENODO_DOI = "10.5281/zenodo.3342253"
CSV_FILE = DATA_DIR / "lorawan_antwerp_2019_dataset.csv"
JSON_FILE = DATA_DIR / "lorawan_antwerp_2019_dataset.json.txt"
#: MD5 checksums published on the Zenodo record page.
MD5 = {
    "lorawan_antwerp_2019_dataset.csv": "9c336dd917074c19b2868dfd71f3774b",
    "lorawan_antwerp_2019_dataset.json.txt": "c3a6f09bfa1d5029c236a93e83e05196",
}

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
SEED = 42  # [S] random_state=42 in the original split and LightGBM parameters.

# Train / validation / test = 70 / 15 / 15  [P, Sec. V]
TEST_VAL_FRACTION = 0.30
TEST_FRACTION_OF_HOLDOUT = 0.50

# --------------------------------------------------------------------------------------
# Signal constants
# --------------------------------------------------------------------------------------
NO_SIGNAL_DBM = -200  # placeholder used in the dataset for "gateway did not receive"
# [P, Sec. IV]: measured RSSI lies in [-127, -60] dBm, so -200 is replaced by -128 dBm.
NO_SIGNAL_REPLACEMENT_DBM = -128

#: Error thresholds (metres) used for the CDF table [P, Table 12].
CDF_THRESHOLDS_M = (50, 100, 150, 200, 250, 300)

# --------------------------------------------------------------------------------------
# Model hyperparameters (final values from the paper's grid search)
# --------------------------------------------------------------------------------------
KNN = dict(n_neighbors=11, p=1)  # [P, Table 1] Manhattan, k = 11 -> 338.56 m

CNN = dict(  # [P, Sec. V-B] + [S]
    filters=(32, 64),
    kernel_size=3,
    dense_units=128,
    dropout=0.2,
    learning_rate=1e-3,
    epochs=100,  # [P] "for further analysis ... epochs was limited to 100"
    batch_size=32,  # [S]
)

SVR = dict(kernel="rbf", C=0.3, epsilon=0.01)  # [P, Sec. V-C] -> 331.92 m
SVR_GAMMA = "scale"  # [ASSUMPTION] not reported; scikit-learn default.

ANN = dict(  # [P, Sec. V-D, Table 5]
    hidden_units=(256, 256, 128, 64),
    dropout=0.1,
    learning_rate=1e-4,  # [S] value used in the supplementary notebook
    max_epochs=150,
    batch_size=64,
    patience=10,
)

XGB = dict(  # [P, Sec. V-E, Table 6]
    n_estimators=400,
    max_depth=12,
    learning_rate=0.05,
    early_stopping_rounds=10,  # [S]
    # Exact greedy split finding. XGBoost < 2.0 chose this by default for a dataset of this
    # size; XGBoost >= 2.0 defaults to "hist" (256 bins per feature), which coarsens the
    # frame counter (values 1-199,164) and costs ~20 m on FV4 (268.4 m vs 247.1 m exact;
    # paper 248.7 m). On RSSI-only inputs the two methods agree within ~1 m.
    tree_method="exact",
)

LGBM = dict(  # [P, Sec. V-F, Tables 7-8]
    num_leaves=400,
    learning_rate=0.02,
    n_estimators=1000,
    # NOTE: LightGBM only applies `subsample` (bagging) when `subsample_freq > 0`. The
    # paper and the authors' code set subsample=0.8 but no frequency, so row subsampling
    # was in fact *inactive*. We keep that behaviour to stay faithful; set
    # subsample_freq=1 to turn bagging on.
    subsample=0.8,
    subsample_freq=0,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=SEED,
    early_stopping_rounds=10,  # [S]
)

HYBRID = dict(  # [P, Sec. III-G / V-G]
    filters=(32, 64),
    kernel_size=3,  # [ASSUMPTION] not reported for the hybrid; same as the CNN.
    embedding_units=64,
    learning_rate=1e-3,  # [ASSUMPTION] Adam default.
    epochs=100,
    batch_size=32,
    xgb_n_estimators=400,
    xgb_max_depth=12,
    xgb_learning_rate=0.09,
    xgb_tree_method="exact",  # see the note in XGB above
    early_stopping_rounds=10,
)
