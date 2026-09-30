# Outdoor Localization Based on LoRa Networks: an educational, reproducible implementation

Machine-learning **RSSI fingerprinting** for outdoor localization in a city-scale LoRaWAN
network. This repository accompanies:

> N. Keleşoğlu, M. Halama, A. Strzoda, "Enhancing LoRa-Based Outdoor Localization Accuracy
> Using Machine Learning," *IEEE Access*, vol. 13, pp. 129432–129450, 2025.
> doi:[10.1109/ACCESS.2025.3589032](https://doi.org/10.1109/ACCESS.2025.3589032) (open access, CC BY 4.0)

It implements the paper's pipeline end to end: data download and checksum, preprocessing, all
seven models (k-NN, CNN, SVR, ANN, XGBoost, LightGBM, and the Hybrid CNN + XGBoost), evaluation
in metres, and a side-by-side comparison with the paper's tables. The pipeline is split into
four notebooks that can be used for teaching.

---

## Repository layout

```
├── lora_loc/                  reusable, documented Python package
│   ├── config.py              paths, seed, and every hyperparameter (with its source)
│   ├── data.py                CSV/JSON loaders, MD5 verification, active-gateway detection
│   ├── preprocessing.py       −200→−128 dBm, log10(|x|+1), min-max scaling, 70/15/15 split
│   ├── features.py            feature vectors FV1–FV16 (paper Table 9)
│   ├── models.py              the 7 models behind one fit/predict interface
│   ├── geo.py                 haversine distances (m), local metric projection
│   ├── evaluation.py          mean/median/SD/CDF/R²/inference-time metrics
│   ├── experiment.py          cached train→evaluate runs
│   ├── paper_results.py       numbers reported in the paper, for comparison
│   └── plotting.py            figures used in the notebooks
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   what is in the data; why fingerprinting works
│   ├── 02_preprocessing.ipynb         each preprocessing step, visualised
│   ├── 03_models.ipynb                theory + code + training for all 7 models
│   └── 04_results.ipynb               reproduced vs. paper: tables, CDF, error map, timing
├── scripts/
│   ├── download_data.py       fetch the dataset from Zenodo and verify checksums
│   ├── run_experiments.py     command-line training of any model × feature vector
│   ├── sensitivity_ann_lr.py  ANN learning-rate sensitivity (not reported in the paper)
│   └── build_notebooks.py     notebooks are generated from this file (easy to diff)
├── tests/test_pipeline.py     unit tests (no dataset needed)
├── results/                   summary.csv, k-NN grid, cached run metrics
├── data/README.md             dataset provenance and checksums
└── legacy/                    the original notebooks, kept unchanged for reference
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py              # 56 MB CSV, MD5-verified (FV1, FV2)
python scripts/download_data.py --json       # optional: 96 MB JSON for FV3–FV16

pytest -q                                    # 9 unit tests, ~2 s
python scripts/run_experiments.py --fv FV1 --quick   # smoke test, ~3 min
python scripts/run_experiments.py --fv FV1 FV2       # full reproduction, ~1–2 h on 4 CPU cores

jupyter lab notebooks/                       # read the notebooks in order 01 → 04
```

Every run is cached under `results/runs/<FV>/<model>.{json,npz}`, so after the full run the
notebooks open in seconds. Use `--force` (or `FORCE = True` in notebook 3) to retrain.

## The method in one page

| Step | What | Paper |
|---|---|---|
| Data | LoRaWAN Antwerp 2019, Zenodo record [3342253](https://doi.org/10.5281/zenodo.3342253) v1.2; 130,429 messages, 72 gateway columns | Sec. IV |
| Gateways | 28 columns are −200 in every row → **44 active gateways** | Sec. IV |
| No signal | −200 dBm → **−128 dBm** (one below the weakest observed −127 dBm) | Sec. IV |
| Transform | x̂ = log₁₀(\|x\|+1) | Eq. 12 |
| Scaling | RSSI: one global min-max; other features: per-column min-max; targets: min-max | Eqs. 13–16 |
| Split | 70/15/15, `random_state=42` → **91,300 / 19,564 / 19,565** (identical to paper) | Sec. V |
| Targets | latitude, longitude; errors computed as distance in metres | Sec. V-H |
| Metrics | mean, median, SD of error; R²; CDF at 50–300 m; per-sample inference time | Sec. V-H |

Final hyperparameters (all in `lora_loc/config.py`):

| Model | Setting |
|---|---|
| k-NN | k = 11, Manhattan (p = 1) |
| SVR | RBF, C = 0.3, ε = 0.01, one model per coordinate |
| ANN | 256-256-128-64 ReLU, Dropout 0.1, Adam, batch 64, ≤150 epochs, early stop (patience 10) |
| CNN | Conv1D(32,3)-Conv1D(64,3)-Dense(128)-Dropout(0.2)-Dense(2), Adam 1e-3, 100 epochs, batch 32 |
| XGBoost | 400 trees, depth 12, η = 0.05, early stop 10 |
| LightGBM | 400 leaves, η = 0.02, 1000 trees, colsample 0.8, L1 = L2 = 0.1, one model per coordinate |
| Hybrid | CNN (32, 64 filters → Dense 64 embedding) + XGBoost (400 trees, depth 12, η = 0.09) on [x, z] |

## Reproduction report

All numbers are on the **test split** (19,565 messages) from one run with seed 42, on a 4-core
CPU (TensorFlow 2.21, XGBoost 3.2, LightGBM 4.7, scikit-learn 1.9). They were produced by
`python scripts/run_experiments.py --fv FV1 FV2` (full table in `results/summary.csv`).

**What matches exactly:** dataset checksum (Zenodo v1.2), 44 active gateways, the RSSI range
[−127, −60] dBm, and the split sizes 91,300 / 19,564 / 19,565.

### Mean localization error (m): reproduction vs. paper (Table 10)

| Model | FV1 ours | FV1 paper | Δ | FV2 ours | FV2 paper | Δ |
|---|---:|---:|---:|---:|---:|---:|
| k-NN | 336.51 | 338.56 | −2.05 | 331.74 | 326.26 | +5.48 |
| CNN | 354.01 | 343.88 | +10.13 | 343.34 | 341.39 | +1.95 |
| SVR | 334.35 | 331.92 | +2.43 | **376.22** | 321.93 | **+54.29** |
| ANN | 354.43 | 332.74 | **+21.69** | 347.11 | 329.84 | +17.27 |
| XGBoost | 329.33 | 327.66 | +1.67 | 320.43 | 318.95 | +1.48 |
| LightGBM | 326.75 | 325.43 | +1.32 | **317.19** | 316.59 | +0.60 |
| Hybrid | **326.72** | 323.38 | +3.34 | 321.53 | 314.21 | +7.32 |

Median / SD on FV1 (ours; paper in brackets where reported): k-NN 207.0 / 414.8 (209.15 / 417.30),
CNN 244.9 / 386.3 (234.16 / 380.84), ANN 248.3 / 392.3 (218 / 384).

The **k-NN hyperparameter grid (Table 1)** also reproduces: Manhattan, k = 11 → 336.51 m
(paper 338.56 m); Euclidean k = 11 → 335.96 m (340.75 m); Chebyshev k = 11 → 350.02 m (350.40 m).
The one exception is cosine at small k, which is 19 m better than the paper at k = 3.
See `results/knn_grid_table1.csv`.

### Reading these results

- **Reproduced well (within ≈2 %):** k-NN, XGBoost and LightGBM on both FV1 and FV2, plus SVR
  and the Hybrid on FV1. Our trends agree with the paper's: the tree ensembles and the Hybrid
  are best, and adding SF (FV2) helps every model except SVR.
- **Not reproduced:**
  - *ANN* is 17–22 m worse than reported. The paper does not give its learning rate. We use the
    authors' notebook value (1e-4), and higher rates are worse (1e-4: 354.4 m, 3e-4: 376.3 m,
    1e-3: 449.3 m on FV1; `results/sensitivity_ann_lr.csv`). The gap remains unexplained.
  - *SVR on FV2* is 54 m worse. With our setup, adding SF makes SVR *worse* than on FV1. The paper
    says SVR received extra feature selection and random search (Sec. V-D) that it does not specify.
  - *CNN on FV1* is 10 m worse (neural-network variance plus unreported batch size).
- **Model differences are small.** On FV1 the top three models (Hybrid, LightGBM, XGBoost) are within 3 m of each other. With
  one seed, we cannot rank them reliably.
- **Heavy tail.** Median errors are about 200 m, but the 90th percentile is about 800 m and the
  maximum about 5 km, set by messages heard by a single gateway (31 % of all messages).
- **Inference time.** The per-sample ranking matches Fig. 5: trees and k-NN take about 2–3 ms, and
  the Keras models about 70 ms when called one sample at a time. Batched prediction is over 1000×
  faster for the neural networks, so Fig. 5-style per-sample timings mostly measure framework
  overhead, not model cost.

## Deviations, assumptions and caveats

These are stated openly so that the numbers above can be interpreted correctly.

**Not reported in the paper, so we chose a value (marked `[ASSUMPTION]` in `config.py`):**
- SVR kernel width γ: scikit-learn default `"scale"`.
- ANN learning rate: 1e-4, taken from the authors' supplementary notebook (`legacy/`).
- Hybrid: kernel size 3, CNN learning rate 1e-3, temporal window T = 1 (Eq. 7's T is never given).
- Metres from degrees: haversine. The paper says "Euclidean distance (in meters)" without giving
  the projection; over a 9 km × 6 km area the two differ by far less than 1 m.
- Scalers are fit on the training split only (the paper does not specify).
- SNR/ESP (FV9–FV16) are treated as per-gateway vectors with the same "neighbouring value"
  rule for missing entries. Not verified: see below.

**Faithful to the authors even where it is unusual:**
- LightGBM `subsample=0.8` has no effect without `subsample_freq>0`, and neither the paper nor
  the authors' code sets it, so bagging stays off here as well.
- Random per-message split (not by time or device), and no HDOP filtering.

**Data availability for FV3–FV16.** Channel, frame counter, airtime, SNR and ESP exist only in
the JSON release (`lorawan_antwerp_2019_dataset.json.txt`). The JSON loader is unit-tested on the
message format published on the Zenodo record page, but it has **not** been run on the full JSON
file in producing the results above (the file could not be downloaded in our build environment).
Results for FV3–FV16 are therefore not reported here. Notebook 4 produces them automatically once
the file is in `data/`.

**Interpretation caveat: the frame counter.** The paper's best results (≈245–255 m) all use CNT.
With a random per-message split, the monotonically increasing per-device counter lets a model
match test messages to temporally adjacent training messages from the same vehicle, which were
sent from nearly the same place. Part of that gain may be track continuity rather than radio
information that transfers to new devices or new time periods. A time-based split is the right
check before using CNT-based models in practice.

**Variance.** Differences of a few metres between the tree models, or between our run and the
paper's, are about the size of the run-to-run variation of the neural models. We report a single
seed, as the paper does. Use repeated seeds and confidence intervals before drawing conclusions
from small gaps.

## Dataset

M. Aernouts, R. Berkvens, K. Van Vlaenderen, M. Weyn, *Sigfox and LoRaWAN Datasets for
Fingerprint Localization in Large Urban and Rural Areas*, Zenodo, v1.2, 2019,
doi:[10.5281/zenodo.3342253](https://doi.org/10.5281/zenodo.3342253). Collection methodology:
*Data* 3(2):13, 2018, doi:[10.3390/data3020013](https://doi.org/10.3390/data3020013).
The data files are not in git; see [`data/README.md`](data/README.md).

## Legacy material

`legacy/Data_Description.ipynb` and
`legacy/Enhancing_LoRa_based_Outdoor_Localization_Accuracy_Using_Machine_Learning.ipynb` are the
original notebooks, unchanged. The second one was an architecture illustration, not a runnable
pipeline: it relied on unpublished pre-scaled files and missing imports. Its content is now in
`lora_loc/models.py` and `notebooks/03_models.ipynb`.
