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
│   ├── split_check.py         random vs. chronological vs. device-held-out split
│   ├── queue_json.sh          the exact sequence of runs used for the results below
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
python scripts/download_data.py --json       # 96 MB JSON for FV3–FV16 (or copy it into data/ by hand)

pytest -q                                    # 9 unit tests, ~2 s
python scripts/run_experiments.py --fv FV1 --quick   # smoke test, ~3 min
python scripts/run_experiments.py --fv FV1 FV2       # all 7 models on FV1/FV2 (CSV), ~3 h on 4 CPU cores
python scripts/run_experiments.py --source json --fv FV1 FV2 FV3 FV4 FV5 FV6 FV7 FV8 FV9 FV10 FV11 FV12 FV13 FV14 FV15 FV16 \
       --models k-NN XGBoost LightGBM ANN            # all 16 feature vectors (JSON), ~2 h
python scripts/split_check.py                         # random vs. chronological vs. device split

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
| XGBoost | 400 trees, depth 12, η = 0.05, early stop 10, exact split finding |
| LightGBM | 400 leaves, η = 0.02, 1000 trees, colsample 0.8, L1 = L2 = 0.1, one model per coordinate |
| Hybrid | CNN (32, 64 filters → Dense 64 embedding) + XGBoost (400 trees, depth 12, η = 0.09) on [x, z] |

## Reproduction report

All numbers are on the **test split** from one run with seed 42, on a 4-core CPU (TensorFlow 2.21,
XGBoost 3.2, LightGBM 4.7, scikit-learn 1.9).

**What matches exactly**
- Dataset checksums: CSV and JSON are identical in Zenodo v1.2 and v1.3.
- 44 active gateways, and the RSSI range [−127, −60] dBm.
- The JSON release has **130,430** messages, the number the paper reports.
- The random-split sizes are 91,300 / 19,564 / 19,565 on the CSV; the JSON has one more training message.
- RSSI + per-gateway SNR (FV9) gives 88 inputs, the input width of the authors' original notebook.

### 1. All 16 feature vectors from the JSON release (paper Tables 10–11)

`python scripts/run_experiments.py --source json --fv FV1 ... FV16` → `results/summary_json.csv`.
Mean error in metres; Δ = ours − paper.

| FV | k-NN | Δ | ANN | Δ | XGBoost | Δ | LightGBM | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FV1 | 334.4 | -4.1 | 369.0 | +36.3 | 327.0 | -0.6 | 324.3 | -1.1 |
| FV2 | 330.9 | +4.7 | 349.3 | +19.5 | 318.1 | -0.8 | 314.8 | -1.8 |
| FV3 | 350.2 | +17.6 | 355.4 | +35.6 | 318.1 | -0.9 | 314.8 | -0.3 |
| FV4 | 295.6 | +4.2 | 325.9 | +10.6 | 247.1 | -1.6 | 253.4 | +0.6 |
| FV5 | 332.5 | +3.5 | 359.0 | +40.7 | 318.0 | -0.9 | 314.8 | -1.2 |
| FV6 | 328.9 | +23.8 | 331.0 | +27.5 | 250.7 | +0.3 | 255.0 | +0.4 |
| FV7 | 334.6 | +25.5 | 338.0 | +32.0 | 250.1 | -1.3 | 255.2 | -0.1 |
| FV8 | 298.5 | +3.4 | 320.5 | +13.1 | 248.2 | -1.2 | 253.2 | +1.7 |
| FV9 | 316.2 | +0.2 | 316.6 | +8.4 | 312.5 | -1.2 | 309.8 | -1.8 |
| FV10 | 314.7 | -3.0 | 316.6 | +5.8 | 311.5 | -1.9 | 309.8 | -1.7 |
| FV11 | 313.1 | +1.5 | 309.8 | +6.2 | 309.3 | +0.4 | 306.1 | -1.1 |
| FV12 | 310.0 | -2.5 | 307.1 | +4.4 | 307.4 | -1.2 | 304.8 | -1.1 |
| FV13 | 308.2 | -0.7 | 300.6 | -7.2 | 308.4 | -0.9 | 305.3 | -1.1 |
| FV14 | 310.0 | +1.3 | 306.9 | +6.6 | 307.8 | -1.1 | 305.4 | -0.4 |
| FV15 | 282.8 | -1.8 | 282.4 | -3.4 | 249.2 | -0.5 | 248.3 | -1.9 |
| FV16 | 284.0 | -1.7 | 285.4 | +5.7 | 251.0 | -0.0 | 248.3 | -1.2 |

- **XGBoost and LightGBM reproduce the paper within 2 m on every one of the 16 feature vectors.**
  The paper's feature-vector trends therefore reproduce for the tree models: SF helps a little,
  SNR and ESP help a little, and the frame counter (FV4, FV6–FV8, FV15, FV16) helps a lot.
  Under a stricter split the counter does *not* help (section 3).
- **XGBoost needs exact split finding** (`tree_method="exact"`, which XGBoost < 2.0 chose by
  default at this data size). With the ≥ 2.0 default, 256-bin histograms, the counter (values
  1–199,164) is coarsened and FV4 is 268.4 m instead of 247.1 m. See the note in `config.py`.
- **k-NN** matches within ≈5 m except on FV3, FV6 and FV7, which add the channel to RSSI/SF/CNT
  (+18 to +25 m). In FV14 the channel is diluted by the 88 SNR/ESP columns, and there it matches. Min-max scaling a categorical channel index adds a large, location-free term
  to the Manhattan distance. The paper must have encoded or weighted the channel differently,
  and it does not say how.
- **ANN** is 11–41 m worse on FV1–FV8 and within ≈8 m on FV9–FV16. Its learning rate is not reported
  (see section 2).

Slow models, run only on the feature vector where the paper reports each one's best result:

| Model | FV | Ours | Paper | Δ |
|---|---|---:|---:|---:|
| Hybrid | FV4 | 259.1 | 244.5 | +14.6 |
| SVR | FV4 | 384.6 | 320.9 | +63.7 |
| CNN | FV15 | 337.7 | 319.6 | +18.2 |

None of the three reproduce the paper's best-FV numbers. SVR in particular gets *worse* with
extra features in our setup. The paper says SVR used additional feature selection and random
search (Sec. V-D) without specifying them.

### 2. FV1 and FV2 from the CSV release (all seven models)

`python scripts/run_experiments.py --fv FV1 FV2` → `results/summary.csv`.

| Model | FV1 ours | FV1 paper | Δ | FV2 ours | FV2 paper | Δ |
|---|---:|---:|---:|---:|---:|---:|
| k-NN | 336.51 | 338.56 | -2.05 | 331.74 | 326.26 | +5.48 |
| CNN | 354.01 | 343.88 | +10.13 | 343.34 | 341.39 | +1.95 |
| SVR | 334.35 | 331.92 | +2.43 | 376.22 | 321.93 | +54.29 |
| ANN | 354.43 | 332.74 | +21.69 | 347.11 | 329.84 | +17.27 |
| XGBoost | 328.30 | 327.66 | +0.64 | 320.11 | 318.95 | +1.16 |
| LightGBM | 326.75 | 325.43 | +1.32 | 317.19 | 316.59 | +0.60 |
| Hybrid | 327.64 | 323.38 | +4.26 | 320.99 | 314.21 | +6.78 |

- The **k-NN grid (Table 1)** reproduces: Manhattan k = 11 gives 336.51 m (paper 338.56 m), Euclidean
  335.96 m (340.75 m), Chebyshev 350.02 m (350.40 m). Cosine at small k comes out 19 m better than the
  paper (`results/knn_grid_table1.csv`).
- **ANN learning rate.** It is not reported. We use the authors' notebook value, 1e-4. Higher rates
  are worse: 3e-4 gives 376.3 m and 1e-3 gives 449.3 m on FV1 (`results/sensitivity_ann_lr.csv`).
  The ≈20 m ANN gap remains unexplained, and the ANN also varies by ≈15 m between the CSV and JSON
  row orders (354 m vs 369 m on FV1), so it is sensitive to initialization.
- **Heavy tail.** Median errors are about 200 m, but the 90th percentile is about 800 m and the
  maximum about 5 km, driven by messages heard by a single gateway (31 % of all messages).
- **Inference time.** The per-sample ranking matches Fig. 5: trees and k-NN take 2–3 ms, Keras
  models about 70 ms one sample at a time. Batched prediction is more than 1000× faster for the
  networks, so Fig. 5-style timings mostly measure framework overhead.

### 3. Do the results survive a stricter split? (not in the paper)

`python scripts/split_check.py` → `results/split_check.csv`. LightGBM, the model that reproduces the
paper most closely, retrained under three protocols. The data contains 23 devices recorded from
16 Nov 2018 to 11 Feb 2019.

| FV | Random split (paper) | Chronological (train on earliest 70 %, test on latest 15 %) | Unseen devices (9 held-out vehicles, 31,125 test messages) |
|---|---:|---:|---:|
| FV1 RSSI | 324 m | 286 m | 573 m |
| FV2 +SF | 315 m | 280 m | 574 m |
| **FV4 +SF+CNT** | 253 m | 555 m | 775 m |
| FV9 +SNR | 310 m | 273 m | 562 m |
| FV13 +SF+SNR+ESP | 305 m | 270 m | 563 m |
| **FV15 +SF+SNR+ESP+CNT** | 248 m | 327 m | 666 m |
| **FV16 … +CNT+AT** | 248 m | 326 m | 664 m |

1. **The frame-counter gain is an artefact of the random split.** Under the chronological split,
   FV4 goes from the best feature vector (253 m) to the worst (555 m), almost twice as bad as RSSI
   alone. The counter only lets the model find the same vehicle's neighbouring messages, which are
   absent when the test period comes later.
2. **The random split tests on places it has already seen.** The median distance from a test
   message to the nearest training position is **0.4 m**, and 96.5 % are within 10 m. One 50 m cell,
   probably a depot, holds 15 % of all test messages.
3. **The chronological numbers without CNT look *better* than random (≈270–286 m), but for a
   different reason.** The February test period is even more concentrated at stationary hotspots
   (the top 10 cells hold 54 % of test messages, against 32 % for the random split), which pulls
   the median down to ≈60 m. It is a different test distribution, not better generalization.
4. **New devices are much harder.** With whole vehicles held out, errors nearly double, to
   ≈560–575 m without CNT, even though 61 % of test positions lie within 10 m of a training
   position. This points to device-specific RSSI offsets (antenna or mounting) that a fingerprint
   model trained on other vehicles does not cover.

For any follow-up work on this dataset (for example a journal comparison), report a
device-held-out or time-based split next to the random split, and treat the frame counter as a
leakage feature rather than a radio feature.

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
  rule for missing entries. This is supported, though not proven: it gives the authors' 88-input
  width for FV9, and the tree models then match the paper within 2 m on FV9–FV16.

**Faithful to the authors even where it is unusual:**
- LightGBM `subsample=0.8` has no effect without `subsample_freq>0`, and neither the paper nor
  the authors' code sets it, so bagging stays off here as well.
- Random per-message split (not by time or device), and no HDOP filtering.

**JSON vs. CSV.** Channel, frame counter, airtime, SNR and ESP exist only in the JSON release. The
JSON holds the same messages as the CSV in a different row order, plus one extra message, so the
random split (and hence individual errors) differs slightly between section 1 and section 2.

**Dates in the paper.** The paper says the data were collected from 17 November 2017 to 5 February
2018. Those are the dates of the *first* dataset version. The v1.2/v1.3 files it actually uses span
16 November 2018 to 11 February 2019 (earliest and latest reception timestamps in the JSON).

**Interpretation caveat: the frame counter.** Confirmed by section 3: the counter's large gain does not
survive a chronological or device-held-out split.

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
