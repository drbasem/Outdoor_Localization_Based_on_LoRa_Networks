"""Numbers reported in the paper, transcribed so that we can compare against them.

Source: Keleşoğlu, Halama, Strzoda, IEEE Access 13 (2025) 129432-129450,
doi:10.1109/ACCESS.2025.3589032. Table numbers refer to that article.
"""
import pandas as pd

MODELS = ["k-NN", "CNN", "SVR", "ANN", "XGBoost", "LightGBM", "Hybrid"]

#: Mean localization error (m) per model and feature vector (Tables 10 and 11).
MEAN_ERROR_M = pd.DataFrame(
    {
        "FV1": [338.56, 343.88, 331.92, 332.74, 327.66, 325.43, 323.38],
        "FV2": [326.26, 341.39, 321.93, 329.84, 318.95, 316.59, 314.21],
        "FV3": [332.62, 342.55, 327.01, 319.83, 318.92, 315.09, 316.49],
        "FV4": [291.49, 321.71, 320.90, 315.33, 248.72, 252.82, 244.51],
        "FV5": [329.08, 344.02, 322.91, 318.26, 318.95, 315.95, 312.24],
        "FV6": [305.10, 321.71, 325.89, 303.47, 250.43, 254.58, 248.32],
        "FV7": [309.06, 335.78, 326.69, 305.95, 251.43, 255.37, 248.85],
        "FV8": [295.08, 322.83, 321.49, 307.46, 249.40, 251.49, 246.87],
        "FV9": [316.01, 345.15, 409.42, 308.19, 313.64, 311.54, 311.72],
        "FV10": [317.72, 337.45, 433.84, 310.86, 313.38, 311.49, 306.61],
        "FV11": [311.58, 333.53, 403.33, 303.60, 308.82, 307.17, 308.27],
        "FV12": [312.48, 330.63, 428.56, 302.66, 308.65, 305.89, 303.12],
        "FV13": [308.87, 334.38, 396.91, 307.77, 309.30, 306.32, 309.85],
        "FV14": [308.69, 347.72, 396.69, 300.29, 308.87, 305.79, 308.67],
        "FV15": [284.58, 319.57, 395.30, 285.74, 249.77, 250.17, 254.30],
        "FV16": [285.65, 320.65, 395.15, 279.66, 251.05, 249.57, 255.07],
    },
    index=MODELS,
)

#: Fraction of test samples within a distance threshold, best FV per model (Table 12).
CDF_TABLE12 = pd.DataFrame(
    {50: [.33, .03, .15, .28, .32, .32, .34], 100: [.41, .25, .35, .37, .42, .41, .44],
     150: [.48, .40, .43, .45, .50, .50, .52], 200: [.55, .47, .40, .53, .58, .59, .60],
     250: [.61, .54, .57, .60, .65, .65, .66], 300: [.66, .61, .63, .66, .71, .71, .72]},
    index=MODELS,
)

#: Best configuration per model: mean / median / SD in metres (Table 13, "This work").
BEST_TABLE13 = pd.DataFrame(
    {"FV": ["FV15", "FV15", "FV4", "FV16", "FV4", "FV16", "FV4"],
     "mean_m": [284.58, 319.57, 320.90, 279.66, 248.72, 249.57, 244.51],
     "median_m": [160.01, 219.96, 194.38, 171.70, 145.54, 146.72, 130.39],
     "sd_m": [380.81, 352.20, 376.45, 350.84, 326.86, 328.14, 337.11]},
    index=MODELS,
)

#: Median / SD on FV1 (RSSI only) where the paper reports them (Tables 1, 2, 5).
FV1_DETAIL = pd.DataFrame(
    {"mean_m": [338.56, 343.88, 331.92, 332.74, 327.66, 325.43, 323.38],
     "median_m": [209.15, 234.16, None, 218.0, None, None, None],
     "sd_m": [417.30, 380.84, None, 384.0, None, None, None],
     "r2": [0.9147, None, None, None, 0.9214, None, None]},
    index=MODELS,
)

#: R^2 per coordinate reported for LightGBM (Tables 7-8) and SVR (Table 4) on FV1.
R2_FV1 = {"LightGBM": (0.9505, 0.8962), "SVR": (0.9488, 0.8918)}

#: Mean per-sample inference time (s), Fig. 5 text. For the two-model methods (SVR,
#: LightGBM) the LAT and LON times reported separately in the text are summed here.
INFERENCE_S = {"k-NN": 0.010593, "CNN": 0.1490, "SVR": 0.00614 + 0.005203, "ANN": 0.05685,
               "XGBoost": 0.00096, "LightGBM": 0.00116 + 0.001296, "Hybrid": 0.3004}

#: Earlier results cited in Table 13 (mean error, m) for context.
PRIOR_WORK = pd.DataFrame(
    [("k-NN", "Aernouts et al. 2018 [43]", 398.40), ("k-NN", "Janssen et al. 2020 [44]", 343),
     ("k-NN", "Bagherian et al. 2025 [32]", 313.26), ("ANN", "Bagherian et al. 2025 [32]", 277.61),
     ("SVR", "Janssen et al. 2020 [44]", 1148), ("RF", "Janssen et al. 2020 [44]", 340)],
    columns=["model", "study", "mean_m"],
)
