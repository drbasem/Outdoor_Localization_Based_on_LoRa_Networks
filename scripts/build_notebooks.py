#!/usr/bin/env python
"""Generate the tutorial notebooks in notebooks/ from the cell lists below.

Keeping the notebooks as source code here makes them easy to review and diff. After
editing, rebuild and execute them::

    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

REPO = "https://github.com/drbasem/Outdoor_Localization_Based_on_LoRa_Networks"
BRANCH = "claude/keen-hamilton-7kcqml"

COLAB = f"""\
# ▶ Run this cell first. On your own computer it does nothing; on Google Colab it downloads the
#   code and the dataset (about 1-2 minutes the first time).
import os, sys, subprocess
if "google.colab" in sys.modules:
    if not os.path.exists("/content/repo"):
        subprocess.run(["git", "clone", "--depth", "1", "-b", "{BRANCH}", "{REPO}", "/content/repo"], check=True)
    os.chdir("/content/repo/notebooks")
    subprocess.run([sys.executable, "../scripts/download_data.py", "--json"], check=True)
print("working folder:", os.getcwd())"""

SETUP = """\
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # make `lora_loc` importable from notebooks/
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lora_loc import config, data, preprocessing as pp, features, geo, plotting
plotting.style()
pd.set_option("display.precision", 2)"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


# ======================================================================================
NB1 = [
    md(r"""
# 1 · The LoRaWAN Antwerp dataset

**Series:** reproducing *Keleşoğlu, Halama & Strzoda, "Enhancing LoRa-Based Outdoor
Localization Accuracy Using Machine Learning", IEEE Access 13 (2025), doi:10.1109/ACCESS.2025.3589032.*

| Notebook | Topic |
|---|---|
| **1 · Dataset** ← you are here | what is in the data, and why fingerprinting works |
| 2 · Preprocessing | −200 dBm handling, log transform, scaling, splits, feature vectors |
| 3 · Models | the seven models: theory, code and the paper's hyperparameters |
| 4 · Results | reproduced numbers next to the paper's, CDFs, error maps, timing |

### What problem are we solving?
A LoRaWAN end device sends an uplink packet. Every gateway in range receives it and reports
the **RSSI** (received signal strength, dBm). For each message we therefore get a vector

$$\mathbf{r} = [\,\mathrm{RSSI}_1, \dots, \mathrm{RSSI}_G\,],$$

with a placeholder value for gateways that did not hear it. **Fingerprinting** treats this
vector as a signature of the place it was sent from. In an offline phase we collect
(fingerprint, GPS position) pairs. Online, we learn a regression
$f:\mathbf{r}\mapsto(\text{lat},\text{lon})$. No gateway coordinates, no path-loss model and no
time synchronisation are needed, which is why this approach suits low-cost LPWAN networks.
"""),
    code(SETUP),
    md("""
## 1.1 Load the data and check it

We use version **1.2** of the Zenodo record *Sigfox and LoRaWAN Datasets for Fingerprint
Localization in Large Urban and Rural Areas* (Aernouts et al., doi:10.5281/zenodo.3342253).
Messages were sent by GPS-equipped trackers on postal vehicles in Antwerp. First we check
that our file is byte-identical to the published one.
"""),
    code("""
print("file :", config.CSV_FILE.name)
print("MD5 matches Zenodo v1.2:", data.verify(config.CSV_FILE))
df = data.load_csv()
print("shape:", df.shape)
df.head()"""),
    md("""
There are **130,429 messages × 77 columns**:

* `BS 1` … `BS 72`: RSSI (dBm) at each of the 72 gateways; **−200 means "not received"**,
* `RX Time`: timestamp of reception,
* `SF`: LoRa spreading factor (7–12), chosen by adaptive data rate (ADR),
* `HDOP`: horizontal dilution of precision of the GPS fix (a quality indicator for the ground truth),
* `Latitude`, `Longitude`: the ground truth, which is our regression target.
"""),
    md("## 1.2 How many gateways actually hear anything?"),
    code("""
bs = data.rssi_columns(df)
active = data.active_gateways(df)
dead = sorted(set(bs) - set(active), key=lambda c: int(c.split()[1]))
print(f"{len(bs)} gateway columns, {len(active)} active, {len(dead)} never receive a message")
print("dead:", ", ".join(dead))"""),
    md("""
28 gateway columns are −200 in *every* row, so they carry no information and are dropped.
This leaves the **44 gateways** used in the paper. Next: how often each gateway receives,
and how many gateways receive each message.
"""),
    code("""
received = df[active] != config.NO_SIGNAL_DBM
per_gw = received.mean().sort_values(ascending=False) * 100
per_msg = received.sum(axis=1)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].bar(range(len(per_gw)), per_gw.values, color=plotting.MODEL_COLORS["k-NN"], width=0.8)
ax[0].set(xlabel="Gateway (sorted)", ylabel="% of messages received",
          title="Gateway reception rate")
ax[0].grid(axis="x", visible=False)
counts = per_msg.value_counts().sort_index()
ax[1].bar(counts.index, counts.values, color=plotting.MODEL_COLORS["k-NN"], width=0.8)
ax[1].set(xlabel="Gateways receiving the message", ylabel="Messages",
          title="Gateways per message")
ax[1].grid(axis="x", visible=False)
plt.tight_layout(); plt.show()
print(f"gateways per message: mean {per_msg.mean():.2f}, median {per_msg.median():.0f}, "
      f"max {per_msg.max()}; heard by only one gateway: {(per_msg == 1).mean():.1%}")"""),
    md("""
**Takeaway.** The fingerprints are **very sparse**: on average a message reaches fewer than 3 of the
44 gateways (93.7 % of the RSSI matrix is "no signal"), and about 31 % of messages reach a single gateway, and then the fingerprint only says
"somewhere in that gateway's coverage area". This alone puts a floor of a few hundred metres
under the error of *any* RSSI-only method. Keep it in mind when reading the results.
"""),
    md("## 1.3 RSSI and spreading-factor distributions"),
    code("""
vals = df[active].to_numpy().ravel()
vals = vals[vals != config.NO_SIGNAL_DBM]
print(f"observed RSSI range: [{vals.min()}, {vals.max()}] dBm "
      f"-> nothing between -200 and {vals.min()} dBm")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].hist(vals, bins=np.arange(-128, -58, 1), color=plotting.MODEL_COLORS["k-NN"])
ax[0].set(xlabel="RSSI (dBm)", ylabel="Count", title="Received RSSI values (−200 excluded)")
sf = df["SF"].value_counts().sort_index()
ax[1].bar(sf.index, sf.values / len(df) * 100, color=plotting.MODEL_COLORS["k-NN"], width=0.7)
ax[1].set(xlabel="Spreading factor", ylabel="% of messages", title="Spreading factor (ADR)")
ax[1].grid(axis="x", visible=False)
plt.tight_layout(); plt.show()"""),
    md("""
* RSSI lies between −127 and −60 dBm and piles up near the receiver sensitivity. The large gap
  down to the −200 placeholder is why notebook 2 replaces −200 with **−128 dBm**, not with a
  value far away.
* Most traffic uses SF7. A high SF usually means the device was far from gateways or poorly
  covered, so SF carries some location information. This is the motivation for FV2 = RSSI + SF.
"""),
    md("## 1.4 Where were the messages sent from?"),
    code("""
x, y = geo.to_local_xy_m(df.Latitude, df.Longitude)
fig, ax = plt.subplots(figsize=(7, 5.5))
hb = ax.hexbin(x / 1000, y / 1000, gridsize=70, bins="log", cmap="viridis", mincnt=1)
ax.set(xlabel="East (km)", ylabel="North (km)", title="Message density (log scale)", aspect="equal")
ax.grid(False); plt.colorbar(hb, ax=ax, label="messages per cell (log)")
plt.show()
print(f"lat [{df.Latitude.min():.4f}, {df.Latitude.max():.4f}], "
      f"lon [{df.Longitude.min():.4f}, {df.Longitude.max():.4f}]; "
      f"extent ≈ {(x.max()-x.min())/1000:.1f} km × {(y.max()-y.min())/1000:.1f} km")"""),
    md("""
The messages follow the vehicles' routes through the city streets, so coverage is uneven.
Streets that are driven often are sampled densely, and other areas are barely sampled. This
matters for data-driven methods: a model can only interpolate where it has seen fingerprints.

## 1.5 A note on ground-truth quality (HDOP)
"""),
    code("""
print(df["HDOP"].describe(percentiles=[.5, .9, .99]).round(2).to_string())
print(f"messages with HDOP > 2: {(df.HDOP > 2).sum()}")"""),
    md("""
Nearly all fixes have HDOP < 1, which means good GPS geometry, but a few are extreme outliers.
The paper does not filter on HDOP, so neither do we. That choice is part of reproducing the
paper faithfully. Removing the outliers would be a reasonable improvement for your own work.

**Summary.** 130,429 messages, 44 useful gateways, sparse fingerprints, RSSI in [−127, −60] dBm.
→ Continue with **notebook 2**.
"""),
]

# ======================================================================================
NB2 = [
    md(r"""
# 2 · Preprocessing: from raw RSSI to model inputs

This notebook walks through Section IV of the paper one step at a time. Every function
used here is in `lora_loc/preprocessing.py`, so the models in notebook 3 see exactly the
same data.

| Step | Operation | Paper |
|---|---|---|
| 1 | keep the 44 active gateways | Sec. IV |
| 2 | replace −200 dBm (no reception) with −128 dBm | Sec. IV |
| 3 | $\hat x = \log_{10}(\lvert x\rvert+1)$ | Eq. 12 |
| 4 | min–max scale RSSI with **one global** min/max | Eq. 13 |
| 5 | min–max scale each extra feature (SF, …) **separately** | Eq. 14 |
| 6 | min–max scale latitude and longitude | Eq. 16 |
| 7 | 70 / 15 / 15 train / validation / test split | Sec. V |
"""),
    code(SETUP),
    code("""
df = data.load_csv()
active = data.active_gateways(df)
X_raw = df[active].to_numpy(dtype=float)
print("raw RSSI matrix:", X_raw.shape)"""),
    md(r"""
## Step 2 — the "no signal" placeholder

The dataset uses −200 dBm for "not received". Passed to a model as is, the value creates a
huge artificial gap. Relative to the real range [−127, −60], "not heard" would look far more
different from "barely heard" (−127) than −127 looks from −60. That is physically wrong, because
a gateway that just missed a packet is almost the same observation as one that just caught it.
Following Bagherian et al., the paper replaces −200 with the **neighbouring value −128 dBm**.
"""),
    code("""
X = pp.replace_no_signal(X_raw)
print("after replacement: min", X.min(), " max", X.max())
print(f"share of entries that are 'no signal': {(X_raw == -200).mean():.1%}")"""),
    md(r"""
## Step 3 — logarithmic transform (Eq. 12)

$$\hat x_{\mathrm{RSSI},g} = \log_{10}\!\left(\lvert x_{\mathrm{RSSI},g}\rvert + 1\right),\qquad g=1,\dots,44 .$$

RSSI in dBm is already logarithmic in power. Taking the log again compresses the strong-signal
end relative to the weak end. Note the sign: because of $\lvert\cdot\rvert$, a **strong** signal
(−60 dBm) maps to a **small** value (1.785) and "no signal" (−128) maps to the largest value
(2.111). Tree models do not care about the direction, and neural networks learn the sign.
"""),
    code("""
X_log = pp.log_transform(X)
received = X_raw != -200
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].hist(X[received], bins=68, color=plotting.MODEL_COLORS["k-NN"])
ax[0].set(title="Received RSSI (dBm)", xlabel="dBm", ylabel="count")
ax[1].hist(X_log[received], bins=68, color=plotting.MODEL_COLORS["CNN"])
ax[1].set(title="After log10(|x|+1)", xlabel="value")
plt.tight_layout(); plt.show()
print("mapping: -60 dBm ->", pp.log_transform(-60.0).round(4), "  -128 dBm ->", pp.log_transform(-128.0).round(4))"""),
    md(r"""
## Steps 4–6 — min–max scaling

$$\bar x = \frac{\hat x - \min(\hat X)}{\max(\hat X) - \min(\hat X)} \in [0,1].$$

* **RSSI** uses **one** min and max over *all* gateways (Eq. 13 has no gateway subscript).
  This keeps relative levels across gateways, and the pattern of relative levels *is* the fingerprint.
  Per-gateway scaling would, for example, stretch a gateway that only ever sees −120…−110 dBm
  over the full [0, 1] range.
* **Other features** (SF, and for FV3–FV16 also channel, counter, airtime, SNR, ESP) are scaled
  column by column (Eq. 14).
* **Targets** are scaled per coordinate (Eq. 16), and predictions are mapped back to degrees
  before we measure errors in metres.

> **Leakage.** All scalers are fit on the **training split only** and then applied to validation
> and test. The paper does not say which it did. For RSSI the difference is negligible: the
> training split spans [−128, −65] dBm and the full dataset [−128, −60] dBm, so a few very strong
> test readings are scaled slightly below 0.
"""),
    code("""
d = pp.prepare(df, active)          # FV1: RSSI only
print("train / val / test:", len(d.X_train), len(d.X_val), len(d.X_test))
print("X range:", d.X_train.min(), "→", d.X_train.max(), "  features:", d.X_train.shape[1])
print("global RSSI scaler (log domain): min %.4f  max %.4f" % (d.pipeline.rssi_scaler.min_, d.pipeline.rssi_scaler.max_))
pd.DataFrame(d.y_train[:5], columns=["lat (scaled)", "lon (scaled)"])"""),
    md("""
## Step 7 — the split

`train_test_split` is applied twice with `random_state = 42`: first 70 % / 30 %, then the 30 %
is halved. This gives **91,300 / 19,564 / 19,565** samples, exactly the numbers in the paper,
which is good evidence that the split matches the authors'.

The split is **random per message**. Consecutive messages from the same vehicle, seconds apart,
can land in train and test. That favours methods that memorise, such as k-NN and deep trees.
It is the paper's protocol and we keep it. A split by time or by region would give a more
conservative estimate. Notebook 4 comes back to this.

## Feature vectors (Table 9)
"""),
    code("""
fv = pd.DataFrame({k: ", ".join(v) for k, v in features.FEATURE_VECTORS.items()}, index=["features"]).T
fv["available from CSV"] = ["yes" if k in features.CSV_FEATURE_VECTORS else "JSON needed" for k in fv.index]
fv"""),
    md("""
The CSV release has RSSI and SF only, so **FV1 and FV2** can be built from it. Channel (CH),
frame counter (CNT), airtime (AT), SNR and ESP are only in the JSON release. `lora_loc.data.load_json`
parses it (130,430 messages, 44 gateways, the counts the paper reports), and notebook 4 runs
FV1–FV16 from it when `data/lorawan_antwerp_2019_dataset.json.txt` is present.

> **A caution about CNT (frame counter).** FV4, FV6–FV8, FV15 and FV16 contain the uplink
> counter and give the paper's best results (≈245–255 m versus ≈325 m for RSSI only). The counter
> is a device-specific, steadily increasing number. Together with a random per-message split, it
> lets a model find *temporally adjacent messages from the same vehicle* in the training set, and
> those were sent from almost the same place. The improvement may therefore reflect track
> continuity rather than radio information that would carry over to new devices or new time periods.
> Treat CNT-based numbers with care. Notebook 4 (section 4.6) checks this with a chronological
> split: the CNT advantage disappears, and FV4 becomes worse than RSSI alone.

→ Continue with **notebook 3**.
"""),
]

# ======================================================================================
NB3 = [
    md(r"""
# 3 · The seven models

Each section gives the idea, the hyperparameters the paper selected and where they come
from, and a call that trains the model and evaluates it on the **test** split (FV1 = RSSI only).

Training every model from scratch on a 4-core CPU takes about 1–2 h in total; SVR and the two
CNNs take the longest. `experiment.run` therefore **caches** each result in `results/runs/`.
If a result is already there it is loaded, so the notebook runs in seconds. Set `FORCE = True`
to retrain.

All models share one interface (`lora_loc/models.py`):
```python
model = models.build("XGBoost")
model.fit(X_train, y_train, X_val, y_val)   # targets are scaled [lat, lon]
y_hat = model.predict(X_test)
```
"""),
    code(SETUP + """
from lora_loc import models, experiment, paper_results as paper

FORCE = False
FV = "FV1"
d = experiment.prepare_fv(FV)
results = {}

def report(name):
    r = experiment.run(name, FV, prepared=d, force=FORCE, verbose=False, metrics_only=not FORCE)
    results[name] = r
    m = r["metrics"]
    ref = paper.MEAN_ERROR_M.loc[name, FV]
    print(f"{name:9s} mean {m['mean_m']:7.2f} m  (paper {ref:.2f} m, Δ {m['mean_m']-ref:+.2f})  "
          f"median {m['median_m']:6.2f} m  SD {m['sd_m']:6.2f} m  "
          f"R² lat/lon {m['r2_lat']:.4f}/{m['r2_lon']:.4f}  train {m['train_s']:.0f} s")"""),
    md(r"""
## 3.1 k-Nearest Neighbours (k-NN)

The classic fingerprinting baseline. It stores all training fingerprints and predicts the
**mean position of the $k$ closest ones**:

$$\hat{\mathbf p}(\mathbf x)=\frac1k\sum_{i\in\mathcal N_k(\mathbf x)}\mathbf p_i,\qquad
d_p(\mathbf x,\mathbf x')=\Big(\sum_g |x_g-x'_g|^p\Big)^{1/p}.$$

The paper's grid (Table 1) searched $k\in\{3,5,7,11\}$ and Manhattan / Euclidean / Chebyshev /
cosine distances. The best was **Manhattan ($p=1$), $k=11$ → 338.56 m**. Manhattan is less
dominated by one large per-gateway difference than Euclidean. That helps here, because
"heard vs. not heard" produces exactly such large jumps.

k-NN is cheap to train, so we can reproduce the **whole of Table 1**:
"""),
    code("""
from sklearn.neighbors import KNeighborsRegressor
metrics = {"Manhattan": dict(p=1), "Euclidean": dict(p=2), "Chebyshev": dict(metric="chebyshev"),
           "Cosine": dict(metric="cosine", algorithm="brute")}
cache = config.RESULTS_DIR / "knn_grid_table1.csv"
if cache.exists() and not FORCE:
    grid = pd.read_csv(cache, index_col=0)
else:
    grid = pd.DataFrame(index=list(metrics), columns=[3, 5, 7, 11], dtype=float)
    for mname, kw in metrics.items():
        for k in grid.columns:
            knn = KNeighborsRegressor(n_neighbors=int(k), n_jobs=-1, **kw).fit(d.X_train, d.y_train)
            err = geo.distance_errors_m(d.y_test_deg, d.pipeline.unscale_y(knn.predict(d.X_test)))
            grid.loc[mname, k] = err.mean()
    grid.to_csv(cache)
grid.columns = [f"k={c}" for c in grid.columns]
paper_t1 = pd.DataFrame({"k=3": [360.26, 362.53, 371.10, 381.90], "k=5": [347.36, 349.98, 358.65, 363.08],
                         "k=7": [342.62, 344.50, 354.39, 360.23], "k=11": [338.56, 340.75, 350.40, 352.74]},
                        index=["Manhattan", "Euclidean", "Chebyshev", "Cosine"])
print("Reproduced mean error (m):"); display(grid.round(2))
print("Paper, Table 1 (m):"); display(paper_t1)"""),
    code('report("k-NN")'),
    md(r"""
## 3.2 Support Vector Regression (SVR)

SVR fits a function that stays within an $\varepsilon$-tube around the targets while keeping the
model as simple as possible:

$$\min_{w,b}\ \tfrac12\lVert w\rVert^2 + C\sum_i(\xi_i+\xi_i^*)\quad\text{s.t.}\quad
|y_i - w^\top\phi(\mathbf x_i) - b|\le\varepsilon+\xi_i^{(*)}.$$

With the RBF kernel $K(\mathbf x,\mathbf x')=\exp(-\gamma\lVert\mathbf x-\mathbf x'\rVert^2)$ the
fitted function is a weighted sum of Gaussian bumps centred on support vectors. SVR has a
single output, so there are **two models**, SVR-LAT and SVR-LON.
Final values (Table 4): **RBF, $C=0.3$, $\varepsilon=0.01$** (in scaled-target units).
The paper does not report $\gamma$, so we use scikit-learn's default `"scale"`.
Training time grows with $n^2$ or worse, which makes this the slowest model on the full 91k set.
"""),
    code('report("SVR")'),
    md(r"""
## 3.3 Artificial Neural Network (ANN)

A fully connected network, $\mathbb R^{44}\to\mathbb R^2$, with ReLU hidden layers of
**256–256–128–64** units, each followed by **Dropout(0.1)**, and a linear 2-unit output. Training
uses Adam with batch size 64 and MSE loss, for up to 150 epochs, with **early stopping** (patience 10
on validation loss). The architecture comes from the paper's Table 5 search. The paper does not
state the learning rate; we use 1e-4, the value in the authors' supplementary notebook.
"""),
    code('report("ANN")'),
    md(r"""
## 3.4 1-D Convolutional Neural Network (CNN)

The 44-gateway vector is treated as a 1-D sequence:
`Conv1D(32, k=3) → Conv1D(64, k=3) → Flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(2)`,
trained with Adam (lr = 1e-3) for 100 epochs, batch 32 (Table 2).

*A critical remark for students:* a convolution assumes that neighbouring inputs are related.
Here the neighbours are `BS 4`, `BS 5`, …, which are dataset column indices, **not** spatial
neighbours. The CNN therefore cannot exploit real locality. In the paper it is the weakest model
on FV1 (343.88 m, worse than the plain ANN and all tree models), and in our reproduction it does
not beat the tree models either.
"""),
    code('report("CNN")'),
    md(r"""
## 3.5 XGBoost

Gradient-boosted decision trees. Tree $t$ is fit to the gradient of the loss with respect to the
current ensemble's prediction, and the prediction is $\hat y=\sum_t \eta\, f_t(\mathbf x)$.
Trees split on thresholds such as "RSSI at BS 14 > −112 dBm". That suits sparse fingerprints
naturally, and it is invariant to the monotone log transform.
Final values (Table 6): **400 trees, depth 12, $\eta=0.05$**, early stopping (10 rounds) on the
validation RMSE. One multi-output model predicts both coordinates.
"""),
    code('report("XGBoost")'),
    md(r"""
## 3.6 LightGBM

Also gradient boosting, but trees grow **leaf-wise** (the leaf with the largest loss reduction is
split next) on histogram-binned features, which makes it fast on large tabular data. There are two
models, LAT and LON.
Final values (Tables 7–8): **400 leaves, $\eta=0.02$, 1000 trees**, colsample 0.8, L1 = L2 = 0.1.

> The paper also lists `subsample = 0.8`. In LightGBM row subsampling only takes effect if
> `subsample_freq > 0`, which neither the paper nor the authors' code sets. It was therefore
> *not* active, and we keep it off (`config.LGBM`) to match.
"""),
    code('report("LightGBM")'),
    md(r"""
## 3.7 Hybrid: CNN feature extractor + XGBoost

The paper's proposed model (Eq. 8):
1. train a CNN `Conv1D(32) → Conv1D(64) → Flatten → Dense(64, ReLU) → Dense(2)` for 100 epochs;
2. take the 64-dimensional `Dense(64)` activations as a learned embedding $\mathbf z$;
3. train XGBoost (400 trees, depth 12, $\eta=0.09$) on $\tilde{\mathbf x}=[\mathbf x;\mathbf z]$,
   one model per coordinate.

The idea is that the CNN learns smooth non-linear combinations of gateways, and the trees handle the
sharp heard / not-heard splits. **Assumptions** (not stated in the paper): kernel size 3, CNN
learning rate 1e-3, and temporal window $T=1$ (Eq. 7 mentions a window of $T$ messages but never
gives $T$).
"""),
    code('report("Hybrid")'),
    md("## 3.8 Summary on FV1 (RSSI only)"),
    code("""
rows = []
for name in models.MODEL_NAMES:
    m = results[name]["metrics"]
    rows.append({"model": name, "ours mean (m)": m["mean_m"], "paper mean (m)": paper.MEAN_ERROR_M.loc[name, FV],
                 "Δ (m)": m["mean_m"] - paper.MEAN_ERROR_M.loc[name, FV], "ours median (m)": m["median_m"],
                 "ours SD (m)": m["sd_m"], "train (s)": m["train_s"]})
summary = pd.DataFrame(rows).set_index("model")
display(summary.round(2))
ours = summary["ours mean (m)"]
plotting.compare_bars(ours, paper.MEAN_ERROR_M[FV], title="FV1 (RSSI only): mean error, reproduction vs. paper")
plt.ylim(0, max(ours.max(), 350) * 1.15); plt.show()"""),
    md("→ **Notebook 4** looks at the full error distributions, FV2, error maps and timing."),
]

# ======================================================================================
NB4 = [
    md(r"""
# 4 · Results: reproduction vs. paper

This notebook gathers the cached runs (created by notebook 3 or by
`python scripts/run_experiments.py --fv FV1 FV2`) and compares them with the paper's
Tables 10–13 and Figures 3–5.
"""),
    code(SETUP + """
from lora_loc import models, experiment, paper_results as paper
res = experiment.collect()
res = res.set_index(["fv", "model"]).sort_index()
available_fvs = sorted(res.index.get_level_values(0).unique(), key=lambda s: int(s[2:]))
print("feature vectors with results:", available_fvs)"""),
    md("## 4.1 Mean error per feature vector (Tables 10–11)"),
    code("""
rows = []
for fv in available_fvs:
    for name in models.MODEL_NAMES:
        if (fv, name) in res.index:
            ours = res.loc[(fv, name), "mean_m"]; ref = paper.MEAN_ERROR_M.loc[name, fv]
            rows.append({"FV": fv, "model": name, "ours (m)": ours, "paper (m)": ref,
                         "Δ (m)": ours - ref, "Δ (%)": 100 * (ours - ref) / ref})
cmp = pd.DataFrame(rows).set_index(["FV", "model"])
display(cmp.round(2))
print("mean |Δ| per FV (m):"); print(cmp["Δ (m)"].abs().groupby(level=0).mean().round(2).to_string())"""),
    code("""
fig, axes = plt.subplots(1, len(available_fvs), figsize=(7.5 * len(available_fvs), 4), squeeze=False)
for ax, fv in zip(axes[0], available_fvs):
    ours = res.loc[fv, "mean_m"]
    plotting.compare_bars(ours, paper.MEAN_ERROR_M[fv], ax=ax, title=f"{fv}: mean error")
    ax.set_ylim(0, max(ours.max(), 350) * 1.15)
plt.tight_layout(); plt.show()"""),
    md("## 4.2 Error distributions (Fig. 3, Table 12)"),
    code("""
best_fv = available_fvs[-1]
errs = {name: experiment.run(name, best_fv, verbose=False)["errors_m"]
        for name in models.MODEL_NAMES if (best_fv, name) in res.index}
plotting.cdf(errs, xmax=1500, thresholds=config.CDF_THRESHOLDS_M)
plt.title(f"Error CDF on {best_fv}"); plt.show()

cols = [f"within_{t}m" for t in config.CDF_THRESHOLDS_M]
tab = res.loc[best_fv, cols].copy(); tab.columns = [f"≤{t} m" for t in config.CDF_THRESHOLDS_M]
print(f"Reproduced, {best_fv} (fraction of test samples):"); display((tab * 100).round(0).astype(int))
print("Paper Table 12 (best FV per model, mostly FV4/FV15/FV16 — not directly comparable):")
display((paper.CDF_TABLE12 * 100).round(0).astype(int))"""),
    md("""
Summary statistics. The paper's Eq. 20 SD is the population SD of the per-sample errors.
"""),
    code("""
stats = res[["mean_m", "median_m", "sd_m", "p90_m", "max_m", "r2_lat", "r2_lon"]]
display(stats.round(3))"""),
    md("## 4.3 Where are the errors? (Fig. 4)"),
    code("""
best = res.loc[best_fv, "mean_m"].idxmin()
r = experiment.run(best, best_fv, verbose=False)
ax, share = plotting.error_heatmap(r["y_true_deg"], r["errors_m"], cell_m=150)
ax.set_title(f"{best} on {best_fv}: mean error per 150 m cell"); plt.show()
print({k: f"{v:.1%}" for k, v in share.items()})
print("paper (Hybrid, best FV): 21.7% of cells < 250 m, 33% < 300 m, 42.6% < 350 m")"""),
    md("""
Large errors cluster on the periphery of the covered area, most visibly in the sparsely
sampled western part. There, few training fingerprints exist and messages are often heard by a
single gateway. The dense city centre has the lowest errors.

The paper's cell statistics (21.7 % / 33 % / 42.6 % of cells below 250 / 300 / 350 m) are for
the Hybrid on its best FV (FV4, mean 244.5 m), so they are not directly comparable with an
FV1/FV2 map.

## 4.4 Inference time (Fig. 5)
Per-sample prediction time, measured by predicting one sample at a time on 200 random test
samples. Absolute numbers depend on the hardware. Our CPU is not the authors' machine, so compare
the **ranking**, not the values.
"""),
    code("""
t = res.loc[best_fv, ["infer_single_mean_s", "infer_batched_s", "train_s"]].copy()
t["paper per-sample (s)"] = pd.Series(paper.INFERENCE_S)
display(t.rename(columns={"infer_single_mean_s": "ours per-sample (s)",
                          "infer_batched_s": "ours batched, per sample (s)",
                          "train_s": "ours training (s)"}).round(5))"""),
    md("""
## 4.5 All 16 feature vectors (JSON release, paper Tables 10–11)

The paper builds every feature vector from the JSON release. Running
`python scripts/run_experiments.py --source json --fv FV1 … FV16 --models k-NN XGBoost LightGBM ANN`
caches the results in `results/runs_json/`. The cell below trains whatever is missing, which is
fast for k-NN / XGBoost / LightGBM and takes a few minutes per FV for the ANN.
"""),
    code("""
json_dir = config.RESULTS_DIR / "runs_json"
ALL_FV = [f"FV{i}" for i in range(1, 17)]
if config.JSON_FILE.exists():
    for fv in ALL_FV:
        d = experiment.prepare_fv(fv, source="json")
        for name in ["k-NN", "XGBoost", "LightGBM", "ANN"]:
            experiment.run(name, fv, prepared=d, runs_dir=json_dir, verbose=False, metrics_only=True)
rj = experiment.collect(json_dir)
if rj.empty:
    print("No JSON results: put lorawan_antwerp_2019_dataset.json.txt in data/ and re-run.")
else:
    ours = rj.pivot(index="model", columns="fv", values="mean_m")
    ours = ours[[f for f in ALL_FV if f in ours.columns]]
    delta = ours - paper.MEAN_ERROR_M.loc[ours.index, ours.columns]
    print("Reproduced mean error (m):"); display(ours.round(1))
    print("Δ = ours − paper (m):"); display(delta.round(1))"""),
    code("""
if not rj.empty:
    models4 = [m for m in ["k-NN", "ANN", "XGBoost", "LightGBM"] if m in delta.index]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(delta.loc[models4].to_numpy(dtype=float), cmap="RdBu_r", vmin=-40, vmax=40, aspect="auto")
    ax.set_xticks(range(delta.shape[1]), delta.columns); ax.set_yticks(range(len(models4)), models4)
    for i, m in enumerate(models4):
        for j, fv in enumerate(delta.columns):
            v = delta.loc[m, fv]
            if pd.notna(v): ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8,
                                    color="white" if abs(v) > 25 else plotting.INK)
    ax.grid(False); ax.set_title("Reproduced − paper mean error (m); white ≈ exact")
    plt.colorbar(im, ax=ax, label="Δ (m)"); plt.show()"""),
    md("""
**Reading the heatmap.**
* **XGBoost and LightGBM are within 2 m of the paper on every feature vector.** The paper's
  feature-vector study therefore reproduces for the tree models.
* For XGBoost this needs `tree_method="exact"`. With the 256-bin histogram default of
  XGBoost ≥ 2.0, the frame counter (values up to 199,164) is coarsened, and the CNT feature
  vectors come out about 20 m worse. This is a good example of how a library default can quietly
  change a published result.
* k-NN is off only where the channel index enters a Manhattan distance (FV3, FV6, FV7).
* The ANN is the least stable model: it is 11–41 m worse on FV1–FV8 and close on FV9–FV16.

## 4.6 Do the results survive a stricter split? (beyond the paper)

`python scripts/split_check.py` retrains LightGBM under three protocols:
**random** (the paper's), **chronological** (train on the earliest 70 % of messages, test on the
latest 15 %), and **unseen devices** (whole vehicles held out).
"""),
    code("""
sc_file = config.RESULTS_DIR / "split_check.csv"
if sc_file.exists():
    sc = pd.read_csv(sc_file)
    tab = sc.pivot(index="fv", columns="split", values="mean_m")
    tab = tab.loc[[f for f in ALL_FV if f in tab.index], ["random", "chronological", "device"]]
    display(tab.round(1))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(tab)); w = 0.27
    for k, (col, colr) in enumerate(zip(tab.columns, ["#c3c2b7", plotting.MODEL_COLORS["k-NN"], plotting.MODEL_COLORS["CNN"]])):
        ax.bar(x + (k - 1) * (w + 0.01), tab[col], w, color=colr, label=col)
    ax.set_xticks(x, tab.index); ax.set(ylabel="Mean error (m)", title="LightGBM under three split protocols")
    ax.grid(axis="x", visible=False); ax.legend(); plt.show()
else:
    print("Run `python scripts/split_check.py` first (needs the JSON release).")"""),
    md("""
Why does the random split look so good? Measure how close each test message is to the nearest
*training* position:
"""),
    code("""
if config.JSON_FILE.exists():
    from sklearn.neighbors import BallTree
    dfj, _ = experiment.load_source("FV1", source="json")
    xy = np.column_stack(geo.to_local_xy_m(dfj.Latitude, dfj.Longitude))
    splits = {"random": pp.split_indices(len(dfj)), "chronological": pp.chronological_split(dfj.RxTime),
              "device": pp.device_split(dfj.DevEUI)}
    rows = []
    for name, (tr, va, te) in splits.items():
        dist = BallTree(xy[tr]).query(xy[te], k=1)[0][:, 0]
        cells = pd.Series(list(zip((xy[te, 0] // 50).astype(int), (xy[te, 1] // 50).astype(int)))).value_counts()
        rows.append({"split": name, "test messages": len(te), "test devices": dfj.DevEUI.iloc[te].nunique(),
                     "median dist. to nearest train point (m)": np.median(dist),
                     "within 10 m of a train point": np.mean(dist < 10),
                     "share in 10 busiest 50 m cells": cells.iloc[:10].sum() / len(te)})
    display(pd.DataFrame(rows).set_index("split").round(3))"""),
    md("""
**What this shows**
1. **The random split tests on places already in the training set.** The median test message is
   less than a metre from a training position. This is interpolation between near-duplicates,
   not localization of new places.
2. **The frame counter (CNT) is a leakage feature.** Under the random split it gives the largest
   gain (FV4 ≈ 253 m). Under the chronological split FV4 becomes the *worst* feature vector
   (≈ 555 m, almost twice the RSSI-only error), because there are no neighbouring messages from
   the same vehicle to look up.
3. **Chronological errors without CNT are lower than random**, but only because the February test
   period is dominated by a few stationary hotspots (a depot). The test distribution changed.
4. **Unseen devices nearly double the error** (≈ 560–575 m without CNT), even though most test
   positions are geographically covered. A fingerprint learned on some vehicles transfers poorly
   to others, which is consistent with device- or mounting-specific RSSI offsets.

**Lesson for your own work:** always report a split that matches the deployment scenario
(new time period, new devices) next to the random split.
"""),
    md("""
## 4.7 Discussion: what reproduces, what does not, and why

See the **Reproduction report** section of the repository `README.md` for the final numbers
from our run and a full list of deviations. The key points for students:

1. **The protocol reproduces exactly.** Dataset checksum, 44 active gateways, −200 → −128 dBm,
   and split sizes (91,300 / 19,564 / 19,565) all match the paper.
2. **XGBoost and LightGBM reproduce within 2 m on all 16 feature vectors** (section 4.5), and k-NN
   within ≈5 m except where the channel enters its distance; the full k-NN grid of Table 1 also
   reproduces. SVR and the Hybrid reproduce within about 1 % on FV1.
   The larger gaps are the ANN (+22 m on FV1; the learning rate is not reported, and 1e-4 was the
   best of the rates we tried), the CNN (+10 m on FV1), and **SVR on FV2 (+54 m)**. For SVR the paper
   says it used additional feature selection and random search beyond the grid, without giving
   details, so the FV2 configuration cannot be reproduced from the text. Other sources of gaps are
   unreported details (SVR γ, hybrid kernel size / T), random initialisation of the neural networks,
   and library versions.
3. **Differences between the tree models are within a few metres,** which is about the size of the
   run-to-run variation of the neural models. Rankings between close models should not be
   over-interpreted without repeated runs (different seeds) and confidence intervals.
4. **RSSI-only fingerprinting in this dataset has a floor of roughly 300 m on average**, set by
   sparse coverage (many messages heard by one gateway). The paper's gains to ≈245 m come from
   FVs with the **frame counter**, and section 4.6 shows they disappear under a chronological or
   device-held-out split.
5. **Median ≪ mean:** the error distribution has a heavy tail, so report median, 90th percentile
   and the CDF, not just the mean.
"""),
]


def write(name, cells):
    nb = nbf.v4.new_notebook()
    badge = md(f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
               f"(https://colab.research.google.com/github/drbasem/Outdoor_Localization_Based_on_LoRa_Networks/blob/{BRANCH}/notebooks/{name})")
    nb.cells = [cells[0], badge, code(COLAB)] + cells[1:]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    path = NB_DIR / name
    nbf.write(nb, path)
    print("wrote", path.relative_to(NB_DIR.parent))


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    write("01_dataset_exploration.ipynb", NB1)
    write("02_preprocessing.ipynb", NB2)
    write("03_models.ipynb", NB3)
    write("04_results.ipynb", NB4)
