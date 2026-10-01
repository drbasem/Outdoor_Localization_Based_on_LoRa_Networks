"""Preprocessing pipeline, following Section IV of the paper.

Steps, in order
---------------
1. **Drop dead gateways.** Keep the 44 gateways that received at least one message.
2. **Replace the "no signal" placeholder.** Observed RSSI lies in [-127, -60] dBm and nothing
   falls between -200 and -127, so -200 is replaced by the neighbouring value -128 dBm
   ([P] Sec. IV, following Bagherian et al.). The RSSI range becomes [-128, -60] dBm.
3. **Logarithmic transform** ([P] Eq. 12)::

        x_hat = log10(|x_RSSI| + 1)

   RSSI values in dB are already logarithmic and are compressed near the noise floor.
   The transform spreads out weak values relative to strong ones before scaling.
   Because ``|x|`` is taken, *strong* signals (-60 dBm -> log10(61) = 1.785) map to
   *small* values and "no signal" (-128 -> log10(129) = 2.111) maps to the largest value.
4. **Min-max scale RSSI to [0, 1]** using one global minimum and maximum over all RSSI
   columns ([P] Eq. 13, which has no per-gateway subscript on min/max). A single range
   keeps the relative level between gateways, which is what a fingerprint encodes.
5. **Min-max scale every other feature separately** (SF, SNR, ESP, channel, counter,
   airtime) ([P] Eq. 14).
6. **Min-max scale the targets** (latitude, longitude) to [0, 1] ([P] Eq. 16). Predictions
   are converted back to degrees before computing errors in metres.

Leakage note
------------
The paper does not say whether the scalers were fit on the whole dataset or only on the
training split. We **fit on the training split only**, which is the standard practice.
For RSSI the choice barely matters: the training split spans [-128, -65] dBm versus
[-128, -60] dBm for the full data, so a handful of very strong test readings map slightly
below 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


# --------------------------------------------------------------------------------------
# Elementary transforms (pure functions; easy to test and to show in the notebooks)
# --------------------------------------------------------------------------------------
def replace_no_signal(x: np.ndarray, placeholder: float = config.NO_SIGNAL_DBM,
                      replacement: float | None = config.NO_SIGNAL_REPLACEMENT_DBM) -> np.ndarray:
    """Replace the ``placeholder`` value (no reception).

    If ``replacement`` is ``None``, use one unit below the smallest observed value. For
    RSSI that is exactly the paper's -128 dBm.
    """
    x = np.asarray(x, dtype=float).copy()
    mask = x == placeholder
    if replacement is None:
        observed = x[~mask]
        replacement = (observed.min() - 1.0) if observed.size else 0.0
    x[mask] = replacement
    return x


def log_transform(x: np.ndarray) -> np.ndarray:
    """Paper Eq. (12): ``log10(|x| + 1)``."""
    return np.log10(np.abs(np.asarray(x, dtype=float)) + 1.0)


@dataclass
class MinMax:
    """Min-max scaler to [0, 1].

    ``global_range=True`` uses one (min, max) pair for the whole matrix (paper Eq. 13);
    otherwise each column gets its own range (paper Eq. 14 and 16).
    """

    global_range: bool = False
    min_: np.ndarray | float | None = None
    max_: np.ndarray | float | None = None

    def fit(self, x: np.ndarray) -> "MinMax":
        x = np.asarray(x, dtype=float)
        axis = None if self.global_range else 0
        self.min_, self.max_ = x.min(axis=axis), x.max(axis=axis)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        span = np.where(np.asarray(self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        return (np.asarray(x, dtype=float) - self.min_) / span

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) * (self.max_ - self.min_) + self.min_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# --------------------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------------------
def split_indices(n: int, seed: int = config.SEED):
    """70/15/15 split of ``range(n)``.

    With ``n = 130,429`` and ``seed = 42`` this gives exactly the paper's sizes: 91,300
    train, 19,564 validation and 19,565 test.
    """
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=config.TEST_VAL_FRACTION, random_state=seed)
    va, te = train_test_split(tmp, test_size=config.TEST_FRACTION_OF_HOLDOUT, random_state=seed)
    return tr, va, te


# --------------------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------------------
@dataclass
class FeaturePipeline:
    """Fit on the training rows, then transform any split.

    Parameters
    ----------
    rssi_cols
        Gateway RSSI columns. They get the placeholder replacement, the log transform and
        one global min-max range.
    other_cols
        Extra features (SF, SNR_*, ESP_*, Channel, Counter, Airtime). Each column is
        min-max scaled on its own. Per-gateway SNR/ESP columns that contain the -200
        placeholder get the paper's "neighbouring value" rule: placeholder -> (observed
        minimum - 1). **[ASSUMPTION]** The paper describes this rule for RSSI only.
    """

    rssi_cols: list[str]
    other_cols: list[str] = field(default_factory=list)
    rssi_scaler: MinMax = field(default_factory=lambda: MinMax(global_range=True))
    other_scaler: MinMax = field(default_factory=MinMax)
    target_scaler: MinMax = field(default_factory=MinMax)
    _other_fill: dict = field(default_factory=dict)

    # -- features ----------------------------------------------------------------------
    def _rssi(self, df: pd.DataFrame) -> np.ndarray:
        return log_transform(replace_no_signal(df[self.rssi_cols].to_numpy()))

    def _other(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if not self.other_cols:
            return np.empty((len(df), 0))
        x = df[self.other_cols].to_numpy(dtype=float).copy()
        for j, c in enumerate(self.other_cols):
            col = x[:, j]
            mask = col == config.NO_SIGNAL_DBM
            if fit:
                obs = col[~mask]
                self._other_fill[c] = (obs.min() - 1.0) if obs.size else 0.0
            col[mask] = self._other_fill[c]
        return x

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "FeaturePipeline":
        self.rssi_scaler.fit(self._rssi(df))
        if self.other_cols:
            self.other_scaler.fit(self._other(df, fit=True))
        self.target_scaler.fit(y)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Paper Eq. (15): ``X = [X_RSSI_scaled, X_other_scaled]``."""
        parts = [self.rssi_scaler.transform(self._rssi(df))]
        if self.other_cols:
            parts.append(self.other_scaler.transform(self._other(df, fit=False)))
        return np.hstack(parts).astype(np.float32)

    # -- targets -----------------------------------------------------------------------
    def scale_y(self, y: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(y).astype(np.float32)

    def unscale_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(y_scaled)


@dataclass
class PreparedData:
    """Everything a model needs: scaled splits, the fitted pipeline and raw targets."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray  # scaled to [0, 1]
    y_val: np.ndarray
    y_test: np.ndarray
    y_test_deg: np.ndarray  # ground truth in degrees, for metre-level errors
    pipeline: FeaturePipeline
    feature_names: list[str]
    idx_test: np.ndarray


def chronological_split(times: pd.Series, fractions=(0.70, 0.15, 0.15)):
    """Train on the earliest 70 % of messages, validate on the next 15 %, test on the last 15 %.

    This removes the "temporal neighbour" shortcut of the random split: no test message has
    a training message from the same vehicle sent seconds earlier.
    """
    t = pd.to_datetime(times, utc=True, format="ISO8601")
    order = np.argsort(t.to_numpy(), kind="stable")
    n = len(order)
    a, b = int(round(fractions[0] * n)), int(round((fractions[0] + fractions[1]) * n))
    return order[:a], order[a:b], order[b:]


def device_split(devices: pd.Series, test_fraction: float = 0.15, seed: int = config.SEED):
    """Hold out whole devices: test devices are never seen during training.

    Devices are shuffled and added to the test set until it holds ~``test_fraction`` of the
    messages; the remaining messages are split 85/15 into train/validation at random.
    """
    rng = np.random.default_rng(seed)
    dev = devices.to_numpy()
    uniq = rng.permutation(pd.unique(dev))
    counts = pd.Series(dev).value_counts()
    test_devs, total = [], 0
    for u in uniq:
        if total >= test_fraction * len(dev):
            break
        test_devs.append(u)
        total += counts[u]
    is_test = np.isin(dev, test_devs)
    rest = np.where(~is_test)[0]
    tr, va = train_test_split(rest, test_size=0.15 / 0.85, random_state=seed)
    return tr, va, np.where(is_test)[0]


def prepare(df: pd.DataFrame, rssi_cols: list[str], other_cols: list[str] | None = None,
            seed: int = config.SEED, split=None) -> PreparedData:
    """Split ``df``, fit the pipeline on the training rows and transform all three splits.

    ``split`` may be a precomputed ``(train_idx, val_idx, test_idx)`` tuple, e.g. from
    :func:`chronological_split`; by default the paper's random 70/15/15 split is used.
    """
    other_cols = list(other_cols or [])
    y = df[["Latitude", "Longitude"]].to_numpy(dtype=float)
    tr, va, te = split if split is not None else split_indices(len(df), seed)
    pipe = FeaturePipeline(rssi_cols=list(rssi_cols), other_cols=other_cols)
    pipe.fit(df.iloc[tr], y[tr])
    return PreparedData(
        X_train=pipe.transform(df.iloc[tr]),
        X_val=pipe.transform(df.iloc[va]),
        X_test=pipe.transform(df.iloc[te]),
        y_train=pipe.scale_y(y[tr]),
        y_val=pipe.scale_y(y[va]),
        y_test=pipe.scale_y(y[te]),
        y_test_deg=y[te],
        pipeline=pipe,
        feature_names=list(rssi_cols) + other_cols,
        idx_test=te,
    )
