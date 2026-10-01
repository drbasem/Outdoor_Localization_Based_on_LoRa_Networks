"""The seven localization models evaluated in the paper.

Every model follows the same small interface so that the experiment code can treat them
all the same way::

    model = build("XGBoost")
    model.fit(X_train, y_train, X_val, y_val)   # y is min-max scaled [lat, lon]
    y_hat = model.predict(X_test)               # (n, 2), still scaled

The validation split is used only by models that early-stop (ANN, XGBoost, LightGBM and
the Hybrid). The others ignore it, as in the paper.

Hyperparameters come from :mod:`lora_loc.config`. See that module for where each value
comes from.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

# LightGBM >= 4.6 renamed `eval_set`; we keep the older spelling for compatibility.
warnings.filterwarnings("ignore", message=".*eval_set.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*np.ndarray subset.*")

from . import config

MODEL_NAMES = ("k-NN", "CNN", "SVR", "ANN", "XGBoost", "LightGBM", "Hybrid")


class Localizer:
    """Base class: fit on scaled ``[lat, lon]`` and predict scaled ``[lat, lon]``."""

    name = "base"

    def fit(self, X, y, X_val=None, y_val=None):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


# ======================================================================================
# 1. k-Nearest Neighbours
# ======================================================================================
class KNN(Localizer):
    """k-NN regression: the position is the mean position of the k closest fingerprints.

    This is the classic fingerprinting baseline. There is no training step: the model
    stores the training set, and ``p`` selects the Minkowski distance (1 = Manhattan,
    2 = Euclidean).
    """

    name = "k-NN"

    def __init__(self, n_neighbors=config.KNN["n_neighbors"], p=config.KNN["p"]):
        from sklearn.neighbors import KNeighborsRegressor

        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, p=p, n_jobs=-1)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ======================================================================================
# 2. Support Vector Regression (one model per coordinate)
# ======================================================================================
class SVRLocalizer(Localizer):
    """RBF-kernel SVR, with separate models SVR-LAT and SVR-LON.

    SVR cannot predict two outputs at once, so we fit one model per coordinate, as the
    paper does. Training cost grows roughly as O(n^2) to O(n^3) in the number of samples,
    so on the full 91,300-sample training set this is by far the slowest model on CPU.
    Pass ``max_train`` to fit on a random subset when you just want to try it out.
    """

    name = "SVR"

    def __init__(self, max_train: int | None = None, **kw):
        from sklearn.svm import SVR

        params = {**config.SVR, "gamma": config.SVR_GAMMA, "cache_size": 4000, **kw}
        self.lat, self.lon = SVR(**params), SVR(**params)
        self.max_train = max_train

    def fit(self, X, y, X_val=None, y_val=None):
        if self.max_train and len(X) > self.max_train:
            idx = np.random.default_rng(config.SEED).choice(len(X), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self.lat.fit(X, y[:, 0])
        self.lon.fit(X, y[:, 1])
        return self

    def predict(self, X):
        return np.column_stack([self.lat.predict(X), self.lon.predict(X)])


# ======================================================================================
# 3. XGBoost (one multi-output model)
# ======================================================================================
class XGB(Localizer):
    """Gradient-boosted trees, one model with two outputs, early-stopped on validation RMSE."""

    name = "XGBoost"

    def __init__(self, **kw):
        from xgboost import XGBRegressor

        p = {**config.XGB, **kw}
        self.model = XGBRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], early_stopping_rounds=p["early_stopping_rounds"],
            eval_metric="rmse", tree_method=p["tree_method"], random_state=config.SEED, n_jobs=-1,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ======================================================================================
# 4. LightGBM (one model per coordinate)
# ======================================================================================
class LGBM(Localizer):
    """Leaf-wise gradient boosting, with separate LightGBM-LAT and LightGBM-LON models."""

    name = "LightGBM"

    def __init__(self, **kw):
        from lightgbm import LGBMRegressor

        p = {**config.LGBM, **kw}
        self.rounds = p.pop("early_stopping_rounds")
        self.lat = LGBMRegressor(**p, n_jobs=-1, verbose=-1)
        self.lon = LGBMRegressor(**p, n_jobs=-1, verbose=-1)

    def fit(self, X, y, X_val=None, y_val=None):
        from lightgbm import early_stopping

        for j, m in enumerate((self.lat, self.lon)):
            yj, yvj = np.ascontiguousarray(y[:, j]), np.ascontiguousarray(y_val[:, j])
            m.fit(X, yj, eval_set=[(X_val, yvj)], eval_metric="rmse",
                  callbacks=[early_stopping(self.rounds, verbose=False)])
        return self

    def predict(self, X):
        return np.column_stack([self.lat.predict(X), self.lon.predict(X)])


# ======================================================================================
# Neural networks (Keras)
# ======================================================================================
def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(config.SEED)
    return tf.keras


class _KerasLocalizer(Localizer):
    """Shared training loop for the CNN and the ANN."""

    epochs = 100
    batch_size = 32
    patience: int | None = None
    verbose = 0

    def _build(self, n_features: int):  # pragma: no cover - interface
        raise NotImplementedError

    def _reshape(self, X):
        return X

    def fit(self, X, y, X_val=None, y_val=None):
        keras = _keras()
        self.model = self._build(X.shape[1])
        callbacks = []
        if self.patience:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True))
        val = (self._reshape(X_val), y_val) if X_val is not None else None
        self.history = self.model.fit(
            self._reshape(X), y, validation_data=val, epochs=self.epochs,
            batch_size=self.batch_size, callbacks=callbacks, verbose=self.verbose)
        return self

    def predict(self, X):
        return self.model.predict(self._reshape(X), batch_size=4096, verbose=0)


class CNN(_KerasLocalizer):
    """1-D CNN that treats the gateway vector as a sequence.

    Conv1D(32, k=3) -> Conv1D(64, k=3) -> Flatten -> Dense(128, ReLU) -> Dropout(0.2) ->
    Dense(2, linear). Trained with Adam (lr = 1e-3) and MSE loss for a fixed number of
    epochs (100 in the paper's final comparison).

    Note that neighbouring columns (``BS 4``, ``BS 5``, ...) are not necessarily
    neighbouring gateways in space, so the "local patterns" a 1-D convolution picks up
    depend on the column order.
    """

    name = "CNN"

    def __init__(self, epochs=config.CNN["epochs"], batch_size=config.CNN["batch_size"]):
        self.epochs, self.batch_size = epochs, batch_size

    def _reshape(self, X):
        return None if X is None else X[..., None]

    def _build(self, n_features):
        keras = _keras()
        c = config.CNN
        model = keras.Sequential([
            keras.Input(shape=(n_features, 1)),
            keras.layers.Conv1D(c["filters"][0], c["kernel_size"], activation="relu"),
            keras.layers.Conv1D(c["filters"][1], c["kernel_size"], activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(c["dense_units"], activation="relu"),
            keras.layers.Dropout(c["dropout"]),
            keras.layers.Dense(2, activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(c["learning_rate"]), loss="mse")
        return model


class ANN(_KerasLocalizer):
    """Fully connected network: 256-256-128-64 ReLU units with Dropout(0.1) after each.

    Trained with Adam, batch size 64, for at most 150 epochs, stopping early when the
    validation loss has not improved for 10 epochs (the best weights are restored).
    """

    name = "ANN"

    def __init__(self, max_epochs=config.ANN["max_epochs"], batch_size=config.ANN["batch_size"]):
        self.epochs, self.batch_size = max_epochs, batch_size
        self.patience = config.ANN["patience"]

    def _build(self, n_features):
        keras = _keras()
        c = config.ANN
        layers = [keras.Input(shape=(n_features,))]
        for units in c["hidden_units"]:
            layers += [keras.layers.Dense(units, activation="relu"), keras.layers.Dropout(c["dropout"])]
        layers.append(keras.layers.Dense(2, activation="linear"))
        model = keras.Sequential(layers)
        model.compile(optimizer=keras.optimizers.Adam(c["learning_rate"]), loss="mse")
        return model


# ======================================================================================
# 5. Hybrid: CNN feature extractor + XGBoost regressors
# ======================================================================================
class Hybrid(Localizer):
    """CNN embeddings concatenated with the raw features, fed to XGBoost ([P] Eq. 8).

    1. Train a CNN: Conv1D(32) -> Conv1D(64) -> Flatten -> Dense(64, ReLU) -> Dense(2).
    2. Take the 64-dimensional output of the ``Dense(64)`` layer as an embedding ``z``.
    3. Build ``x_tilde = [x, z]`` and train two XGBoost models (400 trees, depth 12,
       lr = 0.09), one for latitude and one for longitude.

    The paper's Eq. (7) mentions a temporal window of T consecutive messages, but T is not
    reported. We use T = 1 (each message on its own), which is what the rest of the paper
    evaluates. **[ASSUMPTION]**
    """

    name = "Hybrid"

    def __init__(self, epochs=config.HYBRID["epochs"], batch_size=config.HYBRID["batch_size"]):
        self.epochs, self.batch_size = epochs, batch_size

    def _embed(self, X):
        return self.encoder.predict(X[..., None], batch_size=4096, verbose=0)

    def _augment(self, X):
        return np.hstack([X, self._embed(X)])

    def fit(self, X, y, X_val=None, y_val=None):
        from xgboost import XGBRegressor

        keras = _keras()
        c = config.HYBRID
        inp = keras.Input(shape=(X.shape[1], 1))
        h = keras.layers.Conv1D(c["filters"][0], c["kernel_size"], activation="relu")(inp)
        h = keras.layers.Conv1D(c["filters"][1], c["kernel_size"], activation="relu")(h)
        h = keras.layers.Flatten()(h)
        z = keras.layers.Dense(c["embedding_units"], activation="relu", name="embedding")(h)
        out = keras.layers.Dense(2, activation="linear")(z)
        self.cnn = keras.Model(inp, out)
        self.cnn.compile(optimizer=keras.optimizers.Adam(c["learning_rate"]), loss="mse")
        self.cnn.fit(X[..., None], y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.encoder = keras.Model(inp, z)

        Xa, Xva = self._augment(X), self._augment(X_val)
        self.xgb = []
        for j in range(2):
            m = XGBRegressor(n_estimators=c["xgb_n_estimators"], max_depth=c["xgb_max_depth"],
                             learning_rate=c["xgb_learning_rate"], tree_method=c["xgb_tree_method"],
                             early_stopping_rounds=c["early_stopping_rounds"], eval_metric="rmse",
                             random_state=config.SEED, n_jobs=-1)
            m.fit(Xa, y[:, j], eval_set=[(Xva, y_val[:, j])], verbose=False)
            self.xgb.append(m)
        return self

    def predict(self, X):
        Xa = self._augment(X)
        return np.column_stack([m.predict(Xa) for m in self.xgb])


# ======================================================================================
_REGISTRY = {"k-NN": KNN, "CNN": CNN, "SVR": SVRLocalizer, "ANN": ANN,
             "XGBoost": XGB, "LightGBM": LGBM, "Hybrid": Hybrid}


def build(name: str, **kwargs) -> Localizer:
    """Create a model by the name used in the paper, e.g. ``build("LightGBM")``."""
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError:
        raise KeyError(f"Unknown model {name!r}; choose from {list(_REGISTRY)}") from None
