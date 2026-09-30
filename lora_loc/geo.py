"""Converting between GPS coordinates and metres.

The paper reports "the Euclidean distance (in meters) between the predicted device
location and the ground truth", without saying how degrees are turned into metres. We
use the **haversine** great-circle distance on a sphere of radius 6,371,008.8 m (the
IUGG mean Earth radius). Over the ~9 km x 6 km Antwerp area, haversine and a local
Euclidean (east-north) distance differ by far less than a metre, so the choice does not
affect any reported number.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres between arrays of points given in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distance_errors_m(y_true_deg: np.ndarray, y_pred_deg: np.ndarray) -> np.ndarray:
    """Per-sample localization error (m) for ``(n, 2)`` arrays of ``[lat, lon]``."""
    return haversine_m(y_true_deg[:, 0], y_true_deg[:, 1], y_pred_deg[:, 0], y_pred_deg[:, 1])


def to_local_xy_m(lat, lon, lat0: float | None = None, lon0: float | None = None):
    """Equirectangular projection to local east (x) / north (y) metres, used for plotting."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    lat0 = float(np.mean(lat)) if lat0 is None else lat0
    lon0 = float(np.mean(lon)) if lon0 is None else lon0
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y
