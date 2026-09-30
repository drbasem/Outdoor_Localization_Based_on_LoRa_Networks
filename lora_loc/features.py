"""Feature-vector (FV) definitions from Table 9 of the paper.

Each FV is a list of feature *groups*:

=======  ===================================================  ============  =============
Group    Meaning                                              Width         Source file
=======  ===================================================  ============  =============
RSSI     received signal strength per gateway                 44            CSV or JSON
SF       spreading factor (7-12)                              1             CSV or JSON
CH       TX channel (EU868 channel index)                     1             JSON only
CNT      device uplink frame counter                          1             JSON only
AT       airtime of the packet (s)                            1             JSON only
SNR      signal-to-noise ratio per gateway                    44            JSON only
ESP      estimated signal power per gateway                   44            JSON only
=======  ===================================================  ============  =============

**[ASSUMPTION]** The paper does not say whether SNR and ESP are per-gateway vectors or
scalars. We treat them as per-gateway vectors (44 columns), like RSSI. Then RSSI + SNR
has 88 inputs, which matches the input width seen in the authors' supplementary notebook
(88 inputs to the first ANN/CNN layer).
"""
from __future__ import annotations

import pandas as pd

FEATURE_VECTORS: dict[str, list[str]] = {
    "FV1": ["RSSI"],
    "FV2": ["RSSI", "SF"],
    "FV3": ["RSSI", "SF", "CH"],
    "FV4": ["RSSI", "SF", "CNT"],
    "FV5": ["RSSI", "SF", "AT"],
    "FV6": ["RSSI", "SF", "CH", "CNT"],
    "FV7": ["RSSI", "SF", "CH", "CNT", "AT"],
    "FV8": ["RSSI", "SF", "CNT", "AT"],
    "FV9": ["RSSI", "SNR"],
    "FV10": ["RSSI", "ESP"],
    "FV11": ["RSSI", "SF", "SNR"],
    "FV12": ["RSSI", "SF", "ESP"],
    "FV13": ["RSSI", "SF", "SNR", "ESP"],
    "FV14": ["RSSI", "SF", "SNR", "ESP", "CH"],
    "FV15": ["RSSI", "SF", "SNR", "ESP", "CNT"],
    "FV16": ["RSSI", "SF", "SNR", "ESP", "CNT", "AT"],
}

#: Feature vectors that can be built from the CSV release alone.
CSV_FEATURE_VECTORS = ("FV1", "FV2")

_SCALAR = {"SF": "SF", "CH": "Channel", "CNT": "Counter", "AT": "Airtime"}


def columns_for(fv: str, df: pd.DataFrame, rssi_cols: list[str]) -> tuple[list[str], list[str]]:
    """Return ``(rssi_cols, other_cols)`` for feature vector ``fv`` in ``df``.

    Raises ``KeyError`` with a clear message if ``df`` lacks a required group, e.g. when
    FV4 (which needs the counter) is asked for on the CSV release.
    """
    other: list[str] = []
    for group in FEATURE_VECTORS[fv][1:]:
        if group in _SCALAR:
            col = _SCALAR[group]
            if col not in df.columns:
                raise KeyError(f"{fv} needs '{col}', which is only in the JSON release.")
            other.append(col)
        else:  # SNR / ESP per gateway
            cols = [c for c in df.columns if c.startswith(f"{group}_")]
            if not cols:
                raise KeyError(f"{fv} needs per-gateway {group}, which is only in the JSON release.")
            other.extend(cols)
    return list(rssi_cols), other
