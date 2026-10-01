"""Loading the LoRaWAN Antwerp 2019 dataset (Zenodo record 3342253, v1.2).

The record ships the same 130,429 messages in two formats:

``lorawan_antwerp_2019_dataset.csv``
    One row per message, 77 columns: ``BS 1`` ... ``BS 72`` (RSSI in dBm, ``-200`` = not
    received), ``RX Time``, ``SF``, ``HDOP``, ``Latitude``, ``Longitude``.
    This file is enough for the paper's feature vectors **FV1** (RSSI) and **FV2** (RSSI+SF).

``lorawan_antwerp_2019_dataset.json.txt``
    Same messages with the full per-gateway metadata (``rssi``, ``snr``, ``esp``, ...) and
    per-message fields (``sf``, ``channel``, ``counter``, ``airtime``, ...). It is needed
    for **FV3-FV16**, which use SNR, ESP, channel, counter or airtime.

The schema below follows the description on the Zenodo record page::

    {"hdop": 0.7, "dev_addr": "...", "payload": "...", "adr": 1,
     "latitude": 51.2085, "longitude": 4.4005, "counter": 31952, "airtime": 0.112896,
     "gateways": [{"rssi": -115, "esp": -115.83, "snr": 6.75,
                   "rx_time": {"ts_type": "None", "time": "..."}, "id": "08060716"}, ...],
     "dev_eui": "...", "sf": 7, "channel": 8}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TARGET_COLUMNS = ["Latitude", "Longitude"]


# --------------------------------------------------------------------------------------
# Integrity check
# --------------------------------------------------------------------------------------
def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hexadecimal MD5 digest of ``path``."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path) -> bool:
    """Compare a downloaded file with the checksum published on Zenodo."""
    expected = config.MD5.get(Path(path).name)
    return expected is not None and md5sum(path) == expected


# --------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------
def load_csv(path: Path | str = config.CSV_FILE) -> pd.DataFrame:
    """Load the CSV release as a DataFrame (130,429 x 77)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` or see data/README.md."
        )
    return pd.read_csv(path)


def rssi_columns(df: pd.DataFrame) -> list[str]:
    """Names of the gateway RSSI columns (``BS 1`` ... ``BS 72``)."""
    return [c for c in df.columns if c.startswith("BS ")]


def active_gateways(df: pd.DataFrame, no_signal: int = config.NO_SIGNAL_DBM) -> list[str]:
    """Gateways that received at least one message.

    Of the 72 ``BS`` columns, 28 are ``-200`` in every row, so only **44** gateways carry
    information. This matches the paper ("messages were received by only 44 gateways").
    """
    cols = rssi_columns(df)
    return [c for c in cols if (df[c] != no_signal).any()]


# --------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------
def _iter_json_messages(path: Path):
    """Yield message dicts from a JSON array file or a JSON-lines file."""
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        while head and head.isspace():
            head = f.read(1)
        f.seek(0)
        if head == "[":
            yield from json.load(f)
        else:
            for line in f:
                line = line.strip().rstrip(",")
                if line:
                    yield json.loads(line)


def load_json(path: Path | str = config.JSON_FILE,
              no_signal: float = config.NO_SIGNAL_DBM) -> pd.DataFrame:
    """Flatten the JSON release to one row per message.

    Output columns
    --------------
    ``RSSI_<id>``, ``SNR_<id>``, ``ESP_<id>``
        One column per gateway that ever received a message, sorted by gateway id. As in
        the paper, missing entries get the placeholder ``-200``. (The paper does not say
        what placeholder it used for missing SNR/ESP; we use the same one so every
        per-gateway block is handled the same way. The -200 is replaced later by
        :func:`lora_loc.preprocessing.replace_no_signal`.)
    ``SF``, ``Channel``, ``Counter``, ``Airtime``, ``HDOP``
        Per-message scalars.
    ``Latitude``, ``Longitude``
        Ground truth.
    ``DevEUI``, ``RxTime``
        Device identifier and earliest gateway reception time. They are not model inputs;
        they are used only for the chronological and device-held-out split checks.

    Notes
    -----
    The CSV numbers gateways ``BS 1`` ... ``BS 72``, while the JSON uses hexadecimal gateway
    ids. The two releases do not publish a mapping, so JSON-derived features are kept
    separate from the CSV and should not be mixed column by column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The JSON file is required for FV3-FV16; see data/README.md."
        )
    rows, gw_ids = [], set()
    for msg in _iter_json_messages(path):
        row = {
            "SF": msg.get("sf"),
            "Channel": msg.get("channel"),
            "Counter": msg.get("counter"),
            "Airtime": msg.get("airtime"),
            "HDOP": msg.get("hdop"),
            "Latitude": msg.get("latitude"),
            "Longitude": msg.get("longitude"),
            "DevEUI": msg.get("dev_eui"),
            # earliest reception time over all gateways (ISO-8601 string)
            "RxTime": min((gw.get("rx_time", {}).get("time", "") for gw in msg.get("gateways", [])),
                          default=None),
        }
        for gw in msg.get("gateways", []):
            gid = str(gw["id"])
            gw_ids.add(gid)
            row[f"RSSI_{gid}"] = gw.get("rssi")
            row[f"SNR_{gid}"] = gw.get("snr")
            row[f"ESP_{gid}"] = gw.get("esp")
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered = sorted(gw_ids)
    blocks = []
    for prefix in ("RSSI", "SNR", "ESP"):
        cols = [f"{prefix}_{g}" for g in ordered]
        blocks.append(df.reindex(columns=cols).astype(float).fillna(no_signal))
    scalars = df[["SF", "Channel", "Counter", "Airtime", "HDOP"] + TARGET_COLUMNS + ["DevEUI", "RxTime"]]
    return pd.concat(blocks + [scalars.reset_index(drop=True)], axis=1)


def targets(df: pd.DataFrame) -> np.ndarray:
    """Return an ``(n, 2)`` array of ``[latitude, longitude]`` in degrees."""
    return df[TARGET_COLUMNS].to_numpy(dtype=float)
