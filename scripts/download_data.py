#!/usr/bin/env python
"""Download the LoRaWAN Antwerp 2019 dataset from Zenodo and verify its MD5 checksums.

    python scripts/download_data.py            # CSV only (enough for FV1, FV2)
    python scripts/download_data.py --json     # also the 96 MB JSON (needed for FV3-FV16)

Files go to ./data. Source: Aernouts et al., "Sigfox and LoRaWAN Datasets for Fingerprint
Localization in Large Urban and Rural Areas", Zenodo, doi:10.5281/zenodo.3342253 (v1.2).
If your network blocks zenodo.org, download the files in a browser from
https://zenodo.org/records/3342253 and put them in ./data. The script then only checks them.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lora_loc import config, data  # noqa: E402

URL = "https://zenodo.org/records/{rec}/files/{name}?download=1"


def fetch(name: str) -> None:
    dest = config.DATA_DIR / name
    if dest.exists() and data.verify(dest):
        print(f"OK   {name} (checksum verified)")
        return
    url = URL.format(rec=config.ZENODO_RECORD, name=name)
    print(f"GET  {url}")
    config.DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    status = "verified" if data.verify(dest) else "MISMATCH - file differs from Zenodo v1.2!"
    print(f"     -> {dest} ({status})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="also download the JSON release")
    args = ap.parse_args()
    fetch("lorawan_antwerp_2019_dataset.csv")
    if args.json:
        fetch("lorawan_antwerp_2019_dataset.json.txt")
