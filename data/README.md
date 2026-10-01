# Data

The dataset files are **not** stored in git (they are 56 MB and 96 MB). Get them with:

```bash
python scripts/download_data.py          # CSV  -> FV1, FV2 experiments
python scripts/download_data.py --json   # JSON -> all 16 feature vectors
```

| File | Size | MD5 (from Zenodo) | Used for |
|---|---|---|---|
| `lorawan_antwerp_2019_dataset.csv` | 56.5 MB | `9c336dd917074c19b2868dfd71f3774b` | RSSI of 72 gateways + SF, HDOP, lat/lon → **FV1, FV2** |
| `lorawan_antwerp_2019_dataset.json.txt` | 96.1 MB | `c3a6f09bfa1d5029c236a93e83e05196` | per-gateway RSSI/SNR/ESP + channel, counter, airtime → **FV3–FV16** |

Version **1.3** of the record ([10.5281/zenodo.3904158](https://doi.org/10.5281/zenodo.3904158))
contains byte-identical CSV and JSON files (same MD5). It only adds
`lorawan_antwerp_gateway_locations.json.txt` (MD5 `5f80a045617ae7b889737aa6850b7008`), the gateway
coordinates, which you can also place here.

The JSON has 130,430 messages and 44 gateways; the CSV has 130,429 rows. They contain the same
messages in a different order, and the JSON has one extra message.

**Source.** M. Aernouts, R. Berkvens, K. Van Vlaenderen, M. Weyn, *Sigfox and LoRaWAN
Datasets for Fingerprint Localization in Large Urban and Rural Areas*, Zenodo, v1.2
(19 July 2019), doi:[10.5281/zenodo.3342253](https://doi.org/10.5281/zenodo.3342253).
Collection method: M. Aernouts et al., *Data* 3(2):13, 2018,
doi:[10.3390/data3020013](https://doi.org/10.3390/data3020013).

If `zenodo.org` is blocked on your network, download the files in a browser from the record
page and copy them here. `download_data.py` then only verifies the checksums.
