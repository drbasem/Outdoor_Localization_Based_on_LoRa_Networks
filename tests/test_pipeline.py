"""Unit tests: run with `pytest -q`. They do not need the dataset."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lora_loc import data, features, geo, preprocessing as pp  # noqa: E402


def test_replace_no_signal_uses_paper_value():
    x = np.array([[-200, -60], [-127, -200]])
    assert (pp.replace_no_signal(x) == [[-128, -60], [-127, -128]]).all()


def test_replace_no_signal_neighbour_rule():
    x = np.array([-200.0, -9.0, 6.75])
    assert pp.replace_no_signal(x, replacement=None)[0] == -10.0


def test_log_transform_eq12():
    assert pp.log_transform(np.array([-60.0]))[0] == pytest.approx(np.log10(61))
    assert pp.log_transform(np.array([-128.0]))[0] == pytest.approx(np.log10(129))


def test_minmax_global_vs_per_column():
    x = np.array([[0.0, 10.0], [5.0, 20.0]])
    g = pp.MinMax(global_range=True).fit_transform(x)
    c = pp.MinMax().fit_transform(x)
    assert g.max() == 1 and g[0, 0] == 0 and g[1, 0] == pytest.approx(0.25)
    assert (c == [[0, 0], [1, 1]]).all()
    s = pp.MinMax().fit(x)
    np.testing.assert_allclose(s.inverse_transform(s.transform(x)), x)


def test_split_sizes_match_paper():
    tr, va, te = pp.split_indices(130_429)
    assert (len(tr), len(va), len(te)) == (91_300, 19_564, 19_565)
    assert len(set(tr) | set(va) | set(te)) == 130_429


def test_haversine_known_distance():
    # One degree of latitude is about 111.2 km on the mean-radius sphere.
    assert geo.haversine_m(51.0, 4.4, 52.0, 4.4) == pytest.approx(111_195, rel=1e-3)
    assert geo.haversine_m(51.2, 4.4, 51.2, 4.4) == 0


def test_feature_vector_table():
    assert len(features.FEATURE_VECTORS) == 16
    df = pd.DataFrame({"BS 1": [-100], "SF": [7], "Latitude": [51.2], "Longitude": [4.4]})
    assert features.columns_for("FV2", df, ["BS 1"]) == (["BS 1"], ["SF"])
    with pytest.raises(KeyError):
        features.columns_for("FV4", df, ["BS 1"])


def test_json_loader_on_documented_example(tmp_path):
    """The example message from the Zenodo record page, plus a second one."""
    msg = {"hdop": 0.7, "dev_addr": "07000EFE", "payload": "00", "adr": 1,
           "latitude": 51.20856475830078, "counter": 31952, "longitude": 4.400575637817383,
           "airtime": 0.112896,
           "gateways": [{"rssi": -115, "esp": -115.832695, "snr": 6.75,
                         "rx_time": {"ts_type": "None", "time": "t"}, "id": "08060716"},
                        {"rssi": -116, "esp": -125.51497, "snr": -9.0,
                         "rx_time": {"ts_type": "GPS_RADIO", "time": "t"}, "id": "FF0178DF"}],
           "dev_eui": "3432333853376B18", "sf": 7, "channel": 8}
    msg2 = {**msg, "gateways": msg["gateways"][:1], "sf": 9}
    for layout in ("array", "lines"):
        f = tmp_path / f"{layout}.json.txt"
        f.write_text(json.dumps([msg, msg2]) if layout == "array"
                     else json.dumps(msg) + "\n" + json.dumps(msg2) + "\n")
        df = data.load_json(f)
        assert df.shape[0] == 2
        assert df.loc[0, "SNR_FF0178DF"] == -9.0
        assert df.loc[1, "RSSI_FF0178DF"] == -200  # not received -> placeholder
        assert list(df["SF"]) == [7, 9]
        rssi = [c for c in df.columns if c.startswith("RSSI_")]
        r, o = features.columns_for("FV16", df, rssi)
        assert len(r) == 2 and len(o) == 1 + 2 + 2 + 1 + 1  # SF, SNRx2, ESPx2, CNT, AT


def test_pipeline_end_to_end_synthetic():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(rng.integers(-127, -60, size=(n, 3)), columns=["BS 1", "BS 2", "BS 3"])
    df.iloc[::3, 0] = -200
    df["SF"] = rng.integers(7, 13, n)
    df["Latitude"] = 51.2 + rng.random(n) * 0.05
    df["Longitude"] = 4.4 + rng.random(n) * 0.1
    d = pp.prepare(df, ["BS 1", "BS 2", "BS 3"], ["SF"])
    assert d.X_train.shape[1] == 4
    assert d.X_train.min() >= 0 and d.X_train.max() <= 1
    np.testing.assert_allclose(d.pipeline.unscale_y(d.y_test), d.y_test_deg, atol=1e-5)
