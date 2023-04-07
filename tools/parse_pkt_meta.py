#!/usr/bin/env python3

"""Parse raw csi logs (*.csi) and prints packet metadata of all packets in input file.

    Example usage: ./parse_pkt_meta.py dataset/raw/jump-3.csi 
"""

import pandas as pd
import sys


CSI_COL_NAMES = [
    "type",
    "id",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data",
]

# Ref: [1] https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_wifi.html#_CPPv418wifi_pkt_rx_ctrl_t
# Ref: [2] https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#wi-fi-channel-state-information

sch_map = {
    0: "none",
    1: "above",
    2: "below",
}

sig_mode_map = {
    0: "non HT(11bg) packet",
    1: "HT(11n) packet",
    3: "VHT(11ac) packet",
}

cwb_map = {
    0: "20 MHz",
    1: "40 MHz",
}

stbc_map = {
    0: "non STBC packet",
    1: "STBC packet",
}


if __name__ == "__main__":
    filename = sys.argv[1]

    print(f"IN file: {filename}")
    csi_raw = pd.read_csv(filename, header=None, names=CSI_COL_NAMES)

    for sch in csi_raw["secondary_channel"].unique():
        print(f"Detected secondary channel: \t\t{sch_map[sch]}")

    for sig_mode in csi_raw["sig_mode"].unique():
        print(f"Detected signal mode: \t\t{sig_mode_map[sig_mode]}")

    for bw in csi_raw["bandwidth"].unique():
        print(f"Detected channel bandwidth: \t\t{cwb_map[bw]}")

    for stbc in csi_raw["stbc"].unique():
        print(f"Detected STBC : \t\t{stbc_map[stbc]}")

    for fwi in csi_raw["first_word"].unique():
        print(f"Detected first word invalid: \t\t{fwi}")
