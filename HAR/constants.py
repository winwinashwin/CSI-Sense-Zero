# Header names for CSI data acquired from ESP32
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

# Indices of subcarriers (actually indices of bytes, each subcarrier data is registered as two
# signed characters - first imaginary and the second real) to remove - null + HTLTF
NULL_SUBCARRIERS = list(range(0, 10)) + list(range(118, 256))
