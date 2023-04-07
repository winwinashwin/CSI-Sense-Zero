#!/usr/bin/env python3

"""Parses raw *.csi logs and generates a MATLAB style .mat file that could conveniently be loaded for data analysis and model training.

    NOTE: All parameters should be configured in main function before calling script.
    Example usage: ./genmat.py
"""

import scipy.io
import pandas as pd
import numpy as np
from skimage.restoration import denoise_wavelet
import json

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

NULL_SUBCARRIERS = list(range(0, 10)) + list(range(118, 256))


def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))


def hampel(data, half_win_length=10, n=3):
    windows = np.lib.stride_tricks.sliding_window_view(data, 2 * half_win_length)
    k = 1.4826
    median = np.median(windows, axis=1)
    median = np.concatenate(
        [[median[0]] * (half_win_length), median, [median[-1]] * (half_win_length - 1)]
    )
    sigma = k * np.apply_along_axis(median_absolute_deviation, axis=1, arr=windows)
    sigma = np.concatenate(
        [[sigma[0]] * (half_win_length), sigma, [sigma[-1]] * (half_win_length - 1)]
    )

    outliers = np.array(np.where(np.abs(data - median) >= (n * sigma))).flatten()

    data[outliers] = median[outliers]
    return data


class GenMat:
    """Parse multiple raw CSI capture files, strip all unwanted metadata and prepare a MATLAB style .mat file to be used as dataset."""

    DAQ_HZ = 100

    def __init__(self, preprocess=False, winsize=256, max_samples_per_class=-1):
        self.preprocess = preprocess
        self.winsize = winsize
        self.max_samples_per_class = max_samples_per_class

        self.classnames = []
        self.nsamples = []

        self.X = None
        self.dim = None

    def add_class(self, classname, sources):
        cmat = None
        for infile, strip_sec_begin, strip_sec_end in sources:
            amp = self._parse_csi(infile)
            n_sc, duration = amp.shape

            if self.preprocess:
                print("Preprocessing data. This may take a while...")
                amp = self._preprocess_csi(amp)

            isamples = range(
                self.DAQ_HZ * strip_sec_begin,
                duration - self.DAQ_HZ * strip_sec_end - self.winsize,
                self.winsize,
            )

            mat = np.zeros((n_sc * self.winsize, len(isamples)))

            for i, idx in enumerate(isamples):
                mat[:, i] = amp[:, idx : idx + self.winsize].reshape(-1)

            self.dim = (n_sc, self.winsize)

            if cmat is None:
                cmat = mat
            else:
                cmat = np.hstack((cmat, mat))

        np.random.shuffle(cmat.T)
        if self.max_samples_per_class > 0:
            cmat = cmat[:, : self.max_samples_per_class]

        if self.X is None:
            self.X = cmat
        else:
            self.X = np.hstack((self.X, cmat))

        self.nsamples.append(cmat.shape[1])
        self.classnames.append(classname)

        print("-" * 10)

    def dump(self, outfile):
        scipy.io.savemat(
            outfile,
            {
                "dim": self.dim,
                "nsamples": self.nsamples,
                "classnames": self.classnames,
                "csi": self.X,
            },
        )

    def summary(self):
        print(
            json.dumps(
                {
                    "dim": self.dim,
                    "nsamples": self.nsamples,
                    "classnames": self.classnames,
                    "X.shape": self.X.shape,
                },
                indent=4,
            )
        )

    def _preprocess_csi(self, csi):
        csi = np.apply_along_axis(hampel, arr=csi, axis=1)
        csi = np.apply_along_axis(
            lambda s: denoise_wavelet(
                s,
                method="BayesShrink",
                mode="soft",
                wavelet_levels=1,
                wavelet="sym4",
                rescale_sigma=True,
            ),
            arr=csi,
            axis=1,
        )
        return csi

    def _parse_csi(self, infile):
        print(f"[+] Reading file: {infile}")
        df = pd.read_csv(infile, header=None, names=CSI_COL_NAMES)

        # Ref: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_wifi.html?highlight=rx_ctrl#_CPPv4N18wifi_pkt_rx_ctrl_t8sig_modeE
        n_nonHT = df.loc[df["sig_mode"] == 0].size
        n_HT = df.loc[df["sig_mode"] == 1].size
        n_VHT = df.loc[df["sig_mode"] == 3].size

        print(
            f"Found {n_HT} HT pkts, {n_nonHT} non HT pkts (Dropped), {n_VHT} VHT pkts (Dropped)"
        )

        fs = (1e6 / df["local_timestamp"].diff()[1:]).mean()
        print(f"CSI DAQ frequency: {fs:0.2f} Hz")

        df = df.loc[df["sig_mode"] == 1]

        csi_raw = df["data"].copy()
        csi_data = np.array(
            [
                np.fromstring(csi_record.strip("[]"), dtype=int, sep=",")
                for csi_record in csi_raw
            ]
        )

        csi_data = np.delete(csi_data.T, NULL_SUBCARRIERS, 0).T
        csi_amp = np.array(
            [np.sqrt(data[::2] ** 2 + data[1::2] ** 2) for data in csi_data]
        )

        return csi_amp.T


if __name__ == "__main__":
    g = GenMat(preprocess=False, max_samples_per_class=100, winsize=256)

    g.add_class(
        "empty",
        sources=[
            ("dataset/raw/empty-1.csi", 10, 10),
            # ("dataset/raw/empty-2.csi", 10, 10),
            # ("dataset/raw/empty-3.csi", 5, 5),
            # ("dataset/raw/empty-4.csi", 3, 3),
            # ("dataset/raw/empty-5.csi", 3, 3),
            # ("dataset/raw/empty-6.csi", 3, 3),
        ],
    )

    # g.add_class(
    #     "idle",
    #     sources=[
    #         # ("dataset/raw/idle-1.csi", 10, 10),
    #         # ("dataset/raw/idle-2.csi", 10, 10),
    # ("dataset/raw/idle-3.csi", 5, 5),
    #         # ("dataset/raw/idle-4.csi", 3, 3),
    #     ],
    # )

    g.add_class(
        "walk",
        sources=[
            ("dataset/raw/walk-1.csi", 10, 10),
            # ("dataset/raw/walk-2.csi", 5, 5),
            # ("dataset/raw/walk-3.csi", 5, 5),
            # ("dataset/raw/walk-4.csi", 3, 3),
            # ("dataset/raw/walk-5.csi", 3, 3),
        ],
    )

    g.add_class(
        "jump",
        sources=[
            ("dataset/raw/jump-1.csi", 40, 40),
            # ("dataset/raw/jump-2.csi", 40, 40),
            # ("dataset/raw/jump-3.csi", 40, 40),
            # ("dataset/raw/jump-4.csi", 5, 5),
            # ("dataset/raw/jump-5.csi", 5, 5),
            # ("dataset/raw/jump-6.csi", 5, 5),
            # ("dataset/raw/jump-7.csi", 5, 5),
            # ("dataset/raw/jump-8.csi", 5, 5),
            # ("dataset/raw/jump-9.csi", 3, 3),
            # ("dataset/raw/jump-10.csi", 3, 3),
            # ("dataset/raw/jump-11.csi", 3, 3),
            # ("dataset/raw/jump-12.csi", 3, 3),
        ],
    )

    g.summary()

    g.dump("dataset/rCSI-100-same-env.mat")
