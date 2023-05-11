#!/usr/bin/env python3

"""Parses raw *.csi logs and generates a MATLAB style .mat file that could conveniently be loaded for data analysis and model training.

    NOTE: All parameters should be configured in main function before calling script.
    Example usage: ./genmat.py
"""

import sys

sys.path.insert(0, ".")

import yaml
import json
import pathlib
from functools import lru_cache

import pandas as pd
import numpy as np
import scipy.io

from HAR.constants import CSI_COL_NAMES, NULL_SUBCARRIERS


# Seeding for reproducible results, remove if not required
rng = np.random.default_rng(seed=42)


@lru_cache(maxsize=None)  # cache, read each file only once
def parse_raw_csi(infile):
    """Reads a CSV file containing raw CSI (Channel State Information) data and returns a NumPy
    array of CSI amplitudes, after processing the data.

    Args:
        infile (str): Path to the CSV file containing raw CSI data.

    Returns:
        numpy.ndarray: A NumPy array of CSI amplitudes with shape `(n_subcarriers, n_packets)`,
                       where `n_subcarriers` is the number of valid subcarriers in the CSI data
                       (after removing invalid subcarriers), and `n_packets` is the number of
                       HT packets in the CSI data.
    """
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

    # only interested in HT packets
    df = df.loc[df["sig_mode"] == 1]

    # convert array in string format to numpy.ndarray
    csi_raw = df["data"].copy()
    csi_data = np.array(
        [
            np.fromstring(csi_record.strip("[]"), dtype=int, sep=",")
            for csi_record in csi_raw
        ]
    )

    # remove invalid subcarriers
    csi_data = np.delete(csi_data.T, NULL_SUBCARRIERS, 0).T
    # compute csi amplitude
    csi_amp = np.array([np.sqrt(data[::2] ** 2 + data[1::2] ** 2) for data in csi_data])

    return csi_amp.T


class GenMat:
    DAQ_HZ = 100

    def __init__(self, winsize=256, max_samples_per_class=-1) -> None:
        self.winsize = winsize
        self.max_samples_per_class = max_samples_per_class

        self.classnames = []
        self.nsamples = []

        self.X = None
        self.dim = None

    def add_class(self, classname, sources):
        cmat = None  # holds final data for entire class accumulated over all sources
        for infile, strip_sec_begin, strip_sec_end in sources:
            amp = parse_raw_csi(infile)
            n_sc, duration = amp.shape

            # indices of samples of interest
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
                # add data from this source to class
                cmat = np.hstack((cmat, mat))

        rng.shuffle(cmat.T)
        if self.max_samples_per_class > 0:
            cmat = cmat[:, : self.max_samples_per_class]

        if self.X is None:
            self.X = cmat
        else:
            self.X = np.hstack((self.X, cmat))

        self.nsamples.append(cmat.shape[1])
        self.classnames.append(classname)

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
        print(f"[+] Created {outfile}")

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


def main(args):
    with open(args.recipe, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = pathlib.Path(cfg["data_dir"])
    dest_dir = pathlib.Path(cfg["dest_dir"])

    for target, recipe in cfg["targets"].items():
        target = dest_dir / target

        g = GenMat(
            winsize=recipe["winsize"],
            max_samples_per_class=recipe["max_samples_per_class"],
        )

        for clsname, sources in recipe["classes"].items():
            sources = list(
                map(lambda src: [data_dir / src[0], src[1], src[2]], sources)
            )
            g.add_class(clsname, sources=sources)

        print("-" * 20)
        print(f"Summary: {target}")
        print()
        g.summary()

        if args.dry_run:
            continue

        g.dump(target)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSI Dataset generator")

    parser.add_argument(
        "--recipe", help="Path to recipe file (YAML)", required=True, type=str
    )

    parser.add_argument(
        "--dry-run",
        help="Dry run, do not generate dataset but just show final summary",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("Terminated")
        pass
