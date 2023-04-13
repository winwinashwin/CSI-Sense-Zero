# CSI-Sense-Zero

## Overview

CSI-Sense-Zero offers a realtime human activity recognition solution capable of running on a Raspberry Pi Zero using Wi-Fi CSI data acquired from an ESP32 Wi-Fi module/devkit.

A large pool of randomly initialized kernels are used to extract features from each valid subcarrier. The extracted features are then used for activity classification using a simple ridge regression classifier followed by majority voting.

Approach based on papers [LiteHAR](https://arxiv.org/pdf/2201.09310.pdf) and [Rocket](https://arxiv.org/pdf/1910.13051.pdf).

## Get Started

### Download dataset

```bash
./tools/download_dataset.sh
```

### Install runtime requirements

```bash
pip3 install -r requires/runtime.txt
```

### Train parameters

```bash
python3 train.py --main-set ./dataset/rCSI-5.mat --hold-set ./dataset/rCSI-3.mat --train-size 0.8 --dump artifacts/v1
```

### Run HAR realtime

- Log CSI data to FIFO

> The following command creates a log file `/tmp/csififo` with 235k buffer (holds ~256 CSI records) with file permissions 0644, owned by a user with UID==1000, reads serial device `/dev/ttyUSB0` at baud 921600 and populates the log file

```bash
./tools/populate_csififo.sh -d /dev/ttyUSB0 -b 921600 -n /tmp/csififo -s 235 -p 0644 -u 1000
```

- Run HAR

> Loads parameters from `artifacts/v1` and broadcasts predictions using a websocket server serving at 127.0.0.1:9999 at a frequency of 2 Hz

```bash
python3 main.py --load artifacts/v1 --host 127.0.0.1 --port 9999 --frequency 2
```
