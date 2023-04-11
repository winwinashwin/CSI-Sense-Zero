# CSI-Sense-Zero

## Overview

CSI-Sense-Zero offers a realtime human activity recognition solution capable of running on a Raspberry Pi Zero using Wi-Fi CSI data acquired from an ESP32 Wi-Fi module/devkit.

## Get Started

### Download dataset

```bash
./download_dataset.sh
```

### Train parameters

```bash
python3 train.py --main-set ./dataset/rCSI-5.mat --hold-set ./dataset/rCSI-3.mat --train-size 0.8 --dump artifacts/v1
```

### Run HAR realtime

- Log CSI data to FIFO

```bash
./populate_csififo.sh -d /dev/ttyUSB0 -b 921600 -n /tmp/csififo -s 235 -p 0644 -u 1000
```

- Run HAR

```bash
python3 main.py --load artifacts/v1 --host 127.0.0.1 --port 9999 --frequency 2
```
