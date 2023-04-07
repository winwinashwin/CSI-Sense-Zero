#!/bin/bash -e

wget https://github.com/winwinashwin/CSI-Sense-Zero/releases/download/dataset-release-tag/csi-sense-dataset.tar.gz
wget https://github.com/winwinashwin/CSI-Sense-Zero/releases/download/dataset-release-tag/csi-sense-dataset.tar.gz.sha256sum
sha256sum -c csi-sense-dataset.tar.gz.sha256sum

tar -xvzf csi-sense-dataset.tar.gz
