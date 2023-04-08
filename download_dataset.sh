#!/bin/bash -e

wget https://github.com/winwinashwin/CSI-Sense-Zero/releases/download/dataset-release-tag/dataset.tar.gz
wget https://github.com/winwinashwin/CSI-Sense-Zero/releases/download/dataset-release-tag/dataset.tar.gz.sha256sum
sha256sum -c dataset.tar.gz.sha256sum

tar -xvzf dataset.tar.gz
