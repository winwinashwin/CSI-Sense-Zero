#!/bin/bash -e

wget https://github.com/winwinashwin/CSI-Sense-Zero/releases/download/dataset%2Fv1.0/dataset.tar.gz
wget https://github.com/winwinashwin/CSI-Sense-Zero/releases/download/dataset%2Fv1.0/dataset.tar.gz.sha256sum
sha256sum -c dataset.tar.gz.sha256sum

tar -xvzf dataset.tar.gz
