#!/bin/bash -e

# Program serial device
# 921600 baud rate, 8 data bits, 1 stop bit, no parity bit
stty -F /dev/serial0 921600 cs8 -cstopb -parenb

# Validate presence of character special
if [[ ! -c /tmp/csififo ]]; then
        mkemlog /tmp/csififo 235 0644
fi

exec cat < /dev/serial0 | awk NF > /tmp/csififo

