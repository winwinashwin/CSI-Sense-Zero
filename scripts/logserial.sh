#!/bin/bash -e

# Defaults
DEVICE=/dev/ttyACM0
BAUDRATE=921600
LOGFILE="serial.csi"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--device)
        DEVICE="$2"
        shift
        shift
        ;;
        -b|--baudrate)
        BAUDRATE="$2"
        shift
        shift
        ;;
        -l|--logfile)
        LOGFILE="$2"
        shift
        shift
        ;;
        *)    # unknown option
        shift
        ;;
    esac
done

# Updated values
echo
echo "[*] Serial device: $DEVICE"
echo "[*] Baud rate    : $BAUDRATE"
echo "[*] Log file     : $LOGFILE"
echo

for i in {10..1}; do
  echo "Starting recording in $i"
  sleep 1
done

exec picocom -b "$baud" -g "$logfile" "$device"
