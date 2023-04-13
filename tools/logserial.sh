#!/bin/bash

# Set default values
device="/dev/ttyUSB0"
baud="921600"
logfile="serial.csi"

# Parse command-line arguments
while getopts ":p:b:l:h" opt; do
  case "$opt" in
    p) device="$OPTARG" ;;
    b) baud="$OPTARG" ;;
    l) logfile="$OPTARG" ;;
    h|*) 
       echo "Usage: $(basename "$0") [-p port] [-b baud] [-l logfile]"
       exit 1 
       ;;
  esac
done
shift $((OPTIND-1))

# Output configuration settings
echo "[*] Serial device: $device"
echo "[*] Baud rate: $baud"
echo "[*] Log file: $logfile"

# Countdown before starting
for i in {10..1}; do
  echo "Starting recording in $i"
  sleep 1
done

# Start the recording
exec picocom -b "$baud" -g "$logfile" "$device"
