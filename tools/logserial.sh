#!/bin/bash

while getopts 'p:b:l:h' opt; do
    case "$opt" in
        p)
            device=$OPTARG
            ;;

        b)
            baud=$OPTARG
            ;;

        l)
            logfile=$OPTARG
            ;;

        ?|h)
            echo "Usage: $(basename $0) [-p port] [-b baud] [-l logfile]"
            exit 1
            ;;
    esac
done
shift "$(($OPTIND -1))"

echo "[*] Serial device: $device"
echo "[*] Baud rate : $baud"
echo "[*] Log: $logfile"

for i in {10..1}; do
    echo "Starting record in $i"; sleep 1
done

exec picocom -b $baud -g $logfile $device
