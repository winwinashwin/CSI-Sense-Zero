#!/bin/bash -e

DEVICE=/dev/ttyACM0
BAUDRATE=921600

CSIFIFO_NAME=/tmp/csififo
CSIFIFO_BUFSIZ=235  # 235k buffer
CSIFIFO_PERM=0644   # file permission 0644 - owner (read + write), group (read only), others (read only)
CSIFIFO_USER=1000   # set user with UID==1000 as owner

if grep --quiet 'emlog' /proc/modules; then
    echo "[+] Kernel module loaded"
else
    echo "[-] Kernel module not detected. Loading..."
    sudo modprobe emlog
fi

if [[ -c $CSIFIFO_NAME ]]; then
    echo "[+] Detected FIFO $CSIFIFO"
else
    echo "[-] Log device not found. Creating..."
    sudo mkemlog $CSIFIFO_NAME $CSIFIFO_BUFSIZ $CSIFIFO_PERM $CSIFIFO_USER
fi

echo "[+] Programming serial port $DEVICE"
stty -F $DEVICE $BAUDRATE cs8 -cstopb -parenb

echo "[+] Populating FIFO"
exec cat < $DEVICE | awk NF > $CSIFIFO_NAME
