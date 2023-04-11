#!/bin/bash -e

# Set default values
DEVICE=/dev/ttyACM0
BAUDRATE=921600
CSIFIFO_NAME=/tmp/csififo
CSIFIFO_BUFSIZ=235
CSIFIFO_PERM=0644
CSIFIFO_USER=1000

# Parse command-line arguments
while [[ $# -gt 0 ]]
do
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
    -n|--csififo-name)
    CSIFIFO_NAME="$2"
    shift
    shift
    ;;
    -s|--csififo-bufsize)
    CSIFIFO_BUFSIZ="$2"
    shift
    shift
    ;;
    -p|--csififo-perm)
    CSIFIFO_PERM="$2"
    shift
    shift
    ;;
    -u|--csififo-user)
    CSIFIFO_USER="$2"
    shift
    shift
    ;;
    *)    # unknown option
    shift
    ;;
esac
done

# Use variables with the updated values
echo
echo "[*] DEVICE=$DEVICE"
echo "[*] BAUDRATE=$BAUDRATE"
echo "[*] CSIFIFO_NAME=$CSIFIFO_NAME"
echo "[*] CSIFIFO_BUFSIZ=$CSIFIFO_BUFSIZ"
echo "[*] CSIFIFO_PERM=$CSIFIFO_PERM"
echo "[*] CSIFIFO_USER=$CSIFIFO_USER"
echo

# --- 

if grep --quiet 'emlog' /proc/modules; then
    echo "[+] Kernel module loaded"
else
    echo "[-] Kernel module not detected. Loading..."
    sudo modprobe emlog
fi

if [[ -c $CSIFIFO_NAME ]]; then
    echo "[+] Detected FIFO $CSIFIFO_NAME"
else
    echo "[-] Log device not found. Creating..."
    sudo mkemlog $CSIFIFO_NAME $CSIFIFO_BUFSIZ $CSIFIFO_PERM $CSIFIFO_USER
fi

echo "[+] Programming serial port $DEVICE"
stty -F $DEVICE $BAUDRATE cs8 -cstopb -parenb

echo "[+] Populating FIFO"
exec cat < $DEVICE | awk NF > $CSIFIFO_NAME
