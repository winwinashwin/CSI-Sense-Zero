# ESP32 CSI toolkits

Adapted from [official source](https://github.com/espressif/esp-csi)

## Compiler Environment

The esp-idf version of the current project is [ESP-IDF Release v4.4.1](https://github.com/espressif/esp-idf/releases/tag/v4.4.1)

## Flashing hardware

```bash
# csi_send
cd esp-csi/examples/get-started/csi_send
idf.py set-target esp32
idf.py flash -b 921600 -p /dev/ttyUSB0 monitor

# csi_recv
cd esp-csi/examples/get-started/csi_recv
idf.py set-target esp32
idf.py flash -b 921600 -p /dev/ttyUSB1
```
