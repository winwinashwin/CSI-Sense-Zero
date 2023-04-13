# CSI-Sense-Zero/RPi

Binaries, scripts and configuration files that go into the Raspberry Pi.

## Setting up the Pi

```bash
sudo apt update
sudo apt install --yes --no-install-recommends supervisor git raspberrypi-kernel-headers

sudo cp serialread.conf /etc/supervisor/conf.d

mkdir -p /home/pi/bin
cp serialread.sh /home/pi/bin/

cd emlog
make -j$(nproc)
sudo make install
sudo depmod

sudo echo emlog >> /etc/modules  # might need to run in root shell

sudo reboot
```
