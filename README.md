# VisionFive2_Car
AI RC car on VisionFive 2 Lite
- github: https://github.com/starfive-tech/VisionFive2
- guild: https://doc-en.rvspace.org/VisionFive2Lite/VisionFive2LiteQSG/
#### Installing the StarFive Debian image: 
https://doc-en.rvspace.org/VisionFive2/Quick_Start_Guide/VisionFive2_QSG/flashing_with_mac_linux.html
- image file: https://debian.starfivetech.com/
#### Installing the StarFive Ubuntu image:
https://canonical-ubuntu-hardware-support.readthedocs-hosted.com/boards/how-to/starfive-visionfive-2/

## Step 1: Flash the VisionFive 2 Image
1. Download the deepin-visionfive-v2.img.zst https://cdimage.deepin.com/RISC-V/VisionFive-v2-image/
2. Flash it into a SD card with at least 16 GB with BelenaEtcher.
3. Insert the SD card to the VisionFive 2 Lite board and connect to bettary.
4. Username: user; Password: starfive

## Step 2: ssh connection
### 1) Enable ssh
https://doc-en.rvspace.org/VisionFive2/Quick_Start_Guide/VisionFive2_QSG/enable_ssh_root_login.html
```bash
$ arp -a //to find the ip address
$ echo 'PermitRootLogin=yes'  | sudo tee -a /etc/ssh/sshd_config 
```
### 2) SSH connection
```bash
$ ssh user@<ip_addr>  
```
pw: starfive

#### (opt) Connection with ssh or UART Serial Port
https://doc-en.rvspace.org/VisionFive2/Quick_Start_Guide/VisionFive2_QSGLite/logging_into_distro.html#logging_into_distro__section_v2l_jfw_mhc

## Step 3: Packages Installlation
### 1) Installing basic packages: python3, i2c, libcamera
```bash
ls /dev/i2c-*
sudo apt update
sudo apt install -y python3-pip python3-smbus i2c-tools
sudo apt install -y v4l-utils libcamera-apps
```

### 2) Setup Virtual Environment
```bash
sudo apt install -y python3-venv python3-dev
python3 -m venv tflite-env
source tflite-env/bin/activate
```
Later, to use again: source tflite-env/bin/activate


### 3) Insetalling packages under the virtual environment
```bash
pip3 install --user smbus2
pip3 install Adafruit-PCA9685
pip3 install adafruit-circuitpython-pca9685
```

### 4) Configure I2C PCA9685 servo board (under venv)
sudo i2cdetect -y X  (replace X with your bus number)
```bash
sudo i2cdetect -y 0
sudo i2cdetect -y 1
```
 You should see 40 if your board is at address 0x40.

### 5) Camera
```bash
cam -l
```
```bash
libcamera-hello --list-cameras
libcamera-hello -t 0
libcamera-hello -o test.jpg
```
