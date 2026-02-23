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
sudo apt update
sudo apt install -y python3-flask python3-opencv python3-pip python3-smbus i2c-tools
sudo apt install -y v4l-utils libcamera-apps
```

### 2) Setup Virtual Environment
```bash
sudo apt install -y python3-venv python3-dev
python3 -m venv car-venv
source car-venv/bin/activate
```
Later, to use again: source car-venv/bin/activate


### 3) Insetalling packages under the virtual environment
```bash
pip3 install smbus2
pip3 install Adafruit-PCA9685 adafruit-circuitpython-pca9685
pip install flask opencv-python
```

### 4) Configure I2C PCA9685 servo board (under venv)
sudo i2cdetect -y X  (replace X with your bus number)
```bash
ls -l /dev/i2c*
lsmod | grep i2c
sudo i2cdetect -y 0
```
 You should see 40 if your board is at address 0x40.

### 5) Camera (USB Webcam)
#### 5.1 check the camera device:
```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```
#### 5.2 change permission
check if you see video in the output:
```bash
id
groups
```
if not: 
```bash
sudo usermod -aG video $USER
sudo usermod -aG render $USER
# IMPORTANT: log out and log back in (or reboot)
```
confirm the device node permissions/ACL:
```bash
id
v4l2-ctl --list-devices
getfacl /dev/video0
```
#### 5.3 quick check
take a photo:
```bash
ffmpeg -f video4linux2 -i /dev/video4 -frames:v 1 -y frame.jpg
ls -lh frame.jpg
```
take a video:
```bash
ffmpeg -f video4linux2 -i /dev/video4 -frames:v 1 -y frame.jpg
ls -lh frame.jpg
```
### (ribbon camera IMX219)
```bash
cam -l
libcamera-hello --list-cameras
libcamera-hello -t 0
libcamera-hello -o test.jpg
```

## Step 3: PWM value
- to check the motor and servo of the car
- find the range of throttle and steering
```bash
python3 pwm.py
```

## Step 4: fpv driving
### 1) run the program on the board
```bash
python app.py
```
### 2) connect a phone/tablet to the same network as the board and go to
```bash
http://<board-ip>:5000
```
### 1) run the program on the board
```bash
python3 vf2_web_drive_usbcam.py
```
### 2) connect a phone/tablet to the same network as the board and go to
```bash
http://<board-ip>:8000
```
