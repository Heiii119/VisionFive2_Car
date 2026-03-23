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
sudo apt install -y ffmpeg
```

### 2) Setup Virtual Environment
```bash
sudo apt install -y python3-venv python3-dev
python3 -m venv --system-site-packages car-venv
source car-venv/bin/activate
```
Later, to use again: source car-venv/bin/activate


### 3) Insetalling packages under the virtual environment
```bash
python -m pip install smbus2 Adafruit-PCA9685 adafruit-circuitpython-pca9685
python -m pip install --upgrade pip
python -m pip install flask flask-socketio eventlet
sudo apt install -y python3-opencv
```
test if OpenCV is installed correctly:
```bash
python3 -c "import cv2; print('cv2', cv2.__version__)"
```

### 4) Configure I2C PCA9685 servo board (under venv)
sudo i2cdetect -y X  (replace X with your bus number)
```bash
ls -l /dev/i2c*
lsmod | grep i2c
sudo i2cdetect -y 0
```
 You should see 40 if your board is at address 0x40.



## Step 3: PWM value
- to check the motor and servo of the car
- find the range of throttle and steering
```bash
python3 pwm.py
```

## Step 4: camera check
### 4.1 quick check
#### 1) check the camera device: Camera (USB Webcam)
```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```
#### 2) check permission
check if you see video in the output:
```bash
id
groups
```
if not, change permission: 
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
#### 3) quick check
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
#### (ribbon camera IMX219)
```bash
cam -l
libcamera-hello --list-cameras
libcamera-hello -t 0
libcamera-hello -o test.jpg
```
#### 4) quick check with preview window
```bash
sudo apt-get install cheese
cheese
```
#### 4) quick check with preview window from V4L2 devices
```bash
sudo apt-get install mpv
mpv /dev/video4
```

### 4.2 check live streaming 
#### 0) confirm packages
```bash
python -c "import flask; print(flask.__version__)"
```
#### 1) run the program on the board
```bash
python app_ffmpeg_multipart.py
```
#### 2) connect a phone/tablet to the same network as the board and go to
```bash
http://<board-ip>:5000
```
## Step 5: fpv driving

## Step 6: AI module
### 6.0 Preperation: https://doc.rvspace.org/VisionFive2/Application_Notes/AI_Kit/VisionFive_2/complie_ai.html
Admendment before compilation:
1. 在tappas/core/requirements/gstreamer_requirements.txt 文件中把pandas 版本由1.5.2 改为：2.3.3
2. 以普通用户user 编译安装Tappas, 命令：$ ./install.sh --skip-hailort --target-platform vf2

### 6.1 start environment 
```bash
source /home/user/.hailo/tappas/tappas_env
```
### 6.2 start 
```bash
su -
```
