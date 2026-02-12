# VisionFive2_Car
AI RC car on VisionFive 2 Lite
- github: https://github.com/starfive-tech/VisionFive2
- guild: https://doc-en.rvspace.org/VisionFive2Lite/VisionFive2LiteQSG/
#### Installing the StarFive Debian image: 
https://doc-en.rvspace.org/VisionFive2/Quick_Start_Guide/VisionFive2_QSG/flashing_with_mac_linux.html
- image file: https://debian.starfivetech.com/
####Installing the StarFive Ubuntu image:
https://canonical-ubuntu-hardware-support.readthedocs-hosted.com/boards/how-to/starfive-visionfive-2/

### Flash the VisionFive 2 Image
1. Download the deepin-visionfive-v2.img.zst https://cdimage.deepin.com/RISC-V/VisionFive-v2-image/
2. Flash it into a SD card with at least 16 GB with BelenaEtcher.
3. Insert the SD card to the VisionFive 2 Lite board and connect to bettary.
4. Username: user; Password: starfive


### Enable ssh
https://doc-en.rvspace.org/VisionFive2/Quick_Start_Guide/VisionFive2_QSG/enable_ssh_root_login.html
```bash
$ arp -a //to find the ip address
$ echo 'PermitRootLogin=yes'  | sudo tee -a /etc/ssh/sshd_config //to enable ssh
$ ssh user@<ip_addr> //pw: starfive
```

#### Connection with ssh or UART Serial Port
https://doc-en.rvspace.org/VisionFive2/Quick_Start_Guide/VisionFive2_QSGLite/logging_into_distro.html#logging_into_distro__section_v2l_jfw_mhc

### Installing packages 
```bash
sudo apt install -y libcamera-apps
```
### Camera
```bash
cam -l
```
```bash
libcamera-hello --list-cameras
libcamera-hello -t 0
libcamera-hello -o test.jpg
```
