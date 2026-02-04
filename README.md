# VisionFive2_Car
AI RC car on VisionFive 2 Lite

## FLashing the VisionFive 2 Lite
Method 1: Flashing SD Card/eMMC (Recommended) This is the standard method for installing the OS.
1. Download the Image: Download the latest SD card image (.img or .img.bz2 format) from the StarFive Tech GitHub releases. https://github.com/starfive-tech/VisionFive2
```bash
$ git clone https://github.com/starfive-tech/VisionFive2.git
$ cd VisionFive2
$ git checkout --track origin/JH7110_VisionFive2_devel
$ git submodule update --init --recursive
$ cd linux && git branch JH7110_VisionFive2_devel origin/JH7110_VisionFive2_devel && cd ..

$ cd buildroot && git checkout --track origin/JH7110_VisionFive2_devel && cd ..
$ cd u-boot && git checkout --track origin/JH7110_VisionFive2_devel && cd ..
$ cd linux && git checkout --track origin/JH7110_VisionFive2_devel && cd ..
$ cd opensbi && git checkout --track origin/JH7110_VisionFive2_devel && cd ..
$ cd soft_3rdpart && git checkout JH7110_VisionFive2_devel && cd ..
```
Quick Build Instructions
Below are the quick building for the initramfs image image.fit which could be translated to board through tftp and run on board. The completed toolchain, u-boot-spl.bin.normal.out, visionfive2_fw_payload.img, image.fit will be generated under work/ directory. The completed build tree will consume about 18G of disk space.
```bash
make -j$(nproc)
```
2. Install BalenaEtcher: Download and install BalenaEtcher on your Mac.
3. Insert Card: Insert your Micro-SD card into your Mac using a reader.
4. Flash the Image:
- Open BalenaEtcher.
- Click Flash from file and select the downloaded .img file.
- Click Select target and choose your Micro-SD card.
- Click Flash!.
5. Finalize: Once finished, macOS may display a warning that the disk is unreadable (this is normal for Linux partitions). Remove the card and insert it into the VisionFive 2.

### Install StarFive Image
To export GPG keys on macOS using Homebrew, replace apt-key functionality with the gpg command. Install it via brew install gnupg, then use gpg --export --armor <key_id> to export keys, or gpg --keyserver hkps://keys.openpgp.org --recv-keys <key_id> to import them, bypassing the deprecated apt-key. 

Steps to Manage GPG Keys with Homebrew (macOS):
Install GPG: If not already installed, run:
```bash
brew install gnupg
```
List Existing Keys: Identify the key you want to export:
```bash
gpg --list-keys
```
Export a Key (Replacement for apt-key export):
```bash
gpg --armor --export <KEY_ID_OR_EMAIL> > my-key.pub.asc
```
Import a Key (Replacement for apt-key add):
```bash
gpg --import <key_file>.asc
```

### Install gcc

```bash
brew update
brew upgrade
brew info gcc
brew install gcc
brew cleanup
```

