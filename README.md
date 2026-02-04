# VisionFive2_Car
AI RC car on VisionFive 2 Lite

## FLashing the VisionFive 2 Lite
Method 1: Flashing SD Card/eMMC (Recommended)
This is the standard method for installing the OS.
Download the Image: Download the latest SD card image (.img or .img.bz2 format) from the StarFive Tech GitHub releases.
Install BalenaEtcher: Download and install BalenaEtcher on your Mac.
Insert Card: Insert your Micro-SD card into your Mac using a reader.
Flash the Image:
Open BalenaEtcher.
Click Flash from file and select the downloaded .img file.
Click Select target and choose your Micro-SD card.
Click Flash!.
Finalize: Once finished, macOS may display a warning that the disk is unreadable (this is normal for Linux partitions). Remove the card and insert it into the VisionFive 2.

### Install StarFive Image
To export GPG keys on macOS using Homebrew, replace apt-key functionality with the gpg command. Install it via brew install gnupg, then use gpg --export --armor <key_id> to export keys, or gpg --keyserver hkps://keys.openpgp.org --recv-keys <key_id> to import them, bypassing the deprecated apt-key. 

Steps to Manage GPG Keys with Homebrew (macOS):
Install GPG: If not already installed, run:
brew install gnupg

List Existing Keys: Identify the key you want to export:
gpg --list-keys

Export a Key (Replacement for apt-key export):
gpg --armor --export <KEY_ID_OR_EMAIL> > my-key.pub.asc

Import a Key (Replacement for apt-key add):
gpg --import <key_file>.asc

