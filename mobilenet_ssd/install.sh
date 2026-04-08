#!/bin/bash

# Define the download URL and output filename
URL="https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
FILENAME="MobileNetSSD_deploy.prototxt"

# 1. Download the file
echo "Downloading package..."
wget "$FILENAME" -O "$URL"
wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel -O MobileNetSSD_deploy.caffemodel
# 2. Extract the package
#echo "Extracting files..."
#tar -xzf "$FILENAME"

# 3. Run installation commands (e.g., move files or run a setup script)
echo "Installing..."
# Add your specific install commands here, for example:
# sudo cp -r ./extracted-folder/* /usr/local/bin/

# 4. Cleanup
#rm "$FILENAME"
echo "Installation of prototxt and caffemodel complete!"
