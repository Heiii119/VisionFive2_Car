import cv2
import numpy as np

MODEL_PATH = "best.onnx"
IMG_SIZE = 320

print("Loading model...")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print("Model loaded successfully ✅")

# Create dummy image
dummy = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

# Preprocess
blob = cv2.dnn.blobFromImage(dummy, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=True)

net.setInput(blob)

print("Running forward pass...")
output = net.forward()

print("Forward pass successful ✅")
print("Output shape:", output.shape)
