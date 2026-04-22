# teachable machine model + onnx + open CV
# the model (.tflite) is converted to .onnx on computer before using
import cv2
import numpy as np
import time

MODEL_PATH = "model.onnx"
CAMERA_ID = 4  # /dev/video4

# ===== Load ONNX Model =====
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print("✅ ONNX model loaded")

# ===== Class Names =====
class_names = ["background", "stop", "go", "turn", "slow", "person"]

# ===== Camera Setup =====
cap = cv2.VideoCapture(CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera started")

# ===== Model Input Size =====
# Change this if your model uses different size (e.g., 224 or 192)
INPUT_WIDTH = 224
INPUT_HEIGHT = 224

# ===== RC Car Control Functions (EDIT THESE) =====
def car_stop():
    print("🛑 STOP")

def car_go():
    print("🚗 GO")

def car_turn():
    print("↩ TURN")

def car_slow():
    print("🐢 SLOW")

def car_person():
    print("🚨 PERSON DETECTED - EMERGENCY STOP")

# ===== Main Loop =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # Change HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Run inference
    net.setInput(img)
    output = net.forward()

    output = output[0]
    class_id = np.argmax(output)
    confidence = output[class_id]

    label = class_names[class_id]

    # Show result
    cv2.putText(frame, f"{label} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("RC Car AI", frame)

    # ===== Decision Logic =====
    if confidence > 0.7:
        if label == "stop":
            car_stop()
        elif label == "go":
            car_go()
        elif label == "turn":
            car_turn()
        elif label == "slow":
            car_slow()
        elif label == "person":
            car_person()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
